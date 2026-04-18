import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
import sys
import os
import random

# 将战术根目录加入路径
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

try:
    from tactical_task import TacticalTask
    TACTICAL_TASK_AVAILABLE = True
except ImportError as e:
    TACTICAL_TASK_AVAILABLE = False
    logging.error(f"⚠️ TacticalTask 导入失败，无法使用旧版战术模板: {e}")

log = logging.getLogger(__name__)

class MockAgentWrapper:
    """代理单个 Agent，拦截并转换 ID"""
    def __init__(self, agent, virtual_id, real_id):
        self._agent = agent
        self.uid = virtual_id
        self.real_id = real_id
        
    def __getattr__(self, name):
        if name == 'uid':
            return self.uid
        return getattr(self._agent, name)

class MockDeadAgentWrapper:
    """代理死亡或不存在的 Agent"""
    def __init__(self, virtual_id, real_id):
        self.uid = virtual_id
        self.real_id = real_id
        self.is_alive = False
        self.num_missiles = 0
        
    def get_position(self):
        return [0.0, 0.0, -100.0]
        
    def get_velocity(self):
        return [0.0, 0.0, 0.0]
        
    def get_property_value(self, prop):
        return 0.0
        
    def __getattr__(self, name):
        return None

class MockEnv:
    """包装环境，向 2v2 战术模板隐藏 4v4 中的其它飞机，并注入融合数据，强制代理ID映射"""
    def __init__(self, original_env, mapping_dict):
        """
        mapping_dict: { virtual_id: real_id }
        例如: {'A0100': 'A0300', 'A0200': 'A0400', 'B0100': 'B0300', 'B0200': 'B0400'}
        """
        self._original = original_env
        self.mapping_dict = mapping_dict
        self.current_step = original_env.current_step
        self.time_interval = getattr(original_env, 'time_interval', 0.2)
        self._missile_filter_real_id = None
        
        # 🔥 反向映射: virtual_id -> real_id，方便日志输出真实ID
        self.real_id_map = dict(mapping_dict)  # {virtual: real}
        
        # 预先构建 agent wrapper 词典，避免每次调用时新生成
        self._agents = {}
        for virtual_id, real_id in mapping_dict.items():
            if real_id in self._original.agents:
                self._agents[virtual_id] = MockAgentWrapper(self._original.agents[real_id], virtual_id, real_id)
            else:
                # 🔥 修复：如果真实飞机已被清理或不存在，提供一个死亡的代理，防止 KeyError
                self._agents[virtual_id] = MockDeadAgentWrapper(virtual_id, real_id)
        
    @property
    def agents(self):
        # 仅暴露出映射后的飞机，键为 A0100 / A0200 格式
        return self._agents

    @property
    def _jsbsims(self):
        # 同样暴露映射后的飞机，防止底层代码直接访问真实的 _jsbsims 导致越界
        return self._agents

    def _collect_missile_target_ids(self, missile_sim) -> set[str]:
        target_ids = set()
        for attr_name in ('target_id', 'target_uid'):
            target_id = getattr(missile_sim, attr_name, None)
            if isinstance(target_id, str) and target_id:
                target_ids.add(target_id)

        target_aircraft = getattr(missile_sim, 'target_aircraft', None)
        if target_aircraft is not None:
            for attr_name in ('uid', 'real_id'):
                target_id = getattr(target_aircraft, attr_name, None)
                if isinstance(target_id, str) and target_id:
                    target_ids.add(target_id)

        for target_id in list(target_ids):
            resolved_virtual = self.resolve_virtual_agent_id(target_id)
            if isinstance(resolved_virtual, str) and resolved_virtual:
                target_ids.add(resolved_virtual)

        return target_ids

    @property
    def _tempsims(self):
        source = getattr(self._original, '_tempsims', {})
        if not self._missile_filter_real_id:
            return source

        filtered = {}
        virtual_filter_id = self.resolve_virtual_agent_id(self._missile_filter_real_id)
        for missile_id, missile_sim in source.items():
            target_ids = self._collect_missile_target_ids(missile_sim)
            if self._missile_filter_real_id in target_ids or virtual_filter_id in target_ids:
                filtered[missile_id] = missile_sim
        return filtered

    @property
    def task(self):
        return self._original.task

    def resolve_real_agent_id(self, agent_id: str) -> str:
        return self.real_id_map.get(agent_id, agent_id)

    def resolve_virtual_agent_id(self, real_agent_id: str) -> str:
        for virtual_id, mapped_real_id in self.real_id_map.items():
            if mapped_real_id == real_agent_id:
                return virtual_id
        return real_agent_id

    def __getattr__(self, name):
        return getattr(self._original, name)


class TacticalBridge:
    """
    战术桥接器: 将 4v4 任务拆分为2组 2v2 任务，代理调用旧版的 TacticalTask 进行机动判定。
    """
    def __init__(self):
        self.group_1_task: Optional['TacticalTask'] = None
        self.group_2_task: Optional['TacticalTask'] = None
        self.enemy_group_1_task: Optional['TacticalTask'] = None
        self.enemy_group_2_task: Optional['TacticalTask'] = None
        
        self.group_1_friendly = ['A0100', 'A0200']
        self.group_2_friendly = ['A0300', 'A0400']
        self.group_1_enemy = ['B0100', 'B0200']
        self.group_2_enemy = ['B0300', 'B0400']
        self.initialized = False
        self._enemy_random_states: Dict[str, object] = {}
        self._enemy_np_random_states: Dict[str, tuple] = {}
        
    def _sync_distances_to_task(self, task, cap_task_instance):
        """将 cap_task 的 ControlRanges (km) 同步到旧版 TacticalTask 的 tactical_distances (m)"""
        if task is None or cap_task_instance is None:
            return
        ranges = getattr(cap_task_instance, 'ranges', None)
        if ranges is None:
            return
        try:
            all_ranges = ranges.get_all()  # ControlRanges.get_all() 返回 km 级别的 dict
            if all_ranges and hasattr(task, 'tactical_distances'):
                for k, v_km in all_ranges.items():
                    if k in task.tactical_distances:
                        task.tactical_distances[k] = v_km * 1000.0  # km -> m
        except Exception:
            pass

    def _get_enemy_pair(self, friendly_agents) -> list[str]:
        if friendly_agents == self.group_1_friendly:
            return ['B0100', 'B0200']
        return ['B0300', 'B0400']

    def _get_friendly_pair(self, enemy_agents, env=None) -> list[str]:
        preferred = ['A0100', 'A0200'] if enemy_agents == self.group_1_enemy else ['A0300', 'A0400']
        if env is None:
            return list(preferred)

        other = ['A0300', 'A0400'] if preferred == ['A0100', 'A0200'] else ['A0100', 'A0200']
        alive_preferred = [
            aid for aid in preferred
            if aid in env.agents and getattr(env.agents[aid], 'is_alive', False)
        ]
        alive_other = [
            aid for aid in other
            if aid in env.agents and getattr(env.agents[aid], 'is_alive', False)
        ]
        selected = list(alive_preferred)
        for aid in alive_other:
            if aid not in selected:
                selected.append(aid)
        if not selected:
            selected = list(preferred)
        if len(selected) == 1:
            selected.append(selected[0])
        return selected[:2]

    def _build_enemy_mapping(self, friendly_agents, cap_task_instance: Any) -> Dict[str, str]:
        enemy_pair = self._get_enemy_pair(friendly_agents)
        prioritized_targets = []

        if hasattr(cap_task_instance, '_tactic_assignments_by_agent'):
            for fid in friendly_agents:
                assign = cap_task_instance._tactic_assignments_by_agent.get(fid)
                target_id = getattr(assign, 'target_id', None) if assign else None
                if target_id in enemy_pair and target_id not in prioritized_targets:
                    prioritized_targets.append(target_id)

        visible_targets = list(prioritized_targets)
        for enemy_id in enemy_pair:
            if enemy_id not in visible_targets:
                visible_targets.append(enemy_id)

        if not visible_targets:
            visible_targets = list(enemy_pair)
        elif len(visible_targets) == 1:
            visible_targets.append(visible_targets[0])

        return {
            'B0100': visible_targets[0],
            'B0200': visible_targets[1],
        }

    def _get_enemy_opening_tactic(self, enemy_agents) -> Optional[str]:
        common_override = os.environ.get('CAP_ENEMY_FORCE_TACTIC', '').strip().upper() or None
        if common_override:
            return common_override
        if enemy_agents == self.group_1_enemy:
            return os.environ.get('CAP_ENEMY_GROUP1_FORCE_TACTIC', 'DRAG_SHOOT').strip().upper() or 'DRAG_SHOOT'
        return os.environ.get('CAP_ENEMY_GROUP2_FORCE_TACTIC', 'FRONT_BACK').strip().upper() or 'FRONT_BACK'

    def _enemy_group_key(self, enemy_agents) -> str:
        return 'enemy_group_1' if enemy_agents == self.group_1_enemy else 'enemy_group_2'

    def _run_with_enemy_rng(self, group_key: str, fn):
        saved_random_state = random.getstate()
        saved_np_state = np.random.get_state()
        enemy_random_state = self._enemy_random_states.get(group_key)
        enemy_np_state = self._enemy_np_random_states.get(group_key)
        if enemy_random_state is None or enemy_np_state is None:
            seed = 4101 if group_key == 'enemy_group_1' else 4201
            random.seed(seed)
            np.random.seed(seed)
        else:
            random.setstate(enemy_random_state)
            np.random.set_state(enemy_np_state)
        try:
            return fn()
        finally:
            self._enemy_random_states[group_key] = random.getstate()
            self._enemy_np_random_states[group_key] = np.random.get_state()
            random.setstate(saved_random_state)
            np.random.set_state(saved_np_state)

    def _configure_enemy_task(self, task, enemy_agents) -> None:
        if task is None:
            return
        setattr(task, '_bridge_opening_force_tactic', self._get_enemy_opening_tactic(enemy_agents))
        selector = getattr(getattr(task, 'complete_tactical_system', None), 'tactical_selector', None)
        if selector is not None:
            setattr(selector, 'disable_randomization', True)

    def _prepare_enemy_task_for_step(self, task) -> None:
        if task is None:
            return
        opening_tactic = getattr(task, '_bridge_opening_force_tactic', None)
        if not opening_tactic:
            return
        decision_made = getattr(task, 'decision_made', {}) or {}
        opening_locked = bool(
            decision_made.get('NLT')
            or decision_made.get('MELD')
            or decision_made.get('MTR')
            or getattr(task, 'selected_tactic', None)
        )
        task.force_tactic = None if opening_locked else opening_tactic

    def initialize_tactical_tasks(self, config, env, cap_task_instance=None):
        if not TACTICAL_TASK_AVAILABLE:
            return
        # 初始化2个分开的战术任务，用于维护各自的战术状态、雷达距离状态等
        force_tactic = os.environ.get('CAP_FORCE_TACTIC', '').strip().upper() or None
        self.group_1_task = TacticalTask(config, force_tactic=force_tactic)
        self.group_2_task = TacticalTask(config, force_tactic=force_tactic)
        self._enemy_random_states.clear()
        self._enemy_np_random_states.clear()
        self.enemy_group_1_task = self._run_with_enemy_rng(
            'enemy_group_1',
            lambda: TacticalTask(config, force_tactic=None),
        )
        self.enemy_group_2_task = self._run_with_enemy_rng(
            'enemy_group_2',
            lambda: TacticalTask(config, force_tactic=None),
        )
        self._configure_enemy_task(self.enemy_group_1_task, self.group_1_enemy)
        self._configure_enemy_task(self.enemy_group_2_task, self.group_2_enemy)
        if cap_task_instance is not None:
            # 让旧版 TacticalTask 能把真实发射/阻塞事件回传给 CAPTask 验收统计
            self.group_1_task._cap_task_sink = cap_task_instance
            self.group_2_task._cap_task_sink = cap_task_instance
            self.enemy_group_1_task._cap_task_sink = cap_task_instance
            self.enemy_group_2_task._cap_task_sink = cap_task_instance
        self._patch_executor_missile_filter(self.group_1_task)
        self._patch_executor_missile_filter(self.group_2_task)
        self._patch_executor_missile_filter(self.enemy_group_1_task)
        self._patch_executor_missile_filter(self.enemy_group_2_task)
        self.initialized = True
        # 🔥 初始化时立即同步距离，避免第一帧使用旧的 120km NLT
        if cap_task_instance is not None:
            self._sync_distances_to_task(self.group_1_task, cap_task_instance)
            self._sync_distances_to_task(self.group_2_task, cap_task_instance)
            self._sync_distances_to_task(self.enemy_group_1_task, cap_task_instance)
            self._sync_distances_to_task(self.enemy_group_2_task, cap_task_instance)
        log.info(
            "Enemy TacticalTask bridge profiles: group1=%s group2=%s",
            getattr(self.enemy_group_1_task, '_bridge_opening_force_tactic', 'AUTO'),
            getattr(self.enemy_group_2_task, '_bridge_opening_force_tactic', 'AUTO'),
        )
        log.info("TacticalBridge 成功初始化了两组 2v2 的战术模板管理器(TacticalTask).")

    def _patch_executor_missile_filter(self, task):
        executor = getattr(task, 'executor', None)
        if executor is None or getattr(executor, '_bridge_missile_filter_patched', False):
            return

        original_check = executor._check_and_evade_missile

        def filtered_check(env, agent_id, _original=original_check):
            if not hasattr(env, 'resolve_real_agent_id'):
                return _original(env, agent_id)

            previous_filter = getattr(env, '_missile_filter_real_id', None)
            env._missile_filter_real_id = env.resolve_real_agent_id(agent_id)
            try:
                return _original(env, agent_id)
            finally:
                env._missile_filter_real_id = previous_filter

        executor._check_and_evade_missile = filtered_check
        executor._bridge_missile_filter_patched = True
        
    def reset(self, env):
        self._enemy_random_states.clear()
        self._enemy_np_random_states.clear()
        if self.group_1_task:
            try:
                self.group_1_task.reset(env)
            except Exception as e:
                log.warning(f"TacticalBridge: group_1 reset failed: {e}")
        if self.group_2_task:
            try:
                self.group_2_task.reset(env)
            except Exception as e:
                log.warning(f"TacticalBridge: group_2 reset failed: {e}")
        if self.enemy_group_1_task:
            try:
                self.enemy_group_1_task.reset(env)
            except Exception as e:
                log.warning(f"TacticalBridge: enemy_group_1 reset failed: {e}")
        if self.enemy_group_2_task:
            try:
                self.enemy_group_2_task.reset(env)
            except Exception as e:
                log.warning(f"TacticalBridge: enemy_group_2 reset failed: {e}")

    def get_agent_tactic_info(self, agent_id: str) -> dict:
        """获取指定飞机在TacticalTask内部的实际战术名和阶段
        
        Returns:
            dict: {'tactic': str, 'phase': str} 或空字典
        """
        if agent_id in self.group_1_friendly:
            task = self.group_1_task
        elif agent_id in self.group_2_friendly:
            task = self.group_2_task
        else:
            return {}
        
        if task is None:
            return {}

        # 使用映射后的虚拟ID获取该分组在 TacticalTask 内的真实生效战术
        if agent_id in self.group_2_friendly:
            virtual_id = 'A0100' if agent_id == 'A0300' else 'A0200'
        else:
            virtual_id = agent_id

        tactic = None
        if hasattr(task, '_get_agent_tactic'):
            try:
                tactic = task._get_agent_tactic(virtual_id)
            except Exception:
                tactic = None
        if not tactic:
            tactic = getattr(task, 'selected_tactic', None)
        tactic = tactic or 'unknown'
        # 获取当前阶段
        phase = ''
        if hasattr(task, 'phase_tracker'):
            pt = task.phase_tracker
            # 映射虚拟ID获取阶段
            phase = str(getattr(pt, '_agent_phases', {}).get(virtual_id, ''))
            if not phase:
                phase = str(getattr(pt, 'current_phase', ''))
        if not phase and hasattr(task, 'current_phase'):
            phase = str(task.current_phase)
        
        return {'tactic': tactic, 'phase': phase}

    def get_action(self, env, agent_id: str, cap_task_instance: Any) -> Tuple[int, int, int]:
        if not self.initialized:
            return 7, 8, 3

        # 确定归属的分组
        if agent_id in self.group_1_friendly:
            task = self.group_1_task
            friendly_agents = self.group_1_friendly
        elif agent_id in self.group_2_friendly:
            task = self.group_2_task
            friendly_agents = self.group_2_friendly
        else:
            return 7, 8, 3
            
        if task is None:
            return 7, 8, 3
            
        # 🔥 每帧同步 cap_task 的动态战术节点（使用正确的 get_all() API）
        self._sync_distances_to_task(task, cap_task_instance)

        # 根据协同探测与战术分配获得锁定的敌人
        enemy_targets = []
        if hasattr(cap_task_instance, '_tactic_assignments_by_agent'):
            for fid in friendly_agents:
                assign = cap_task_instance._tactic_assignments_by_agent.get(fid)
                if assign and hasattr(assign, 'target_id') and assign.target_id:
                    enemy_targets.append(assign.target_id)
        
        # 构建稳定的友机ID映射：虚拟A0100/A0200固定绑定到该对子两架真机
        # 不能随 shooter/support 动态换位，否则底层状态机会在两架真机间“串台”。
        mapping_dict = {}
        mapping_dict['A0100'] = friendly_agents[0]
        mapping_dict['A0200'] = friendly_agents[1]
        mapping_dict.update(self._build_enemy_mapping(friendly_agents, cap_task_instance))
        
        real_b1 = mapping_dict['B0100']
        real_b2 = mapping_dict['B0200']
        mapping_dict['B0100'] = real_b1
        mapping_dict['B0200'] = real_b2
        
        virtual_agent_id = 'A0100' if agent_id == friendly_agents[0] else 'A0200'
        
        # 用假环境包装原环境 (将所有数据映射成 A0100/A0200 vs B0100/B0200 以满足老模板的硬编码需求)
        mock_env = MockEnv(env, mapping_dict)
        
        filter_added = False
        log_filter = None

        class MappingLogFilter(logging.Filter):
            def __init__(self, id_mapping):
                super().__init__()
                self.id_mapping = id_mapping

            def filter(self, record):
                if isinstance(record.msg, str):
                    msg = record.msg
                    for vid, rid in self.id_mapping.items():
                        msg = msg.replace(vid, rid)
                    record.msg = msg
                return True

        log_filter = MappingLogFilter(mapping_dict)
        # 作用于root logger以拦截所有模块产生的日志
        logging.getLogger().addFilter(log_filter)
        filter_added = True

        try:
            # TODO: 如果后续要求基于协同探测数据覆盖敌方真实坐标，可以在这里或者 MockAgentWrapper 中进行重载
            res = task.get_action(mock_env, virtual_agent_id)
            if res and len(res) >= 3:
                action = (int(res[0]), int(res[1]), int(res[2]))
                # 🔥 记录真实ID映射关系（仅在与虚拟ID不同时标注）
                if agent_id != virtual_agent_id:
                    log.debug(f"[TacticalBridge] {agent_id}(伪装为{virtual_agent_id}) -> action={action}")
                return action
        except Exception as e:
            import traceback
            log.error(f"TacticalBridge_get_action报错为 {agent_id}: {e}\n{traceback.format_exc()}")
        finally:
            if filter_added and log_filter is not None:
                logging.getLogger().removeFilter(log_filter)
        
        # 出错时的Fallback
        return 7, 8, 3

    def get_enemy_action(self, env, agent_id: str, cap_task_instance: Any) -> Tuple[int, int, int]:
        # 敌方桥接我方 CAPTask 逻辑已整体注释保留，默认不再使用 MockEnv 映射、sticky takeover 或 pair 接管链路。
        # if not self.initialized:
        #     return 7, 8, 3
        #
        # if agent_id in self.group_1_enemy:
        #     task = self.enemy_group_1_task
        #     enemy_agents = self.group_1_enemy
        # elif agent_id in self.group_2_enemy:
        #     task = self.enemy_group_2_task
        #     enemy_agents = self.group_2_enemy
        # else:
        #     return 7, 8, 3
        #
        # if task is None:
        #     return 7, 8, 3
        #
        # self._sync_distances_to_task(task, cap_task_instance)
        # self._prepare_enemy_task_for_step(task)
        # friendly_targets = self._get_friendly_pair(enemy_agents, env)
        # mapping_dict = {
        #     'A0100': enemy_agents[0],
        #     'A0200': enemy_agents[1],
        #     'B0100': friendly_targets[0],
        #     'B0200': friendly_targets[1],
        # }
        # virtual_agent_id = 'A0100' if agent_id == enemy_agents[0] else 'A0200'
        # mock_env = MockEnv(env, mapping_dict)
        #
        # filter_added = False
        # log_filter = None
        #
        # class MappingLogFilter(logging.Filter):
        #     def __init__(self, id_mapping):
        #         super().__init__()
        #         self.id_mapping = id_mapping
        #
        #     def filter(self, record):
        #         if isinstance(record.msg, str):
        #             msg = record.msg
        #             for vid, rid in self.id_mapping.items():
        #                 msg = msg.replace(vid, rid)
        #             record.msg = msg
        #         return True
        #
        # log_filter = MappingLogFilter(mapping_dict)
        # logging.getLogger().addFilter(log_filter)
        # filter_added = True
        #
        # try:
        #     group_key = self._enemy_group_key(enemy_agents)
        #     res = self._run_with_enemy_rng(
        #         group_key,
        #         lambda: task.get_action(mock_env, virtual_agent_id),
        #     )
        #     if res and len(res) >= 3:
        #         action = (int(res[0]), int(res[1]), int(res[2]))
        #         if agent_id != virtual_agent_id:
        #             log.debug(f"[TacticalBridge:enemy] {agent_id}(virtual={virtual_agent_id}) -> action={action}")
        #         return action
        # except Exception as e:
        #     import traceback
        #     log.error(f"TacticalBridge_get_enemy_action error {agent_id}: {e}\n{traceback.format_exc()}")
        # finally:
        #     if filter_added and log_filter is not None:
        #         logging.getLogger().removeFilter(log_filter)

        return 7, 8, 3
