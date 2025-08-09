#!/usr/bin/env python3
"""
钳形夹击战术任务 - 完全基于drag_shoot_tactical_task架构模式
严格照抄drag_shoot_tactical_task的实现，只修改战术逻辑部分
"""

import logging
import numpy as np
import torch
from enum import Enum
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.termination_conditions.termination_condition_base import BaseTerminationCondition
from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers, CompositeManeuverExecutor


class PincerAttackTermination(BaseTerminationCondition):
    """钳形夹击专用终止条件 - 只有双方全灭才终止"""

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 1000)  # 1000米
        self.max_steps = getattr(config, 'max_steps', 2500)

    def get_termination(self, task, env, agent_id, info={}):
        """
        自定义终止条件：
        1. 高度过低
        2. 时间限制
        3. 极端状态
        4. 过载
        5. 双方全灭（不是单架飞机被击落）
        """
        # 检查高度过低
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= self.altitude_limit:
            self.log(f"{agent_id} altitude too low: {current_alt:.1f}m")
            return True, False, info

        # 检查时间限制
        if env.current_step >= self.max_steps:
            self.log(f"Time limit reached: {env.current_step} steps")
            return True, False, info

        # 检查极端状态
        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            self.log(f"{agent_id} extreme state detected")
            return True, False, info

        # 检查过载状态
        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            self.log(f"{agent_id} overload detected")
            return True, False, info

        # 检查双方存活情况 - 只有当一方全灭时才终止
        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]

        # 只有当一方全灭时才终止
        if len(red_alive) == 0:
            self.log("Red team eliminated")
            return True, True, {"termination_reason": "red_eliminated", "winner": "blue"}
        elif len(blue_alive) == 0:
            self.log("Blue team eliminated")
            return True, True, {"termination_reason": "blue_eliminated", "winner": "red"}

        # 继续仿真 - 不因为单架飞机被击落而终止
        return False, False, info


class TacticalPhase(Enum):
    """钳形夹击战术阶段 - 基于控制距离的精确定义"""
    NLT_MELD = "NLT_MELD"    # 90-81km: 钳形展开阶段
    MELD_MTR = "MELD_MTR"    # 81-45km: 钳形收拢阶段
    MTR_TR = "MTR_TR"        # 45-41km: 导弹发射阶段
    TR_DOR = "TR_DOR"        # 41-19.6km: 内侧脱离阶段
    DOR_DR = "DOR_DR"        # 19.6-14.5km: 返航阶段


class PincerAttackTacticalTask(MultipleCombatTask):
    """
    钳形夹击战术任务 - 基于drag_shoot_tactical_task架构模式

    动作空间架构：
    - 定义的动作空间: [41, 41, 41, 30] (继承自MultipleCombatTask，但实际不使用)
    - 实际使用的动作空间: [15, 17, 7] (高层战术指令)
    - 转换机制: 高层指令 → baseline模型 → 底层飞行控制

    主要函数调用链：
    normalize_action() → _process_pincer_attack_tactics() → _get_tactical_command_indices()
    → _get_friendly_command_indices() / _get_enemy_command_indices()
    → _use_lowlevel_policy() → baseline模型输出底层控制指令

    平稳飞行指令: [7, 8, 3] = [高度0m变化, 航向0°变化, 速度0m/s变化]
    """

    def __init__(self, config):
        """初始化钳形夹击战术任务"""
        super().__init__(config)

        # 钳形夹击战术距离配置 - 严格按照控制距离定义
        self.tactical_distances = {
            'NLT_MELD_min': 81000,   # 81km - 钳形展开阶段
            'MELD_MTR_min': 45000,   # 45km - 钳形收拢阶段
            'MTR_TR_min': 41000,     # 41km - 导弹发射阶段
            'TR_DOR_min': 19600,     # 19.6km - 内侧脱离阶段
            'DOR_DR_min': 14500,     # 14.5km - 返航阶段
        }

        # 钳形夹击战术参数 - 确保敌机保持在雷达照射范围内
        self.pincer_config = {
            'crank_angle_moderate': 30.0,   # 适中的Crank角度，确保雷达照射
            'leader_left_crank': True,      # 长机执行左侧Crank
            'wingman_right_crank': True,    # 僚机执行右侧Crank
            'short_skate_angle': 45.0,      # Short Skate转弯角度
            'rtb_heading': 180.0,           # 返航航向
        }

        # 当前战术阶段
        self.current_phase = TacticalPhase.NLT_MELD
        self.phase_start_time = 0.0
        self.phase_start_step = 0

        # 底层策略网络 - 完全照抄drag_shoot_tactical_task
        self.my_lowlevel_policy = BaselineActor()
        self._inner_rnn_states = {}
        self._load_baseline_models()

        # 动作空间定义：[15, 17, 7] - 高层战术指令空间
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0  # 索引7 = 0m变化（平稳飞行）

        self.norm_delta_heading = np.array([
            -60, -45, -30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30, 45, 60
        ]) * np.pi / 180.0  # 索引8 = 0°变化（保持航向）

        self.norm_delta_velocity = np.array([
            -100, -50, -20, 0, 20, 50, 100
        ])  # 索引3 = 0m/s变化（保持速度）

        # 基础机动系统 - 照抄drag_shoot_tactical_task
        self.basic_maneuvers = BasicManeuvers()
        self.composite_executor = CompositeManeuverExecutor()

        # 状态跟踪
        self.initial_heading = {}
        self.initial_altitude = {}
        self.active_maneuvers = {}
        self.maneuver_start_times = {}

        # 导弹发射管理
        self.last_missile_launch_time = {}
        self.friendly_missile_cooldown = 2.0  # 友方2秒冷却
        self.enemy_missile_cooldown = 10.0    # 敌方10秒冷却

        # Short Skate机动状态跟踪 - 复制拖曳射击项目
        self.short_skate_states = {}

        # 导弹发射管理 - 复制拖曳射击项目
        self.last_missile_launch_time = {"A0100": -999, "A0200": -999, "B0100": -999, "B0200": -999}
        self.friendly_missile_cooldown = 2.0  # 友方2秒冷却
        self.enemy_missile_cooldown = 10.0    # 敌方10秒冷却

        logging.info("PincerAttackTacticalTask initialized")

    def get_termination(self, env, agent_id, info={}):
        """
        完全重写终止条件 - 绕过父类的termination_conditions列表
        只有双方全灭、高度过低、或达到时间限制才终止
        """
        # 检查高度过低
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= 1000:  # 1000米以下
            logging.info(f"{agent_id} altitude too low: {current_alt:.1f}m - terminating")
            return True, {"termination_reason": "low_altitude"}

        # 检查时间限制
        if env.current_step >= 2500:  # 最大步数
            logging.info(f"Time limit reached: {env.current_step} steps - terminating")
            return True, {"termination_reason": "timeout"}

        # 检查极端状态
        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            logging.info(f"{agent_id} extreme state detected - terminating")
            return True, {"termination_reason": "extreme_state"}

        # 检查过载状态
        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            logging.info(f"{agent_id} overload detected - terminating")
            return True, {"termination_reason": "overload"}

        # 检查双方存活情况 - 只有当一方全灭时才终止
        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]

        # 添加调试日志
        if env.current_step % 50 == 0:  # 每50步输出一次状态
            logging.info(f"TERMINATION CHECK - Step {env.current_step}: Red alive: {red_alive}, Blue alive: {blue_alive}")

        # 只有当一方全灭时才终止
        if len(red_alive) == 0:
            logging.info("Red team eliminated - terminating")
            return True, {"termination_reason": "red_eliminated", "winner": "blue"}
        elif len(blue_alive) == 0:
            logging.info("Blue team eliminated - terminating")
            return True, {"termination_reason": "blue_eliminated", "winner": "red"}

        # 继续仿真 - 不因为单架飞机被击落而终止
        return False, info

    def _load_baseline_models(self):
        """加载baseline模型 - 完全学习drag_shoot_tactical_task的实现"""
        try:
            model_path = get_root_dir() + '/model/baseline_model.pt'
            if torch.cuda.is_available():
                device = torch.device("cuda")
                checkpoint = torch.load(model_path, weights_only=True)
            else:
                device = torch.device("cpu")
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            self.my_lowlevel_policy.load_state_dict(checkpoint)
            self.my_lowlevel_policy.eval()
            logging.info("Successfully loaded baseline model for pincer-attack task")
        except Exception as e:
            logging.error(f"加载baseline模型失败: {e}")
            self.my_lowlevel_policy = None

    def normalize_action(self, env, agent_id, action):
        """
        钳形夹击战术的normalize_action实现 - 完全照抄drag_shoot_tactical_task的架构
        """
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        current_time = env.current_step * env.time_interval

        # 处理钳形夹击战术逻辑
        result = self._process_pincer_attack_tactics(env, agent_id, current_time)

        # 处理导弹发射 - 添加友方导弹发射功能
        self._handle_missile_launch(env, agent_id, current_time)

        return result

    def _process_pincer_attack_tactics(self, env, agent_id, current_time):
        """处理钳形夹击战术逻辑 - 照抄drag_shoot_tactical_task的_process_drag_shoot_tactics"""
        try:
            # 更新战术阶段
            self._update_tactical_phase(env)

            # 获取战术指令索引
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id)

            # 使用底层策略
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        except Exception as e:
            logging.error(f"钳形夹击战术逻辑失败 {agent_id}: {e}")
            return self._use_lowlevel_policy(env, agent_id, 7, 8, 3)  # 平稳飞行

    def _update_tactical_phase(self, env):
        """更新战术阶段 - 照抄drag_shoot_tactical_task的实现"""
        # 获取主要对抗双方
        leader_red = env._jsbsims.get("A0100")
        leader_blue = env._jsbsims.get("B0100")

        if not leader_red or not leader_blue or not leader_red.is_alive or not leader_blue.is_alive:
            return

        # 计算距离
        distance = self._calculate_distance(leader_red, leader_blue)

        # 确定当前阶段
        new_phase = self._get_phase_by_distance(distance)

        if new_phase != self.current_phase:
            current_time = env.current_step * env.time_interval
            logging.info(f"Phase transition: {self.current_phase.value} -> {new_phase.value} "
                        f"at t={current_time:.1f}s, distance={distance/1000:.1f}km")
            self.current_phase = new_phase
            self.phase_start_time = current_time
            self.phase_start_step = env.current_step

    def _get_phase_by_distance(self, distance):
        """根据距离确定战术阶段"""
        if distance >= self.tactical_distances['NLT_MELD_min']:
            return TacticalPhase.NLT_MELD
        elif distance >= self.tactical_distances['MELD_MTR_min']:
            return TacticalPhase.MELD_MTR
        elif distance >= self.tactical_distances['MTR_TR_min']:
            return TacticalPhase.MTR_TR
        elif distance >= self.tactical_distances['TR_DOR_min']:
            return TacticalPhase.TR_DOR
        else:
            return TacticalPhase.DOR_DR

    def _calculate_distance(self, aircraft1, aircraft2):
        """计算两架飞机之间的距离"""
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        return np.linalg.norm(pos1 - pos2)

    def _get_tactical_command_indices(self, env, agent_id):
        """获取钳形夹击战术指令索引"""
        # 根据当前阶段和角色确定战术行为
        if agent_id.startswith('A'):  # 友方
            return self._get_friendly_command_indices(env, agent_id)
        else:  # 敌方
            return self._get_enemy_command_indices(env, agent_id)

    def _get_friendly_command_indices(self, env, agent_id):
        """友方钳形夹击指令 - 严格按照战术规格实现"""
        if agent_id == "A0100":  # 友方长机
            return self._get_friendly_leader_command_indices(env, agent_id)
        elif agent_id == "A0200":  # 友方僚机
            return self._get_friendly_wingman_command_indices(env, agent_id)
        else:
            return 7, 8, 3  # 默认平稳飞行

    def _get_friendly_leader_command_indices(self, env, agent_id):
        """友方长机钳形夹击指令"""
        if self.current_phase == TacticalPhase.NLT_MELD:
            # 阶段1 (90-81km): 长机执行LEFT crank机动
            return self._maintain_heading_precise(env, agent_id, 330.0)  # 左转30°

        elif self.current_phase == TacticalPhase.MELD_MTR:
            # 阶段2 (81-45km): Crank恢复，指向敌机方向，平行飞行0°
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.MTR_TR:
            # 阶段3 (45-41km): 导弹发射阶段，保持0°航向
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 阶段4 (41-19.6km): 长机执行RIGHT转弯进行Short Skate
            return self._execute_short_skate_precise(env, agent_id, "right")

        else:  # DOR_DR
            # 阶段5 (19.6-14.5km): 返航，航向180°
            return self._maintain_heading_precise(env, agent_id, 180.0)

    def _get_friendly_wingman_command_indices(self, env, agent_id):
        """友方僚机钳形夹击指令"""
        if self.current_phase == TacticalPhase.NLT_MELD:
            # 阶段1 (90-81km): 僚机执行RIGHT crank机动
            return self._maintain_heading_precise(env, agent_id, 30.0)  # 右转30°

        elif self.current_phase == TacticalPhase.MELD_MTR:
            # 阶段2 (81-45km): Crank恢复，指向敌机方向，平行飞行0°
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.MTR_TR:
            # 阶段3 (45-41km): 导弹发射阶段，僚机可能稍微滞后
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 阶段4 (41-19.6km): 僚机执行LEFT转弯进行Short Skate
            return self._execute_short_skate_precise(env, agent_id, "left")

        else:  # DOR_DR
            # 阶段5 (19.6-14.5km): 返航，航向180°
            return self._maintain_heading_precise(env, agent_id, 180.0)

    def _get_enemy_command_indices(self, env, agent_id):
        """敌方战术指令索引 - 直接复制拖曳射击项目的敌方AI逻辑"""
        current_time = env.current_step * env.time_interval

        # 导入并使用拖曳射击项目的敌方AI系统
        try:
            import os
            import sys
            # 添加当前目录到Python路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            from enemy_tactical_ai import get_enemy_tactical_command
            commands = get_enemy_tactical_command(env, agent_id, current_time)
            logging.info(f"✅ {agent_id} 敌方AI指令: {commands}")
            return commands
        except ImportError as e:
            logging.warning(f"❌ Enemy tactical AI import failed: {e}, using fallback logic")
            # 回退到原有逻辑
            return self._get_enemy_command_indices_fallback(env, agent_id, current_time)
        except Exception as e:
            logging.error(f"❌ {agent_id} 敌方AI执行错误: {e}, using fallback logic")
            import traceback
            logging.error(f"详细错误信息: {traceback.format_exc()}")
            # 回退到原有逻辑
            return self._get_enemy_command_indices_fallback(env, agent_id, current_time)

    def _get_enemy_command_indices_fallback(self, env, agent_id, current_time):
        """敌方战术指令索引 - 强制使用完整BVR战术循环"""
        # 强制使用我的新BVR循环，不再尝试调用拖曳射击项目的AI
        print(f"🎯 {agent_id}: 使用新BVR战术循环 (时间: {current_time:.1f}s)")

        # 初始化敌方BVR状态管理
        if not hasattr(self, 'enemy_bvr_states'):
            self.enemy_bvr_states = {}

        if agent_id not in self.enemy_bvr_states:
            self.enemy_bvr_states[agent_id] = {
                'phase': 'approach',  # approach, engage, cold_turn, rtb, re_engage
                'phase_start_time': current_time,
                'engagement_count': 0,
                'last_rtb_time': 0,
                'rtb_completed': False
            }

        # 获取敌机状态
        enemy_aircraft = env.agents[agent_id]
        enemy_pos = enemy_aircraft.get_position()
        current_heading = np.rad2deg(enemy_aircraft.get_property_value(c.attitude_psi_rad))

        # 寻找最近的友方目标
        min_distance = float('inf')
        closest_friendly_pos = None
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                friendly_pos = env.agents[friendly_id].get_position()
                distance = np.linalg.norm(friendly_pos - enemy_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_friendly_pos = friendly_pos

        if closest_friendly_pos is None:
            return 7, 8, 3  # 保持当前状态

        # 获取当前BVR状态
        bvr_state = self.enemy_bvr_states[agent_id]
        phase_duration = current_time - bvr_state['phase_start_time']

        print(f"🎯 {agent_id}: BVR状态={bvr_state['phase']}, 距离={min_distance/1000:.1f}km, 阶段时长={phase_duration:.1f}s")

        # 执行完整的BVR战术循环
        return self._execute_enemy_bvr_cycle(env, agent_id, current_time, min_distance, closest_friendly_pos, bvr_state, phase_duration)

    def _execute_enemy_bvr_cycle(self, env, agent_id, current_time, distance, target_pos, bvr_state, phase_duration):
        """执行敌方完整BVR战术循环 - 修复版：确保真正的0°北向RTB"""
        enemy_aircraft = env.agents[agent_id]
        enemy_pos = enemy_aircraft.get_position()
        current_heading = np.rad2deg(enemy_aircraft.get_property_value(c.attitude_psi_rad))

        current_phase = bvr_state['phase']

        # 强制RTB条件检查 - 优先级最高
        force_rtb = False
        if distance < 15000:  # 15km以下强制RTB
            force_rtb = True
            print(f"🚨 {agent_id}: 距离过近强制RTB - 距离{distance/1000:.1f}km")
        elif bvr_state['engagement_count'] >= 2:  # 2轮交战后强制RTB
            force_rtb = True
            print(f"🚨 {agent_id}: 交战轮数达到上限强制RTB - 第{bvr_state['engagement_count']}轮")
        elif current_time > 180.0:  # 3分钟后强制RTB
            force_rtb = True
            print(f"🚨 {agent_id}: 时间到强制RTB - {current_time:.1f}s")

        if force_rtb:
            bvr_state['phase'] = 'final_rtb'
            bvr_state['phase_start_time'] = current_time
            target_heading = 0.0  # 强制北向返航
            print(f"🏠 {agent_id}: 强制RTB返航 - 目标航向{target_heading:.1f}°")
            return self._maintain_heading_precise(env, agent_id, target_heading)

        # 阶段1：接敌阶段 (Approach Phase)
        if current_phase == 'approach':
            if distance > 40000:  # 40km以上：继续接敌
                target_heading = self._calculate_bearing_to_target(enemy_pos, target_pos)
                print(f"🎯 {agent_id}: 接敌阶段 - 距离{distance/1000:.1f}km，目标航向{target_heading:.1f}°")
                return self._maintain_heading_precise(env, agent_id, target_heading)
            else:  # 进入交战阶段
                bvr_state['phase'] = 'engage'
                bvr_state['phase_start_time'] = current_time
                print(f"⚔️ {agent_id}: 进入交战阶段 - 距离{distance/1000:.1f}km")

        # 阶段2：交战阶段 (Engagement Phase)
        elif current_phase == 'engage':
            if distance < 25000:  # 25km以下：立即Cold Turn
                bvr_state['phase'] = 'cold_turn'
                bvr_state['phase_start_time'] = current_time
                print(f"🚨 {agent_id}: 距离过近，执行Cold Turn - 距离{distance/1000:.1f}km")
            elif phase_duration > 20.0:  # 交战20秒后主动Cold Turn
                bvr_state['phase'] = 'cold_turn'
                bvr_state['phase_start_time'] = current_time
                print(f"🔄 {agent_id}: 交战时间到，执行Cold Turn - 距离{distance/1000:.1f}km")
            else:  # 继续BVR交战机动
                # 执行Crank机动保持雷达照射
                if agent_id == "B0100":
                    crank_angle = 45.0  # 右Crank
                else:
                    crank_angle = -45.0  # 左Crank
                target_heading = (current_heading + crank_angle) % 360
                print(f"⚔️ {agent_id}: BVR交战机动 - Crank{crank_angle:.0f}°，目标航向{target_heading:.1f}°")
                return self._maintain_heading_precise(env, agent_id, target_heading)

        # 阶段3：Cold Turn阶段 (Cold Turn Phase)
        elif current_phase == 'cold_turn':
            if phase_duration < 10.0:  # Cold Turn持续10秒
                target_heading = 0.0  # 转向北方（敌方基地方向）
                print(f"❄️ {agent_id}: Cold Turn机动 - 目标航向{target_heading:.1f}°，持续{phase_duration:.1f}s")
                return self._maintain_heading_precise(env, agent_id, target_heading)
            else:  # Cold Turn完成，进入RTB
                bvr_state['phase'] = 'rtb'
                bvr_state['phase_start_time'] = current_time
                bvr_state['last_rtb_time'] = current_time
                print(f"🏃 {agent_id}: Cold Turn完成，开始RTB返航")

        # 阶段4：返航基地阶段 (RTB Phase)
        elif current_phase == 'rtb':
            if phase_duration < 30.0:  # RTB持续30秒
                target_heading = 0.0  # 持续北向返航
                print(f"🏠 {agent_id}: RTB返航 - 目标航向{target_heading:.1f}°，返航{phase_duration:.1f}s")
                return self._maintain_heading_precise(env, agent_id, target_heading)
            else:  # RTB完成，考虑重新接敌
                bvr_state['rtb_completed'] = True
                bvr_state['engagement_count'] += 1

                if bvr_state['engagement_count'] < 2 and distance > 40000:  # 最多2轮，距离足够远
                    bvr_state['phase'] = 're_engage'
                    bvr_state['phase_start_time'] = current_time
                    print(f"🔄 {agent_id}: RTB完成，准备重新接敌 - 第{bvr_state['engagement_count']}轮")
                else:  # 继续返航
                    bvr_state['phase'] = 'final_rtb'
                    target_heading = 0.0
                    print(f"🏠 {agent_id}: 交战结束，最终返航基地 - 目标航向{target_heading:.1f}°")
                    return self._maintain_heading_precise(env, agent_id, target_heading)

        # 阶段5：重新接敌阶段 (Re-engagement Phase)
        elif current_phase == 're_engage':
            if distance > 45000:  # 距离足够远，重新接敌
                bvr_state['phase'] = 'approach'
                bvr_state['phase_start_time'] = current_time
                print(f"⚔️ {agent_id}: 重新接敌开始，第{bvr_state['engagement_count']}轮")
            else:  # 距离不够，继续返航
                target_heading = 0.0
                print(f"🏃 {agent_id}: 距离不够，继续返航 - 距离{distance/1000:.1f}km，目标航向{target_heading:.1f}°")
                return self._maintain_heading_precise(env, agent_id, target_heading)

        # 阶段6：最终返航阶段 (Final RTB Phase)
        elif current_phase == 'final_rtb':
            target_heading = 0.0  # 持续北向返航
            print(f"🏠 {agent_id}: 最终返航基地 - 目标航向{target_heading:.1f}°")
            return self._maintain_heading_precise(env, agent_id, target_heading)

        # 默认返航
        target_heading = 0.0
        print(f"🏠 {agent_id}: 默认返航 - 目标航向{target_heading:.1f}°")
        return self._maintain_heading_precise(env, agent_id, target_heading)

    # ========== 精确航向保持和机动方法 ==========
    def _maintain_heading_precise(self, env, agent_id, target_heading):
        """精确航向保持 - 复制拖曳射击项目的实现"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        # 计算航向差值
        heading_diff = target_heading - current_heading
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360

        # 精确航向控制
        if abs(heading_diff) > 2.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
        else:
            heading_cmd_id = 8  # 保持当前航向

        print(f"[钳形夹击] {agent_id}: 当前航向{current_heading:.1f}° -> 目标航向{target_heading:.1f}° (差值{heading_diff:.1f}°)")

        return 7, heading_cmd_id, 3  # 保持高度，调整航向，保持速度

    def _execute_short_skate_precise(self, env, agent_id, direction):
        """执行精确的Short Skate机动 - 复制拖曳射击项目的实现"""
        if direction == "right":
            # 长机右转Short Skate
            target_heading = 45.0  # 右转45°然后逐渐转向180°
        elif direction == "left":
            # 僚机左转Short Skate
            target_heading = 315.0  # 左转45°然后逐渐转向180°
        else:  # direction == "north" for enemy
            # 敌方返航北方
            target_heading = 0.0

        # 初始化Short Skate状态
        if agent_id not in self.short_skate_states:
            self.short_skate_states[agent_id] = {
                'start_time': env.current_step * env.time_interval,
                'phase': 'turn',
                'target_heading': target_heading
            }

        # 执行Short Skate机动
        skate_state = self.short_skate_states[agent_id]
        current_time = env.current_step * env.time_interval
        elapsed_time = current_time - skate_state['start_time']

        if elapsed_time < 10.0:  # 前10秒执行转弯
            return self._maintain_heading_precise(env, agent_id, skate_state['target_heading'])
        else:  # 10秒后转向返航
            if direction in ["right", "left"]:
                return self._maintain_heading_precise(env, agent_id, 180.0)  # 友方返航南方
            else:
                return self._maintain_heading_precise(env, agent_id, 0.0)    # 敌方返航北方

    # ========== 辅助方法保持不变 ==========

    # ========== 辅助方法 ==========
    def _get_primary_enemy_position(self, env):
        """获取主要敌机位置"""
        for agent_id, agent in env.agents.items():
            if agent_id.startswith('B') and agent.is_alive:
                return agent.get_position()
        return None

    def _get_primary_friendly_position(self, env):
        """获取主要友机位置"""
        for agent_id, agent in env.agents.items():
            if agent_id.startswith('A') and agent.is_alive:
                return agent.get_position()
        return None

    def _calculate_bearing_to_target(self, my_pos, target_pos):
        """计算指向目标的方位角"""
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        bearing = np.rad2deg(np.arctan2(dx, dy))
        return bearing % 360

    def _convert_heading_to_index(self, heading_diff_rad):
        """将航向差值转换为指令索引"""
        diff_deg = np.rad2deg(heading_diff_rad)
        heading_options = [-60, -45, -30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30, 45, 60]

        closest_idx = 0
        min_diff = abs(diff_deg - heading_options[0])

        for i, option in enumerate(heading_options):
            if abs(diff_deg - option) < min_diff:
                min_diff = abs(diff_deg - option)
                closest_idx = i

        return closest_idx

    def _use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """使用底层策略 - 完全照抄drag_shoot_tactical_task的实现"""
        if self.my_lowlevel_policy is None:
            return np.array([0.0, 0.0, 0.0, 0.7])

        try:
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            # 安全索引访问，防止越界
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)

            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.nan_to_num(input_obs, nan=0.0)
            input_obs = np.expand_dims(input_obs, axis=0)

            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))

            _action, _rnn_states = self.my_lowlevel_policy(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4

            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 1000:
                norm_act[1] = max(norm_act[1], 0.0)
                norm_act[3] = max(norm_act[3], 0.8)

            return norm_act
        except Exception as e:
            logging.error(f"底层策略错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def step(self, env):
        """执行钳形夹击战术步骤 - 基于DragShootTacticalTask的step方法"""
        # 为每个智能体生成战术动作
        obs = {}
        share_obs = {}
        rewards = {}
        dones = {}
        infos = {}

        for agent_id in env._jsbsims.keys():
            if not env._jsbsims[agent_id].is_alive:
                obs[agent_id] = np.zeros(self.obs_length)
                share_obs[agent_id] = np.zeros(self.obs_length)
                rewards[agent_id] = [-10.0]
                dones[agent_id] = [True]
                infos[agent_id] = {"agent_id": agent_id, "alive": False}
                continue

            # 获取观测
            agent_obs = self.get_obs(env, agent_id)
            obs[agent_id] = agent_obs
            share_obs[agent_id] = agent_obs

            # 计算奖励
            rewards[agent_id] = [0.0]  # 简化奖励

            # 检查终止条件
            done, info = self.get_termination(env, agent_id, {})
            dones[agent_id] = [done]
            infos[agent_id] = {"agent_id": agent_id, "alive": True, "phase": self.current_phase.value}

        return obs, share_obs, rewards, dones, infos

    def _handle_missile_launch(self, env, agent_id: str, current_time: float):
        """
        处理导弹发射 - 修复版：更宽松的发射条件，参考拖曳射击项目
        """
        # 检查导弹数量
        if env.agents[agent_id].num_missiles <= 0:
            return

        # 友方返航期间禁止发射导弹
        if agent_id.startswith('A') and self.current_phase == TacticalPhase.DOR_DR:
            logging.info(f"{agent_id} 返航期间禁止发射导弹")
            return

        # 检查冷却时间
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if agent_id.startswith('A'):
            cooldown = self.friendly_missile_cooldown
        else:
            cooldown = self.enemy_missile_cooldown

        if current_time - last_launch < cooldown:
            return

        # 寻找目标
        target = None
        min_distance = float('inf')
        for other_id, other_aircraft in env.agents.items():
            if self._is_enemy_agent(agent_id, other_id) and other_aircraft.is_alive:
                distance = self._calculate_distance(env.agents[agent_id], other_aircraft)
                if distance < min_distance:
                    min_distance = distance
                    target = other_aircraft

        if target is None:
            return

        # 修复版导弹发射逻辑：更宽松的条件，敌方也能发射
        should_launch = False

        if agent_id == "A0100":  # 友方长机
            # 更宽松的发射条件：多个阶段都可以发射
            if (self.current_phase in [TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR, TacticalPhase.TR_DOR] and
                min_distance <= 60000):  # 60km内都可以发射
                should_launch = True
                logging.info(f"🚀 A0100长机发射: 阶段={self.current_phase.value}, 距离={min_distance/1000:.1f}km")

        elif agent_id == "A0200":  # 友方僚机
            # 僚机稍微滞后，但条件也很宽松
            if (self.current_phase in [TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR, TacticalPhase.TR_DOR] and
                min_distance <= 55000):  # 55km内可以发射
                should_launch = True
                logging.info(f"🚀 A0200僚机发射: 阶段={self.current_phase.value}, 距离={min_distance/1000:.1f}km")

        elif agent_id.startswith('B'):  # 敌方
            # 敌方智能发射逻辑：基于BVR状态
            if hasattr(self, 'enemy_bvr_states') and agent_id in self.enemy_bvr_states:
                bvr_state = self.enemy_bvr_states[agent_id]
                current_bvr_phase = bvr_state['phase']

                # 敌方在交战阶段发射导弹
                if current_bvr_phase in ['approach', 'engage'] and min_distance <= 50000:
                    should_launch = True
                    logging.info(f"🚀 {agent_id}敌方发射: BVR阶段={current_bvr_phase}, 距离={min_distance/1000:.1f}km")
                # 敌方在接敌阶段也可以发射
                elif current_bvr_phase == 'approach' and min_distance <= 45000:
                    should_launch = True
                    logging.info(f"🚀 {agent_id}敌方接敌发射: BVR阶段={current_bvr_phase}, 距离={min_distance/1000:.1f}km")
            else:
                # 回退逻辑：简单的距离判断
                if min_distance <= 45000:
                    should_launch = True
                    logging.info(f"🚀 {agent_id}敌方发射: 距离={min_distance/1000:.1f}km")

        if should_launch:
            # 执行发射
            if agent_id.startswith('A'):
                self._launch_friendly_missiles(env, agent_id, target, current_time)
            else:
                self._launch_missile(env, agent_id, target, current_time)

    def _is_enemy_agent(self, agent_id1: str, agent_id2: str) -> bool:
        """判断是否为敌方"""
        return (agent_id1.startswith('A') and agent_id2.startswith('B')) or \
               (agent_id1.startswith('B') and agent_id2.startswith('A'))

    def _launch_friendly_missiles(self, env, agent_id: str, target, current_time: float):
        """友方连续发射机制 - 复制拖曳射击项目的实现"""
        aircraft = env.agents[agent_id]
        distance = self._calculate_distance(aircraft, target)

        # 确定发射数量
        missiles_to_launch = 1  # 默认发射1枚

        # 在MTR-TR阶段且距离合适时，考虑发射2枚导弹
        if (self.current_phase == TacticalPhase.MTR_TR and
            aircraft.num_missiles >= 2 and
            41000 <= distance <= 45000):  # MTR-TR阶段最佳发射距离

            # 检查是否已经进行过连续发射
            if not hasattr(self, 'friendly_burst_launch'):
                self.friendly_burst_launch = {"A0100": 0, "A0200": 0}

            burst_count = self.friendly_burst_launch.get(agent_id, 0)
            if burst_count == 0:  # 第一次连续发射机会
                missiles_to_launch = 2
                self.friendly_burst_launch[agent_id] = 1
                logging.info(f"{agent_id} 钳形夹击积极发射策略：一次性发射2枚导弹")

        # 执行发射
        for i in range(missiles_to_launch):
            if aircraft.num_missiles > 0:
                self._launch_missile(env, agent_id, target, current_time)
                if i < missiles_to_launch - 1:  # 不是最后一枚导弹
                    # 短暂延迟，模拟连续发射
                    current_time += 0.5  # 0.5秒间隔

    def _launch_missile(self, env, agent_id: str, target, current_time: float):
        """发射导弹 - 修复版：完全复制拖曳射击项目的导弹创建逻辑"""
        try:
            aircraft = env.agents[agent_id]

            # 创建导弹ID - 使用正确的格式 A0100 → A1001, A1002
            missile_count = 2 - aircraft.num_missiles + 1  # 第1枚或第2枚导弹
            # A0100 → A100, B0100 → B100
            base_id = agent_id[0] + agent_id[2:]  # A0100 → A100
            missile_uid = f"{base_id}{missile_count}"  # A100 → A1001

            # 根据发射平台选择导弹类型 - 完全复制拖曳射击项目
            if agent_id.startswith('A'):  # 我方飞机 - 使用AIM-120C7
                try:
                    from envs.JSBSim.core.simulatior import MissileSimulator
                    missile = MissileSimulator.create(
                        parent=aircraft,
                        target=target,
                        uid=missile_uid
                    )
                    missile_type = "AIM-120C-7"

                    # 添加到环境的临时模拟器 - 关键步骤！
                    env.add_temp_simulator(missile)
                    logging.info(f"✅ 友方导弹{missile_uid}已添加到环境")

                except Exception as missile_error:
                    logging.error(f"友方导弹创建失败: {missile_error}")
                    missile_type = "AIM-120C-7"

            else:  # 敌方飞机 - 使用R-27ER
                try:
                    from r27er_missile import R27ERMissileSimulator
                    missile = R27ERMissileSimulator.create(
                        parent=aircraft,
                        target=target,
                        uid=missile_uid
                    )
                    missile_type = "R-27ER"

                    # 添加到环境的临时模拟器 - 关键步骤！
                    env.add_temp_simulator(missile)
                    logging.info(f"✅ 敌方导弹{missile_uid}已添加到环境")

                except Exception as missile_error:
                    logging.error(f"敌方导弹创建失败: {missile_error}")
                    missile_type = "R-27ER"

            # 初始化导弹记录系统 - 完全复制拖曳射击项目
            if not hasattr(env, '_missile_records'):
                env._missile_records = {}

            # 记录导弹信息
            env._missile_records[missile_uid] = {
                'launcher': agent_id,
                'target': target.uid,
                'type': missile_type,
                'status': 'LAUNCHED',
                'launch_time': current_time,
                'launch_position': aircraft.get_position().copy(),
                'launch_velocity': aircraft.get_velocity().copy()
            }

            # 更新飞机状态
            aircraft.num_missiles -= 1

            # 更新发射时间记录
            self.last_missile_launch_time[agent_id] = current_time

            # 详细的导弹发射日志
            distance_km = self._calculate_distance(aircraft, target)/1000
            logging.info(f"🚀 MISSILE LAUNCH: {agent_id} -> {target.uid} at t={current_time:.1f}s, "
                        f"distance={distance_km:.1f}km, missile_id={missile_uid}, remaining_missiles={aircraft.num_missiles}")

            # 记录发射时间线
            if not hasattr(self, 'missile_launch_timeline'):
                self.missile_launch_timeline = {}
            self.missile_launch_timeline[agent_id] = {
                'launch_time': current_time,
                'phase': self.current_phase.value,
                'distance': distance_km,
                'target': target.uid,
                'missile_type': missile_type,
                'missile_id': missile_uid
            }

            # 发射协调分析
            if agent_id == "A0100":
                logging.info(f"   🎯 长机A0100发射: {current_time:.1f}s, 阶段={self.current_phase.value}, 距离={distance_km:.1f}km")
            elif agent_id == "A0200":
                logging.info(f"   🎯 僚机A0200发射: {current_time:.1f}s, 阶段={self.current_phase.value}, 距离={distance_km:.1f}km")
                # 检查与长机的协调
                if "A0100" in self.missile_launch_timeline:
                    leader_launch = self.missile_launch_timeline["A0100"]
                    time_diff = current_time - leader_launch['launch_time']
                    logging.info(f"   📊 僚机发射延迟: {time_diff:.1f}s (相对于长机)")
            elif agent_id.startswith("B"):
                logging.info(f"   🎯 敌方{agent_id}发射: {current_time:.1f}s, 阶段={self.current_phase.value}, 距离={distance_km:.1f}km")

            # 打印发射状态汇总
            logging.info(f"📊 导弹发射状态汇总:")
            for launcher, data in self.missile_launch_timeline.items():
                logging.info(f"   {launcher}: {data['launch_time']:.1f}s, {data['phase']}, {data['distance']:.1f}km -> {data['target']}")

        except Exception as e:
            logging.error(f"导弹发射失败 {agent_id}: {e}")
            import traceback
            traceback.print_exc()
