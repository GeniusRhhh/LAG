"""Reengagement and DR decision helpers extracted from TacticalTask."""

import logging

from envs.JSBSim.core.catalog import Catalog as c
from core.target_assignment import get_enemy_team


def _start_formation_reset_procedure(self, env, agent_id: str, current_time: float):
    """启动队形重置程序 - 新的二次进攻准备"""
    formation_label = self._get_formation_label(agent_id)
    logging.info(f"🔄 [{formation_label}] {agent_id} 启动队形重置程序，准备二次进攻")

    # 通知战术执行器开始队形重置
    formation_agents = self._get_alive_formation_agents(
        env,
        agent_id,
        preserve_nominal_pair=True,
    )
    for aid in formation_agents:
        # 标记该飞机进入队形重置状态
        if not hasattr(self, 'formation_reset_agents'):
            self.formation_reset_agents = set()
        self.formation_reset_agents.add(aid)

    logging.info(f"📋 [{formation_label}] 队形重置参与飞机: {list(formation_agents)}")


def _handle_dr_reengage_unified(self, env, agent_id: str, current_time: float,
                                enemies_alive: int, threat_level: float):
    """🔥 修复2: 统一的DR重新交战处理"""
    formation_label = self._get_formation_label(agent_id, env)
    formation_agents = self._get_alive_formation_agents(
        env,
        agent_id,
        preserve_nominal_pair=True,
    )
    formation_manager = self._get_formation_reset_manager(agent_id)

    override_plan = {}
    if agent_id.startswith('A') and hasattr(self, '_build_red_reengage_override_plan'):
        try:
            override_plan = self._build_red_reengage_override_plan(
                env,
                agent_id,
                current_phase_name="DR_MAR",
                reason="dr_reengage",
                allow_template_reset=True,
            )
            formation_agents = override_plan.get("formation_agents", formation_agents) or formation_agents
        except Exception:
            override_plan = {}
    replacement_tactic = str(override_plan.get("replacement_tactic", "") or "")
    if replacement_tactic == 'DEFENSIVE_GUARD':
        if hasattr(self, '_deescalate_red_formation_to_defense'):
            self._deescalate_red_formation_to_defense(
                env,
                agent_id,
                reason="dr_reengage_guard",
            )
        else:
            for aid in formation_agents or [agent_id]:
                self.returning_agents.discard(aid)
                self.set_agent_second_attack(aid, False)
                self._set_agent_tactic(aid, 'DEFENSIVE_GUARD')
        logging.info(f"[DR决策-{formation_label}] {agent_id} 保持守区防御，不转入前出追击")
        return

    for aid in formation_agents or [agent_id]:
        self.returning_agents.discard(aid)
        self.set_agent_second_attack(aid, True)

    if len(formation_agents) < 2:
        solo_tactic = replacement_tactic or (
            'ADAPTIVE_ATTACK'
            if bool((override_plan.get("posture", {}) or {}).get("release_for_engage", False))
            else 'DEFENSIVE_GUARD'
        )
        if solo_tactic == 'DEFENSIVE_GUARD':
            if hasattr(self, '_deescalate_red_formation_to_defense'):
                self._deescalate_red_formation_to_defense(
                    env,
                    agent_id,
                    reason="dr_reengage_single_guard",
                )
            self.set_agent_second_attack(agent_id, False)
        self._set_agent_tactic(agent_id, solo_tactic)
        logging.info(f"🎯 [DR决策-{formation_label}] {agent_id} 当前仅单机存活，切换到{solo_tactic}")
        return

    if replacement_tactic == 'ADAPTIVE_ATTACK':
        for aid in formation_agents:
            self._set_agent_tactic(aid, 'ADAPTIVE_ATTACK')
        try:
            formation_manager.active = False
            formation_manager.phase = 'inactive'
            formation_manager.active_agents = []
            formation_manager.zero_heading_hold_start = None
        except Exception:
            pass
        logging.info(f"[DR决策-{formation_label}] {agent_id} 进入守区内反击窗口，直接切换ADAPTIVE_ATTACK")
        return

    if not self._second_attack_tactic_selected_by_formation.get(formation_label, False):
        # 选择二次进攻战术
        second_attack_tactic = self._select_second_attack_tactic(env, agent_id, enemies_alive, threat_level)
        # 保存选择的战术
        if not hasattr(self, '_selected_second_attack_tactic'):
            self._selected_second_attack_tactic = {}
        for aid in formation_agents:
            self._selected_second_attack_tactic[aid] = second_attack_tactic
        self._second_attack_tactic_selected_by_formation[formation_label] = True
        role = "长机" if self._is_formation_lead(env, agent_id) else "僚机"
        logging.info(f"⚡ [DR决策-{formation_label}-{role}] 选择二次进攻战术: {second_attack_tactic}")

    # 🔥 关键修复：无论长机还是僚机，都应该启动/加入队形重置
    if not formation_manager.is_reset_active():
        # 队形重置未激活，主动启动
        success = formation_manager.start_formation_reset(env, current_time, formation_agents)
        if success:
            role = "长机" if self._is_formation_lead(env, agent_id) else "僚机"
            logging.info(f"⚡ [DR决策-{formation_label}-{role}] 启动队形重置程序")
            logging.info(f"   → 决定重新进攻！敌机{enemies_alive}架，威胁{threat_level:.2f}")
            # 切换到队形重置模式
            self._set_agent_tactic(agent_id, 'FORMATION_RESET')
        else:
            logging.warning(f"⚠️ [DR决策] 队形重置启动失败，切换到直接二次进攻")
            # 🔥 修复：启动失败时直接切换到UNIFIED_SECOND_ATTACK
            self._set_agent_tactic(agent_id, 'UNIFIED_SECOND_ATTACK')
            logging.info(f"⚡ [DR决策-{formation_label}] 切换到UNIFIED_SECOND_ATTACK战术")
    else:
        # 队形重置已在进行中，加入
        role = "长机" if self._is_formation_lead(env, agent_id) else "僚机"
        logging.info(f"⚡ [DR决策-{formation_label}-{role}] 加入进行中的队形重置程序")
        self._set_agent_tactic(agent_id, 'FORMATION_RESET')


def _execute_formation_reset_procedure(self, env, agent_id: str) -> tuple:
    """🔥 修复: 执行队形重置程序 - 确保真正执行编队重整"""
    # 🔥 调试：只打印我方（A开头）的函数调用追踪
    current_time = env.current_step * env.time_interval
    if agent_id.startswith('A') and env.current_step % 60 == 0:
        pos = env.agents[agent_id].get_position()
        heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        logging.warning(
            f"🔍 [函数调用-{agent_id}] 进入_execute_formation_reset_procedure, "
            f"位置=({pos[0]/1000:.1f}, {pos[1]/1000:.1f}, {pos[2]:.1f}m), 航向={heading:.1f}°, 时间={current_time:.1f}s"
        )

    # 🔥 关键修复：确保在FORMATION_RESET状态下不会执行其他战术
    if agent_id.startswith('A'):
        logging.debug(f"🔄 [{agent_id}] 执行编队重整机动")

    # 使用统一的队形重置管理器
    formation_label = self._get_formation_label(agent_id)
    formation_manager = self._get_formation_reset_manager(agent_id)
    maneuver_command = formation_manager.execute_formation_reset(env, agent_id)

    # 获取我方存活飞机列表
    our_agents = self._get_alive_formation_agents(
        env,
        agent_id,
        preserve_nominal_pair=True,
    )

    # 检查队形重置是否完成 / 失败
    if formation_manager.is_formation_reset_complete(env, our_agents):
        # ✅ 只有在几何条件全部满足时，才认为编队重整真正完成
        # 🔥 问题5修复：使用DR节点选择的二次进攻战术
        if hasattr(self, '_selected_second_attack_tactic') and agent_id in self._selected_second_attack_tactic:
            second_attack_tactic = self._selected_second_attack_tactic[agent_id]
            logging.info(f"✅ [{formation_label}] {agent_id} 队形重整完成，使用DR节点选择的战术: {second_attack_tactic}")
            # 将选择的战术传递给二次进攻系统
            if not hasattr(self.executor, 'second_attack_coordination'):
                self.executor.second_attack_coordination = {}
            if agent_id not in self.executor.second_attack_coordination:
                self.executor.second_attack_coordination[agent_id] = {}
            self.executor.second_attack_coordination[agent_id]['tactic'] = second_attack_tactic
            # 为所有我方飞机统一设置战术
            for aid in our_agents:
                if aid not in self.executor.second_attack_coordination:
                    self.executor.second_attack_coordination[aid] = {}
                self.executor.second_attack_coordination[aid]['tactic'] = second_attack_tactic
                self.executor.second_attack_coordination[aid]['ready'] = True
                self.executor.second_attack_coordination[aid]['tactical_info'] = {'mtr_prime_completed': True}

        self._set_agent_tactic(agent_id, 'UNIFIED_SECOND_ATTACK')
        logging.info(f"✅ [{formation_label}] {agent_id} 队形重整完成（满足几何条件），切换到统一二次进攻模式")
        return self._execute_unified_second_attack(env, agent_id)
    if not formation_manager.is_reset_active():
        # ⏱️ 重整过程已结束但未收敛（例如 final_convergence 超时）：视为失败，直接进入二次进攻
        self._set_agent_tactic(agent_id, 'UNIFIED_SECOND_ATTACK')
        logging.warning(f"⚠️ [{formation_label}] {agent_id} 编队重整未达到目标条件（例如间距/前后/速度未收敛），使用当前队形直接进入统一二次进攻")
        return self._execute_unified_second_attack(env, agent_id)

    # 🔥 仍在重整过程中的阶段，继续执行编队重整机动
    return maneuver_command


def _execute_unified_second_attack(self, env, agent_id: str) -> tuple:
    """执行统一的二次进攻"""
    return self.executor.execute_unified_second_attack(env, agent_id)


def _check_intelligent_reengagement_active(self, agent_id: str) -> bool:
    """检查是否正在执行智能重新交战"""
    if not hasattr(self, 'intelligent_reengagement_states'):
        return False
    return agent_id in self.intelligent_reengagement_states


def _select_second_attack_tactic(self, env=None, agent_id=None, enemies_alive=0, threat_level=0.0) -> str:
    """
    选择二次进攻战术 - 基于敌方态势智能选择

    可选战术模板：
    1. DRAG_SHOOT (拖曳射击) - 长机诱敌前出，僚机后方射击
       适用场景：敌机数量较多、敌方采取进攻姿态
    2. PINCER_ATTACK (钳形攻势) - 双机左右包夹
       适用场景：敌机分散、敌方采取防御/中立姿态
    3. HIGH_LOW_ATTACK (上下夹击) - 双机高低协同攻击
       适用场景：敌机集中、需要立体打击
    4. SIDE_BY_SIDE (并排射击) - 双机并排同时射击
       适用场景：2v1局面
    """
    # 🔥 问题5修复：如果已有选择的战术，直接返回
    if hasattr(self, '_selected_second_attack_tactic') and agent_id and agent_id in self._selected_second_attack_tactic:
        return self._selected_second_attack_tactic[agent_id]

    # 🔥 问题5修复：在DR节点决策时就选择战术，使用传入的参数
    if enemies_alive > 0 or threat_level > 0:
        if enemies_alive == 1:
            # 2v1局面：优先使用并排射击
            return 'SIDE_BY_SIDE'
        if threat_level < 0.5:
            # 威胁值低：使用拖曳射击
            return 'DRAG_SHOOT'
        if threat_level < 0.7:
            # 威胁值中等：使用钳形攻势
            return 'PINCER_ATTACK'
        # 威胁值高：使用高低攻击
        return 'HIGH_LOW_ATTACK'

    # 如果提供了env，使用env获取信息
    if env is not None:
        # 如果是单机剩余，选择DRAG_SHOOT (单机也能执行)
        my_formation_agents = self._get_alive_formation_agents(
            env,
            agent_id or 'A0100',
            preserve_nominal_pair=True,
        )
        enemy_formation_agents = [eid for eid in get_enemy_team(agent_id or 'A0100')]

        our_alive_count = len(my_formation_agents)

        if our_alive_count == 1:
            logging.info("🎯 [二次进攻战术] 单机剩余，选择ADAPTIVE_ATTACK")
            return "ADAPTIVE_ATTACK"

        # 🔥 基于敌方态势智能选择战术
        try:
            if hasattr(self, 'situation_algorithm_switcher'):
                # 获取敌方存活情况
                enemy_alive = [aid for aid in enemy_formation_agents
                              if aid in env.agents and env.agents[aid].is_alive]
                enemy_count = len(enemy_alive)

                # 获取敌方意图（使用态势识别算法）
                enemy_intent = "ATTACK"  # 默认
                if enemy_alive:
                    my_aircraft_list = list(my_formation_agents)
                    my_aircraft_obj = env.agents.get(my_aircraft_list[0]) if my_aircraft_list else None
                    enemy_aircraft_obj = env.agents.get(enemy_alive[0])
                    if my_aircraft_obj is not None and enemy_aircraft_obj is not None:
                        enemy_intent = self.situation_algorithm_switcher.recognize_intent(
                            enemy_aircraft_obj, my_aircraft_obj, env
                        )

                # 根据敌方意图选择战术
                logging.info(f"🔍 [二次进攻决策] 敌机存活:{enemy_count}, 敌方意图:{enemy_intent}")

                if enemy_intent in ["RETREAT", "DEFENSIVE"]:
                    # 敌方撤退/防御：使用钳形攻势追击包夹
                    selected = "PINCER_ATTACK"
                    logging.info(f"🎯 [二次进攻战术] 敌方{enemy_intent}，选择钳形攻势追击: {selected}")
                elif enemy_intent == "ATTACK" and enemy_count >= 2:
                    # 敌方进攻且数量多：使用拖曳射击，利用诱敌战术
                    selected = "DRAG_SHOOT"
                    logging.info(f"🎯 [二次进攻战术] 敌方进攻(敌{enemy_count}机)，选择拖曳射击: {selected}")
                elif enemy_count == 1:
                    # 2v1局面：使用并排射击
                    selected = "SIDE_BY_SIDE"
                    logging.info(f"🎯 [二次进攻战术] 2v1局面，选择并排射击: {selected}")
                else:
                    # 默认：拖曳射击（最稳妥）
                    selected = "DRAG_SHOOT"
                    logging.info(f"🎯 [二次进攻战术] 默认选择拖曳射击: {selected}")

                return selected
        except Exception as e:
            logging.warning(f"二次进攻战术智能选择失败: {e}")

    # 兜底策略：默认选择DRAG_SHOOT
    logging.info("🎯 [二次进攻战术] 兜底策略，选择DRAG_SHOOT")
    return "DRAG_SHOOT"


def _is_second_attack_ready_to_fire(self, agent_id: str = None) -> bool:
    """检查二次进攻是否准备好发射导弹（agent_id=None表示按全编队聚合检查）"""
    # 检查队形重置是否完成
    if agent_id is not None:
        formation_manager = self._get_formation_reset_manager(agent_id)
        if formation_manager.is_reset_active():
            return False
    elif any(manager.is_reset_active() for manager in getattr(self, '_formation_reset_managers', {}).values()):
        # 队形重置仍在进行中，不允许发射
        return False

    # ✅ 修复：原实现错误使用了不存在的阶段名'MTR_TR'，导致二次进攻长期“未准备好”
    # 二次进攻的LR'/TR'节点仍复用全局阶段（MTR_LR/LR_TR/TR_DOR...），因此只要不在编队重置中即可允许发射，
    # 由 missile_manager.should_launch_missile() 再做窗口/雷达/朝向的发射门限控制。
    if hasattr(self, 'state_manager'):
        if agent_id is not None:
            current_phase = self.state_manager.get_agent_phase(agent_id)
            if not current_phase:
                return False
            allowed = {'MTR_LR', 'LR_TR', 'TR_DOR', 'DOR_DR', 'DR_MAR', 'BEYOND_MAR'}
            if current_phase.value not in allowed:
                return False
        else:
            our_agents = [aid for aid in getattr(self, 'agent_phases', {}).keys()
                          if str(aid).startswith('A') and hasattr(self, 'env') and self.env and aid in self.env.agents and self.env.agents[aid].is_alive]
            allowed = {'MTR_LR', 'LR_TR', 'TR_DOR', 'DOR_DR', 'DR_MAR', 'BEYOND_MAR'}
            for aid in our_agents:
                current_phase = self.state_manager.get_agent_phase(aid)
                if (not current_phase) or (current_phase.value not in allowed):
                    return False

    # 🔥 修复：控制日志输出频率，避免每步都打印
    if not hasattr(self, '_second_attack_ready_logged'):
        self._second_attack_ready_logged = False
        self._last_ready_log_step = -1

    # 获取当前步数
    current_step = getattr(self, 'env', None) and getattr(self.env, 'current_step', 0) or 0

    # 只在状态首次变为准备就绪时打印，或者每200步（40秒）打印一次
    if (not self._second_attack_ready_logged or
        current_step - self._last_ready_log_step >= 200):
        if agent_id is not None:
            logging.info(f"✅ [导弹控制-{self._get_formation_label(agent_id)}] 二次进攻准备就绪，允许导弹发射")
        else:
            logging.info("✅ [导弹控制] 二次进攻准备就绪，允许导弹发射")
        self._second_attack_ready_logged = True
        self._last_ready_log_step = current_step

    return True


def _handle_dr_decision_unified(self, env, agent_id: str, current_time: float):
    """
    统一处理DR节点重新进攻决策，避免重复和冲突
    """
    # 检查是否已经做出DR进攻决策
    formation_label = self._get_formation_label(agent_id)
    formation_agents = self._get_alive_formation_agents(
        env,
        agent_id,
        preserve_nominal_pair=True,
    )

    if self._dr_reengage_decision_made.get(formation_label, False):
        return  # 已决策，跳过

    # 只有长机负责DR进攻决策，僚机跟随
    if not self._is_formation_lead(env, agent_id):
        return  # 非长机，跳过决策

    # 标记决策已完成，防止重复
    self._dr_reengage_decision_made[formation_label] = True

    # 执行统一的威胁评估
    enemies_alive = _count_alive_enemies(self, env)
    threat_level = _calculate_comprehensive_threat_level(self, env)

    logging.info(f"⚡ [DR决策-统一-{formation_label}] 长机{agent_id}决定重新进攻！敌机{enemies_alive}架，威胁{threat_level:.2f}")

    # 启动队形重置程序
    _start_formation_reset_procedure(self, env, agent_id, current_time)

    # 通知所有友方飞机切换到队形重置模式
    for aid in formation_agents:
        # 统一设置战术
        self._set_agent_tactic(aid, 'FORMATION_RESET')
        logging.info(f"📋 [{formation_label}] {aid} 接收DR进攻决策，切换到FORMATION_RESET模式")


def _handle_dr_retreat_unified(self, env, agent_id: str, current_time: float):
    """
    统一处理DR节点撤退决策，避免重复和冲突
    """
    # 🔥 关键修复：如果正在执行FORMATION_RESET，不允许撤退决策干扰
    if self._get_agent_tactic(agent_id) == 'FORMATION_RESET':
        logging.info(f"🛡️ [DR撤退决策] {agent_id} 正在执行编队重整，忽略撤退决策")
        return

    # 检查是否已经做出DR撤退决策
    formation_label = self._get_formation_label(agent_id)
    formation_agents = self._get_alive_formation_agents(
        env,
        agent_id,
        preserve_nominal_pair=True,
    )

    alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)

    if agent_id.startswith('A') and alive_enemy_exists:
        preserve_state = self._preserve_red_defense_if_needed(
            env,
            agent_id,
            current_phase_name="dr_retreat",
            reason="dr_retreat",
            log_tag="DR_RETREAT_PRESERVE",
        ) if hasattr(self, '_preserve_red_defense_if_needed') else None
        preserve_defense = bool(preserve_state)
        if preserve_state is None:
            try:
                preserve_defense = bool(
                    self._evaluate_red_defensive_posture(
                        env,
                        agent_id,
                        current_phase_name="dr_retreat",
                        reason="dr_retreat",
                    ).get("preserve", False)
                )
            except Exception:
                preserve_defense = False
        if preserve_defense:
            self._dr_retreat_decision_made[formation_label] = True
            if preserve_state is not None:
                logging.info(
                    "🛡️ [DR撤退保留-%s] reason=%s",
                    formation_label,
                    preserve_state.get("reason", "dr_retreat"),
                )
            elif hasattr(self, '_deescalate_red_formation_to_defense'):
                self._deescalate_red_formation_to_defense(
                    env,
                    agent_id,
                    reason="dr_retreat",
                )
            else:
                for aid in formation_agents:
                    self.returning_agents.discard(aid)
                    self.set_agent_second_attack(aid, False)
                    self._set_agent_tactic(aid, 'DEFENSIVE_GUARD')
            logging.info(f"🛡️ [DR撤退保留-{formation_label}] 敌机仍存活，但当前保持守区防御态势")
            return

        override_plan = {}
        if hasattr(self, '_build_red_reengage_override_plan'):
            try:
                override_plan = self._build_red_reengage_override_plan(
                    env,
                    agent_id,
                    current_phase_name="dr_retreat",
                    reason="dr_retreat",
                    allow_template_reset=True,
                )
                formation_agents = override_plan.get("formation_agents", formation_agents) or formation_agents
            except Exception:
                override_plan = {}
        replacement_tactic = str(
            override_plan.get(
                "replacement_tactic",
                'ADAPTIVE_ATTACK' if len(formation_agents) < 2 else 'FORMATION_RESET',
            )
        )

        self._dr_retreat_decision_made[formation_label] = True
        if replacement_tactic == 'DEFENSIVE_GUARD':
            if hasattr(self, '_deescalate_red_formation_to_defense'):
                self._deescalate_red_formation_to_defense(
                    env,
                    agent_id,
                    reason="dr_retreat_guard",
                )
            else:
                for aid in formation_agents:
                    self.returning_agents.discard(aid)
                    self.set_agent_second_attack(aid, False)
                    self._set_agent_tactic(aid, 'DEFENSIVE_GUARD')
            logging.info(f"[DR撤退拦截-{formation_label}] 敌机仍存活，但当前保持守区防御")
            return

        for aid in formation_agents:
            self.returning_agents.discard(aid)
            self.set_agent_second_attack(aid, True)
            self._set_agent_tactic(aid, replacement_tactic)

        if replacement_tactic == 'FORMATION_RESET' and formation_agents:
            formation_manager = self._get_formation_reset_manager(agent_id)
            if hasattr(formation_manager, 'start_formation_reset') and (not formation_manager.is_reset_active()):
                try:
                    formation_manager.start_formation_reset(env, current_time, formation_agents)
                except Exception:
                    pass
        elif formation_agents:
            formation_manager = self._get_formation_reset_manager(agent_id)
            if formation_manager is not None:
                try:
                    formation_manager.active = False
                    formation_manager.phase = 'inactive'
                    formation_manager.active_agents = []
                    formation_manager.zero_heading_hold_start = None
                except Exception:
                    pass

        logging.info(f"🛑 [DR撤退拦截-{formation_label}] 敌机仍存活，撤退决策改为{replacement_tactic}")
        return

    if self._dr_retreat_decision_made.get(formation_label, False):
        return  # 已决策，跳过

    # 只有长机负责DR撤退决策，僚机跟随
    if not self._is_formation_lead(env, agent_id):
        return  # 非长机，跳过决策

    # 标记决策已完成，防止重复
    self._dr_retreat_decision_made[formation_label] = True

    # 执行统一的威胁评估
    enemies_alive = _count_alive_enemies(self, env)
    threat_level = _calculate_comprehensive_threat_level(self, env)

    logging.info(f"⚡ [DR决策-统一-{formation_label}] 长机{agent_id}决定撤退。敌机{enemies_alive}架，威胁{threat_level:.2f}")

    # 通知所有友方飞机执行统一撤退
    for aid in formation_agents:
        self._set_agent_tactic(aid, 'TACTICAL_TURN')
        logging.info(f"📋 [{formation_label}] {aid} 接收DR撤退决策，执行180°回转返航")


def _count_alive_enemies(self, env) -> int:
    """计算存活敌机数量"""
    return sum(1 for aid in env.agents.keys() if aid.startswith('B') and env._jsbsims[aid].is_alive)


def _calculate_comprehensive_threat_level(self, env) -> float:
    """计算综合威胁等级"""
    # 简化威胁计算（可以根据需要扩展）
    enemies_alive = _count_alive_enemies(self, env)
    our_alive = sum(1 for aid in env.agents.keys() if aid.startswith('A') and env._jsbsims[aid].is_alive)

    if our_alive == 0:
        return 1.0  # 最高威胁

    base_threat = enemies_alive / (our_alive + enemies_alive)

    # 可以根据距离、导弹数量等因素进一步调整
    return min(base_threat, 1.0)
