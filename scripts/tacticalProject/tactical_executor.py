"""
战术执行模块
包含所有战术执行函数：拖曳射击、钳形攻势、上下夹击、前后攻击、并排射击等
从tactical_task.py中提取，提高代码可维护�?
"""
import logging
import numpy as np
from tactical_types import TacticalPhase
from envs.JSBSim.core.catalog import Catalog as c
from core.target_assignment import get_target_with_fallback
from formation_reset_manager import FormationResetManager
from nodes.mtr_prime_node import MTRPrimeNode


class TacticalExecutor:
    """战术执行器 - 负责执行各种战术"""

    def __init__(self, tactical_task):
        """
        Args:
            tactical_task: TacticalTask实例，用于访问共享状态和工具函数
        """
        self.task = tactical_task
        self.evasion_states = {}
        self.turn_states = {}

        # 初始化队形重置管理器
        self.formation_reset_manager = FormationResetManager()

        # 初始化MTR'节点
        self.mtr_prime_node = MTRPrimeNode()

        # 初始化二次进攻协调状态
        self.second_attack_coordination = {}
        self.second_attack_tactic_selected = False

        # 🎯 日志频率控制 - 减少高频非关键日志
        self.last_log_times = {}  # 记录上次打印时间
        self.log_intervals = {
            'tactical_turn_protection': 10.0,  # TACTICAL_TURN高度保护日志间隔10秒
            'high_low_attack_info': 15.0,      # HIGH_LOW_ATTACK战术信息间隔15秒
            'front_back_info': 15.0,           # FRONT_BACK战术信息间隔15秒
            'debug_heading': 20.0,             # 航向调试信息间隔20秒
            'missile_status': 5.0,             # 导弹状态信息间隔5秒
        }

    def should_log(self, log_key: str, current_time: float) -> bool:
        """
        检查是否应该打印日志（基于时间间隔控制）

        Args:
            log_key: 日志类型键
            current_time: 当前时间

        Returns:
            bool: 是否应该打印日志
        """
        if log_key not in self.log_intervals:
            return True  # 未配置的日志类型默认打印

        last_time = self.last_log_times.get(log_key, 0)
        interval = self.log_intervals[log_key]

        if current_time - last_time >= interval:
            self.last_log_times[log_key] = current_time
            return True
        return False
    
    def execute_drag_shoot(self, env, agent_id: str) -> tuple:
        """
        战术1: 拖曳射击 (DRAG_SHOOT)
        核心思想: 长机诱敌，僚机射击
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        
        # 二次进攻特殊处理
        if getattr(self.task, 'is_second_attack', False):
            selected_tactic = getattr(self.task, 'selected_tactic', 'DRAG_SHOOT')
            if selected_tactic == 'DRAG_SHOOT':
                return self._execute_second_attack_drag_shoot(env, agent_id, is_lead, current_time)
            elif selected_tactic == 'PINCER_ATTACK':
                return self._execute_second_attack_pincer(env, agent_id, is_lead, current_time)
            elif selected_tactic == 'HIGH_LOW_ATTACK':
                return self._execute_second_attack_high_low(env, agent_id, is_lead, current_time)
            else:
                # 默认使用拖曳射击
                return self._execute_second_attack_drag_shoot(env, agent_id, is_lead, current_time)
        
        # 调试：确认方法被调用
        if env.current_step % 60 == 0 and agent_id == 'A0100':
            logging.info(f"🎯 [DRAG_SHOOT] {agent_id} execute_drag_shoot被调用，当前阶段: {self.task.current_phase.value}")
        
        # 检查Short Skate状态
        if agent_id in self.task.short_skate_states:
            return self.task._execute_short_skate_precise(env, agent_id, current_time)
        
        # 长机动作序列
        if is_lead:
            # 调试：打印长机全局阶段
            if env.current_step % 30 == 0:
                logging.info(f"🔍 [拖曳射击-长机] {agent_id} 全局阶段: {self.task.current_phase.value}")
            
            if self.task.current_phase.value in ['NLT_MELD', 'MELD_MTR']:
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'MTR_LR':
                hdg = self._heading_to_attack_wp_or_default(env, agent_id, default_heading=0.0)
                return self.task._maintain_heading_precise(env, agent_id, hdg)
            elif self.task.current_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    logging.info(f"🚀 [DRAG_SHOOT-长机] {agent_id} 设置导弹发射标记")
                    self.task.missile_launched[agent_id] = True
                
                # 🎯 战术优先：DRAG_SHOOT长机在LR_TR阶段保持直飞，不执行Crank
                if env.current_step % 60 == 0:
                    logging.info(f"🎯 [DRAG_SHOOT-长机LR_TR] {agent_id} 战术特定：保持直飞诱敌")
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'TR_DOR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time)
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self.task._execute_short_skate_precise(env, agent_id, current_time)
        
        # 僚机动作序列（独立阶段判断）
        else:
            target_id = get_target_with_fallback(agent_id, env)
            target_aircraft = env._jsbsims.get(target_id)
            if target_aircraft and target_aircraft.is_alive:
                distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
                wingman_phase = self.task._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.task.current_phase
            
            if wingman_phase.value == 'NLT_MELD':
                return self.task._maintain_heading_precise(env, agent_id, 35.0)
            elif wingman_phase.value == 'MELD_MTR':
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase.value == 'MTR_LR':
                hdg = self._heading_to_attack_wp_or_default(env, agent_id, default_heading=0.0)
                return self.task._maintain_heading_precise(env, agent_id, hdg)
            elif wingman_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                # 🎯 战术优先：DRAG_SHOOT僚机在LR_TR阶段轻微偏转350°进行射击
                if env.current_step % 60 == 0:
                    logging.info(f"🎯 [DRAG_SHOOT-僚机LR_TR] {agent_id} 战术特定：50°射击角度")
                return self.task._maintain_heading_precise(env, agent_id, 350.0)
            elif wingman_phase.value == 'TR_DOR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time)
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 350.0)
            else:
                return self.task._execute_short_skate_precise(env, agent_id, current_time)
    
    def execute_pincer_attack(self, env, agent_id: str) -> tuple:
        """
        战术2: 钳形攻势 (Pincer Attack)
        # 核心思想: 双机从两侧包夹敌机
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        
        # 二次进攻特殊处理
        if getattr(self.task, 'is_second_attack', False):
            return self._execute_second_attack_pincer(env, agent_id, is_lead, current_time)
        
        skate_direction = 'right' if is_lead else 'left'
        
        if agent_id in self.task.short_skate_states:
            return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        if is_lead:
            lead_phase = self.task.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
            
            if env.current_step % 60 == 0:
                logging.info(f"  🎯 [钳形攻势-长机] {agent_id} 阶段:{lead_phase.value}")
            
            lead_pos = env.agents[agent_id].get_position()
            wing_pos = env.agents['A0200'].get_position() if env.agents['A0200'].is_alive else lead_pos
            lateral_separation = abs(wing_pos[1] - lead_pos[1])
            target_lateral_separation = 10000
            
            lead_y = lead_pos[1]
            approaching_boundary = abs(lead_y) > 50000
            
            if lead_phase.value == 'NLT_MELD':
                current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                target_heading = 315.0
                heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
                hdg_idx = self.task._convert_heading_to_index(np.deg2rad(heading_diff))
                
                if approaching_boundary or lateral_separation >= target_lateral_separation:
                    if env.current_step % 60 == 0:
                        logging.info(f"     ✅已达到分离，保持315°")
                else:
                    if env.current_step % 60 == 0:
                        logging.info(f"     ✅目标航向:315° (展开)，当前{current_heading:.1f}°")
                return 7, hdg_idx, 3
            elif lead_phase.value == 'MELD_MTR':
                current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                target_heading = 345.0
                heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
                hdg_idx = self.task._convert_heading_to_index(np.deg2rad(heading_diff))
                if env.current_step % 60 == 0:
                    logging.info(f"📍 [MELD分离] 长机目标345°(-15°)，当前{current_heading:.1f}°")
                return 7, hdg_idx, 3
            elif lead_phase.value == 'MTR_LR':
                # 🎯 修复：钳形攻势MTR阶段：长机从345°回调到0°，与僚机的回调形成对称
                if env.current_step % 60 == 0:
                    logging.info(f"[PINCER_ATTACK-长机MTR_LR] {agent_id} 从345°回调到0°")
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif lead_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True

                if last_launch > 0 and (current_time - last_launch) < 15.0:
                    remaining_time = 15.0 - (current_time - last_launch)
                    if env.current_step % 60 == 0:
                        logging.info(f"🎯 [{agent_id}] 导弹制导保护: 剩余{remaining_time:.1f}秒，保持当前姿态")
                    return 7, 8, 3

                # 🎯 修复：PINCER_ATTACK长机在LR_TR阶段执行右侧crank机动（10°），形成钳形夹击
                if env.current_step % 60 == 0:
                    logging.info(f"[PINCER_ATTACK-长机LR_TR] {agent_id} 执行右侧crank机动10°")
                return self.task._maintain_heading_precise(env, agent_id, 10.0)
            elif lead_phase.value == 'TR_DOR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 15.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    # 制导保护期间或其他情况，保持当前航向
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif lead_phase.value == 'DOR_DR':
                # DOR_DR阶段：钳形夹击收拢，长机从左侧向内侧（东向90°）收拢
                if env.current_step % 60 == 0:
                    logging.info(f"[PINCER_ATTACK-长机DOR_DR] {agent_id} 从左侧向内收拢90°")
                return self.task._maintain_heading_precise(env, agent_id, 90.0)
            elif lead_phase.value == 'DR_MAR':
                # DR_MAR阶段：继续保持收拢姿态，不重置航向
                if env.current_step % 60 == 0:
                    logging.info(f"[PINCER_ATTACK-长机DR_MAR] {agent_id} 保持收拢姿态90°")
                return self.task._maintain_heading_precise(env, agent_id, 90.0)
            elif lead_phase.value == 'BEYOND_MAR':
                # 超越MAR阶段：执行Short Skate机动返航180°
                if env.current_step % 60 == 0:
                    logging.info(f"[PINCER_ATTACK-长机BEYOND_MAR] {agent_id} 执行Short Skate返航")
                return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
            else:
                # 默认情况：直�?
                return 7, 8, 3

    # 占位：辅助函数在类末尾定义，避免打断当前方法的else分支

        # 僚机逻辑
        else:
            target_id = get_target_with_fallback(agent_id, env)
            target_aircraft = env._jsbsims.get(target_id)
            if target_aircraft and target_aircraft.is_alive:
                distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
                wingman_phase = self.task._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.task.current_phase
            
            lead_pos = env.agents['A0100'].get_position() if env.agents['A0100'].is_alive else env.agents[agent_id].get_position()
            wing_pos = env.agents[agent_id].get_position()
            lateral_separation = abs(wing_pos[1] - lead_pos[1])
            target_lateral_separation = 10000
            
            wing_y = wing_pos[1]
            approaching_boundary = abs(wing_y) > 50000
            
            if wingman_phase.value == 'NLT_MELD':
                # 初期小角度右�?5°（原版设定），缩短Crank时间
                target_heading = 35.0
                return self.task._maintain_heading_precise(env, agent_id, target_heading)
            elif wingman_phase.value == 'MELD_MTR':
                current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                target_heading = 15.0
                heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
                hdg_idx = self.task._convert_heading_to_index(np.deg2rad(heading_diff))
                if env.current_step % 60 == 0:
                    logging.info(f"📍 [MELD分离] 僚机目标15°(+15°)，当前{current_heading:.1f}°")
                return 7, hdg_idx, 3
            elif wingman_phase.value == 'MTR_LR':
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase.value == 'LR_TR':
                # LR阶段�?8km）：发射主动雷达弹，中制导开�?
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.task.missile_launched[agent_id] = True  # 标记需要发�?
                
                # 🎯 战术优先：PINCER_ATTACK僚机在LR_TR阶段保持右分离姿态50°，不执行额外Crank
                if env.current_step % 60 == 0:
                    logging.info(f"🎯 [PINCER_ATTACK-僚机LR_TR] {agent_id} 战术特定：保持右分离姿态50°")
                return self.task._maintain_heading_precise(env, agent_id, 350.0)
            elif wingman_phase.value == 'TR_DOR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 15.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    return 7, 8, 3
            elif wingman_phase.value == 'DOR_DR':
                # DOR_DR阶段：钳形夹击收拢，僚机从右侧向内侧（西向270°）收拢
                if env.current_step % 60 == 0:
                    logging.info(f"[PINCER_ATTACK-僚机DOR_DR] {agent_id} 从右侧向内收拢270°")
                return self.task._maintain_heading_precise(env, agent_id, 270.0)
            elif wingman_phase.value == 'DR_MAR':
                # DR_MAR阶段：继续保持收拢姿态，不重置航向
                if env.current_step % 60 == 0:
                    logging.info(f"[PINCER_ATTACK-僚机DR_MAR] {agent_id} 保持收拢姿态270°")
                return self.task._maintain_heading_precise(env, agent_id, 270.0)
            elif wingman_phase.value == 'BEYOND_MAR':
                # 超越Mar阶段：执行Short Skate机动返航180°
                if env.current_step % 60 == 0:
                    logging.info(f"[PINCER_ATTACK-僚机BEYOND_MAR] {agent_id} 执行Short Skate返航")
                return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
            else:
                return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)

    def execute_tactical_evasion(self, env, agent_id: str) -> tuple:
        current_time = env.current_step * env.time_interval
        state = self.evasion_states.get(agent_id)
        if state is None:
            ttl = 8.0 if agent_id.endswith('200') else 10.0
            self.evasion_states[agent_id] = {'start': current_time, 'ttl': ttl}
            if not hasattr(self.task, '_defense_prev_tactic'):
                self.task._defense_prev_tactic = self.task.selected_tactic
        # 🎯 修复：钳形攻势期间保持钳形攻势的返航逻辑
        if getattr(self.task, '_defense_prev_tactic', None) == 'PINCER_ATTACK':
            is_lead = agent_id.endswith('100')
            current_phase = self.task.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)

            # 在DOR_DR或DR_MAR阶段，执行钳形攻势的收拢返航
            if current_phase.value in ['DOR_DR', 'DR_MAR']:
                if is_lead:
                    # 长机：从左侧向内收拢90°（向东）
                    if env.current_step % 60 == 0:
                        logging.info(f"[TACTICAL_EVASION-钳形攻势-长机] {agent_id} 保持钳形收拢90°")
                    return self.task._maintain_heading_precise(env, agent_id, 90.0)
                else:
                    # 僚机：从右侧向内收拢270°（向西）
                    if env.current_step % 60 == 0:
                        logging.info(f"[TACTICAL_EVASION-钳形攻势-僚机] {agent_id} 保持钳形收拢270°")
                    return self.task._maintain_heading_precise(env, agent_id, 270.0)

        # 基于RWR威胁等级选择规避方式
        try:
            from simulation.radar_manager import get_unified_radar_manager
            rm = get_unified_radar_manager()
            rwr_level = rm.get_rwr_threat_level(agent_id)
        except Exception:
            rwr_level = 0
        # RWR�?（导弹威胁）：执行Notch Back（快速80°回转下降�?
        if rwr_level >= 4:
            action = self.task.maneuver_lib.execute_notch_back(env, agent_id)
        # RWR=2~3：执行Beam�?/9机动�?
        elif rwr_level >= 2:
            action = self.task.maneuver_lib.execute_beam_maneuver(env, agent_id)
        # 其余：执行简化Short Skate
        else:
            action = self.task._execute_short_skate_precise(env, agent_id, current_time)
        state = self.evasion_states.get(agent_id)
        if state and (current_time - state['start'] >= state['ttl']):
            self.evasion_states.pop(agent_id, None)
            if getattr(self.task, '_defense_prev_tactic', None):
                self.task.selected_tactic = self.task._defense_prev_tactic
        return action

    def execute_tactical_turn(self, env, agent_id: str) -> tuple:
        """执行战术回转机动 - 两次180度转向，最终朝向敌方"""
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        # 初始化回转状态
        if agent_id not in self.turn_states:
            self.turn_states[agent_id] = {
                'start': current_time, 
                'phase': 'first_turn',  # 第一次80度转向
                'phase_start': current_time,
                'initial_heading': current_heading,
                'turn_type': 'standard'
            }
            # 保存之前的战术
            if not hasattr(self.task, '_defense_prev_tactic'):
                self.task._defense_prev_tactic = self.task.selected_tactic
            
            # 检查是否是重新交战回转
            if hasattr(self.task.node_decision_maker, 'last_decision'):
                last_decision = getattr(self.task.node_decision_maker, 'last_decision', {})
                if last_decision.get('turn_type') == 'reengage_turn':
                    self.turn_states[agent_id]['turn_type'] = 'reengage_turn'
                    logging.info(f"🔄 {agent_id} 开始重新交战回转（两次180度）")
                else:
                    logging.info(f"🔄 {agent_id} 开始标准战术回转")
        
        state = self.turn_states[agent_id]
        phase_time = current_time - state['phase_start']
        total_time = current_time - state['start']
        
        # 第一阶段：第一次80度转向（12秒）- 增加时间确保完成180度转向
        if state['phase'] == 'first_turn':
            if phase_time < 12.0:
                # 🛡️ 超强化高度保护检查（特别针对急转弯）
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                if current_altitude < 2000:  # 危险高度：禁止TACTICAL_TURN，直接爬升
                    alt_cmd = 14  # 最大爬升+1500m
                    hdg_cmd = 8   # 保持航向，不转弯
                    vel_cmd = 2   # 减速以确保安全
                    if env.current_step % 30 == 0:
                        logging.error(f"🚨🚨 [{agent_id}] TACTICAL_TURN危险高度 {current_altitude:.0f}m < 2000m，禁止转弯，强制爬升")
                    return alt_cmd, hdg_cmd, vel_cmd  # 🎯 修复：正确的参数顺序
                elif current_altitude < 3000:  # 警告高度：限制转弯幅度
                    alt_cmd = 12  # 大幅爬升+500m
                    if self.should_log('tactical_turn_protection', current_time):
                        logging.warning(f"🛡️ [{agent_id}] TACTICAL_TURN警告高度: {current_altitude:.0f}m < 3000m，限制转弯+爬升")
                elif current_altitude < 4000:  # 预警高度：轻微爬升
                    alt_cmd = 9  # 轻微爬升+50m
                    if self.should_log('tactical_turn_protection', current_time):
                        logging.warning(f"🛡️ [{agent_id}] TACTICAL_TURN预警高度: {current_altitude:.0f}m < 4000m，轻微爬升")
                else:
                    alt_cmd = 7  # 保持高度

                # 执行急转弯实现80度转向 - 🎯 修复：正确的参数顺序(alt_cmd, hdg_cmd, vel_cmd)
                if agent_id.endswith('100'):  # 长机左转
                    return alt_cmd, 0, 3  # 高度保护 + 急左转 + 保持速度
                else:  # 僚机右转
                    return alt_cmd, 16, 3  # 高度保护 + 急右转 + 保持速度
            else:
                # 第一次转向完成，进入间隔阶段
                state['phase'] = 'hold_heading'
                state['phase_start'] = current_time
                logging.info(f"🔄 {agent_id} 第一次180度转向完成，保持航向5秒")
        
        # 间隔阶段：保持当前航向（5秒）- 增加稳定时间
        elif state['phase'] == 'hold_heading':
            if phase_time < 5.0:
                # 强化高度保护检查（间隔阶段也要保护
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                if current_altitude < 1000:
                    alt_cmd = 10  # 大幅爬升
                    if env.current_step % 30 == 0:
                        logging.error(f"🚨🚨 [{agent_id}] TACTICAL_TURN间隔极危{current_altitude:.0f}m < 1000m")
                elif current_altitude < 2000:
                    alt_cmd = 9  # 爬升
                    if env.current_step % 60 == 0:
                        logging.warning(f"🚨 [{agent_id}] TACTICAL_TURN间隔危险高度: {current_altitude:.0f}m < 2000m")
                elif current_altitude < 4000:
                    alt_cmd = 8  # 轻微爬升
                    if env.current_step % 60 == 0:
                        logging.warning(f"🛡{agent_id}] TACTICAL_TURN间隔高度保护: {current_altitude:.0f}m < 4000m")
                else:
                    alt_cmd = 7  # 保持高度
                
                return alt_cmd, 7, 3  # 高度保护 + 直飞 + 保持速度
            else:
                # 间隔完成，进入第二次转向
                state['phase'] = 'second_turn'
                state['phase_start'] = current_time
                logging.info(f"🔄 {agent_id} 间隔完成，开始第二次180度转向")
        
        # 第二阶段：第二次180度转向（12秒）- 充足时间完成回转
        elif state['phase'] == 'second_turn':
            if phase_time < 12.0:
                # 强化高度保护检查（第二次转向更要小心）
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                if current_altitude < 1000:  # 极危险：最强爬升
                    alt_cmd = 10  # 大幅爬升
                    if env.current_step % 30 == 0:
                        logging.error(f"🚨🚨 [{agent_id}] TACTICAL_TURN第二次转向极危险: {current_altitude:.0f}m < 1000m")
                elif current_altitude < 2000:  # 很危险：强力爬升
                    alt_cmd = 9  # 爬升
                    if env.current_step % 60 == 0:
                        logging.warning(f"🚨 [{agent_id}] TACTICAL_TURN第二次转向危{current_altitude:.0f}m < 2000m")
                elif current_altitude < 4000:  # 警告：普通爬�?
                    alt_cmd = 8  # 轻微爬升
                    if env.current_step % 60 == 0:
                        logging.warning(f"🛡{agent_id}] TACTICAL_TURN第二次转向保�? {current_altitude:.0f}m < 4000m")
                else:
                    alt_cmd = 7  # 保持高度
                
                # 执行反向急转弯，回到初始方向 - 🎯 修复：正确的参数顺序
                if agent_id.endswith('100'):  # 长机右转回到0度
                    return alt_cmd, 16, 3  # 高度保护 + 急右转 + 保持速度
                else:  # 僚机左转回到0度
                    return alt_cmd, 0, 3  # 高度保护 + 急左转 + 保持速度
            else:
                # 第二次转向完成，进入航向调整阶段
                state['phase'] = 'heading_adjust'
                state['phase_start'] = current_time
                logging.info(f"🔄 {agent_id} 第二次180度转向完成，调整航向到0度")
        
        # 第三阶段：精确调整航向至0度（6秒）- 确保精确到位
        elif state['phase'] == 'heading_adjust':
            if phase_time < 6.0:
                # 最终调整阶段的高度保护
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                if current_altitude < 1000:  # 极危险
                    alt_cmd = 10  # 大幅爬升
                    if env.current_step % 30 == 0:
                        logging.error(f"🚨🚨 [{agent_id}] TACTICAL_TURN调整阶段极危险{current_altitude:.0f}m < 1000m")
                elif current_altitude < 2000:  # 危险
                    alt_cmd = 9   # 爬升
                    if env.current_step % 60 == 0:
                        logging.warning(f"🚨 [{agent_id}] TACTICAL_TURN调整阶段危险: {current_altitude:.0f}m < 2000m")
                elif current_altitude < 3000:  # 警告
                    alt_cmd = 8   # 轻微爬升
                    if env.current_step % 60 == 0:
                        logging.warning(f"🛡️ [{agent_id}] TACTICAL_TURN调整阶段保护: {current_altitude:.0f}m < 3000m")
                else:
                    alt_cmd = 7   # 保持高度
                
                # 精确调整�?度航向（修正目标航向�?
                target_heading = 0.0  # 明确目标航向为北向（0�?360度）
                heading_diff = (target_heading - current_heading + 360) % 360
                if heading_diff > 180:
                    heading_diff -= 360
                
                if abs(heading_diff) < 5:
                    return alt_cmd, 8, 3  # 高度保护 + 直飞 + 保持速度
                elif heading_diff > 0:
                    return alt_cmd, 10, 3  # 高度保护 + 轻微右转 + 保持速度
                else:
                    return alt_cmd, 6, 3  # 高度保护 + 轻微左转 + 保持速度
            else:
                # 战术回转完全完成
                turn_type = state.get('turn_type', 'standard')
                self.turn_states.pop(agent_id, None)
                
                if turn_type == 'reengage_turn':
                    # 重新交战回转完成，切换到攻击战术
                    logging.info(f"✅{agent_id} 重新交战回转完成，开始二次进攻")
                    if hasattr(self.task.node_decision_maker, 'last_decision'):
                        last_decision = getattr(self.task.node_decision_maker, 'last_decision', {})
                        second_attack_tactic = last_decision.get('second_attack_tactic', 'PINCER_ATTACK')
                        self.task.selected_tactic = second_attack_tactic
                        self.task.is_second_attack = True
                        logging.info(f"🎯 {agent_id} 切换到二次进攻战术 {second_attack_tactic}")
                    else:
                        self.task.selected_tactic = 'PINCER_ATTACK'  # 默认战术
                else:
                    # 标准战术回转完成，恢复之前战术
                    if getattr(self.task, '_defense_prev_tactic', None):
                        self.task.selected_tactic = self.task._defense_prev_tactic
                        logging.info(f"✅{agent_id} 标准战术回转完成，恢复战术 {self.task.selected_tactic}")
                        
        return 8, 7, 3  # 默认直飞（修正索引=直飞�?
    
    def _heading_to_attack_wp_or_default(self, env, agent_id: str, default_heading: float) -> float:
        try:
            wp = self.task.attack_waypoint.get(agent_id)
            if not wp:
                return default_heading
            cur = env.agents[agent_id].get_position()
            vec = np.array(wp) - np.array(cur)
            if np.linalg.norm(vec[:2]) < 1.0:
                return default_heading
            desired = float(np.rad2deg(np.arctan2(vec[1], vec[0])) % 360.0)
            current_heading = float(env.agents[agent_id].get_property_value(c.attitude_psi_deg))
            delta = self.task._normalize_angle_diff(desired - current_heading)
            delta = float(np.clip(delta, -10.0, 10.0))
            target = (current_heading + delta) % 360.0
            return target
        except Exception:
            return default_heading
    
    def execute_high_low_attack(self, env, agent_id: str) -> tuple:
        """
        战术3: 上下夹击 (High-Low Attack)
        # 核心思想: 僚机高空，长机低飞
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        current_alt = env.agents[agent_id].get_position()[2]
        
        if agent_id in self.task.short_skate_states:
            skate_direction = 'left' if is_lead else 'right'
            return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 长机动作序列（低空）
        if is_lead:
            target_altitude = self.task.vertical_split_targets.get(agent_id, 6096.0)
            
            if self.task.current_phase.value in ['NLT_MELD', 'MELD_MTR', 'MTR_LR']:
                alt_diff = target_altitude - current_alt
                if abs(alt_diff) > 200:
                    if alt_diff > 1000:
                        alt_cmd_value = 1000
                    elif alt_diff > 500:
                        alt_cmd_value = 500
                    elif alt_diff > 200:
                        alt_cmd_value = 300
                    elif alt_diff < -1000:
                        alt_cmd_value = -1000
                    elif alt_diff < -500:
                        alt_cmd_value = -500
                    elif alt_diff < -200:
                        alt_cmd_value = -300
                    else:
                        alt_cmd_value = alt_diff
                    
                    alt_cmd = self.task._convert_altitude_to_index(alt_cmd_value)
                    return alt_cmd, 8, 3
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'LR_TR':
                # LR阶段�?8km）：发射主动雷达弹，中制导开�?
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                # 🎯 战术优先：HIGH_LOW_ATTACK长机在LR_TR阶段保持低空直飞，不执行Crank
                if self.should_log('high_low_attack_info', current_time):
                    logging.info(f"🎯 [HIGH_LOW_ATTACK-长机LR_TR] {agent_id} 战术特定：低空直飞")
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'TR_DOR':
                # TR阶段�?5km）：中制导结束，准备规避
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time, 'left')
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self.task._execute_short_skate_precise(env, agent_id, current_time, 'left')
        
        # 僚机动作序列（高空）
        else:
            target_id = get_target_with_fallback(agent_id, env)
            target_aircraft = env._jsbsims.get(target_id)
            if target_aircraft and target_aircraft.is_alive:
                distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
                wingman_phase = self.task._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.task.current_phase
            
            target_altitude = self.task.vertical_split_targets.get(agent_id, 9144.0)
            
            # 🎯 战术优先：HIGH_LOW_ATTACK僚机纯粹高度分离，不进行任何横向偏转
            # 使用固定的初始航向，避免航向漂移
            if agent_id in self.task.initial_heading:
                target_heading = self.task.initial_heading[agent_id]  # 使用度数
            else:
                target_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
                self.task.initial_heading[agent_id] = target_heading
                
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            if env.current_step % 200 == 0:
                logging.info(f"🎯 [HIGH_LOW_ATTACK-僚机] {agent_id} 战术特定：纯粹高度变化，固定航向{target_heading:.1f}° (当前{current_heading:.1f}°)")
            
            if wingman_phase.value == 'NLT_MELD':
                action = self.task._maintain_heading_precise(env, agent_id, target_heading)
                if env.current_step % 100 == 0 and agent_id == "A0200":
                    logging.info(f"🔍 [DEBUG-A0200-NLT_MELD] 目标航向={target_heading:.1f}°, 返回动作: {action}")
                return action
            elif wingman_phase.value == 'MELD_MTR':
                alt_diff = target_altitude - current_alt
                if abs(alt_diff) > 200:
                    if alt_diff > 1000:
                        alt_cmd_value = 1000
                    elif alt_diff > 500:
                        alt_cmd_value = 500
                    elif alt_diff > 200:
                        alt_cmd_value = 300
                    elif alt_diff < -1000:
                        alt_cmd_value = -1000
                    elif alt_diff < -500:
                        alt_cmd_value = -500
                    elif alt_diff < -200:
                        alt_cmd_value = -300
                    else:
                        alt_cmd_value = alt_diff
                    
                    alt_cmd = self.task._convert_altitude_to_index(alt_cmd_value)
                    # 纯高度变化，不进行航向调�?
                    hdg_cmd = 8  # 保持航向索引
                    action = (alt_cmd, hdg_cmd, 3)
                    if env.current_step % 100 == 0 and agent_id == "A0200":
                        logging.info(f"🔍 [DEBUG-A0200-MELD_MTR] 高度{alt_diff:.0f}m, 目标航向={target_heading:.1f}°, 返回动作: {action}")
                    return action
                else:
                    action = self.task._maintain_heading_precise(env, agent_id, target_heading)
                    if env.current_step % 100 == 0 and agent_id == "A0200":
                        logging.info(f"🔍 [DEBUG-A0200-MELD_MTR] 高度接近目标，目标航{target_heading:.1f}°, 返回动作: {action}")
                    return action
            elif wingman_phase.value == 'MTR_LR':
                action = self.task._maintain_heading_precise(env, agent_id, target_heading)
                if env.current_step % 100 == 0 and agent_id == "A0200":
                    logging.info(f"🔍 [DEBUG-A0200-MTR_LR] 目标航向={target_heading:.1f}°, 返回动作: {action}")
                return action
            elif wingman_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                # 🎯 战术优先：HIGH_LOW_ATTACK僚机在LR_TR阶段保持高空直飞，不执行Crank
                if self.should_log('high_low_attack_info', current_time):
                    logging.info(f"🎯 [HIGH_LOW_ATTACK-僚机LR_TR] {agent_id} 战术特定：高空直飞，保持固定航向{target_heading:.1f}°")
                return self.task._maintain_heading_precise(env, agent_id, target_heading)
            elif wingman_phase.value == 'TR_DOR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time, 'right')
                else:
                    return 5, 8, 4
            else:
                return self.task._execute_short_skate_precise(env, agent_id, current_time, 'right')
    
    def execute_front_back(self, env, agent_id: str) -> tuple:
        """
        战术4: 前后攻击 (Front-Back Attack)
        # 核心思想: 僚机藏在长机后方，形成一字型纵队
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        
        if agent_id in self.task.short_skate_states:
            skate_direction = 'left' if is_lead else 'right'
            return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 长机动作序列（前机）
        if is_lead:
            if self.task.current_phase.value in ['NLT_MELD', 'MELD_MTR']:
                if current_time < 4.0:
                    return 7, 8, 4
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'MTR_LR':
                hdg = self._heading_to_attack_wp_or_default(env, agent_id, default_heading=0.0)
                return self.task._maintain_heading_precise(env, agent_id, hdg)
            elif self.task.current_phase.value == 'LR_TR':
                # LR阶段�?8km）：发射主动雷达弹，中制导开�?
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                # 🎯 战术优先：FRONT_BACK长机在LR_TR阶段保持直飞，不执行Crank
                if self.should_log('front_back_info', current_time):
                    logging.info(f"🎯 [FRONT_BACK-僚机LR_TR] {agent_id} 战术特定：保持后排队形")
                return self.task._maintain_heading_precise(env, agent_id, 20.0)
            elif self.task.current_phase.value == 'TR_DOR':
                # TR阶段�?5km）：中制导结束，准备规避
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    # 左侧返航
                    return self.task._execute_short_skate_precise(env, agent_id, current_time, 'left')
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self.task._execute_short_skate_precise(env, agent_id, current_time, 'left')
        
        # 僚机动作序列
        else:
            target_id = get_target_with_fallback(agent_id, env)
            target_aircraft = env._jsbsims.get(target_id)
            if target_aircraft and target_aircraft.is_alive:
                distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
                wingman_phase = self.task._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.task.current_phase
            
            if wingman_phase.value in ['NLT_MELD', 'MELD_MTR']:
                return self.task._establish_rear_formation(env, agent_id, current_time)
            elif wingman_phase.value == 'MTR_LR':
                if hasattr(self.task, 'wingman_crank_state') and self.task.wingman_crank_state.get("completed", False):
                    recent_completion = current_time - self.task.wingman_crank_state.get("completed_time", 0) < 60.0
                    if recent_completion:
                        logging.info(f"🔄 [{agent_id}]刚完成一字型，继续使用establish逻辑保持队形")
                        return self.task._establish_rear_formation(env, agent_id, current_time)
                hdg = self._heading_to_attack_wp_or_default(env, agent_id, default_heading=0.0)
                return self.task._maintain_heading_precise(env, agent_id, hdg)
            elif wingman_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                # 🎯 战术优先：FRONT_BACK僚机在LR_TR阶段保持后排队形，不执行Crank
                if env.current_step % 60 == 0:
                    logging.info(f"🎯 [FRONT_BACK-僚机LR_TR] {agent_id} 战术特定：保持后排队形")
                
                if hasattr(self.task, 'wingman_crank_state') and self.task.wingman_crank_state.get("completed", False):
                    recent_completion = current_time - self.task.wingman_crank_state.get("completed_time", 0) < 60.0
                    if recent_completion:
                        return self.task._establish_rear_formation(env, agent_id, current_time)
                return self.task._maintain_rear_formation(env, agent_id)
            elif wingman_phase.value == 'TR_DOR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time, 'right')
                else:
                    if hasattr(self.task, 'wingman_crank_state') and self.task.wingman_crank_state.get("completed", False):
                        recent_completion = current_time - self.task.wingman_crank_state.get("completed_time", 0) < 60.0
                        if recent_completion:
                            return self.task._establish_rear_formation(env, agent_id, current_time)
                    return self.task._maintain_rear_formation(env, agent_id)
            else:
                if hasattr(self.task, 'wingman_crank_state') and self.task.wingman_crank_state.get("completed", False):
                    recent_completion = current_time - self.task.wingman_crank_state.get("completed_time", 0) < 60.0
                    if recent_completion:
                        logging.info(f"🔄 [{agent_id}]刚完成一字型，继续保持队形而非返航")
                        return self.task._establish_rear_formation(env, agent_id, current_time)
                return self.task._execute_short_skate_precise(env, agent_id, current_time, 'right')
    
    def execute_side_by_side(self, env, agent_id: str) -> tuple:
        """
        战术5: 并排射击 (Side-by-Side)
        # 核心思想: 双机保持队形，同时发射导弹
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        
        skate_direction = 'left' if is_lead else 'right'
        
        if agent_id in self.task.short_skate_states:
            return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        if is_lead:
            if self.task.current_phase.value in ['NLT_MELD', 'MELD_MTR', 'MTR_LR']:
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'TR_DOR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 15.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        else:
            target_id = get_target_with_fallback(agent_id, env)
            target_aircraft = env._jsbsims.get(target_id)
            if target_aircraft and target_aircraft.is_alive:
                distance = self.task._calculate_distance_between(env.agents[agent_id], target_aircraft)
                wingman_phase = self.task._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.task.current_phase
            
            if wingman_phase.value in ['NLT_MELD', 'MELD_MTR']:
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase.value == 'MTR_LR':
                hdg = self._heading_to_attack_wp_or_default(env, agent_id, default_heading=0.0)
                return self.task._maintain_heading_precise(env, agent_id, hdg)
            elif wingman_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                lead_launch = self.task.last_missile_launch_time.get('A0100', -999)
                
                if last_launch < 0:
                    if lead_launch > 0 and (current_time - lead_launch) >= 5.0:
                        self.task.missile_launched[agent_id] = True
                        logging.info(f"🚀 [僚机跟随发射] {agent_id} 在长机发射{current_time - lead_launch:.1f}秒后准备发射")
                    elif lead_launch < 0:
                        self.task.missile_launched[agent_id] = True
                
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase.value == 'TR_DOR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 15.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
    # ==================== 统一二次进攻协同系统 ====================
    
    def check_second_attack_readiness(self, env) -> bool:
        """检查是否准备好进行二次进攻"""
        our_agents = [aid for aid in env.agents.keys() if aid.startswith('A') and env._jsbsims[aid].is_alive]

        # 检查队形重置是否完成
        if self.formation_reset_manager.is_formation_reset_complete(env, our_agents):
            # 准备MTR'节点的战术信息
            tactical_info = {
                'second_attack_ready': True,
                'formation_manager': self.formation_reset_manager,
                'second_attack_active': False,
                'mtr_prime_completed': False
            }

            # 检查MTR'节点是否应该激活
            first_agent = our_agents[0] if our_agents else None
            if first_agent and self.mtr_prime_node.should_activate(env, first_agent, tactical_info):
                if not getattr(self, 'second_attack_tactic_selected', False):
                    # 选择统一的二次进攻战术
                    selected_tactic = self.formation_reset_manager.get_second_attack_tactic(env)
                    logging.info(f"🎯 MTR'节点激活，二次进攻战术选择: {selected_tactic}")

                    # 设置统一的战术给所有飞机
                    for agent_id in our_agents:
                        if agent_id not in self.second_attack_coordination:
                            self.second_attack_coordination[agent_id] = {}
                        self.second_attack_coordination[agent_id]['tactic'] = selected_tactic
                        self.second_attack_coordination[agent_id]['ready'] = True
                        self.second_attack_coordination[agent_id]['tactical_info'] = tactical_info.copy()
                        logging.info(f"🎯 {agent_id} 准备MTR'节点验证和二次进攻")

                    self.second_attack_tactic_selected = True
                return True

        return False
    
    def execute_formation_reset(self, env, agent_id: str) -> tuple:
        """执行队形重置"""
        return self.formation_reset_manager.execute_formation_reset(env, agent_id)
    
    def execute_unified_second_attack(self, env, agent_id: str) -> tuple:
        """
        执行统一的二次进攻战术
        集成MTR'节点验证和二次进攻执行
        """
        if agent_id not in self.second_attack_coordination:
            return self.task._maintain_heading_precise(env, agent_id, 0.0)

        coordination = self.second_attack_coordination[agent_id]
        tactical_info = coordination.get('tactical_info', {})

        # 检查MTR'节点是否已完成
        if not tactical_info.get('mtr_prime_completed', False):
            # 执行MTR'节点逻辑
            mtr_command = self.mtr_prime_node.execute_mtr_prime(env, agent_id, tactical_info)

            # 更新战术信息
            coordination['tactical_info'] = tactical_info

            # 如果MTR'节点完成，记录日志
            if tactical_info.get('mtr_prime_completed', False):
                selected_tactic = tactical_info.get('second_attack_tactic', 'DRAG_SHOOT')
                logging.info(f"✅ [{agent_id}] MTR'节点完成，开始执行{selected_tactic}战术")

            return mtr_command

        # MTR'节点已完成，执行具体的二次进攻战术
        selected_tactic = coordination.get('tactic', 'DRAG_SHOOT')
        is_lead = agent_id.endswith('100')

        if selected_tactic == 'DRAG_SHOOT':
            return self._execute_unified_drag_shoot(env, agent_id, is_lead)
        elif selected_tactic == 'PINCER_ATTACK':
            return self._execute_unified_pincer(env, agent_id, is_lead)
        elif selected_tactic == 'HIGH_LOW_ATTACK':
            return self._execute_unified_high_low(env, agent_id, is_lead)
        else:
            return self._execute_unified_drag_shoot(env, agent_id, is_lead)
    
    def _execute_unified_drag_shoot(self, env, agent_id: str, is_lead: bool) -> tuple:
        """统一的拖曳射击二次进攻"""
        if env.current_step % 60 == 0:
            logging.info(f"🔄 [二次进攻-DRAG_SHOOT] {agent_id} ({'长机' if is_lead else '僚机'}) 执行编队机动")
        
        if is_lead:
            # 长机：保持北向0°，主导攻击
            return self.task._maintain_heading_precise(env, agent_id, 0.0)
        else:
            # 僚机：保持350°，支援攻击
            return self.task._maintain_heading_precise(env, agent_id, 350.0)
    
    def _execute_unified_pincer(self, env, agent_id: str, is_lead: bool) -> tuple:
        """统一的钳形攻势二次进攻"""
        if env.current_step % 60 == 0:
            logging.info(f"🔄 [二次进攻-PINCER] {agent_id} ({'左翼' if is_lead else '右翼'}) 执行钳形包夹")
        
        if is_lead:
            # 长机：左翼攻击，航向315°
            return self.task._maintain_heading_precise(env, agent_id, 315.0)
        else:
            # 僚机：右翼攻击，航向45°
            return self.task._maintain_heading_precise(env, agent_id, 45.0)
    
    def _execute_unified_high_low(self, env, agent_id: str, is_lead: bool) -> tuple:
        """统一的高低协同二次进攻"""
        if env.current_step % 60 == 0:
            logging.info(f"🔄 [二次进攻-HIGH_LOW] {agent_id} ({'高位' if is_lead else '低位'}) 执行立体攻击")
        
        aircraft = env.agents[agent_id]
        current_altitude = aircraft.get_position()[2]
        
        if is_lead:
            # 长机：高位攻击，保持较高高度
            target_altitude = 7000.0
            if current_altitude < target_altitude - 200:
                return (8, 9, 3)  # 爬升到目标高度
            else:
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
        else:
            # 僚机：低位攻击，保持较低高度
            target_altitude = 5000.0
            if current_altitude > target_altitude + 200:
                return (8, 6, 3)  # 下降到目标高度
            else:
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
    
    # ==================== 旧的二次进攻函数（保留兼容性）====================
    
    def _execute_second_attack_drag_shoot(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
        """二次进攻 - 拖曳射击编队"""
        if env.current_step % 60 == 0:
            logging.info(f"🔄 [二次进攻-DRAG_SHOOT] {agent_id} ({'长机' if is_lead else '僚机'}) 执行编队机动")
        
        if is_lead:
            # 长机：保持北�?°，主导攻�?
            self.task.missile_launched[agent_id] = True  # 准备发射导弹
            return self.task._maintain_heading_precise(env, agent_id, 0.0)
        else:
            # 僚机：保�?50°，支援攻�?
            self.task.missile_launched[agent_id] = True  # 准备发射导弹
            return self.task._maintain_heading_precise(env, agent_id, 350.0)
    
    def _execute_second_attack_pincer(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
        """二次进攻 - 钳形攻势编队"""
        if env.current_step % 60 == 0:
            logging.info(f"🔄 [二次进攻-PINCER] {agent_id} ({'左翼' if is_lead else '右翼'}) 执行钳形包夹")
        
        if is_lead:
            # 长机：左翼攻击，航向315°
            self.task.missile_launched[agent_id] = True
            return self.task._maintain_heading_precise(env, agent_id, 315.0)
        else:
            # 僚机：右翼攻击，航向45°
            self.task.missile_launched[agent_id] = True
            return self.task._maintain_heading_precise(env, agent_id, 45.0)
    
    def _execute_second_attack_high_low(self, env, agent_id: str, is_lead: bool, current_time: float) -> tuple:
        """二次进攻 - 上下夹击编队"""
        if env.current_step % 60 == 0:
            logging.info(f"🔄 [二次进攻-HIGH_LOW] {agent_id} ({'高位' if is_lead else '低位'}) 执行立体攻击")
        
        if is_lead:
            # 长机：高位攻击，爬升+前进
            self.task.missile_launched[agent_id] = True
            return (9, 8, 3)  # 爬升+直飞+标准速度
        else:
            # 僚机：低位攻击，俯冲+前进
            self.task.missile_launched[agent_id] = True
            return (6, 8, 3)  # 下降+直飞+标准速度
