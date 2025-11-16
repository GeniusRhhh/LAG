"""
机动库模块
包含所有机动执行函数：Short Skate, Beam, Crank, 战术爬升/下降等
从tactical_task.py中提取，提高代码可维护性
"""
import logging
import numpy as np
from envs.JSBSim.core.catalog import Catalog as c


class ManeuverLibrary:
    """机动库 - 负责执行各种机动动作"""
    
    def __init__(self, tactical_task):
        """
        Args:
            tactical_task: TacticalTask实例，用于访问共享状态和工具函数
        """
        self.task = tactical_task
    
    def maintain_heading_precise(self, env, agent_id: str, target_heading: float, duration=10.0) -> tuple:
        """
        精确保持航向 - 严格按照原战术模板
        
        Args:
            env: 环境
            agent_id: 飞机ID
            target_heading: 目标航向（度）
            duration: 持续时间（秒）
        
        Returns:
            (altitude_cmd, heading_cmd, velocity_cmd)
        """
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        
        # 计算航向差值
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        # 默认索引
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 3  # 保持速度
        
        # 精度控制：3度精度（防止过于灵敏）
        if abs(heading_diff) > 3.0:
            heading_cmd_id = self.task._convert_heading_to_index(np.deg2rad(heading_diff))
        
        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id
    
    def init_short_skate(self, agent_id, current_time, direction='auto'):
        """
        初始化short_skate机动状态
        
        Args:
            direction: 'auto'(根据队伍), 'left'(左转), 'right'(右转)
        """
        if direction == 'left':
            crank_angle = -40.0
            turn_cold_angle = -100.0
        elif direction == 'right':
            crank_angle = 40.0
            turn_cold_angle = 100.0
        else:  # 'auto' - 根据队伍
            crank_angle = -40.0 if agent_id.startswith('A') else 40.0
            turn_cold_angle = -100.0 if agent_id.startswith('A') else 100.0
        
        self.task.short_skate_states[agent_id] = {
            "phase": "crank",
            "phase_start_time": current_time,
            "total_start_time": current_time,
            "crank_angle": crank_angle,
            "turn_cold_angle": turn_cold_angle,
            "initial_heading": None,
            "initial_altitude": None
        }
        self.task.short_skate_start_time[agent_id] = current_time
    
    def execute_short_skate_precise(self, env, agent_id, current_time, direction='auto'):
        """
        执行精确的Short Skate机动
        三阶段：小角度Crank → 快速转向目标航向 → 加速逃离
        """
        if agent_id not in self.task.short_skate_states:
            self.init_short_skate(agent_id, current_time, direction)
        
        state = self.task.short_skate_states[agent_id]
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        
        if state["initial_heading"] is None:
            state["initial_heading"] = current_heading
        
        phase_time = current_time - state["phase_start_time"]
        
        # 时间参数
        if agent_id == "A0200":
            crank_duration = 12.0
            turn_cold_duration = 18.0
            escape_duration = 15.0
        else:
            crank_duration = 6.0
            turn_cold_duration = 15.0
            escape_duration = 15.0
        
        # 阶段1：Crank机动
        if state["phase"] == "crank":
            if phase_time < crank_duration:
                target_heading = state["initial_heading"] + state["crank_angle"]
                target_heading = target_heading % 360
                heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
                
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3
                else:
                    return 7, 8, 3
            else:
                state["phase"] = "turn_cold"
                state["phase_start_time"] = current_time
                state["turn_cold_start_heading"] = current_heading
        
        # 阶段2：Turn Cold
        elif state["phase"] == "turn_cold":
            if phase_time < turn_cold_duration:
                target_heading = state["turn_cold_start_heading"] + state["turn_cold_angle"]
                target_heading = target_heading % 360
                heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
                
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3
                else:
                    return 7, 8, 3
            else:
                state["phase"] = "escape"
                state["phase_start_time"] = current_time
        
        # 阶段3：加速逃离
        elif state["phase"] == "escape":
            if phase_time < escape_duration:
                return 7, 8, 5
            else:
                if agent_id.startswith('A'):
                    target_heading = 180.0
                else:
                    target_heading = 0.0
                
                heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3
                else:
                    return 7, 8, 3
        
        return 7, 8, 3
    
    def execute_short_skate(self, env, agent_id: str, direction='left', target_heading=None) -> tuple:
        """执行Short Skate机动的简化接口"""
        current_time = env.current_step * env.time_interval
        return self.execute_short_skate_precise(env, agent_id, current_time, direction)
    
    def execute_beam_maneuver(self, env, agent_id: str) -> tuple:
        """
        执行Beam机动 - 三九机动，将敌机置于自身3/9位置
        """
        current_time = env.current_step * env.time_interval
        last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
        if last_launch > 0 and (current_time - last_launch) < 15.0:
            remaining_time = 15.0 - (current_time - last_launch)
            if env.current_step % 60 == 0:
                logging.info(f"🎯 [{agent_id}] 导弹制导保护: 剩余{remaining_time:.1f}秒，保持直飞")
            return 7, 8, 3
        
        enemy_bearing = self.task._get_enemy_bearing(env, agent_id)
        if enemy_bearing is None:
            return 7, 8, 3
        
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        beam_left = (enemy_bearing - 90) % 360
        beam_right = (enemy_bearing + 90) % 360
        
        diff_left = abs(self.task._normalize_angle_diff(beam_left - current_heading))
        diff_right = abs(self.task._normalize_angle_diff(beam_right - current_heading))
        
        if diff_left < diff_right:
            target_heading = beam_left
            beam_direction = "3点方向"
        else:
            target_heading = beam_right
            beam_direction = "9点方向"
        
        heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
        if abs(heading_diff) > 30:
            if heading_diff > 0:
                target_heading = (current_heading + 30) % 360
            else:
                target_heading = (current_heading - 30) % 360
            if env.current_step % 60 == 0:
                logging.warning(f"⚠️ [{agent_id}] BEAM转向限制: 原目标{heading_diff:.1f}°→限制±30°")
        
        if not hasattr(self.task, 'beam_maneuver_state'):
            self.task.beam_maneuver_state = {}
        
        if agent_id in self.task.beam_maneuver_state:
            state = self.task.beam_maneuver_state[agent_id]
            if 'start_time' in state:
                elapsed = current_time - state['start_time']
                
                if elapsed < 5.0:
                    if env.current_step % 60 == 0:
                        logging.info(f"📍 [BEAM执行] {agent_id} 敌机{enemy_bearing:.1f}° → {beam_direction} {target_heading:.1f}° (剩{5.0-elapsed:.1f}s)")
                    return self.task._turn_to_heading(env, agent_id, target_heading, speed_cmd=3)
                
                elif elapsed < 15.0:
                    if env.current_step % 60 == 0:
                        logging.info(f"⏸️ [BEAM冷却] {agent_id} 保持航向 (剩{15.0-elapsed:.1f}s)")
                    return 7, 8, 3
                
                else:
                    state['start_time'] = current_time
                    logging.info(f"📍 [BEAM启动] {agent_id} 敌机{enemy_bearing:.1f}° → {beam_direction} {target_heading:.1f}°")
                    return self.task._turn_to_heading(env, agent_id, target_heading, speed_cmd=3)
        else:
            self.task.beam_maneuver_state[agent_id] = {'start_time': current_time}
            logging.info(f"📍 [BEAM启动] {agent_id} 敌机{enemy_bearing:.1f}° → {beam_direction} {target_heading:.1f}°")
            return self.task._turn_to_heading(env, agent_id, target_heading, speed_cmd=3)
    
    def execute_tactical_crank(self, env, agent_id: str, direction='left', climb=True) -> tuple:
        """
        执行战术Crank机动
        飞机做crank机动的同时做爬升或下降机动
        """
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        if agent_id not in self.task.maneuver_states or self.task.maneuver_states[agent_id].get('type') != 'tactical_crank':
            self.task.maneuver_states[agent_id] = {
                'type': 'tactical_crank',
                'phase': 'crank_climb',
                'start_time': current_time,
                'phase_start_time': current_time,
                'direction': direction,
                'climb': climb,
                'initial_heading': current_heading
            }
            action = "爬升" if climb else "下降"
            logging.info(f"🔄 【战术Crank】{agent_id}开始{direction}侧Crank+{action}机动")
        
        state = self.task.maneuver_states[agent_id]
        phase_time = current_time - state['phase_start_time']
        direction = state.get('direction', 'left')
        climb = state.get('climb', True)
        
        if state['phase'] == 'crank_climb':
            if phase_time < 5.0:
                alt_cmd = 11 if climb else 3
                hdg_cmd = 4 if direction == 'left' else 12
                return alt_cmd, hdg_cmd, 4
            else:
                state['phase'] = 'level_off'
                state['phase_start_time'] = current_time
                logging.info(f"🔄 【战术Crank】{agent_id}: Crank→平飞恢复")
        
        if state['phase'] == 'level_off':
            if phase_time < 2.0:
                return 7, 8, 3
            else:
                del self.task.maneuver_states[agent_id]
                logging.info(f"✅ 【战术Crank】{agent_id}完成机动，方位角已变化")
                return 7, 8, 3
        
        return 7, 8, 3
    
    def execute_tactical_climb(self, env, agent_id: str, direction='left') -> tuple:
        """执行战术爬升机动 - crank+爬升+反向crank+平飞"""
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        current_alt = env.agents[agent_id].get_position()[2]
        
        if agent_id not in self.task.maneuver_states or self.task.maneuver_states[agent_id].get('type') != 'tactical_climb':
            crank_angle = -30.0 if direction == 'left' else 30.0
            target_alt = current_alt + 1000
            
            self.task.maneuver_states[agent_id] = {
                'type': 'tactical_climb',
                'phase': 'initial_crank',
                'start_time': current_time,
                'direction': direction,
                'initial_heading': current_heading,
                'initial_alt': current_alt,
                'target_alt': target_alt,
                'crank_angle': crank_angle
            }
            logging.info(f"📍 [战术爬升] {agent_id}开始战术爬升: {direction}侧crank+爬升{target_alt-current_alt:.0f}m")
        
        state = self.task.maneuver_states[agent_id]
        elapsed_time = current_time - state['start_time']
        
        if state['phase'] == 'initial_crank':
            if elapsed_time < 5.0:
                target_heading = (state['initial_heading'] + state['crank_angle']) % 360
                heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
                
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                return 7, hdg_cmd, 3
            else:
                state['phase'] = 'climb_reverse_crank'
                state['phase_start_time'] = current_time
                logging.info(f"📍 [战术爬升] {agent_id}: 初始Crank→爬升+反向Crank")
        
        elif state['phase'] == 'climb_reverse_crank':
            phase_time = current_time - state['phase_start_time']
            if phase_time < 10.0:
                alt_diff = state['target_alt'] - current_alt
                if alt_diff > 200:
                    alt_cmd = 11
                elif alt_diff > 50:
                    alt_cmd = 9
                else:
                    alt_cmd = 7
                
                reverse_target = (state['initial_heading'] - state['crank_angle']) % 360
                heading_diff = self.task._normalize_angle_diff(reverse_target - current_heading)
                
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                
                return alt_cmd, hdg_cmd, 3
            else:
                state['phase'] = 'level_off'
                state['phase_start_time'] = current_time
                logging.info(f"📍 [战术爬升] {agent_id}: 爬升+反向Crank→平飞")
        
        elif state['phase'] == 'level_off':
            phase_time = current_time - state['phase_start_time']
            if phase_time < 5.0:
                heading_diff = self.task._normalize_angle_diff(state['initial_heading'] - current_heading)
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                return 7, hdg_cmd, 3
            else:
                del self.task.maneuver_states[agent_id]
                logging.info(f"✅ [战术爬升] {agent_id}完成战术爬升机动 (最终航向{current_heading:.1f}°)")
                return 7, 8, 3
        
        return 7, 8, 3
    
    def execute_tactical_descent(self, env, agent_id: str, direction='left') -> tuple:
        """执行战术下降机动 - crank+下降+反向crank+平飞"""
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        current_alt = env.agents[agent_id].get_position()[2]
        
        if agent_id not in self.task.maneuver_states or self.task.maneuver_states[agent_id].get('type') != 'tactical_descent':
            crank_angle = -30.0 if direction == 'left' else 30.0
            target_alt = max(current_alt - 500, 3000)
            
            self.task.maneuver_states[agent_id] = {
                'type': 'tactical_descent',
                'phase': 'initial_crank',
                'start_time': current_time,
                'direction': direction,
                'initial_heading': current_heading,
                'initial_alt': current_alt,
                'target_alt': target_alt,
                'crank_angle': crank_angle
            }
            logging.info(f"📍 [战术下降] {agent_id}开始战术下降: {direction}侧crank+下降{current_alt-target_alt:.0f}m")
        
        state = self.task.maneuver_states[agent_id]
        elapsed_time = current_time - state['start_time']
        
        if state['phase'] == 'initial_crank':
            if elapsed_time < 5.0:
                target_heading = (state['initial_heading'] + state['crank_angle']) % 360
                heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
                
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                return 7, hdg_cmd, 3
            else:
                state['phase'] = 'descent_reverse_crank'
                state['phase_start_time'] = current_time
                logging.info(f"📍 [战术下降] {agent_id}: 初始Crank→下降+反向Crank")
        
        elif state['phase'] == 'descent_reverse_crank':
            phase_time = current_time - state['phase_start_time']
            if phase_time < 10.0:
                alt_diff = current_alt - state['target_alt']
                if alt_diff > 200:
                    alt_cmd = 3
                elif alt_diff > 50:
                    alt_cmd = 5
                else:
                    alt_cmd = 7
                
                reverse_target = (state['initial_heading'] - state['crank_angle']) % 360
                heading_diff = self.task._normalize_angle_diff(reverse_target - current_heading)
                
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                
                return alt_cmd, hdg_cmd, 3
            else:
                state['phase'] = 'level_off'
                state['phase_start_time'] = current_time
                logging.info(f"📍 [战术下降] {agent_id}: 下降+反向Crank→平飞")
        
        elif state['phase'] == 'level_off':
            phase_time = current_time - state['phase_start_time']
            if phase_time < 5.0:
                heading_diff = self.task._normalize_angle_diff(state['initial_heading'] - current_heading)
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                return 7, hdg_cmd, 3
            else:
                del self.task.maneuver_states[agent_id]
                logging.info(f"✅ [战术下降] {agent_id}完成战术下降机动 (最终航向{current_heading:.1f}°)")
                return 7, 8, 3
        
        return 7, 8, 3
    
    def execute_notch_back(self, env, agent_id: str, direction='left') -> tuple:
        """执行Notch Back机动 - 180°回旋下降"""
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        if agent_id not in self.task.maneuver_states or self.task.maneuver_states[agent_id].get('type') != 'notch_back':
            self.task.maneuver_states[agent_id] = {
                'type': 'notch_back',
                'phase': 'turn_descend',
                'start_time': current_time,
                'phase_start_time': current_time,
                'direction': direction,
                'initial_heading': current_heading
            }
            logging.info(f"🔄 【Notch Back】{agent_id}开始180°回旋下降机动")
        
        state = self.task.maneuver_states[agent_id]
        phase_time = current_time - state['phase_start_time']
        
        if state['phase'] == 'turn_descend':
            if phase_time < 8.0:
                return 3, 16, 5
            else:
                del self.task.maneuver_states[agent_id]
                logging.info(f"✅ 【Notch Back】{agent_id}完成机动")
                return 7, 8, 4
        
        return 7, 8, 4
    
    def establish_rear_formation(self, env, agent_id: str, current_time: float) -> tuple:
        """僚机建立后方队形（前后攻击战术专用）"""
        leader = env._jsbsims.get("A0100")
        wingman = env._jsbsims.get(agent_id)
        
        altitude_cmd_id = 7
        heading_cmd_id = 8
        velocity_cmd_id = 3
        
        if not leader or not wingman:
            return altitude_cmd_id, heading_cmd_id, velocity_cmd_id
        
        leader_pos = leader.get_position()
        wingman_pos = wingman.get_position()
        leader_x, leader_y, leader_z = leader_pos[0], leader_pos[1], leader_pos[2]
        wingman_x, wingman_y, wingman_z = wingman_pos[0], wingman_pos[1], wingman_pos[2]
        
        leader_heading = np.rad2deg(leader.get_property_value(c.attitude_psi_rad))
        wingman_heading = np.rad2deg(wingman.get_property_value(c.attitude_psi_rad))
        
        alt_diff = wingman_z - leader_z
        
        if not hasattr(self.task, 'formation_state'):
            self.task.formation_state = {}
        if agent_id not in self.task.formation_state:
            self.task.formation_state[agent_id] = 'LATERAL_ALIGN'
        
        current_state = self.task.formation_state.get(agent_id, 'LATERAL_ALIGN')
        
        if current_state in ['LATERAL_ALIGN', 'HEADING_CORRECT']:
            alt_threshold = 300
        elif current_state == 'FORMATION_HOLD':
            alt_threshold = 200
        else:
            alt_threshold = 150
        
        if alt_diff > alt_threshold:
            altitude_cmd_id = 5
            logging.warning(f"⚠️ [{agent_id}]高度过高 {alt_diff:.0f}m (阈值{alt_threshold}m) → 下降")
        elif alt_diff < -alt_threshold:
            altitude_cmd_id = 9
            logging.warning(f"⚠️ [{agent_id}]高度过低 {alt_diff:.0f}m (阈值{alt_threshold}m) → 上升")
        
        dx = wingman_x - leader_x
        dy = wingman_y - leader_y
        heading_diff = self.task._normalize_angle_diff(leader_heading - wingman_heading)
        
        TARGET_DY_TOLERANCE = 200
        TARGET_DX_MIN = -12000
        TARGET_DX_MAX = -7000
        TARGET_HEADING_TOLERANCE = 10.0
        
        STABLE_DY_RANGE = 3000
        STABLE_DX_MIN = -15000
        STABLE_DX_MAX = -5000
        STABLE_HEADING_RANGE = 30.0
        
        if current_state == 'LATERAL_ALIGN':
            if abs(dy) < TARGET_DY_TOLERANCE:
                self.task.formation_state[agent_id] = 'HEADING_CORRECT'
                logging.info(f"✅ [{agent_id}]横向对齐完成 dy={dy:.0f}m → 进入航向回正")
            else:
                if dy > 0:
                    if dy < 1500 and heading_diff > 0:
                        if dy < 500:
                            heading_cmd_id = 8
                        else:
                            heading_cmd_id = 9
                        velocity_cmd_id = 3
                        logging.info(f"✅ [{agent_id}]阶段1-右转对齐: dy={dy:.0f}m hdg={heading_diff:.1f}°")
                    elif dy > 2000:
                        heading_cmd_id = 6
                        velocity_cmd_id = 3
                        logging.info(f"🔄 [{agent_id}]阶段1-大幅左转: dy={dy:.0f}m")
                    else:
                        heading_cmd_id = 7
                        velocity_cmd_id = 3
                        logging.info(f"🔄 [{agent_id}]阶段1-中幅左转: dy={dy:.0f}m")
                else:
                    heading_cmd_id = 9
                    velocity_cmd_id = 3
                    logging.warning(f"⚠️ [{agent_id}]阶段1-过头右转: dy={dy:.0f}m")
        
        elif current_state == 'HEADING_CORRECT':
            if abs(heading_diff) < TARGET_HEADING_TOLERANCE:
                self.task.formation_state[agent_id] = 'LONGITUDINAL_ADJUST'
                logging.info(f"✅ [{agent_id}]航向对齐完成 hdg_diff={heading_diff:.1f}° → 进入纵向调整")
            else:
                if heading_diff > 30.0:
                    heading_cmd_id = 10
                    logging.info(f"🔄 [{agent_id}]阶段2-大幅右转: hdg_diff={heading_diff:.1f}°")
                elif heading_diff > TARGET_HEADING_TOLERANCE:
                    heading_cmd_id = 9
                    logging.info(f"🔄 [{agent_id}]阶段2-微幅右转: hdg_diff={heading_diff:.1f}°")
                elif heading_diff < -30.0:
                    heading_cmd_id = 6
                    logging.warning(f"🔄 [{agent_id}]阶段2-异常左转: hdg_diff={heading_diff:.1f}°")
                elif heading_diff < -TARGET_HEADING_TOLERANCE:
                    heading_cmd_id = 7
                    logging.warning(f"🔄 [{agent_id}]阶段2-异常微左转: hdg_diff={heading_diff:.1f}°")
                else:
                    heading_cmd_id = 8
                
                velocity_cmd_id = 3
        
        elif current_state == 'LONGITUDINAL_ADJUST':
            in_stable_zone = (STABLE_DX_MIN <= dx <= STABLE_DX_MAX and 
                            abs(dy) < STABLE_DY_RANGE and 
                            abs(heading_diff) < STABLE_HEADING_RANGE)
            
            if in_stable_zone:
                self.task.formation_state[agent_id] = 'FORMATION_HOLD'
                logging.info(f"✅ [{agent_id}]进入稳定区域 dx={dx/1000:.2f}km dy={dy/1000:.2f}km → 开始平飞")
            else:
                heading_cmd_id = 8
                velocity_cmd_id = 3
                
                if dx > 0:
                    velocity_cmd_id = 2
                elif dx > -5000:
                    velocity_cmd_id = 2
                elif dx < -15000:
                    velocity_cmd_id = 4
                
                logging.info(f"🔄 [{agent_id}]阶段3-只调速度: dx={dx/1000:.2f}km dy={dy/1000:.2f}km vel={velocity_cmd_id}")
        
        elif current_state == 'FORMATION_HOLD':
            seriously_off = (abs(dy) > STABLE_DY_RANGE or 
                           dx > STABLE_DX_MAX or 
                           dx < STABLE_DX_MIN or
                           abs(heading_diff) > STABLE_HEADING_RANGE)
            
            if seriously_off:
                self.task.formation_state[agent_id] = 'LATERAL_ALIGN'
                logging.warning(f"⚠️ [{agent_id}]严重脱离队形 dx={dx/1000:.2f}km dy={dy:.0f}m hdg={heading_diff:.1f}° → 重新对齐")
            else:
                heading_cmd_id = 8
                velocity_cmd_id = 3
                
                if dy < -500:
                    heading_cmd_id = 9
                elif dy > 500:
                    heading_cmd_id = 7
                
                leader_velocity = leader.get_property_value(c.velocities_u_fps)
                wingman_velocity = wingman.get_property_value(c.velocities_u_fps)
                speed_diff = leader_velocity - wingman_velocity
                
                if dx > -6000 and speed_diff > 10:
                    velocity_cmd_id = 4
                elif dx < -12000 and speed_diff < -10:
                    velocity_cmd_id = 2
                elif speed_diff > 30:
                    velocity_cmd_id = 4
                elif speed_diff < -30:
                    velocity_cmd_id = 2
                
                if env.current_step % 60 == 0:
                    logging.info(f"✅ [{agent_id}]队形微调: dx={dx/1000:.2f}km dy={dy/1000:.2f}km vel_diff={speed_diff:.0f}fps hdg={heading_cmd_id} vel={velocity_cmd_id}")
        
        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id
    
    def maintain_rear_formation(self, env, agent_id: str):
        """僚机保持后方队形 - 前后攻击专用"""
        current_time = env.current_step * env.time_interval
        return self.establish_rear_formation(env, agent_id, current_time)
