"""
战术执行模块
包含所有战术执行函数：拖曳射击、钳形攻势、上下夹击、前后攻击、并排射击等
从tactical_task.py中提取，提高代码可维护性
"""
import logging
import numpy as np
from tactical_types import TacticalPhase
from envs.JSBSim.core.catalog import Catalog as c
from core.target_assignment import get_target_with_fallback


class TacticalExecutor:
    """战术执行器 - 负责执行各种战术"""
    
    def __init__(self, tactical_task):
        """
        Args:
            tactical_task: TacticalTask实例，用于访问共享状态和工具函数
        """
        self.task = tactical_task
    
    def execute_drag_shoot(self, env, agent_id: str) -> tuple:
        """
        战术1: 拖曳射击 (DRAG_SHOOT)
        核心思想: 长机诱敌，僚机射击
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        
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
            
            if self.task.current_phase.value in ['NLT_MELD', 'MELD_MTR', 'MTR_LR']:
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    logging.info(f"🚀 [DRAG_SHOOT-长机] {agent_id} 设置导弹发射标记")
                    self.task.missile_launched[agent_id] = True
                
                lr_maneuver = getattr(self.task, 'lr_maneuver', {}).get(agent_id, 'straight')
                if env.current_step % 5 == 0:  # 每秒打印一次
                    logging.info(f"🔍 [拖曳射击-长机LR_TR] {agent_id} lr_maneuver={lr_maneuver}")
                
                if lr_maneuver == 'crank':
                    direction = 'left' if is_lead else 'right'
                    logging.info(f"🔄 【拖曳射击-长机Crank】{agent_id} 开始{direction}侧Crank机动")
                    return self.task._execute_tactical_crank(env, agent_id, direction, climb=False)
                else:
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
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                lr_maneuver = getattr(self.task, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    return self.task._execute_tactical_crank(env, agent_id, 'right', climb=False)
                else:
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
        核心思想: 双机从两侧包夹敌机
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        
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
                        logging.info(f"     → 已达到分离，保持315°")
                else:
                    if env.current_step % 60 == 0:
                        logging.info(f"     → 目标航向:315° (展开)，当前{current_heading:.1f}°")
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
                
                lr_maneuver = getattr(self.task, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    return self.task._execute_tactical_crank(env, agent_id, 'left', climb=False)
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif lead_phase.value == 'TR_DOR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 15.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    return 7, 8, 3
            else:
                return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
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
                # 初期小角度右转35°（原版设定），缩短Crank时间
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
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.task.missile_launched[agent_id] = True  # 标记需要发射
                
                # ✅ 根据LR决策执行Crank或平飞（原版逻辑）
                lr_maneuver = getattr(self.task, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    # 执行Crank机动（偏转30°保持规避余度）
                    return self.task._execute_tactical_crank(env, agent_id, 'right', climb=False)
                else:
                    # 平飞保持航向，完成中制导
                    return self.task._maintain_heading_precise(env, agent_id, 350.0)
            elif wingman_phase.value == 'TR_DOR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 15.0:
                    return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    return 7, 8, 3
            else:
                return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
    
    def execute_high_low_attack(self, env, agent_id: str) -> tuple:
        """
        战术3: 上下夹击 (High-Low Attack)
        核心思想: 僚机高空，长机低空
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
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                lr_maneuver = getattr(self.task, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    return self.task._execute_tactical_crank(env, agent_id, 'left', climb=False)
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'TR_DOR':
                # TR阶段（75km）：中制导结束，准备规避
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
            
            lead_pos = env.agents['A0100'].get_position()
            wing_pos = env.agents[agent_id].get_position()
            lateral_separation = abs(wing_pos[1] - lead_pos[1])
            target_lateral_separation = 5000
            
            wing_y = wing_pos[1]
            approaching_boundary = abs(wing_y) > 50000
            
            if approaching_boundary:
                heading_offset = -10.0
                if env.current_step % 200 == 0:
                    logging.warning(f"⚠️ [{agent_id}] 接近CAP边界(Y={wing_y/1000:.1f}km)，强制左偏返回")
            elif lateral_separation >= target_lateral_separation:
                heading_offset = 0.0
            else:
                heading_offset = 10.0
            
            if wingman_phase.value == 'NLT_MELD':
                return self.task._maintain_heading_precise(env, agent_id, heading_offset)
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
                    hdg_cmd = self.task._convert_heading_to_index(np.deg2rad(heading_offset))
                    return alt_cmd, hdg_cmd, 3
                else:
                    return self.task._maintain_heading_precise(env, agent_id, heading_offset)
            elif wingman_phase.value == 'MTR_LR':
                return self.task._maintain_heading_precise(env, agent_id, heading_offset)
            elif wingman_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                lr_maneuver = getattr(self.task, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    return self.task._execute_tactical_crank(env, agent_id, 'right', climb=False)
                else:
                    return self.task._maintain_heading_precise(env, agent_id, heading_offset)
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
        核心思想: 僚机藏在长机后方，形成一字型纵队
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        
        if agent_id in self.task.short_skate_states:
            skate_direction = 'left' if is_lead else 'right'
            return self.task._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 长机动作序列（前机）
        if is_lead:
            if self.task.current_phase.value in ['NLT_MELD', 'MELD_MTR', 'MTR_LR']:
                if current_time < 4.0:
                    return 7, 8, 4
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'LR_TR':
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                lr_maneuver = getattr(self.task, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    return self.task._execute_tactical_crank(env, agent_id, 'left', climb=False)
                else:
                    return self.task._maintain_heading_precise(env, agent_id, 0.0)
            elif self.task.current_phase.value == 'TR_DOR':
                # TR阶段（75km）：中制导结束，准备规避
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
                return self.task._maintain_rear_formation(env, agent_id)
            elif wingman_phase.value == 'LR_TR':
                last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:
                    self.task.missile_launched[agent_id] = True
                
                lr_maneuver = getattr(self.task, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    return self.task._execute_tactical_crank(env, agent_id, 'right', climb=False)
                else:
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
        核心思想: 双机保持队形，同时发射导弹
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
            
            if wingman_phase.value in ['NLT_MELD', 'MELD_MTR', 'MTR_LR']:
                return self.task._maintain_heading_precise(env, agent_id, 0.0)
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
