"""
机动库模块
包含所有机动执行函数：Short Skate, Beam, Crank, 战术爬升/下降等
从tactical_task.py中提取，提高代码可维护性
"""
import logging
import numpy as np
import logging
from envs.JSBSim.core.catalog import Catalog as c


class ManeuverLibrary:
    """机动库 - 负责执行各种机动动作"""
    
    def __init__(self, tactical_task):
        """
        Args:
            tactical_task: TacticalTask实例，用于访问共享状态和工具函数
        """
        self.task = tactical_task
    
    def _get_dynamic_velocity_cmd(self, env, agent_id: str) -> int:
        """
        动态速度管理：根据当前速度选择合适的速度指令
        
        Args:
            env: 环境
            agent_id: 飞机ID
        
        Returns:
            velocity_cmd_id: 速度指令索引
                - 2: 轻微减速 (delta=-50m/s)
                - 3: 保持当前速度 (delta=0)
                - 4: 轻微加速 (delta=+50m/s)
        """
        current_velocity = np.linalg.norm(env.agents[agent_id].get_velocity())
        
        # 阈值设计：保持速度在220-280 m/s范围内（降低阈值，减缓接近速度）
        VELOCITY_LOW_THRESHOLD = 220.0   # 低速阈值（从240降低到220）
        VELOCITY_HIGH_THRESHOLD = 280.0  # 高速阈值
        VELOCITY_CRITICAL_LOW = 200.0    # 危险低速阈值（从220降低到200）
        
        if current_velocity < VELOCITY_CRITICAL_LOW:
            # 危险低速：轻微加速（限制在安全档位）
            return 4
        elif current_velocity < VELOCITY_LOW_THRESHOLD:
            # 低速：轻微加速
            return 4
        elif current_velocity > VELOCITY_HIGH_THRESHOLD:
            # 高速：轻微减速
            return 2
        else:
            # 正常范围（220-280 m/s）：保持当前速度
            return 3
    
    def maintain_heading_precise(self, env, agent_id: str, target_heading: float, duration=10.0) -> tuple:
        """精确保持航向"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        altitude_cmd_id = 7
        heading_cmd_id = 8
        velocity_cmd_id = self._get_dynamic_velocity_cmd(env, agent_id)
        
        if abs(heading_diff) > 3.0:
            heading_cmd_id = self.task._convert_heading_to_index(np.deg2rad(heading_diff))
        
        # 🔥 JSBSim 底层飞行数据诊断（环境变量 JSBSIM_DIAG=1 开启）
        import os
        if os.environ.get('JSBSIM_DIAG') == '1' and env.current_step % 50 == 0:
            agent = env.agents[agent_id]
            alt_m = agent.get_property_value(c.position_h_sl_m)
            v_down = agent.get_property_value(c.velocities_v_down_fps) * 0.3048  # fps→m/s
            roll_deg = np.rad2deg(agent.get_property_value(c.attitude_phi_rad))
            pitch_deg = np.rad2deg(agent.get_property_value(c.attitude_theta_rad))
            tas_mps = agent.get_property_value(c.velocities_u_mps)
            try:
                g_load = abs(agent.get_property_value(c.accelerations_n_pilot_z_norm))
            except Exception:
                g_load = -1.0
            logging.warning(
                f"📊 [DIAG-{agent_id}] t={env.current_step * env.time_interval:.0f}s "
                f"ALT={alt_m:.0f}m VSI={-v_down:.1f}m/s "
                f"ROLL={roll_deg:.1f}° PITCH={pitch_deg:.1f}° "
                f"TAS={tas_mps:.0f}m/s G={g_load:.1f} "
                f"HDG={current_heading:.1f}→{target_heading:.1f}° "
                f"CMD=({altitude_cmd_id},{heading_cmd_id},{velocity_cmd_id})"
            )
        
        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id
    
    def init_short_skate(self, agent_id, current_time, direction='auto'):
        """
        初始化short_skate机动状态
        
        Args:
            direction: 'auto'(根据战术决定), 'left'(左转), 'right'(右转)
        """
        if direction == 'rtb':
            # Return-To-Base Short Skate: crank for lateral separation, then Turn Cold to base heading.
            # 我方默认返航指向南向180°（与TacticalTask的返航逻辑保持一致）。
            crank_angle = 40.0 if agent_id.endswith('100') else -40.0
            turn_cold_angle = 0.0
            turn_cold_target_heading = 180.0 if agent_id.startswith('A') else 0.0
        elif direction == 'left':
            crank_angle = -40.0
            turn_cold_angle = -140.0  # 🔥 修正：加大转弯角度(100->140)，确保指向南向(180)
        elif direction == 'right':
            crank_angle = 40.0
            turn_cold_angle = 140.0   # 🔥 修正：加大转弯角度(100->140)，确保指向南向(180)
        elif direction == 'auto':
            # 🔥 根据当前选择的战术确定转向策略
            current_tactic = getattr(self.task, 'selected_tactic', 'UNKNOWN')
            
            if current_tactic == 'DRAG_SHOOT':
                # 拖曳射击：长机朝外侧（右），僚机朝内侧（左）
                if agent_id.endswith('100'):  # 长机
                    crank_angle = -40.0   # 朝右外侧
                    turn_cold_angle = 140.0 # 🔥 修正
                    logging.info(f"🎯 [DRAG_SHOOT] {agent_id}(长机) Short Skate朝外侧右转")
                else:  # 僚机
                    crank_angle = -40.0  # 朝左内侧
                    turn_cold_angle = -140.0 # 🔥 修正
                    logging.info(f"🎯 [DRAG_SHOOT] {agent_id}(僚机) Short Skate朝内侧左转")
            elif current_tactic == 'PINCER_ATTACK':
                # 钳形攻势：长机右，僚机左
                if agent_id.endswith('100'):  # 长机
                    crank_angle = 40.0   # 右转
                    turn_cold_angle = 140.0 # 🔥 修正
                    logging.info(f"🎯 [PINCER_ATTACK] {agent_id}(长机) Short Skate右转")
                else:  # 僚机
                    crank_angle = -40.0  # 左转
                    turn_cold_angle = -140.0 # 🔥 修正
                    logging.info(f"🎯 [PINCER_ATTACK] {agent_id}(僚机) Short Skate左转")
            else:
                # 默认策略：友方左转，敌方右转（保持原逻辑兼容性）
                crank_angle = -40.0 if agent_id.startswith('A') else 40.0
                turn_cold_angle = -140.0 if agent_id.startswith('A') else 140.0 # 🔥 修正
                logging.info(f"🎯 [{current_tactic}] {agent_id} Short Skate默认方向")
        else:
            # 其他方向参数，使用默认逻辑
            crank_angle = -40.0 if agent_id.startswith('A') else 40.0
            turn_cold_angle = -140.0 if agent_id.startswith('A') else 140.0 # 🔥 修正
        
        self.task.short_skate_states[agent_id] = {
            "phase": "crank",
            "phase_start_time": current_time,
            "total_start_time": current_time,
            "crank_angle": crank_angle,
            "turn_cold_angle": turn_cold_angle,
            "turn_cold_target_heading": turn_cold_target_heading if direction == 'rtb' else None,
            "initial_heading": None,
            "initial_altitude": None
        }
        self.task.short_skate_start_time[agent_id] = current_time
        if not hasattr(self.task, '_skate_completed'):
            self.task._skate_completed = {}
        self.task._skate_completed[f'{agent_id}_skate_done'] = False
    
    def execute_short_skate_precise(self, env, agent_id, current_time, direction='auto'):
        """
        执行精确的Short Skate机动
        三阶段：小角度Crank → 快速转向目标航向 → 加速逃离
        """
        if agent_id not in self.task.short_skate_states:
            self.init_short_skate(agent_id, current_time, direction)
        
        state = self.task.short_skate_states[agent_id]
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_alt = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
        current_speed = float(np.linalg.norm(env.agents[agent_id].get_velocity()))
        vertical_speed = -float(env.agents[agent_id].get_property_value(c.velocities_v_down_mps))

        # 低空保护：Short Skate在低空下沉时立即中断，切换为返航抬头命令。
        if current_alt < 3200.0 and (current_speed < 220.0 or vertical_speed < -8.0):
            rtb_heading = 180.0 if agent_id.startswith('A') else 0.0
            heading_diff = self.task._normalize_angle_diff(rtb_heading - current_heading)
            hdg_cmd = self.task._convert_heading_to_index(np.deg2rad(heading_diff))
            if current_alt < 1800.0 or vertical_speed < -20.0:
                return 12, 8, 4
            if current_alt < 2600.0 or vertical_speed < -12.0:
                return 11, hdg_cmd, 4
            return 10, hdg_cmd, 4
        
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
                
                vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, vel_cmd
                else:
                    return 7, 8, vel_cmd
            else:
                state["phase"] = "turn_cold"
                state["phase_start_time"] = current_time
                state["turn_cold_start_heading"] = current_heading
        
        # 阶段2：Turn Cold
        elif state["phase"] == "turn_cold":
            if phase_time < turn_cold_duration:
                if state.get("turn_cold_target_heading") is not None:
                    target_heading = float(state["turn_cold_target_heading"]) % 360.0
                else:
                    target_heading = state["turn_cold_start_heading"] + state["turn_cold_angle"]
                    target_heading = target_heading % 360
                heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
                
                vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, vel_cmd
                else:
                    return 7, 8, vel_cmd
            else:
                state["phase"] = "escape"
                state["phase_start_time"] = current_time
        
        # 阶段3：加速逃离
        elif state["phase"] == "escape":
            if phase_time < escape_duration:
                # 逃离阶段仍限制在安全速度档位(2/3/4)
                return 7, 8, 4
            else:
                state["phase"] = "completed"
                state["phase_start_time"] = current_time

        # 完成态：保持脱离航向，不再重启阶段机动
        elif state["phase"] == "completed":
            if agent_id.startswith('A'):
                target_heading = 180.0
            else:
                target_heading = 0.0

            heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
            vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)
            if abs(heading_diff) > 5.0:
                return 7, 6 if heading_diff < 0 else 10, vel_cmd
            else:
                if not hasattr(self.task, '_skate_completed'):
                    self.task._skate_completed = {}
                self.task._skate_completed[f'{agent_id}_skate_done'] = True
                self.task.short_skate_states.pop(agent_id, None)
                self.task.short_skate_start_time.pop(agent_id, None)
                logging.info(f"✅ [{agent_id}] Short Skate完成，释放机动状态")
                return 7, 8, vel_cmd
        
        vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
        return 7, 8, vel_cmd
    
    def execute_short_skate(self, env, agent_id: str, direction='left', target_heading=None) -> tuple:
        """执行Short Skate机动的简化接口"""
        current_time = env.current_step * env.time_interval
        return self.execute_short_skate_precise(env, agent_id, current_time, direction)
    
    def execute_beam_maneuver(self, env, agent_id: str) -> tuple:
        """
        执行Beam机动 - 三九机动，将敌机置于自身3/9位置
        """
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        if hasattr(self.task, '_log_key_event') and agent_id.startswith('A') and env.current_step % 120 == 0:
            phase_obj = None
            try:
                phase_obj = self.task.state_manager.get_agent_phase(agent_id)
            except Exception:
                phase_obj = None
            phase_str = phase_obj.value if hasattr(phase_obj, 'value') else phase_obj
            self.task._log_key_event(
                env,
                agent_id,
                "机动函数进入",
                current_time=current_time,
                当前阶段=phase_str,
                当前战术=getattr(self.task, 'selected_tactic', None),
                进入函数="execute_beam_maneuver",
                执行机动="BEAM",
                当前状态=f"heading={float(current_heading):.1f}deg",
                当前指令="beam_pending",
                退出条件="beam_state超时/导弹威胁释放",
                是否满足退出="否",
            )

        # ✅ 中文文件日志：进入Beam（节流）
        try:
            from utils.trace_logger import trace_throttle
            trace_throttle(
                key=f"beam_enter:{agent_id}",
                min_steps=120,
                标题="Beam-进入",
                env=env,
                我机=agent_id,
                阶段=(self.task.state_manager.get_agent_phase(agent_id).value if hasattr(self.task, 'state_manager') else None),
                战术=str(getattr(self.task, 'selected_tactic', None)),
                说明="进入 execute_beam_maneuver()",
                数据={"当前航向_deg": round(float(current_heading), 1)},
            )
        except Exception:
            pass
        
        # 🔥 修复：导弹制导保护时间进一步缩短为1.5秒，降低Beam固挂时间
        last_launch = self.task.last_missile_launch_time.get(agent_id, -999)
        if last_launch > 0 and (current_time - last_launch) < 1.5:
            remaining_time = 1.5 - (current_time - last_launch)
            try:
                from utils.trace_logger import trace_throttle
                trace_throttle(
                    key=f"beam_guidance_protect:{agent_id}",
                    min_steps=120,
                    标题="Beam-制导保护",
                    env=env,
                    我机=agent_id,
                    阶段=(self.task.state_manager.get_agent_phase(agent_id).value if hasattr(self.task, 'state_manager') else None),
                    战术=str(getattr(self.task, 'selected_tactic', None)),
                    说明="发射后短暂保护窗口内保持直飞",
                    数据={"剩余_s": round(float(remaining_time), 2), "返回动作": (7, 8, 3)},
                )
            except Exception:
                pass
            vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
            return 7, 8, vel_cmd
        
        enemy_bearing = self.task._get_enemy_bearing(env, agent_id)
        if enemy_bearing is None:
            try:
                from utils.trace_logger import trace_throttle
                trace_throttle(
                    key=f"beam_no_bearing:{agent_id}",
                    min_steps=120,
                    标题="Beam-失败",
                    env=env,
                    我机=agent_id,
                    阶段=(self.task.state_manager.get_agent_phase(agent_id).value if hasattr(self.task, 'state_manager') else None),
                    战术=str(getattr(self.task, 'selected_tactic', None)),
                    说明="无法获取敌机方位，回退直飞",
                    数据={"返回动作": (7, 8, 3)},
                )
            except Exception:
                pass
            vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
            return 7, 8, vel_cmd

        # ✅ 中文文件日志：敌机方位（节流）
        try:
            from utils.trace_logger import trace_throttle
            trace_throttle(
                key=f"beam_enemy_bearing:{agent_id}",
                min_steps=120,
                标题="Beam-方位",
                env=env,
                我机=agent_id,
                阶段=(self.task.state_manager.get_agent_phase(agent_id).value if hasattr(self.task, 'state_manager') else None),
                战术=str(getattr(self.task, 'selected_tactic', None)),
                说明="获取敌机方位",
                数据={"敌机方位_deg": round(float(enemy_bearing), 1)},
            )
        except Exception:
            pass
        
        # 🔥 重构：三九机动规范角度为80-90度，随机选择
        import random
        beam_angle = 85.0
        beam_left = (enemy_bearing - beam_angle) % 360
        beam_right = (enemy_bearing + beam_angle) % 360
        
        # 角度/左右目标属于调试信息：不再输出到控制台
        
        diff_left = abs(self.task._normalize_angle_diff(beam_left - current_heading))
        diff_right = abs(self.task._normalize_angle_diff(beam_right - current_heading))
        
        # 🔥 修复：确保长机和僚机方向不同，动态选择
        # 长机固定选择一个方向，僚机选择另一个方向
        is_lead = agent_id.endswith('100')
        import random
        
        # 角色判断属于调试信息：不再输出到控制台
        
        # 初始化Beam方向状态（如果不存在）
        if not hasattr(self.task, 'beam_direction_state'):
            self.task.beam_direction_state = {}
        
        # 🔥 修复：检查是否是钳形攻势
        is_pincer = getattr(self.task, 'selected_tactic', None) == 'PINCER_ATTACK' or \
                   getattr(self.task, '_defense_prev_tactic', None) == 'PINCER_ATTACK'
        
        # 战术判断属于调试信息：不再输出到控制台
        
        # 🔥 修复：确保长机和僚机使用相反方向
        # 获取队友的方向，确保相反
        teammate_id = self.task._get_teammate_id(agent_id) if hasattr(self.task, '_get_teammate_id') else ('A0100' if agent_id == 'A0200' else 'A0200')
        teammate_direction = None
        if teammate_id in self.task.beam_direction_state:
            teammate_direction = self.task.beam_direction_state[teammate_id]
        
        # 🔥 关键修复：如果该机还没有确定Beam方向，根据角色和队友方向分配，确保相反且随机
        # 🔥 但优先使用tactical_evasion设置的方向（如果存在）
        if agent_id not in self.task.beam_direction_state:
            # 检查是否有tactical_evasion设置的方向（通过tactical_evasion_direction_state）
            if hasattr(self.task, 'tactical_evasion_direction_state') and agent_id in self.task.tactical_evasion_direction_state:
                # 优先使用tactical_evasion设置的方向
                self.task.beam_direction_state[agent_id] = self.task.tactical_evasion_direction_state[agent_id]
            elif teammate_direction is not None:
                # 队友有方向，使用相反方向
                self.task.beam_direction_state[agent_id] = 'right' if teammate_direction == 'left' else 'left'
            else:
                # 队友没有方向，根据战术分配
                if is_pincer:
                    # 🔥 修复：钳形攻势时，长机必须右转，僚机左转
                    self.task.beam_direction_state[agent_id] = 'right' if is_lead else 'left'
                else:
                    # 其他战术：随机分配但确保相反
                    import random
                    rand_val = random.random()
                    if rand_val < 0.5:
                        # 50%概率：长机左转，僚机右转
                        self.task.beam_direction_state[agent_id] = 'left' if is_lead else 'right'
                    else:
                        # 50%概率：长机右转，僚机左转（增加多样性）
                        self.task.beam_direction_state[agent_id] = 'right' if is_lead else 'left'
        else:
            # 🔥 修复：如果已设置方向但与队友相同，强制切换
            # 🔥 但如果tactical_evasion设置了方向，优先使用tactical_evasion的方向
            if hasattr(self.task, 'tactical_evasion_direction_state') and agent_id in self.task.tactical_evasion_direction_state:
                tactical_direction = self.task.tactical_evasion_direction_state[agent_id]
                if self.task.beam_direction_state[agent_id] != tactical_direction:
                    self.task.beam_direction_state[agent_id] = tactical_direction
            elif teammate_direction is not None:
                current_direction = self.task.beam_direction_state[agent_id]
                if current_direction == teammate_direction:
                    # 与队友方向相同，强制切换
                    self.task.beam_direction_state[agent_id] = 'right' if teammate_direction == 'left' else 'left'
        
        # 🔥 修复：如果tactical_evasion临时设置了方向，优先使用临时方向
        if hasattr(self.task, 'beam_direction_state') and agent_id in self.task.beam_direction_state:
            assigned_direction = self.task.beam_direction_state[agent_id]
        else:
            # 默认：长机左转，僚机右转
            assigned_direction = 'left' if is_lead else 'right'
        
        # Beam机动状态：统一存放在state_manager，避免多处写入导致“缺start_time/目标航向漂移”
        beam_state = None
        try:
            beam_state = getattr(getattr(self.task, 'state_manager', None), 'beam_maneuver_state', None)
        except Exception:
            beam_state = None
        if not isinstance(beam_state, dict):
            # 兼容极端情况：回退到task自身字段
            if not hasattr(self.task, 'beam_maneuver_state') or not isinstance(getattr(self.task, 'beam_maneuver_state', None), dict):
                self.task.beam_maneuver_state = {}
            beam_state = self.task.beam_maneuver_state
        
        # 如果Beam机动已经开始，使用保存的目标航向；否则计算新的目标航向
        if agent_id in beam_state and 'target_heading' in beam_state[agent_id]:
            # 使用保存的目标航向
            target_heading = beam_state[agent_id]['target_heading']
            assigned_direction = beam_state[agent_id].get('assigned_direction', assigned_direction)
            beam_direction = "3点方向(左转)" if assigned_direction == 'left' else "9点方向(右转)"
            if env.current_step % 30 == 0:
                pass
        else:
            # 计算新的目标航向并保存
            if assigned_direction == 'left':
                target_heading = beam_left
                beam_direction = "3点方向(左转)"
            else:
                target_heading = beam_right
                beam_direction = "9点方向(右转)"
            
            # 🔥 重构：三九机动应该转80-90度，不要限制到±15度
            # 直接使用计算出的80-90度目标航向，不进行限制
            heading_diff = self.task._normalize_angle_diff(target_heading - current_heading)
            
            # 目标航向差属于调试信息：不再输出到控制台
            
            # 保存目标航向和方向到状态中
            if agent_id not in beam_state:
                beam_state[agent_id] = {}
            beam_state[agent_id]['target_heading'] = target_heading
            beam_state[agent_id]['assigned_direction'] = assigned_direction
            # 保证start_time存在（否则会出现“状态存在但没有start_time”日志）
            beam_state[agent_id].setdefault('start_time', current_time)
            
            # 方向分配属于调试信息：不再输出到控制台
        
        # 航向差属于调试信息：不再输出到控制台
        
        if agent_id in beam_state:
            state = beam_state[agent_id]
            # 兼容旧状态：缺start_time则补齐，但不改目标航向
            if 'start_time' not in state:
                # 状态修复属于异常分支：只做状态补齐，不刷屏
                state['start_time'] = current_time
            # 兼容：若target_heading/assigned_direction缺失，使用当前计算值补齐
            state.setdefault('target_heading', target_heading)
            state.setdefault('assigned_direction', assigned_direction)
            state.setdefault('execute_duration_s', 2.5)
            state.setdefault('total_duration_s', 4.0)
            
            elapsed = current_time - state['start_time']
            
            # Beam机动保持更久一些，便于观察真正的三九(beam)
            execute_duration = float(state.get('execute_duration_s', 2.5))
            total_duration = float(max(execute_duration, state.get('total_duration_s', 4.0)))

            # 执行阶段：尽快转到目标航向（80-90度侧向）
            if elapsed < execute_duration:
                # 🔥 关键修复：使用保存的目标航向和方向
                saved_target_heading = state.get('target_heading', target_heading)
                saved_assigned_direction = state.get('assigned_direction', assigned_direction)
                current_heading_now = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                heading_diff_now = self.task._normalize_angle_diff(saved_target_heading - current_heading_now)
                
                # 🔥 重构：根据航向差动态选择转向力度，确保能够完成80-90度转向
                # 如果航向差很大（>45度），使用更大的转向力度
                if abs(heading_diff_now) > 45:
                    # 大角度差：使用最大转向力度
                    if saved_assigned_direction == 'left':
                        hdg_cmd = 5  # 左转（约-60度，更大力度）
                    else:
                        hdg_cmd = 11  # 右转（约+30度，更大力度）
                elif abs(heading_diff_now) > 15:
                    # 中等角度差：使用中等转向力度
                    if saved_assigned_direction == 'left':
                        hdg_cmd = 6  # 左转（-30度）
                    else:
                        hdg_cmd = 10  # 右转（+5度）
                elif abs(heading_diff_now) > 3:
                    # 小角度差：使用小转向力度
                    if saved_assigned_direction == 'left':
                        hdg_cmd = 7  # 微左转（-20度）
                    else:
                        hdg_cmd = 9  # 微右转（+2度）
                else:
                    # 航向差<=3度：保持航向（已到达目标）
                    hdg_cmd = 8
                
                # 🔥 关键修复：Beam机动时必须保持足够速度！速度命令从3改为4（维持升力，防止下沉）
                # 速度命令：0=下降很多, 3-4=保持/缓升, 6+=快速爬升
                heading_cmd_result = (7, hdg_cmd, 4)  # 保持高度、转向、提高速度（防止下沉）
                beam_dir_str = "3点方向(左转)" if saved_assigned_direction == 'left' else "9点方向(右转)"
                try:
                    from utils.trace_logger import trace_throttle
                    trace_throttle(
                        key=f"beam_exec:{agent_id}",
                        min_steps=30,
                        标题="Beam-执行",
                        env=env,
                        我机=agent_id,
                        阶段=(self.task.state_manager.get_agent_phase(agent_id).value if hasattr(self.task, 'state_manager') else None),
                        战术=str(getattr(self.task, 'selected_tactic', None)),
                        说明="转向到3/9位目标航向（速度提高防止下沉）",
                        数据={
                            "敌机方位_deg": round(float(enemy_bearing), 1),
                            "目标航向_deg": round(float(saved_target_heading), 1),
                            "当前航向_deg": round(float(current_heading_now), 1),
                            "航向差_deg": round(float(heading_diff_now), 1),
                            "方向": str(saved_assigned_direction),
                            "hdg_cmd": int(hdg_cmd),
                            "返回动作": tuple(heading_cmd_result),
                            "剩余_s": round(float(execute_duration - elapsed), 2),
                        },
                    )
                except Exception:
                    pass
                return heading_cmd_result

            # 保持阶段：继续维持侧向姿态一段时间（避免“点一下就回去”的观感）
            elif elapsed < total_duration:
                # 🔥 关键修复：即使在冷却期，也保持之前的转向方向，使用保存的目标航向
                saved_target_heading = state.get('target_heading', target_heading)
                saved_assigned_direction = state.get('assigned_direction', assigned_direction)
                current_heading_now = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                heading_diff_now = self.task._normalize_angle_diff(saved_target_heading - current_heading_now)
                
                # 🔥 重构：根据航向差动态选择转向力度，确保能够完成80-90度转向
                if abs(heading_diff_now) > 45:
                    # 大角度差：使用最大转向力度
                    if saved_assigned_direction == 'left':
                        hdg_cmd = 5  # 左转（约-60度）
                    else:
                        hdg_cmd = 11  # 右转（约+30度）
                elif abs(heading_diff_now) > 15:
                    # 中等角度差：使用中等转向力度
                    if saved_assigned_direction == 'left':
                        hdg_cmd = 6  # 左转（-30度）
                    else:
                        hdg_cmd = 10  # 右转（+5度）
                elif abs(heading_diff_now) > 3:
                    # 小角度差：使用小转向力度
                    if saved_assigned_direction == 'left':
                        hdg_cmd = 7  # 微左转（-20度）
                    else:
                        hdg_cmd = 9  # 微右转（+2度）
                else:
                    # 航向差<=3度：保持航向（已到达目标）
                    hdg_cmd = 8
                
                try:
                    from utils.trace_logger import trace_throttle
                    trace_throttle(
                        key=f"beam_hold:{agent_id}",
                        min_steps=60,
                        标题="Beam-保持",
                        env=env,
                        我机=agent_id,
                        阶段=(self.task.state_manager.get_agent_phase(agent_id).value if hasattr(self.task, 'state_manager') else None),
                        战术=str(getattr(self.task, 'selected_tactic', None)),
                        说明="保持侧向姿态（速度命令提高防止下沉）",
                        数据={
                            "目标航向_deg": round(float(saved_target_heading), 1),
                            "当前航向_deg": round(float(current_heading_now), 1),
                            "航向差_deg": round(float(heading_diff_now), 1),
                            "方向": str(saved_assigned_direction),
                            "hdg_cmd": int(hdg_cmd),
                            "剩余_s": round(float(total_duration - elapsed), 2),
                        },
                    )
                except Exception:
                    pass
                # 🔥 关键修复：保持阶段也要提高速度命令（从3改为4）防止下沉
                return self.maintain_heading_precise(
                    env,
                    agent_id,
                    float(saved_target_heading),
                    duration=max(0.8, total_duration - elapsed),
                )
            
            else:
                # Beam机动完成后，清除状态，避免无限循环导致转圈
                # 避免无限循环执行Beam机动导致转圈
                try:
                    from utils.trace_logger import trace_event
                    trace_event(
                        事件="Beam-结束",
                        env=env,
                        我机=agent_id,
                        阶段=(self.task.state_manager.get_agent_phase(agent_id).value if hasattr(self.task, 'state_manager') else None),
                        战术=str(getattr(self.task, 'selected_tactic', None)),
                        说明="Beam机动完成，清除状态并回退默认动作",
                        数据={"总时长_s": float(total_duration), "返回动作": (7, 8, 3)},
                    )
                except Exception:
                    pass
                # 清除Beam机动状态
                if agent_id in beam_state:
                    del beam_state[agent_id]
                # 返回默认动作，不再执行Beam机动
                return self.maintain_heading_precise(
                    env,
                    agent_id,
                    float(target_heading),
                    duration=1.0,
                )
        else:
            # 🔥 修复：初始化Beam机动状态，保存目标航向和方向
            beam_state[agent_id] = {
                'start_time': current_time,
                'target_heading': target_heading,
                'assigned_direction': assigned_direction,
                'execute_duration_s': 2.5,
                'total_duration_s': 4.0,
            }
            # 🔥 修复：直接根据assigned_direction生成转向指令
            current_heading_now = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            heading_diff_now = self.task._normalize_angle_diff(target_heading - current_heading_now)
            
            # 🔥 重构：根据航向差动态选择转向力度，确保能够完成80-90度转向
            if abs(heading_diff_now) > 45:
                # 大角度差：使用最大转向力度
                hdg_cmd = 5 if assigned_direction == 'left' else 11
            elif abs(heading_diff_now) > 15:
                # 中等角度差：使用中等转向力度
                hdg_cmd = 6 if assigned_direction == 'left' else 10
            elif abs(heading_diff_now) > 3:
                # 小角度差：使用小转向力度
                hdg_cmd = 7 if assigned_direction == 'left' else 9
            else:
                # 航向差<=3度：保持航向（已到达目标）
                hdg_cmd = 8
            
            heading_cmd_result = (7, hdg_cmd, 3)
            beam_dir_str = "3点方向(左转)" if assigned_direction == 'left' else "9点方向(右转)"
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="Beam-启动",
                    env=env,
                    我机=agent_id,
                    阶段=(self.task.state_manager.get_agent_phase(agent_id).value if hasattr(self.task, 'state_manager') else None),
                    战术=str(getattr(self.task, 'selected_tactic', None)),
                    说明="初始化Beam目标航向并开始转向",
                    数据={
                        "敌机方位_deg": round(float(enemy_bearing), 1),
                        "目标航向_deg": round(float(target_heading), 1),
                        "当前航向_deg": round(float(current_heading_now), 1),
                        "航向差_deg": round(float(heading_diff_now), 1),
                        "方向": str(assigned_direction),
                        "hdg_cmd": int(hdg_cmd),
                        "返回动作": tuple(heading_cmd_result),
                    },
                )
            except Exception:
                pass
            return heading_cmd_result
        
        # 🔥 关键修复：确保所有代码路径都有返回值（防止返回None）
        # 如果上面的分支都没有执行，使用默认动作
        try:
            from utils.trace_logger import trace_event
            trace_event(
                事件="Beam-异常",
                env=env,
                我机=agent_id,
                阶段=(self.task.state_manager.get_agent_phase(agent_id).value if hasattr(self.task, 'state_manager') else None),
                战术=str(getattr(self.task, 'selected_tactic', None)),
                说明="Beam机动所有分支未命中，回退默认动作",
                数据={"返回动作": (7, 8, 3)},
            )
        except Exception:
            pass
        vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
        return 7, 8, vel_cmd
    
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

        # 🔥 修复：不允许在机动进行中切换方向，避免方向混乱导致转圈
        # 如果外部在机动进行中切换了方向/爬升策略，记录警告但不更新（避免一直沿用初始化方向）
        if state.get('type') == 'tactical_crank' and state.get('phase') == 'crank_climb':
            if state.get('direction') != direction or state.get('climb') != climb:
                prev_direction = state.get('direction')
                # 🔥 关键修复：不允许在机动进行中切换方向，保持初始方向
                logging.warning(f"⚠️ 【战术Crank】{agent_id}方向冲突: 当前={prev_direction}, 请求={direction}, 保持当前方向")
                # 不更新方向，保持初始方向
                # state['direction'] = direction
                # state['climb'] = climb
                # state['initial_heading'] = current_heading
                # state['phase_start_time'] = current_time

        phase_time = current_time - state['phase_start_time']
        direction = state.get('direction', 'left')
        climb = state.get('climb', True)
        
        if state['phase'] == 'crank_climb':
            # ✅ 低空安全门：高度<4000m时放弃Crank机动，防止低空能量耗尽
            _crank_alt = float(env.agents[agent_id].get_position()[2])
            if _crank_alt < 4000.0:
                if agent_id in self.task.maneuver_states:
                    del self.task.maneuver_states[agent_id]
                _vel = int(np.clip(self._get_dynamic_velocity_cmd(env, agent_id), 2, 4))
                if env.current_step % 30 == 0:
                    logging.warning(f"⚠️ [Crank安全门-{agent_id}] 高度{_crank_alt:.0f}m<4000m，放弃Crank爬升恢复")
                return 10, 8, _vel
            # 缩短Crank持续时间：2秒
            if phase_time < 2.0:
                # 🔥 紧急修复转圈问题：Crank机动时必须保持高速度，禁止爬升
                # 爬升会降低速度导致失速转圈，改为保持高度或轻微下降
                alt_cmd = 7  # 保持高度，不爬升不下降
                
                # 🔥 修复：使用非常温和的转向指令，防止大角度转弯失速
                # 转向索引：6=小左(-30°), 7=微左(-20°), 9=微右(+2°), 10=小右(+5°)
                # 🔥 进一步减小角度：使用更小的转向角度，避免转圈
                if direction == 'left':
                    hdg_cmd = 7  # 微左转，避免过度转弯
                else:
                    hdg_cmd = 9  # 微右转，避免过度转弯
                
                # 🔥 关键：速度指令改为加速，确保有足够能量支撑转弯
                result = (alt_cmd, hdg_cmd, 4)  # 保持高度，温和转向，vel=4(避免高能量消耗)
                if env.current_step % 60 == 0:
                    logging.warning(f"🔍 [Crank机动-{agent_id}] 方向={direction}, hdg_cmd={hdg_cmd}({'左' if direction == 'left' else '右'}), 返回动作={result}, 剩余时间={2.0-phase_time:.1f}s")
                return result
            state['phase'] = 'level_off'
            state['phase_start_time'] = current_time
            logging.info(f"🔄 【战术Crank】{agent_id}: Crank→平飞恢复")
        
        if state['phase'] == 'level_off':
            if phase_time < 2.0:
                vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
                return 7, 8, vel_cmd
            else:
                del self.task.maneuver_states[agent_id]
                logging.info(f"✅ 【战术Crank】{agent_id}完成机动，方位角已变化")
                vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
                return 7, 8, vel_cmd
        
        vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
        return 7, 8, vel_cmd
    
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
                vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
                return 7, 8, vel_cmd
        
        vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
        return 7, 8, vel_cmd
    
    def execute_notch_back(self, env, agent_id: str, direction=None) -> tuple:
        """执行Notch Back机动 - 90°偏置+下降规避导弹 - 🔥 修复：根据长机/僚机自动选择方向"""
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        # 🔍 调试：打印进入Notch Back
        logging.warning(f"🔍 [Notch Back-{agent_id}] 进入Notch Back机动，当前航向={current_heading:.1f}°")
        
        # 🔥 修复：如果没有指定方向，根据长机/僚机自动选择相反方向
        if direction is None:
            is_lead = agent_id.endswith('100')
            # 长机左转，僚机右转（确保相反）
            direction = 'left' if is_lead else 'right'
            logging.warning(f"🔍 [Notch Back-{agent_id}] 自动选择方向: 长机={'是' if is_lead else '否'} → {direction}")
        else:
            logging.warning(f"🔍 [Notch Back-{agent_id}] 使用指定方向: {direction}")
        
        # 检查是否在返航状态，返航时不执行Notch Back
        if hasattr(self.task, 'returning_agents') and agent_id in getattr(self.task, 'returning_agents', set()):
            logging.debug(f"{agent_id} 正在返航，跳过Notch Back机动")
            vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
            return 7, 8, vel_cmd  # 直飞+保持高度+动态速度
        
        if agent_id not in self.task.maneuver_states or self.task.maneuver_states[agent_id].get('type') != 'notch_back':
            self.task.maneuver_states[agent_id] = {
                'type': 'notch_back',
                'phase': 'beam_descend',
                'start_time': current_time,
                'phase_start_time': current_time,
                'direction': direction,
                'initial_heading': current_heading
            }
            
            logging.info(f"🔄 【Notch Back】{agent_id}开始90°偏置下降机动，方向={direction}")
        
        state = self.task.maneuver_states[agent_id]
        phase_time = current_time - state['phase_start_time']
        
        if state['phase'] == 'beam_descend':
            # 延长持续时间到15秒，让机动更加充分
            if phase_time < 15.0:  # 15秒而非8秒
                # 90度偏置下降：beam机动(垂直于威胁方向) + 下降 + 动态速度
                vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
                if state['direction'] == 'left':
                    return 3, 16, vel_cmd  # 左转 + 下降 + 动态速度
                else:
                    return 5, 16, vel_cmd  # 右转 + 下降 + 动态速度
            else:
                del self.task.maneuver_states[agent_id]
                logging.info(f"✅ 【Notch Back】{agent_id}完成机动")
                vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
                return 7, 8, vel_cmd  # 直飞+保持高度+动态速度
        
        vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)  # 🔥 使用动态速度管理
        return 7, 8, vel_cmd
    
    def establish_rear_formation(self, env, agent_id: str, current_time: float) -> tuple:
        """僚机建立后方队形（前后攻击战术专用）"""
        leader_id = self.task._get_teammate_id(agent_id) if hasattr(self.task, '_get_teammate_id') else "A0100"
        leader = env._jsbsims.get(leader_id)
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
            # 🔧 降低日志频率：每60步输出一次（约12秒）
            env_obj = getattr(self.task, 'env', None)
            current_step = getattr(env_obj, 'current_step', 0) if env_obj else 0
            if not hasattr(self.task, '_alt_high_log_step') or current_step - self.task._alt_high_log_step > 60:
                logging.warning(f"⚠️ [{agent_id}]高度过高 {alt_diff:.0f}m (阈值{alt_threshold}m) → 下降")
                self.task._alt_high_log_step = current_step
        elif alt_diff < -alt_threshold:
            altitude_cmd_id = 9
            # 🔧 降低日志频率
            env_obj = getattr(self.task, 'env', None)
            current_step = getattr(env_obj, 'current_step', 0) if env_obj else 0
            if not hasattr(self.task, '_alt_low_log_step') or current_step - self.task._alt_low_log_step > 60:
                logging.warning(f"⚠️ [{agent_id}]高度过低 {alt_diff:.0f}m (阈值{alt_threshold}m) → 上升")
                self.task._alt_low_log_step = current_step
        
        # 🔥 关键修复：将全局坐标差旋转到长机本体坐标系
        # dx_body > 0 = 僚机在长机前方, dx_body < 0 = 僚机在长机后方
        # dy_body > 0 = 僚机在长机右方, dy_body < 0 = 僚机在长机左方
        raw_dx = wingman_x - leader_x
        raw_dy = wingman_y - leader_y
        leader_heading_rad = np.deg2rad(leader_heading)
        cos_h = np.cos(leader_heading_rad)
        sin_h = np.sin(leader_heading_rad)
        dx = raw_dx * cos_h + raw_dy * sin_h     # 沿长机前进方向
        dy = -raw_dx * sin_h + raw_dy * cos_h    # 沿长机右侧方向
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
                        logging.debug(f"✅ [{agent_id}]阶段1-右转对齐: dy={dy:.0f}m hdg={heading_diff:.1f}°")
                    elif dy > 2000:
                        heading_cmd_id = 6
                        velocity_cmd_id = 3
                        logging.debug(f"🔄 [{agent_id}]阶段1-大幅左转: dy={dy:.0f}m")
                    else:
                        heading_cmd_id = 7
                        velocity_cmd_id = 3
                        logging.debug(f"🔄 [{agent_id}]阶段1-中幅左转: dy={dy:.0f}m")
                else:
                    heading_cmd_id = 9
                    velocity_cmd_id = 3
                    logging.debug(f"⚠️ [{agent_id}]阶段1-过头右转: dy={dy:.0f}m")
        
        elif current_state == 'HEADING_CORRECT':
            if abs(heading_diff) < TARGET_HEADING_TOLERANCE:
                self.task.formation_state[agent_id] = 'LONGITUDINAL_ADJUST'
                logging.info(f"✅ [{agent_id}]航向对齐完成 hdg_diff={heading_diff:.1f}° → 进入纵向调整")
            else:
                if heading_diff > 30.0:
                    heading_cmd_id = 10
                    logging.debug(f"🔄 [{agent_id}]阶段2-大幅右转: hdg_diff={heading_diff:.1f}°")
                elif heading_diff > TARGET_HEADING_TOLERANCE:
                    heading_cmd_id = 9
                    logging.debug(f"🔄 [{agent_id}]阶段2-微幅右转: hdg_diff={heading_diff:.1f}°")
                elif heading_diff < -30.0:
                    heading_cmd_id = 6
                    logging.debug(f"🔄 [{agent_id}]阶段2-异常左转: hdg_diff={heading_diff:.1f}°")
                elif heading_diff < -TARGET_HEADING_TOLERANCE:
                    heading_cmd_id = 7
                    logging.debug(f"🔄 [{agent_id}]阶段2-异常微左转: hdg_diff={heading_diff:.1f}°")
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
                
                # 减少日志输出频率，每60步输出一次（约12秒）
                if env.current_step % 60 == 0:
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
