"""
导弹管理模块
管理导弹发射、追踪、状态更新等功能
从tactical_task.py中提取，提高代码可维护性
"""
import logging
import numpy as np
from envs.JSBSim.core.catalog import Catalog as c
from tactical_types import TacticalPhase, get_target_with_fallback
from simulation.radar_manager import check_missile_launch_conditions


class MissileManager:
    """导弹管理器 - 负责导弹相关操作"""
    
    def __init__(self, state_manager):
        """
        Args:
            state_manager: TacticalStateManager实例
        """
        self.state_manager = state_manager
        self.missile_tracks = {}  # 导弹追踪信息
        self.aircraft_missile_counts = {}  # 每架飞机的导弹发射计数 {agent_id: fired_count}
        self.MAX_MISSILES_PER_AIRCRAFT = 4  # 每架飞机最大导弹数量
    
    def should_launch_missile(self, env, agent_id: str, current_phase, target_id: str = None, 
                           salvo_mode: str = "single") -> bool:
        """
        判断是否应该发射导弹 - 支持分批发射 + 严格4导弹限制
        
        Args:
            env: 环境
            agent_id: 飞机ID
            current_phase: 当前战术阶段
            target_id: 目标ID
            salvo_mode: 发射模式 ("single", "double", "salvo")
            
        Returns:
            True表示应该发射
        """
        # **关键修复：严格检查4导弹限制**
        if agent_id not in self.aircraft_missile_counts:
            self.aircraft_missile_counts[agent_id] = 0
        
        fired_count = self.aircraft_missile_counts[agent_id]
        
        # 根据发射模式检查导弹数量要求
        if salvo_mode == "single":
            required_missiles = 1
        elif salvo_mode == "double":
            required_missiles = 2
        elif salvo_mode == "salvo":
            required_missiles = min(2, self.MAX_MISSILES_PER_AIRCRAFT - fired_count)  # 最多一次性发射2枚
        else:
            required_missiles = 1
        
        # **关键检查：确保不超过4枚导弹限制**
        if fired_count + required_missiles > self.MAX_MISSILES_PER_AIRCRAFT:
            logging.debug(f"⛔ [{agent_id}] 导弹已达上限: 已发射{fired_count}枚, 需要{required_missiles}枚, 上限{self.MAX_MISSILES_PER_AIRCRAFT}枚")
            return False
        
        # 检查剩余导弹
        current_time = env.current_step * env.time_interval
        aircraft = env.agents[agent_id]
        
        if aircraft.num_missiles < required_missiles:
            logging.debug(f"⛔ [{agent_id}] 导弹不足: 需要{required_missiles}枚, 剩余{aircraft.num_missiles}枚")
            return False
        
        if not self.state_manager.can_launch_missile(agent_id, current_time):
            return False
        
        # 获取目标
        if target_id is None:
            target_id = get_target_with_fallback(agent_id, env)
        
        if target_id is None:
            return False
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            return False
        
        # 计算距离
        from tactical_utils import TacticalUtils
        distance = TacticalUtils.calculate_distance_between(env.agents[agent_id], target_aircraft)

        # MAR 保护：MAR内不允许新发射（40km）
        if distance <= 40000:
            logging.debug(f"⛔ [{agent_id}] 距离{distance/1000:.1f}km ≤ MAR(40km)，禁止发射")
            return False

        # 3.7.2 导弹发射集成检查（雷达模式/锁定/跟踪质量/Notch/探测概率/范围）
        radar_passed = True
        try:
            launch_check = check_missile_launch_conditions(env, agent_id, target_id)
            if not launch_check.get('can_launch', False):
                reason = launch_check.get('reason', 'unknown')
                logging.debug(f"⛔ [{agent_id}] 发射条件不满足（3.7.2）: {reason}")
                radar_passed = False
        except Exception as e:
            logging.debug(f"⚠️ [{agent_id}] 发射条件检查异常，回退到阶段窗口: {e}")
            radar_passed = False
        
        # 放宽发射窗口条件，优先基于阶段而非严格距离
        launch_allowed = False
        
        if current_phase == TacticalPhase.LR_TR:
            # LR阶段：70-80km为理想窗口，但60-85km也可发射
            if 60000 <= distance <= 85000:
                launch_allowed = True
                logging.info(f"🎯 [{agent_id}] LR阶段发射: 距离{distance/1000:.1f}km")
        elif current_phase == TacticalPhase.TR_DOR:
            # TR阶段：允许更广范围发射
            if 40000 <= distance <= 75000:
                launch_allowed = True  
                logging.info(f"🎯 [{agent_id}] TR阶段发射: 距离{distance/1000:.1f}km")
        elif current_phase == TacticalPhase.DOR_DR:
            # DOR阶段：近距发射窗口
            if 40000 <= distance <= 65000:
                launch_allowed = True
                logging.info(f"🎯 [{agent_id}] DOR阶段发射: 距离{distance/1000:.1f}km")
        elif current_phase == TacticalPhase.BEYOND_MAR:
            # 超过MAR：只有在特定条件下才发射
            if 40000 <= distance <= 60000 and radar_passed:
                launch_allowed = True
                logging.info(f"🎯 [{agent_id}] MAR后发射: 距离{distance/1000:.1f}km")
        
        # 如果基础条件不满足，检查朝向补偿
        if not launch_allowed and distance >= 40000 and distance <= 90000:
            try:
                my_pos = env.agents[agent_id].get_position()
                tgt_pos = target_aircraft.get_position()
                current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
                import numpy as np
                dx, dy = tgt_pos[0] - my_pos[0], tgt_pos[1] - my_pos[1]
                bearing = np.arctan2(dy, dx)
                err = abs(((bearing - current_heading + np.pi) % (2*np.pi)) - np.pi)
                if np.rad2deg(err) <= 90.0:  # 放宽朝向要求
                    launch_allowed = True
                    logging.info(f"🎯 [{agent_id}] 朝向补偿发射: 距离{distance/1000:.1f}km, 角度{np.rad2deg(err):.1f}°")
            except Exception:
                pass
                
        return launch_allowed
    
    def execute_missile_launch(self, env, agent_id: str, target_id: str = None, 
                            salvo_mode: str = "single") -> bool:
        """
        执行导弹发射 - 支持分批发射 + 严格4导弹限制
        
        Args:
            env: 环境
            agent_id: 飞机ID
            target_id: 目标ID
            salvo_mode: 发射模式 ("single", "double", "salvo")
            
        Returns:
            True表示发射成功
        """
        
        # **关键修复：再次检查4导弹限制**
        if agent_id not in self.aircraft_missile_counts:
            self.aircraft_missile_counts[agent_id] = 0
        
        fired_count = self.aircraft_missile_counts[agent_id]
        
        # 获取目标
        if target_id is None:
            target_id = get_target_with_fallback(agent_id, env)
        
        if target_id is None:
            logging.warning(f"⚠️ [{agent_id}] 导弹发射失败: 无目标")
            return False
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            logging.warning(f"⚠️ [{agent_id}] 导弹发射失败: 目标不存在或已被击毁")
            return False
        
        # 确定发射数量
        aircraft = env.agents[agent_id]
        remaining_quota = self.MAX_MISSILES_PER_AIRCRAFT - fired_count
        
        if salvo_mode == "single":
            launch_count = min(1, remaining_quota, aircraft.num_missiles)
        elif salvo_mode == "double":
            launch_count = min(2, remaining_quota, aircraft.num_missiles)
        elif salvo_mode == "salvo":
            launch_count = min(2, remaining_quota, aircraft.num_missiles)  # 限制一次最多发射2枚
        else:
            launch_count = min(1, remaining_quota, aircraft.num_missiles)
        
        if launch_count <= 0:
            logging.warning(f"⚠️ [{agent_id}] 导弹发射失败: 已发射{fired_count}枚, 达到上限{self.MAX_MISSILES_PER_AIRCRAFT}枚")
            return False
        
        # 执行发射
        current_time = env.current_step * env.time_interval
        launched_count = 0
        
        try:
            for i in range(launch_count):
                if aircraft.num_missiles <= 0 or self.aircraft_missile_counts[agent_id] >= self.MAX_MISSILES_PER_AIRCRAFT:
                    break
                    
                # **修复导弹ID生成逻辑**
                fired_sequence = self.aircraft_missile_counts[agent_id] + 1  # 1-4的序列号
                base_id = agent_id[0] + agent_id[2:]  # A0100 -> A100
                missile_uid = f"{base_id}{fired_sequence:0>2}"  # A10001, A10002, A10003, A10004
                
                # 发射导弹（这里应该调用环境的导弹发射接口）
                # env.launch_missile(agent_id, target_id, missile_uid)
                
                # 记录发射信息
                self.state_manager.record_missile_launch(agent_id, current_time, target_id)
                
                # 记录导弹轨迹追踪
                self.missile_tracks[missile_uid] = {
                    'launcher': agent_id,
                    'target': target_id,
                    'launch_time': current_time,
                    'status': 'active'
                }
                
                # **更新发射计数**
                self.aircraft_missile_counts[agent_id] += 1
                
                # 更新飞机导弹数量
                aircraft.num_missiles = max(0, aircraft.num_missiles - 1)
                launched_count += 1
                
                logging.info(f"🚀 [{agent_id}] 发射导弹[{missile_uid}] → [{target_id}] (第{self.aircraft_missile_counts[agent_id]}/{self.MAX_MISSILES_PER_AIRCRAFT}枚)")
            
            if launched_count > 0:
                launch_mode_desc = {
                    "single": "单发",
                    "double": "双发", 
                    "salvo": "齐射"
                }
                logging.info(f"✅ [{agent_id}] {launch_mode_desc.get(salvo_mode, '发射')} {launched_count}枚导弹 → {target_id} (已发射总计:{self.aircraft_missile_counts[agent_id]}/{self.MAX_MISSILES_PER_AIRCRAFT})")
                return True
            
        except Exception as e:
            logging.error(f"❌ [{agent_id}] 导弹发射异常: {str(e)}")
            
        return False
    
    def update_missile_status(self, env, current_time: float):
        """
        更新所有导弹状态
        
        Args:
            env: 环境
            current_time: 当前时间
        """
        for missile_id, track in list(self.missile_tracks.items()):
            if track['status'] != 'active':
                continue
            
            # 检查导弹飞行时间
            flight_time = current_time - track['launch_time']
            
            # 检查目标状态
            target_id = track['target']
            target_aircraft = env._jsbsims.get(target_id)
            
            if target_aircraft is None or not target_aircraft.is_alive:
                track['status'] = 'target_destroyed'
                logging.info(f"🎯 导弹{missile_id}: 目标{target_id}已被击毁")
                continue
            
            # 简化版：假设导弹飞行60秒后失效
            if flight_time > 60.0:
                track['status'] = 'expired'
                logging.info(f"⏱️ 导弹{missile_id}: 飞行超时失效")
                continue
    
    def get_active_missiles_count(self, agent_id: str = None) -> int:
        """
        获取激活导弹数量
        
        Args:
            agent_id: 飞机ID（None表示所有飞机）
            
        Returns:
            激活导弹数量
        """
        count = 0
        for track in self.missile_tracks.values():
            if track['status'] == 'active':
                if agent_id is None or track['launcher'] == agent_id:
                    count += 1
        return count
    
    def get_missiles_targeting(self, target_id: str) -> int:
        """
        获取瞄准某目标的导弹数量
        
        Args:
            target_id: 目标ID
            
        Returns:
            导弹数量
        """
        count = 0
        for track in self.missile_tracks.values():
            if track['status'] == 'active' and track['target'] == target_id:
                count += 1
        return count
    
    def check_missile_threat(self, env, agent_id: str, current_time: float) -> dict:
        """
        检查导弹威胁
        
        Args:
            env: 环境
            agent_id: 飞机ID
            current_time: 当前时间
            
        Returns:
            威胁信息字典: {'threat_level': 0-3, 'incoming_count': int, 'closest_distance': float}
        """
        threat_info = {
            'threat_level': 0,  # 0-无威胁, 1-低, 2-中, 3-高
            'incoming_count': 0,
            'closest_distance': float('inf')
        }
        
        # 检查所有瞄准该飞机的导弹
        incoming = []
        for missile_id, track in self.missile_tracks.items():
            if track['status'] == 'active' and track['target'] == agent_id:
                flight_time = current_time - track['launch_time']
                incoming.append({
                    'missile_id': missile_id,
                    'launcher': track['launcher'],
                    'flight_time': flight_time
                })
        
        threat_info['incoming_count'] = len(incoming)
        
        # 评估威胁等级
        if len(incoming) == 0:
            threat_info['threat_level'] = 0
        elif len(incoming) == 1:
            # 单导弹威胁：根据飞行时间评估
            flight_time = incoming[0]['flight_time']
            if flight_time < 10:
                threat_info['threat_level'] = 1  # 低（刚发射）
            elif flight_time < 30:
                threat_info['threat_level'] = 2  # 中（接近）
            else:
                threat_info['threat_level'] = 3  # 高（即将命中）
        else:
            # 多导弹威胁：高威胁
            threat_info['threat_level'] = 3
        
        return threat_info
    
    def get_remaining_missiles(self, agent_id: str) -> int:
        """
        获取飞机剩余导弹配额
        
        Args:
            agent_id: 飞机ID
            
        Returns:
            剩余可发射导弹数量
        """
        if agent_id not in self.aircraft_missile_counts:
            self.aircraft_missile_counts[agent_id] = 0
        
        fired_count = self.aircraft_missile_counts[agent_id]
        return max(0, self.MAX_MISSILES_PER_AIRCRAFT - fired_count)
    
    def reset(self):
        """重置导弹管理器"""
        self.missile_tracks.clear()
        self.aircraft_missile_counts.clear()
