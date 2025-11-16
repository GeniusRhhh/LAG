"""
导弹管理模块
管理导弹发射、追踪、状态更新等功能
从tactical_task.py中提取，提高代码可维护性
"""
import logging
import numpy as np
from envs.JSBSim.core.catalog import Catalog as c
from tactical_types import TacticalPhase, get_target_with_fallback


class MissileManager:
    """导弹管理器 - 负责导弹相关操作"""
    
    def __init__(self, state_manager):
        """
        Args:
            state_manager: TacticalStateManager实例
        """
        self.state_manager = state_manager
        self.missile_tracks = {}  # 导弹追踪信息
    
    def should_launch_missile(self, env, agent_id: str, current_phase, target_id: str = None) -> bool:
        """
        判断是否应该发射导弹
        
        Args:
            env: 环境
            agent_id: 飞机ID
            current_phase: 当前战术阶段
            target_id: 目标ID
            
        Returns:
            True表示应该发射
        """
        
        # 检查是否已发射
        if self.state_manager.missile_launched.get(agent_id, False):
            return False
        
        # 检查剩余导弹
        current_time = env.current_step * env.time_interval
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
        
        # LR阶段发射条件（约78km）
        if current_phase == TacticalPhase.LR_TR:
            if 75000 <= distance <= 82000:  # 75-82km范围
                logging.info(f"🎯 [{agent_id}] LR阶段发射条件满足: 距离{distance/1000:.1f}km")
                return True
        
        # TR阶段补充发射（约40-60km）
        elif current_phase == TacticalPhase.TR_DOR:
            if 35000 <= distance <= 65000:  # 35-65km范围
                logging.info(f"🎯 [{agent_id}] TR阶段补充发射: 距离{distance/1000:.1f}km")
                return True
        
        return False
    
    def execute_missile_launch(self, env, agent_id: str, target_id: str = None) -> bool:
        """
        执行导弹发射
        
        Args:
            env: 环境
            agent_id: 飞机ID
            target_id: 目标ID
            
        Returns:
            True表示发射成功
        """
        
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
        
        # 执行发射
        current_time = env.current_step * env.time_interval
        
        try:
            # 这里应该调用环境的导弹发射接口
            # env.launch_missile(agent_id, target_id)
            # 由于我们只是记录状态，暂时只记录发射信息
            
            self.state_manager.record_missile_launch(agent_id, current_time, target_id)
            
            # 记录导弹轨迹追踪
            missile_id = f"{agent_id}_missile_{len(self.missile_tracks)}"
            self.missile_tracks[missile_id] = {
                'launcher': agent_id,
                'target': target_id,
                'launch_time': current_time,
                'status': 'active'
            }
            
            logging.info(f"🚀 [{agent_id}] 成功发射导弹 → {target_id} (t={current_time:.1f}s)")
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
    
    def reset(self):
        """重置导弹管理器"""
        self.missile_tracks.clear()
