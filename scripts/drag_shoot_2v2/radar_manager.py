"""
拖曳射击项目雷达管理系统
保持现有功能的基础性，我方雷达为基础状态转换器，敌方雷达具有更多功能以贴合实际
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from enum import Enum

# 导入JSBSim catalog
try:
    from envs.JSBSim.core.catalog import Catalog as c
except ImportError:
    # 如果导入失败，创建一个基础的catalog类
    class c:
        attitude_psi_rad = "attitude/psi-rad"
        position_h_sl_m = "position/h-sl-m"


class RadarStatus(Enum):
    """雷达状态枚举"""
    SEARCH = "SEARCH"      # 搜索模式
    TRACK = "TRACK"        # 跟踪模式
    LOCK = "LOCK"          # 锁定模式


class RadarManager:
    """雷达管理器 - 保持现有功能的基础性"""
    
    def __init__(self):
        """初始化雷达管理器"""
        # 我方雷达状态 - 基础的状态转换器
        self.friendly_radar_states = {
            "A0100": RadarStatus.SEARCH,
            "A0200": RadarStatus.SEARCH
        }
        
        # 敌方雷达状态
        self.enemy_radar_states = {
            "B0100": RadarStatus.SEARCH,
            "B0200": RadarStatus.SEARCH
        }
        
        # 敌方雷达数据 - 功能更多，贴合实际型号
        self.enemy_radar_data = {
            "B0100": self._get_default_enemy_radar_data(),
            "B0200": self._get_default_enemy_radar_data()
        }
        
        logging.info("📡 雷达管理器初始化完成")
    
    def _get_default_enemy_radar_data(self) -> Dict[str, Any]:
        """获取默认敌方雷达数据"""
        return {
            "snr": 0.0,
            "doppler_shift": 0.0,
            "lock_quality": 0.0,
            "beam_angle": 0.0
        }
    
    def update_friendly_radar_states(self, env, current_time: float):
        """更新我方雷达状态 - 基础的状态转换器"""
        for agent_id in self.friendly_radar_states.keys():
            if agent_id in env.agents and env.agents[agent_id].is_alive:
                self._update_single_friendly_radar(env, agent_id, current_time)
    
    def update_enemy_radar_states(self, env, current_time: float):
        """更新敌方雷达状态 - 功能更多"""
        for agent_id in self.enemy_radar_states.keys():
            if agent_id in env.agents and env.agents[agent_id].is_alive:
                self._update_single_enemy_radar(env, agent_id, current_time)
    
    def _update_single_friendly_radar(self, env, agent_id: str, current_time: float):
        """更新单个我方雷达状态 - 保持原有的基础逻辑"""
        # 找到最近的敌机
        target = self._find_target(env, agent_id)
        if not target:
            self.friendly_radar_states[agent_id] = RadarStatus.SEARCH
            return

        distance = self._calculate_distance(env.agents[agent_id], target)

        # 根据距离确定雷达状态 - 保持原有的基础逻辑
        if distance > 90000:  # 90km
            self.friendly_radar_states[agent_id] = RadarStatus.SEARCH
        elif distance > 81000:  # 81km
            self.friendly_radar_states[agent_id] = RadarStatus.SEARCH
        elif distance > 45000:  # 45km
            self.friendly_radar_states[agent_id] = RadarStatus.TRACK
        else:  # < 45km
            self.friendly_radar_states[agent_id] = RadarStatus.LOCK
    
    def _update_single_enemy_radar(self, env, agent_id: str, current_time: float):
        """更新单个敌方雷达状态 - 功能更多，贴合实际型号"""
        # 找到目标
        target = self._find_target(env, agent_id)
        if not target:
            self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
            self._reset_enemy_radar_data(agent_id)
            return

        distance = self._calculate_distance(env.agents[agent_id], target)

        # 根据距离确定雷达状态
        if distance > 90000:  # 90km
            self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
        elif distance > 81000:  # 81km
            self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
        elif distance > 45000:  # 45km
            self.enemy_radar_states[agent_id] = RadarStatus.TRACK
        else:  # < 45km
            self.enemy_radar_states[agent_id] = RadarStatus.LOCK

        # 更新敌方雷达数据 - 贴合实际N001VE雷达
        self._update_enemy_radar_data(env, agent_id, target, distance, current_time)
    
    def _find_target(self, env, agent_id: str):
        """找到目标"""
        if not hasattr(env, 'agents') or agent_id not in env.agents:
            return None
        
        agent = env.agents[agent_id]
        enemies = agent.enemies
        
        if not enemies or not isinstance(enemies, (list, tuple)) or len(enemies) == 0:
            return None
        
        # 返回第一个存活的敌人
        for enemy in enemies:
            if enemy and enemy.is_alive:
                return enemy
        
        return None
    
    def _calculate_distance(self, aircraft1, aircraft2) -> float:
        """计算两架飞机之间的距离"""
        try:
            pos1 = aircraft1.get_position()
            pos2 = aircraft2.get_position()
            return np.linalg.norm(pos1 - pos2)
        except Exception as e:
            logging.error(f"计算距离时出错: {e}")
            return np.inf
    
    def _reset_enemy_radar_data(self, agent_id: str):
        """重置敌方雷达数据"""
        if agent_id in self.enemy_radar_data:
            self.enemy_radar_data[agent_id] = self._get_default_enemy_radar_data()
    
    def _update_enemy_radar_data(self, env, agent_id: str, target, distance: float, current_time: float):
        """更新敌方雷达数据 - 贴合实际N001VE雷达"""
        try:
            # 计算SNR - 基于实际雷达方程
            snr = self._calculate_enemy_snr(distance)
            
            # 计算多普勒频移
            doppler_shift = self._calculate_enemy_doppler_shift(env, agent_id)
            
            # 计算锁定质量 - 基于距离和状态
            lock_quality = self._calculate_enemy_lock_quality(distance, self.enemy_radar_states[agent_id])
            
            # 计算波束角度
            beam_angle = self._calculate_enemy_beam_angle(env, agent_id, target)
            
            # 更新雷达数据
            self.enemy_radar_data[agent_id] = {
                "snr": snr,
                "doppler_shift": doppler_shift,
                "lock_quality": lock_quality,
                "beam_angle": beam_angle
            }
            
        except Exception as e:
            logging.error(f"更新敌方雷达数据时出错 {agent_id}: {e}")
            self._reset_enemy_radar_data(agent_id)
    
    def _calculate_enemy_snr(self, distance: float) -> float:
        """计算敌方雷达SNR - 贴合N001VE雷达"""
        if distance <= 0:
            return -np.inf
        
        # N001VE雷达基础参数
        base_snr = 42.0  # 基础SNR (dB) - N001VE略低于AN/APG-68
        # 距离衰减 (R^4 law)
        distance_attenuation = -40 * np.log10(distance / 10000)
        # 大气衰减
        atmospheric_attenuation = -2 * np.log10(max(distance / 10000, 0.1))
        
        total_snr = base_snr + distance_attenuation + atmospheric_attenuation
        return total_snr
    
    def _calculate_enemy_doppler_shift(self, env, agent_id: str) -> float:
        """计算敌方雷达多普勒频移"""
        try:
            if not hasattr(env, 'agents') or agent_id not in env.agents:
                return 0.0
            
            agent = env.agents[agent_id]
            enemies = agent.enemies
            
            if not enemies or not isinstance(enemies, (list, tuple)) or len(enemies) == 0:
                return 0.0
            
            enemy = enemies[0]
            if not enemy or not enemy.is_alive:
                return 0.0
            
            # 计算径向速度 - 添加错误处理
            try:
                ego_vel = agent.get_velocity()
                enemy_vel = enemy.get_velocity()
                relative_pos = enemy.get_position() - agent.get_position()
            except Exception as e:
                logging.error(f"获取速度或位置失败 {agent_id}: {e}")
                return 0.0
            
            if np.linalg.norm(relative_pos) == 0:
                return 0.0
            
            # 径向速度分量
            relative_vel = enemy_vel - ego_vel
            unit_los = relative_pos / np.linalg.norm(relative_pos)
            radial_velocity = np.dot(relative_vel, unit_los)
            
            return radial_velocity
            
        except Exception as e:
            logging.error(f"计算敌方多普勒频移时出错 {agent_id}: {e}")
            return 0.0
    
    def _calculate_enemy_lock_quality(self, distance: float, radar_status: RadarStatus) -> float:
        """计算敌方雷达锁定质量"""
        # 基于距离和状态计算锁定质量
        if radar_status == RadarStatus.LOCK:
            # 距离越近，锁定质量越高
            if distance < 20000:  # 20km内
                return 0.9
            elif distance < 35000:  # 35km内
                return 0.7
            else:  # 35-45km
                return 0.5
        elif radar_status == RadarStatus.TRACK:
            # 跟踪模式下锁定质量较低
            return 0.3
        else:  # SEARCH
            return 0.0
    
    def _calculate_enemy_beam_angle(self, env, agent_id: str, target) -> float:
        """计算敌方雷达波束角度"""
        try:
            if not target:
                return 0.0
            
            agent = env.agents[agent_id]
            agent_pos = agent.get_position()
            target_pos = target.get_position()
            
            # 计算相对位置向量
            relative_pos = target_pos - agent_pos
            
            # 获取飞机航向 - 添加详细调试信息
            try:
                agent_heading = agent.get_property_value(c.attitude_psi_rad)
                logging.debug(f"获取航向值 {agent_id}: {agent_heading} (类型: {type(agent_heading)})")
            except Exception as e:
                logging.error(f"获取航向值失败 {agent_id}: {e}")
                return 0.0
            
            # 确保航向是数值类型
            if not isinstance(agent_heading, (int, float)):
                logging.warning(f"航向值不是数值类型 {agent_id}: {type(agent_heading)} = {agent_heading}")
                # 尝试转换为数值
                try:
                    agent_heading = float(agent_heading)
                    logging.info(f"成功转换航向值为数值 {agent_id}: {agent_heading}")
                except (ValueError, TypeError):
                    logging.error(f"无法转换航向值为数值 {agent_id}: {agent_heading}")
                    return 0.0
            
            # 计算目标相对于飞机的角度
            target_angle = np.arctan2(relative_pos[1], relative_pos[0])
            
            # 计算角度差
            angle_diff = target_angle - agent_heading
            
            # 归一化到[-π, π]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            return angle_diff
            
        except Exception as e:
            logging.error(f"计算敌方波束角度时出错 {agent_id}: {e}")
            return 0.0
    
    def get_friendly_radar_states(self) -> Dict[str, str]:
        """获取我方雷达状态"""
        return {k: v.value for k, v in self.friendly_radar_states.items()}
    
    def get_enemy_radar_states(self) -> Dict[str, str]:
        """获取敌方雷达状态"""
        return {k: v.value for k, v in self.enemy_radar_states.items()}
    
    def get_enemy_radar_data(self) -> Dict[str, Dict[str, Any]]:
        """获取敌方雷达数据"""
        return self.enemy_radar_data
    
    def record_radar_data(self, env, current_time: float) -> List[Dict[str, Any]]:
        """记录雷达数据到CSV格式"""
        radar_data = []
        
        # 记录我方雷达数据 - 基础状态
        for agent_id, radar_state in self.friendly_radar_states.items():
            if agent_id in env._jsbsims and env._jsbsims[agent_id].is_alive:
                # 找到目标
                target_id = self._find_target_id(env, agent_id)
                target_distance = self._get_target_distance(env, agent_id)
                
                if target_id:
                    radar_data.append({
                        'Time_s': current_time,
                        'Agent_ID': agent_id,
                        'Radar_Type': 'AN/APG-68(V)9',
                        'Status': radar_state.value,
                        'Target_ID': target_id,
                        'Target_Distance_km': target_distance / 1000.0
                    })
        
        # 记录敌方雷达数据 - 详细数据
        for agent_id, radar_state in self.enemy_radar_states.items():
            if agent_id in env._jsbsims and env._jsbsims[agent_id].is_alive:
                # 找到目标
                target_id = self._find_target_id(env, agent_id)
                target_distance = self._get_target_distance(env, agent_id)
                
                if target_id:
                    radar_info = self.enemy_radar_data.get(agent_id, {})
                    radar_data.append({
                        'Time_s': current_time,
                        'Agent_ID': agent_id,
                        'Radar_Type': 'N001VE',  # SU-27雷达型号
                        'Status': radar_state.value,
                        'Target_ID': target_id,
                        'Target_Distance_km': target_distance / 1000.0,
                        'SNR_dB': radar_info.get('snr', 0.0),
                        'Doppler_Shift_m_s': radar_info.get('doppler_shift', 0.0),
                        'Lock_Quality': radar_info.get('lock_quality', 0.0),
                        'Beam_Angle_deg': np.rad2deg(radar_info.get('beam_angle', 0.0)),
                        'Side': 'Enemy'
                    })
        
        return radar_data
    
    def _find_target_id(self, env, agent_id: str) -> Optional[str]:
        """找到目标ID"""
        for enemy_id, enemy in env._jsbsims.items():
            if ((agent_id.startswith('A') and enemy_id.startswith('B')) or
                (agent_id.startswith('B') and enemy_id.startswith('A'))) and enemy.is_alive:
                return enemy_id
        return None
    
    def _get_target_distance(self, env, agent_id: str) -> float:
        """获取目标距离"""
        target_id = self._find_target_id(env, agent_id)
        if target_id and agent_id in env._jsbsims and target_id in env._jsbsims:
            pos1 = env._jsbsims[agent_id].get_position()
            pos2 = env._jsbsims[target_id].get_position()
            return np.linalg.norm(pos1 - pos2)
        return 0.0


# 全局雷达管理器实例
_radar_manager = None

def get_radar_manager() -> RadarManager:
    """获取全局雷达管理器实例"""
    global _radar_manager
    if _radar_manager is None:
        _radar_manager = RadarManager()
    return _radar_manager

def update_friendly_radars(env, current_time: float):
    """更新我方雷达状态"""
    radar_manager = get_radar_manager()
    radar_manager.update_friendly_radar_states(env, current_time)

def update_enemy_radars(env, current_time: float):
    """更新敌方雷达状态"""
    radar_manager = get_radar_manager()
    radar_manager.update_enemy_radar_states(env, current_time)

def update_all_radars(env, current_time: float):
    """更新所有雷达状态"""
    update_friendly_radars(env, current_time)
    update_enemy_radars(env, current_time)

def record_radar_data(env, current_time: float) -> List[Dict[str, Any]]:
    """记录雷达数据"""
    radar_manager = get_radar_manager()
    return radar_manager.record_radar_data(env, current_time)

def get_friendly_radar_states() -> Dict[str, str]:
    """获取我方雷达状态"""
    radar_manager = get_radar_manager()
    return radar_manager.get_friendly_radar_states()

def get_enemy_radar_states() -> Dict[str, str]:
    """获取敌方雷达状态"""
    radar_manager = get_radar_manager()
    return radar_manager.get_enemy_radar_states()

def get_enemy_radar_data() -> Dict[str, Dict[str, Any]]:
    """获取敌方雷达数据"""
    radar_manager = get_radar_manager()
    return radar_manager.get_enemy_radar_data() 