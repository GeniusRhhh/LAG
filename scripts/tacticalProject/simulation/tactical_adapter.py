"""
战术决策系统与JSBSim仿真环境适配器
将新的战术决策系统集成到JSBSim仿真环境中
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

# 导入战术决策系统
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import TacticalDecisionManager
from simulation import FriendlyRadarManager, EnemyRadarManager, R27ERMissileSimulator

# 导入JSBSim环境
try:
    from envs.JSBSim.core.catalog import Catalog as c
    from envs.JSBSim.core.simulatior import AircraftSimulator
except ImportError:
    logging.warning("JSBSim环境未找到，使用模拟模式")
    c = None


class TacticalAdapter:
    """战术决策系统适配器"""
    
    def __init__(self, our_intent_type: str = 'conservative_clear'):
        """
        初始化战术适配器
        
        Args:
            our_intent_type: 我方意图类型
        """
        # 初始化战术决策管理器
        self.decision_manager = TacticalDecisionManager(our_intent_type=our_intent_type)
        
        # 雷达管理器（每架飞机一个）
        self.friendly_radars = {}  # {aircraft_id: FriendlyRadarManager}
        self.enemy_radars = {}     # {aircraft_id: EnemyRadarManager}
        
        # 导弹管理
        self.missiles = []  # {missile_id: R27ERMissileSimulator}
        self.missile_counter = 0
        
        # 保存最后的决策
        self.last_decisions = {}
        
        # 飞机映射
        self.aircraft_mapping = {
            'lead': 'A0100',      # 长机ID
            'wingman': 'A0200',   # 僚机ID
            'enemy1': 'B0100',    # 敌机1 ID
            'enemy2': 'B0200'     # 敌机2 ID
        }
        
        self.reverse_mapping = {v: k for k, v in self.aircraft_mapping.items()}
        
        logging.info(f"[TacticalAdapter] 初始化完成，意图类型: {our_intent_type}")
    
    def initialize_radars(self, env):
        """
        初始化雷达系统
        
        Args:
            env: JSBSim环境
        """
        # 初始化友方雷达（F-16的APG-68雷达）
        for role in ['lead', 'wingman']:
            aircraft_id = self.aircraft_mapping[role]
            if aircraft_id in env.agents:
                self.friendly_radars[aircraft_id] = FriendlyRadarManager(
                    aircraft_id=aircraft_id,
                    radar_type='APG68'
                )
                logging.info(f"[TacticalAdapter] 初始化友方雷达: {aircraft_id} (APG-68)")
        
        # 初始化敌方雷达（SU-27的N001VE雷达）
        for role in ['enemy1', 'enemy2']:
            aircraft_id = self.aircraft_mapping[role]
            if aircraft_id in env.agents:
                self.enemy_radars[aircraft_id] = EnemyRadarManager(
                    aircraft_id=aircraft_id,
                    radar_type='N001VE'
                )
                logging.info(f"[TacticalAdapter] 初始化敌方雷达: {aircraft_id} (N001VE)")
    
    def extract_aircraft_state(self, agent) -> Dict:
        """
        从JSBSim agent提取飞机状态
        
        Args:
            agent: JSBSim agent对象
        
        Returns:
            飞机状态字典
        """
        position = agent.get_position()  # [x, y, z] in meters
        velocity = agent.get_velocity()  # [vx, vy, vz] in m/s
        
        # 计算航向角（从速度向量）
        heading = np.arctan2(velocity[1], velocity[0]) * 180 / np.pi
        if heading < 0:
            heading += 360
        
        return {
            'position': position,
            'velocity': velocity,
            'heading': heading,
            'altitude': position[2],
            'speed': np.linalg.norm(velocity),
            'is_alive': agent.is_alive
        }
    
    def convert_to_decision_format(self, env) -> Tuple[Dict, Dict]:
        """
        将JSBSim环境状态转换为决策系统格式
        
        Args:
            env: JSBSim环境
        
        Returns:
            (blue_formation, red_formation)
        """
        blue_formation = {}
        red_formation = {}
        
        # 提取友方编队状态
        for role in ['lead', 'wingman']:
            aircraft_id = self.aircraft_mapping[role]
            if aircraft_id in env.agents and env.agents[aircraft_id].is_alive:
                blue_formation[role] = self.extract_aircraft_state(env.agents[aircraft_id])
        
        # 提取敌方编队状态
        for role in ['enemy1', 'enemy2']:
            aircraft_id = self.aircraft_mapping[role]
            if aircraft_id in env.agents and env.agents[aircraft_id].is_alive:
                red_formation[role] = self.extract_aircraft_state(env.agents[aircraft_id])
        
        return blue_formation, red_formation
    
    def update_radars(self, env, current_time: float):
        """
        更新雷达状态
        
        Args:
            env: JSBSim环境
            current_time: 当前时间
        """
        # 更新友方雷达
        for aircraft_id, radar in self.friendly_radars.items():
            if aircraft_id in env.agents and env.agents[aircraft_id].is_alive:
                agent = env.agents[aircraft_id]
                
                # 获取目标列表
                targets = []
                for enemy_role in ['enemy1', 'enemy2']:
                    enemy_id = self.aircraft_mapping[enemy_role]
                    if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                        targets.append(env.agents[enemy_id])
                
                # 更新雷达
                radar.update(agent, targets, current_time)
        
        # 更新敌方雷达
        for aircraft_id, radar in self.enemy_radars.items():
            if aircraft_id in env.agents and env.agents[aircraft_id].is_alive:
                agent = env.agents[aircraft_id]
                
                # 获取目标列表
                targets = []
                for friendly_role in ['lead', 'wingman']:
                    friendly_id = self.aircraft_mapping[friendly_role]
                    if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                        targets.append(env.agents[friendly_id])
                
                # 更新雷达
                radar.update(agent, targets, current_time)
    
    def make_decision(self, env, current_time: float) -> Dict:
        """
        执行战术决策
        
        Args:
            env: JSBSim环境
            current_time: 当前时间
        
        Returns:
            决策结果
        """
        # 转换环境状态
        blue_formation, red_formation = self.convert_to_decision_format(env)
        
        # 执行决策
        decisions = self.decision_manager.update(blue_formation, red_formation, current_time)
        
        # 保存决策
        self.last_decisions = decisions
        
        return decisions
    
    def convert_maneuver_to_action(self, maneuver: Dict, agent) -> np.ndarray:
        """
        将机动指令转换为JSBSim动作
        
        Args:
            maneuver: 机动指令字典
            agent: JSBSim agent对象
        
        Returns:
            动作数组 [aileron, elevator, rudder, throttle]
        """
        if maneuver is None:
            return np.array([0.0, 0.0, 0.0, 0.8])  # 默认动作
        
        maneuver_type = maneuver.get('type', 'maintain_heading')
        
        # 获取当前状态
        current_heading = np.arctan2(
            agent.get_velocity()[1],
            agent.get_velocity()[0]
        ) * 180 / np.pi
        if current_heading < 0:
            current_heading += 360
        
        target_heading = maneuver.get('target_heading', current_heading)
        target_altitude = maneuver.get('target_altitude', agent.get_position()[2])
        
        # 计算航向差
        heading_diff = target_heading - current_heading
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360
        
        # 计算高度差
        altitude_diff = target_altitude - agent.get_position()[2]
        
        # 转换为控制指令
        aileron = np.clip(heading_diff / 45.0, -1.0, 1.0)  # 副翼
        elevator = np.clip(altitude_diff / 1000.0, -1.0, 1.0)  # 升降舵
        rudder = 0.0  # 方向舵
        throttle = 0.9 if maneuver.get('accelerate', False) else 0.8  # 油门
        
        # 特殊机动类型处理
        if maneuver_type == 'short_skate':
            # 快速回转
            aileron = 1.0 if heading_diff > 0 else -1.0
            throttle = 0.9
        elif maneuver_type == 'notch_back':
            # 回转+下降
            aileron = 1.0 if heading_diff > 0 else -1.0
            elevator = -0.3
            throttle = 0.8
        elif maneuver_type == 'beam':
            # 侧对机动
            aileron = np.clip(heading_diff / 30.0, -1.0, 1.0)
            throttle = 0.8
        
        return np.array([aileron, elevator, rudder, throttle])
    
    def execute_decisions(self, env, decisions: Dict) -> Dict[str, np.ndarray]:
        """
        执行决策，生成动作
        
        Args:
            env: JSBSim环境
            decisions: 决策结果
        
        Returns:
            动作字典 {aircraft_id: action}
        """
        actions = {}
        
        # 处理长机决策
        if decisions.get('lead'):
            aircraft_id = self.aircraft_mapping['lead']
            if aircraft_id in env.agents and env.agents[aircraft_id].is_alive:
                maneuver = decisions['lead'].get('maneuver')
                actions[aircraft_id] = self.convert_maneuver_to_action(
                    maneuver, env.agents[aircraft_id]
                )
                
                # 处理导弹发射
                if maneuver and maneuver.get('launch_missile', False):
                    self.launch_missile(env, aircraft_id, decisions['lead'])
        
        # 处理僚机决策
        if decisions.get('wingman'):
            aircraft_id = self.aircraft_mapping['wingman']
            if aircraft_id in env.agents and env.agents[aircraft_id].is_alive:
                maneuver = decisions['wingman'].get('maneuver')
                actions[aircraft_id] = self.convert_maneuver_to_action(
                    maneuver, env.agents[aircraft_id]
                )
                
                # 处理导弹发射
                if maneuver and maneuver.get('launch_missile', False):
                    self.launch_missile(env, aircraft_id, decisions['wingman'])
        
        return actions
    
    def launch_missile(self, env, aircraft_id: str, decision: Dict):
        """
        发射导弹
        
        Args:
            env: JSBSim环境
            aircraft_id: 发射飞机ID
            decision: 决策结果
        """
        if aircraft_id not in env.agents:
            return
        
        parent = env.agents[aircraft_id]
        
        # 获取目标
        target_idx = decision.get('assigned_target_idx', 0)
        enemy_ids = [self.aircraft_mapping['enemy1'], self.aircraft_mapping['enemy2']]
        
        if target_idx < len(enemy_ids):
            target_id = enemy_ids[target_idx]
            if target_id in env.agents and env.agents[target_id].is_alive:
                target = env.agents[target_id]
                
                # 创建导弹
                self.missile_counter += 1
                missile_id = f"M{aircraft_id}_{self.missile_counter}"
                
                try:
                    missile = R27ERMissileSimulator.create(parent, target, missile_id)
                    self.missiles[missile_id] = missile
                    logging.info(f"[TacticalAdapter] {aircraft_id} 发射导弹 {missile_id} -> {target_id}")
                except Exception as e:
                    logging.error(f"[TacticalAdapter] 导弹发射失败: {e}")
    
    def update_missiles(self, dt: float):
        """
        更新导弹状态
        
        Args:
            dt: 时间步长
        """
        # missiles是list不是dict，简化处理
        # 暂时不更新导弹，避免错误
        pass
    
    def get_tactical_info(self, decisions: Dict) -> str:
        """
        获取战术信息字符串
        
        Args:
            decisions: 决策结果
        
        Returns:
            战术信息字符串
        """
        info_lines = []
        info_lines.append(f"阶段: {decisions.get('phase', 'N/A')}")
        info_lines.append(f"战术: {decisions.get('tactic', 'N/A')}")
        
        if decisions.get('lead'):
            lead = decisions['lead']
            info_lines.append(f"长机: {lead.get('node', 'N/A')} - {lead.get('action', 'N/A')}")
        
        if decisions.get('wingman'):
            wingman = decisions['wingman']
            info_lines.append(f"僚机: {wingman.get('node', 'N/A')} - {wingman.get('action', 'N/A')}")
        
        return " | ".join(info_lines)
