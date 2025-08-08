#!/usr/bin/env python3
"""
2v2拖曳射击战术任务 - 正确继承MultipleCombatTask架构
"""

import logging
import numpy as np
from enum import Enum
from typing import Dict, Any, Optional
import math

# 导入现有模块
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.core.simulatior import AircraftSimulator, MissileSimulator


class TacticalPhase(Enum):
    """拖曳射击战术阶段"""
    NLT_MELD = "NLT_MELD"    # 90-81km: 非致命时间-中距离交战线
    MELD_MTR = "MELD_MTR"    # 81-45km: 中距离交战线-导弹目标范围
    MTR_TR = "MTR_TR"        # 45-41km: 导弹目标范围-目标范围
    TR_DOR = "TR_DOR"        # 41-19.6km: 目标范围-动态攻击范围
    DOR_DR = "DOR_DR"        # 19.6-14.5km: 动态攻击范围-防御范围


class DragShootTacticalTask(MultipleCombatTask):
    """
    2v2拖曳射击战术任务
    正确继承MultipleCombatTask，扩展拖曳射击特定功能
    """
    
    def __init__(self, config):
        """初始化拖曳射击任务"""
        # 调用父类初始化 - 这是关键！
        super().__init__(config)
        
        # 添加父类需要的属性
        self._last_shoot_time = {}
        self._remaining_missiles = {}
        self._shoot_action = {}
        self._last_action = {}
        self._maneuver_history = []
        self._target_allocation = {}
        self.rewards = {}
        self.current_phases = {}
        self.tactical_templates = {}
        
        # 拖曳射击特定配置
        self.current_phase = TacticalPhase.NLT_MELD
        self.phase_start_time = 0.0
        self.phase_start_step = 0
        
        # 战术距离配置
        self.tactical_distances = {
            'NLT_MELD_min': 81000,   # 81km
            'MELD_MTR_min': 45000,   # 45km
            'MTR_TR_min': 41000,     # 41km
            'TR_DOR_min': 19600,     # 19.6km
            'DOR_DR_min': 14500,     # 14.5km
            'leader_launch_range': 45000,   # 长机45km发射
            'wingman_launch_range': 41000,  # 僚机41km发射
            'lock_range': 45000,     # 雷达锁定距离
        }
        
        # 导弹发射状态
        self.missile_launched = {
            "A0100": False,
            "A0200": False,
            "B0100": False,
            "B0200": False
        }
        
        # 数据记录
        self.trajectory_data = []
        self.radar_data = []
        self.missile_data = []
        
        # 任务时间线
        self.mission_timeline = {
            "start_time": 0.0,
            "contact_time": None,
            "engagement_time": None,
            "launch_time": None,
            "end_time": None
        }
        
        logging.info("DragShootTacticalTask initialized with tactical phases")
    
    def reset(self, env):
        """重置任务状态 - 调用父类reset"""
        # 调用父类reset方法 - 这是关键！
        super().reset(env)
        
        # 重置父类需要的属性
        self._last_shoot_time = {}
        self._remaining_missiles = {}
        self._shoot_action = {}
        self._last_action = {}
        self._maneuver_history = []
        self._target_allocation = {}
        self.rewards = {}
        self.current_phases = {agent_id: "contact_guidance" for agent_id in env.agents.keys()}
        self.tactical_templates = {}
        
        # 重置拖曳射击特定状态
        self.current_phase = TacticalPhase.NLT_MELD
        self.phase_start_time = 0.0
        self.phase_start_step = 0
        
        # 重置导弹发射状态
        for agent_id in env.agents.keys():
            self.missile_launched[agent_id] = False
        
        # 重置数据记录
        self.trajectory_data.clear()
        self.radar_data.clear()
        self.missile_data.clear()
        
        # 重置任务时间线
        self.mission_timeline = {
            "start_time": 0.0,
            "contact_time": None,
            "engagement_time": None,
            "launch_time": None,
            "end_time": None
        }
        
        logging.info(f"DragShootTacticalTask reset at step {env.current_step}")
    
    def step(self, env):
        """执行一步仿真 - 正确调用父类step"""
        current_time = env.current_step * env.time_interval
        
        # 更新战术阶段
        self._update_tactical_phase(env, current_time)
        
        # 记录数据
        self._record_data(env, current_time)
        
        # 更新任务时间线
        self._update_mission_timeline(env, current_time)
        
        # 调用父类step方法 - 这是关键！
        # 父类会处理所有标准的多智能体空战逻辑
        obs, share_obs, rewards, dones, infos = super().step(env)
        
        # 在父类处理后，添加拖曳射击特定的行为
        self._execute_tactical_behaviors(env, current_time)
        
        # 添加拖曳射击特定信息到infos
        for agent_id in env.agents.keys():
            if agent_id in infos:
                infos[agent_id].update({
                    "tactical_phase": self.current_phase.value,
                    "missiles_launched": self.missile_launched[agent_id],
                    "phase_time": current_time - self.phase_start_time
                })
        
        return obs, share_obs, rewards, dones, infos
    
    def _update_tactical_phase(self, env, current_time: float):
        """更新战术阶段"""
        new_phase = self._get_current_phase(env)
        
        if new_phase != self.current_phase:
            # 计算距离用于日志
            leader_red = env.agents.get("A0100")
            leader_blue = env.agents.get("B0100")
            distance = 0.0
            if leader_red and leader_blue and leader_red.is_alive and leader_blue.is_alive:
                distance = self._calculate_distance(leader_red, leader_blue)
            
            logging.info(f"Phase transition: {self.current_phase.value} -> {new_phase.value} "
                        f"at t={current_time:.1f}s, distance={distance/1000:.1f}km")
            
            self.current_phase = new_phase
            self.phase_start_time = current_time
            self.phase_start_step = env.current_step
    
    def _get_current_phase(self, env) -> TacticalPhase:
        """根据当前距离确定战术阶段"""
        # 获取主要对抗双方的距离
        leader_red = env.agents.get("A0100")
        leader_blue = env.agents.get("B0100")
        
        if not leader_red or not leader_blue or not leader_red.is_alive or not leader_blue.is_alive:
            return self.current_phase
        
        distance = self._calculate_distance(leader_red, leader_blue)
        
        # 根据距离确定阶段 (距离递减)
        if distance > self.tactical_distances.get('NLT_MELD_min', 81000):
            return TacticalPhase.NLT_MELD
        elif distance > self.tactical_distances.get('MELD_MTR_min', 45000):
            return TacticalPhase.MELD_MTR
        elif distance > self.tactical_distances.get('MTR_TR_min', 41000):
            return TacticalPhase.MTR_TR
        elif distance > self.tactical_distances.get('TR_DOR_min', 19600):
            return TacticalPhase.TR_DOR
        else:
            return TacticalPhase.DOR_DR
    
    def _calculate_distance(self, aircraft1: AircraftSimulator, aircraft2: AircraftSimulator) -> float:
        """计算两架飞机之间的距离"""
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _execute_tactical_behaviors(self, env, current_time: float):
        """执行拖曳射击特定的战术行为"""
        # 这里只处理拖曳射击特定的逻辑
        # 基础的飞机控制由父类MultipleCombatTask处理
        
        for agent_id, aircraft in env.agents.items():
            if not aircraft.is_alive:
                continue
            
            # 处理导弹发射逻辑
            self._handle_missile_launch(env, agent_id, current_time)
    
    def _handle_missile_launch(self, env, agent_id: str, current_time: float):
        """处理导弹发射逻辑"""
        if self.missile_launched[agent_id]:
            return  # 已经发射过导弹
        
        aircraft = env.agents[agent_id]
        
        # 检查导弹数量
        if not hasattr(aircraft, 'num_left_missiles') or aircraft.num_left_missiles <= 0:
            return  # 没有导弹
        
        # 只有己方在特定阶段才发射导弹
        if not agent_id.startswith('A') or self.current_phase != TacticalPhase.MTR_TR:
            return
        
        # 找到最近的敌机
        nearest_enemy = None
        min_distance = float('inf')
        
        for enemy_id, enemy in env.agents.items():
            if enemy_id.startswith('B') and enemy.is_alive:
                distance = self._calculate_distance(aircraft, enemy)
                if distance < min_distance:
                    min_distance = distance
                    nearest_enemy = enemy
        
        if not nearest_enemy:
            return
        
        # 根据角色确定发射距离
        should_launch = False
        if agent_id == "A0100":  # 长机45km发射
            launch_range = self.tactical_distances.get('leader_launch_range', 45000)
            should_launch = (launch_range - 1000 < min_distance <= launch_range)
        elif agent_id == "A0200":  # 僚机41km发射
            launch_range = self.tactical_distances.get('wingman_launch_range', 41000)
            should_launch = (launch_range - 1000 < min_distance <= launch_range)
        
        if should_launch:
            try:
                # 使用现有的导弹发射机制
                new_missile_uid = f"{agent_id}{aircraft.num_left_missiles}"
                
                # 创建导弹
                missile = MissileSimulator.create(
                    parent=aircraft,
                    target=nearest_enemy,
                    uid=new_missile_uid
                )
                
                # 添加到环境
                env.add_temp_simulator(missile)
                
                # 减少导弹数量
                aircraft.num_left_missiles -= 1
                
                self.missile_launched[agent_id] = True
                logging.info(f"t={current_time:.1f}s: {agent_id} launches missile {new_missile_uid} at {nearest_enemy.uid}, "
                           f"distance={min_distance/1000:.1f}km, remaining={aircraft.num_left_missiles}")
                
                # 更新任务时间线
                if self.mission_timeline["launch_time"] is None:
                    self.mission_timeline["launch_time"] = current_time
                    
            except Exception as e:
                logging.warning(f"Failed to launch missile from {agent_id}: {e}")
    
    def _record_data(self, env, current_time: float):
        """记录仿真数据"""
        # 记录轨迹数据
        for agent_id, aircraft in env.agents.items():
            if not aircraft.is_alive:
                continue
                
            pos = aircraft.get_position()
            velocity = aircraft.get_velocity()
            heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
            pitch = np.rad2deg(aircraft.get_property_value(c.attitude_pitch_rad))
            trajectory_record = {
                'time_s': current_time,
                'agent_id': agent_id,
                'aircraft_type': getattr(aircraft, 'model', 'F16').upper(),
                'x_m': pos[0],
                'y_m': pos[1], 
                'z_m': pos[2],
                'heading_deg': heading,
                'pitch_deg': pitch,
                'velocity_m_s': np.linalg.norm(velocity),
                'is_alive': aircraft.is_alive,
                'missiles_remaining': aircraft.num_left_missiles
            }
            self.trajectory_data.append(trajectory_record)
        
        # 记录导弹数据
        for missile_uid, missile in env._tempsims.items():
            if not missile.is_alive:
                continue
                
            pos = missile.get_position()
            velocity = missile.get_velocity()
            
            missile_record = {
                'time_s': current_time,
                'missile_id': missile.uid,
                'launcher_id': getattr(missile, 'parent_uid', None),
                'missile_type': getattr(missile, 'model', 'MISSILE').upper(),
                'status': self._get_missile_status(missile),
                'x_m': pos[0],
                'y_m': pos[1],
                'z_m': pos[2],
                'velocity_m_s': np.linalg.norm(velocity),
                'target_id': getattr(missile, 'target_uid', None),
                'is_active': missile.is_alive
            }
            self.missile_data.append(missile_record)
    
    def _get_missile_status(self, missile) -> str:
        """获取导弹状态"""
        if not missile.is_alive:
            if getattr(missile, 'is_success', False):
                return "HIT"
            else:
                return "MISS"
        
        # 根据导弹的飞行阶段确定状态
        if hasattr(missile, 'stage'):
            stage = missile.stage
            if stage == 0:
                return "LAUNCHED"
            elif stage == 1:
                return "MIDCOURSE"
            elif stage == 2:
                return "TERMINAL"
        
        return "LAUNCHED"
    
    def _update_mission_timeline(self, env, current_time: float):
        """更新任务时间线"""
        # 更新接触时间
        if (self.mission_timeline["contact_time"] is None and 
            self.current_phase != TacticalPhase.NLT_MELD):
            self.mission_timeline["contact_time"] = current_time
        
        # 更新交战时间
        if (self.mission_timeline["engagement_time"] is None and 
            self.current_phase in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR, TacticalPhase.DOR_DR]):
            self.mission_timeline["engagement_time"] = current_time

    def get_state_dict(self, env, agent_id):
        """获取智能体的状态字典 - 必需的父类方法"""
        try:
            aircraft = env.agents[agent_id]
            if not aircraft.is_alive:
                return {
                    "enemy_distance": 100000,
                    "enemy_angle_off": 0,
                    "radar_lock": False,
                    "missile_launched": False,
                    "missile_hit": False,
                    "current_altitude": 0,
                    "has_warning": False
                }
            
            # 找到最近的敌机
            nearest_enemy = None
            min_distance = float('inf')
            
            for other_id, other_aircraft in env.agents.items():
                if (other_id != agent_id and 
                    other_aircraft.is_alive and 
                    other_aircraft.color != aircraft.color):
                    
                    distance = self._calculate_distance(aircraft, other_aircraft)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_enemy = other_aircraft
            
            # 计算角度偏移
            enemy_angle_off = 0
            if nearest_enemy:
                # 计算到敌机的角度偏移
                target_vec = nearest_enemy.get_position() - aircraft.get_position()
                heading = aircraft.get_velocity()
                if np.linalg.norm(heading) > 0 and np.linalg.norm(target_vec) > 0:
                    angle = np.rad2deg(np.arccos(np.clip(
                        np.dot(target_vec, heading) / (np.linalg.norm(target_vec) * np.linalg.norm(heading)),
                        -1, 1)))
                    enemy_angle_off = angle
            
            # 检查雷达锁定状态
            radar_lock = False
            if nearest_enemy and min_distance < self.tactical_distances.get('lock_range', 45000):
                radar_lock = True
            
            # 检查导弹威胁
            has_warning = False
            missile_launched = False
            missile_hit = False
            
            # 检查是否有导弹威胁
            for missile_uid, missile in env._tempsims.items():
                if missile.is_alive and hasattr(missile, 'target_aircraft'):
                    if missile.target_aircraft == aircraft:
                        has_warning = True
                        missile_launched = True
                        if not missile.is_alive and getattr(missile, 'is_success', False):
                            missile_hit = True
            
            return {
                "enemy_distance": min_distance,
                "enemy_angle_off": enemy_angle_off,
                "radar_lock": radar_lock,
                "missile_launched": missile_launched,
                "missile_hit": missile_hit,
                "current_altitude": aircraft.get_position()[2],
                "has_warning": has_warning
            }
            
        except Exception as e:
            logging.error(f"Error in get_state_dict for {agent_id}: {e}")
            return {
                "enemy_distance": 100000,
                "enemy_angle_off": 0,
                "radar_lock": False,
                "missile_launched": False,
                "missile_hit": False,
                "current_altitude": 0,
                "has_warning": False
            }

    def get_tactical_state(self, agent_id):
        """获取战术状态信息 - 必需的父类方法"""
        try:
            return {
                "current_phase": self.current_phase.value,
                "maneuver_history": [],
                "last_template": 0,
                "template_history": []
            }
        except Exception as e:
            logging.error(f"Error in get_tactical_state for {agent_id}: {e}")
            return {
                "current_phase": "unknown",
                "maneuver_history": [],
                "last_template": 0,
                "template_history": []
            }

    def get_termination(self, env, agent_id, info):
        """检查终止条件 - 必需的父类方法"""
        try:
            aircraft = env.agents[agent_id]
            
            # 检查飞机是否存活
            if not aircraft.is_alive:
                return True, info
            
            # 检查高度
            altitude = aircraft.get_position()[2]
            if altitude < 100:  # 低于100米认为撞地
                return True, info
            
            # 检查是否超出边界
            pos = aircraft.get_position()
            if abs(pos[0]) > 100000 or abs(pos[1]) > 100000:  # 超出100km边界
                return True, info
            
            return False, info
            
        except Exception as e:
            logging.error(f"Error in get_termination for {agent_id}: {e}")
            return False, info

    def allocate_targets(self, env):
        """目标分配方法 - 必需的父类方法"""
        try:
            alive_agents = {k: v for k, v in env.agents.items() if v.is_alive}
            red_team = {k: v for k, v in alive_agents.items() if v.color == "Red"}
            blue_team = {k: v for k, v in alive_agents.items() if v.color == "Blue"}

            # 为红方分配目标
            for agent_id, agent in red_team.items():
                if not hasattr(agent, 'is_leader') or not agent.is_leader():
                    continue
                min_threat = float("inf")
                closest_enemy = None
                for enemy_id, enemy in blue_team.items():
                    distance = np.linalg.norm(agent.get_position() - enemy.get_position())
                    threat_score = distance / (1 + np.linalg.norm(enemy.get_velocity()) / 1000)
                    if threat_score < min_threat:
                        min_threat = threat_score
                        closest_enemy = enemy
                if closest_enemy:
                    self._target_allocation[agent_id] = [closest_enemy]
                    for wingman_id in red_team:
                        if wingman_id != agent_id:
                            self._target_allocation[wingman_id] = [closest_enemy]

            # 为蓝方分配目标
            for agent_id, agent in blue_team.items():
                if not hasattr(agent, 'is_leader') or not agent.is_leader():
                    continue
                min_threat = float("inf")
                closest_enemy = None
                for enemy_id, enemy in red_team.items():
                    distance = np.linalg.norm(agent.get_position() - enemy.get_position())
                    threat_score = distance / (1 + np.linalg.norm(enemy.get_velocity()) / 1000)
                    if threat_score < min_threat:
                        min_threat = threat_score
                        closest_enemy = enemy
                if closest_enemy:
                    self._target_allocation[agent_id] = [closest_enemy]
                    for wingman_id in blue_team:
                        if wingman_id != agent_id:
                            self._target_allocation[wingman_id] = [closest_enemy]

        except Exception as e:
            logging.error(f"Error in allocate_targets: {e}")
            # 简单的目标分配：每个智能体攻击最近的敌机
            for agent_id, agent in env.agents.items():
                if not agent.is_alive:
                    continue
                nearest_enemy = None
                min_distance = float('inf')
                for other_id, other_agent in env.agents.items():
                    if (other_id != agent_id and 
                        other_agent.is_alive and 
                        other_agent.color != agent.color):
                        distance = self._calculate_distance(agent, other_agent)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_enemy = other_agent
                if nearest_enemy:
                    self._target_allocation[agent_id] = [nearest_enemy]

    def update_combat_phase(self, env, agent_id):
        """更新作战阶段 - 必需的父类方法"""
        try:
            # 简化的阶段更新逻辑
            # 根据距离更新阶段
            aircraft = env.agents[agent_id]
            if not aircraft.is_alive:
                return
            
            # 找到最近的敌机
            nearest_enemy = None
            min_distance = float('inf')
            for other_id, other_aircraft in env.agents.items():
                if (other_id != agent_id and 
                    other_aircraft.is_alive and 
                    other_aircraft.color != aircraft.color):
                    distance = self._calculate_distance(aircraft, other_aircraft)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_enemy = other_aircraft
            
            if not nearest_enemy:
                self.current_phases[agent_id] = "contact_guidance"
                return
            
            # 根据距离确定阶段
            if min_distance > 80000:
                self.current_phases[agent_id] = "contact_guidance"
            elif min_distance > 45000:
                self.current_phases[agent_id] = "target_search"
            elif min_distance > 25000:
                self.current_phases[agent_id] = "missile_launch"
            else:
                self.current_phases[agent_id] = "tactical_decision"
                
        except Exception as e:
            logging.error(f"Error in update_combat_phase for {agent_id}: {e}")
            self.current_phases[agent_id] = "contact_guidance"
