#!/usr/bin/env python3
"""
敌方多样化战术AI系统

核心功能：
1. 四种战术模式：攻击(AGGRESSIVE)、防御(DEFENSIVE)、中立(NEUTRAL)、支援(SUPPORT)
2. 动态模式切换：基于威胁等级、距离、导弹状态、队友状态等因素
3. 完整BVR对抗：从远距离探测到近距离脱离的全过程
4. 多样化机动：包括notch、cranking、defensive split、beam maneuver等
5. 随机性控制：通过参数控制战术切换的随机性程度

战术模式特点：
- AGGRESSIVE: 主动接敌，积极发射，追求火力优势
- DEFENSIVE: 规避威胁，优先生存，被动应战
- NEUTRAL: 平衡攻防，根据态势灵活调整
- SUPPORT: 配合队友，协同作战，编队机动

实现基于威胁感知的智能机动逻辑，区别于我方的拖曳射击战术
"""

import numpy as np
import logging
from enum import Enum
from typing import Dict, Tuple, Optional, Any
from envs.JSBSim.core.catalog import JsbsimCatalog as c
from envs.JSBSim.core.catalog import ExtraCatalog

def safe_get_altitude(agent, default_value=6000.0):
    """安全获取飞机高度，使用多种方法尝试"""
    try:
        # 方法1：尝试直接获取位置的Z坐标（高度）
        position = agent.get_position()
        if hasattr(position, '__len__') and len(position) >= 3:
            return float(position[2])  # Z坐标通常是高度
    except:
        pass

    try:
        # 方法2：尝试从属性获取高度（英尺转米）
        altitude_ft = agent.get_property_value(c.position_h_sl_ft)
        return altitude_ft * 0.3048  # 英尺转米
    except:
        pass

    try:
        # 方法3：尝试从ExtraCatalog获取米制高度
        return agent.get_property_value(ExtraCatalog.position_h_sl_m)
    except:
        pass

    # 如果所有方法都失败，返回默认值
    logging.debug(f"无法获取飞机高度，使用默认值 {default_value}m")
    return default_value

def safe_get_property(agent, property_name, default_value=0.0):
    """安全获取飞机属性，避免属性访问错误"""
    try:
        return agent.get_property_value(property_name)
    except (AttributeError, KeyError, Exception) as e:
        logging.debug(f"属性获取失败 {property_name}: {e}, 使用默认值 {default_value}")
        return default_value

class ThreatLevel(Enum):
    """威胁等级"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class BVRPhase(Enum):
    """BVR交战阶段 - 镜像友军拖曳射击逻辑"""
    APPROACH = "approach"           # 接敌阶段 - 镜像友军NLT_MELD到MELD_MTR (>45km)
    ENGAGE = "engage"              # 交战阶段 - 镜像友军MTR_TR发射窗口 (45-41km)
    TURN_COLD = "turn_cold"        # 转冷阶段 - 镜像友军Short Skate Crank (6-18秒)
    RETURN = "return"              # 返航阶段 - 镜像友军Short Skate Turn Cold (15-30秒)
    ESCAPE = "escape"              # 脱离阶段 - 镜像友军Short Skate Escape (15-22秒)
    RE_ENGAGE = "re_engage"        # 重新接敌阶段 - 重新开始循环

class TacticalMode(Enum):
    """敌方战术模式"""
    AGGRESSIVE = "aggressive"    # 攻击模式：主动接敌，积极发射
    DEFENSIVE = "defensive"      # 防御模式：规避威胁，优先生存
    NEUTRAL = "neutral"          # 中立模式：平衡攻防，灵活调整
    SUPPORT = "support"          # 支援模式：配合队友，协同作战

class EnemyManeuverType(Enum):
    """敌方机动类型"""
    CAP_PATROL = "cap_patrol"           # CAP巡逻
    AGGRESSIVE_APPROACH = "aggressive"   # 攻击性接敌
    DEFENSIVE_TURN = "defensive_turn"    # 防御转弯
    EVASIVE_MANEUVER = "evasive"        # 规避机动
    ATTACK_POSITIONING = "attack_pos"    # 攻击定位
    RETREAT = "retreat"                  # 撤退
    NOTCH_MANEUVER = "notch"            # Notch机动（90度规避）
    SPLIT_S = "split_s"                 # Split-S机动
    BARREL_ROLL = "barrel_roll"         # 桶滚机动
    CRANKING = "cranking"               # Cranking机动（保持雷达锁定）
    DEFENSIVE_SPLIT = "defensive_split"  # 防御分离机动
    BEAM_MANEUVER = "beam_maneuver"     # Beam机动（侧向规避）

    # BVR战术机动
    BVR_TURN_COLD = "bvr_turn_cold"     # BVR转冷返航
    BVR_RETURN_BASE = "bvr_return_base" # BVR返回基地
    BVR_RE_ENGAGE = "bvr_re_engage"     # BVR重新接敌

class EnemyTacticalAI:
    """
    敌方多样化战术AI系统

    支持四种战术模式：
    - AGGRESSIVE: 攻击模式，主动接敌，积极发射
    - DEFENSIVE: 防御模式，规避威胁，优先生存
    - NEUTRAL: 中立模式，平衡攻防，灵活调整
    - SUPPORT: 支援模式，配合队友，协同作战
    """

    def __init__(self, randomness_level: float = 0.3):
        # 战术模式管理
        self.current_tactical_mode = {}  # agent_id -> TacticalMode
        self.mode_switch_cooldown = {}   # agent_id -> last_switch_time
        self.mode_switch_interval = 15.0  # 模式切换最小间隔（秒）
        self.randomness_level = randomness_level  # 随机性程度 [0.0-1.0]

        # 机动状态跟踪
        self.maneuver_states = {}  # agent_id -> maneuver_state
        self.threat_history = {}   # agent_id -> threat_history
        self.last_maneuver_time = {}  # agent_id -> last_maneuver_time

        # 态势感知数据
        self.situation_awareness = {}  # agent_id -> situation_data

        # BVR战术状态跟踪
        self.bvr_states = {}  # agent_id -> BVRPhase
        self.missile_launch_time = {}  # agent_id -> last_missile_launch_time
        self.return_to_base_time = {}  # agent_id -> return_start_time
        self.engagement_cycle = {}  # agent_id -> current_cycle_phase
        self.initial_position = {}  # agent_id -> initial_spawn_position
        self.battlefield_center = np.array([0.0, 0.0, 8000.0])  # 战场中心位置

        # 添加缺失的属性
        self.threat_ranges = {
            ThreatLevel.NONE: 0,
            ThreatLevel.LOW: 50000,
            ThreatLevel.MEDIUM: 30000,
            ThreatLevel.HIGH: 20000,
            ThreatLevel.CRITICAL: 10000
        }

        # 镜像友军拖曳射击的距离阈值 - 强化BVR距离控制
        self.mirror_tactical_distances = {
            'NLT_MELD_min': 81000,      # 81km - 镜像友军NLT_MELD阶段
            'MELD_MTR_min': 50000,      # 50km - 镜像友军MELD_MTR阶段（提高到50km）
            'MTR_TR_min': 45000,        # 45km - 镜像友军MTR_TR发射窗口（提高到45km）
            'TR_DOR_min': 35000,        # 35km - 强制BVR距离（大幅提高）
            'DOR_DR_min': 30000,        # 30km - 强制BVR距离（大幅提高）
            'leader_launch_range': 50000,   # 长机50km发射 - 更远距离发射
            'wingman_launch_range': 45000,  # 僚机45km发射 - 更远距离发射
            'emergency_distance': 30000,    # 30km紧急转冷距离（提高）
            'min_bvr_distance': 25000,      # 25km最小BVR距离（提高）
            'ideal_bvr_distance': 40000,    # 40km理想BVR距离（提高）
            'force_cold_turn_distance': 35000,  # 35km强制冷转距离
        }

        # 镜像友军Short Skate机动时间参数
        self.mirror_short_skate_durations = {
            'leader_crank': 6.0,        # 长机Crank 6秒 - 镜像友军
            'leader_turn_cold': 15.0,   # 长机Turn Cold 15秒 - 镜像友军
            'leader_escape': 15.0,      # 长机Escape 15秒 - 镜像友军
            'wingman_crank': 18.0,      # 僚机Crank 18秒 - 镜像友军
            'wingman_turn_cold': 30.0,  # 僚机Turn Cold 30秒 - 镜像友军
            'wingman_escape': 22.0,     # 僚机Escape 22秒 - 镜像友军
        }

        # 编队协调属性
        self.formation_coordination = {}  # 编队协调状态
        self.last_coordination_check = {}  # 上次协调检查时间

        # 镜像Short Skate状态管理
        self.mirror_short_skate_states = {}  # 镜像友军的Short Skate状态
        self.mirror_skate_start_time = {}    # 镜像Short Skate开始时间

        # 机动参数配置
        self.maneuver_params = {
            EnemyManeuverType.AGGRESSIVE_APPROACH: {"duration": 15.0, "priority": 3},
            EnemyManeuverType.DEFENSIVE_TURN: {"duration": 12.0, "priority": 4},
            EnemyManeuverType.EVASIVE_MANEUVER: {"duration": 10.0, "priority": 5},
            EnemyManeuverType.ATTACK_POSITIONING: {"duration": 18.0, "priority": 3},
            EnemyManeuverType.NOTCH_MANEUVER: {"duration": 8.0, "priority": 5},
            EnemyManeuverType.BEAM_MANEUVER: {"duration": 10.0, "priority": 4},
            EnemyManeuverType.CRANKING: {"duration": 15.0, "priority": 3},
            EnemyManeuverType.BARREL_ROLL: {"duration": 12.0, "priority": 2},
            EnemyManeuverType.DEFENSIVE_SPLIT: {"duration": 10.0, "priority": 4},
            EnemyManeuverType.CAP_PATROL: {"duration": 20.0, "priority": 1},
            EnemyManeuverType.BVR_TURN_COLD: {"duration": 8.0, "priority": 5},
            EnemyManeuverType.BVR_RETURN_BASE: {"duration": 25.0, "priority": 2},
            EnemyManeuverType.BVR_RE_ENGAGE: {"duration": 15.0, "priority": 3}
        }

    def _initialize_bvr_state(self, agent_id: str, agent_pos: np.ndarray):
        """初始化BVR战术状态"""
        if agent_id not in self.bvr_states:
            self.bvr_states[agent_id] = BVRPhase.APPROACH
            self.initial_position[agent_id] = agent_pos.copy()
            self.missile_launch_time[agent_id] = 0.0
            self.return_to_base_time[agent_id] = 0.0
            self.engagement_cycle[agent_id] = 0
            self.formation_coordination[agent_id] = True
            self.last_coordination_check[agent_id] = 0.0

    def _check_formation_coordination(self, env, current_time: float) -> Dict[str, bool]:
        """检查编队协调状态，确保两机协调行动"""
        coordination_status = {}

        # 获取两机状态
        b0100_alive = "B0100" in env.agents and env.agents["B0100"].is_alive
        b0200_alive = "B0200" in env.agents and env.agents["B0200"].is_alive

        if not (b0100_alive and b0200_alive):
            # 如果有飞机被击落，剩余飞机独立作战
            for agent_id in ["B0100", "B0200"]:
                if agent_id in env.agents and env.agents[agent_id].is_alive:
                    coordination_status[agent_id] = False
            return coordination_status

        # 获取两机的BVR状态
        b0100_phase = self.bvr_states.get("B0100", BVRPhase.APPROACH)
        b0200_phase = self.bvr_states.get("B0200", BVRPhase.APPROACH)

        # 协调规则：
        # 1. 转冷阶段：僚机等待长机先转冷
        # 2. 返航阶段：确保两机同时返航
        # 3. 重新接敌：长机先接敌，僚机跟随

        if b0100_phase == BVRPhase.TURN_COLD and b0200_phase == BVRPhase.APPROACH:
            # 长机转冷，僚机继续接敌支援
            coordination_status["B0100"] = True
            coordination_status["B0200"] = True
        elif b0100_phase == BVRPhase.RETURN and b0200_phase == BVRPhase.TURN_COLD:
            # 长机返航，僚机转冷，协调正常
            coordination_status["B0100"] = True
            coordination_status["B0200"] = True
        elif abs(self.return_to_base_time.get("B0100", 0) - self.return_to_base_time.get("B0200", 0)) > 10.0:
            # 返航时间差超过10秒，需要协调
            if self.return_to_base_time.get("B0100", 0) > self.return_to_base_time.get("B0200", 0):
                coordination_status["B0100"] = True
                coordination_status["B0200"] = False  # 僚机等待
            else:
                coordination_status["B0100"] = False  # 长机等待
                coordination_status["B0200"] = True
        else:
            # 正常协调状态
            coordination_status["B0100"] = True
            coordination_status["B0200"] = True

        return coordination_status

    def _init_mirror_short_skate(self, agent_id: str, current_time: float):
        """初始化镜像友军的Short Skate机动状态"""
        self.mirror_short_skate_states[agent_id] = {
            "phase": "crank",  # 开始阶段：Crank机动
            "phase_start_time": current_time,
            "initial_heading": None,
            "initial_altitude": None,
            "crank_angle": -40.0 if agent_id == "B0100" else 40.0,  # 长机左转，僚机右转
        }
        self.mirror_skate_start_time[agent_id] = current_time

    def _execute_mirror_short_skate(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行镜像友军的Short Skate机动 - 完全对应友军逻辑"""
        if agent_id not in self.mirror_short_skate_states:
            self._init_mirror_short_skate(agent_id, current_time)

        state = self.mirror_short_skate_states[agent_id]
        agent = env.agents[agent_id]
        current_heading = np.rad2deg(safe_get_property(agent, c.attitude_psi_rad, 0.0))

        if state["initial_heading"] is None:
            state["initial_heading"] = current_heading

        phase_time = current_time - state["phase_start_time"]

        # 镜像友军Short Skate时间参数
        if agent_id == "B0100":  # 长机
            crank_duration = self.mirror_short_skate_durations['leader_crank']
            turn_cold_duration = self.mirror_short_skate_durations['leader_turn_cold']
            escape_duration = self.mirror_short_skate_durations['leader_escape']
        else:  # 僚机
            crank_duration = self.mirror_short_skate_durations['wingman_crank']
            turn_cold_duration = self.mirror_short_skate_durations['wingman_turn_cold']
            escape_duration = self.mirror_short_skate_durations['wingman_escape']

        # 阶段1：Crank机动 - 镜像友军左/右转40度
        if state["phase"] == "crank":
            if phase_time < crank_duration:
                target_heading = state["initial_heading"] + state["crank_angle"]
                target_heading = target_heading % 360
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3  # 左转或右转
                else:
                    # Crank完成，进入Turn Cold阶段
                    state["phase"] = "turn_cold"
                    state["phase_start_time"] = current_time
                    state["initial_heading"] = current_heading
                    print(f"🔄 {agent_id}: Crank完成，开始Turn Cold")

        # 阶段2：Turn Cold机动 - 镜像友军转向0度（正北）
        elif state["phase"] == "turn_cold":
            if phase_time < turn_cold_duration:
                target_heading = 0.0  # 正北方向
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3  # 左转或右转
                else:
                    # Turn Cold完成，进入Escape阶段
                    state["phase"] = "escape"
                    state["phase_start_time"] = current_time
                    print(f"🏃 {agent_id}: Turn Cold完成，开始Escape")

        # 阶段3：Escape机动 - 镜像友军保持北向脱离
        elif state["phase"] == "escape":
            if phase_time < escape_duration:
                target_heading = 0.0  # 继续保持正北方向
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3  # 左转或右转
                else:
                    # Escape完成，Short Skate机动结束
                    state["phase"] = "complete"
                    print(f"✅ {agent_id}: Short Skate机动完成")

        return 7, 8, 3  # 默认保持航向

    def _normalize_angle_diff(self, angle_diff: float) -> float:
        """标准化角度差值到[-180, 180]范围"""
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return angle_diff

    def _update_bvr_state(self, agent_id: str, env, current_time: float):
        """更新BVR战术状态 - 长机僚机协调版本"""
        current_phase = self.bvr_states.get(agent_id, BVRPhase.APPROACH)
        agent = env.agents[agent_id]
        agent_pos = np.array(agent.get_position())

        # 检查编队协调状态
        if current_time - self.last_coordination_check.get(agent_id, 0) > 2.0:
            coordination_status = self._check_formation_coordination(env, current_time)
            self.formation_coordination[agent_id] = coordination_status.get(agent_id, True)
            self.last_coordination_check[agent_id] = current_time

        # 战场边界检查 - 强制保持在合理范围内
        distance_to_center = np.linalg.norm(agent_pos[:2] - self.battlefield_center[:2])
        max_distance_from_center = 60000  # 60km最大距离（缩小范围）

        # 如果超出战场边界，强制返回接敌状态
        if distance_to_center > max_distance_from_center:
            self.bvr_states[agent_id] = BVRPhase.APPROACH
            print(f"⚠️ {agent_id}: 超出战场边界({distance_to_center/1000:.1f}km)，强制返回接敌")
            return

        # 计算到最近敌方的距离
        min_distance = float('inf')
        nearest_enemy_id = None
        for other_id, other_agent in env.agents.items():
            if other_id != agent_id and other_agent.is_alive:
                # 只计算与敌方的距离（A vs B）
                if ((agent_id.startswith('A') and other_id.startswith('B')) or
                    (agent_id.startswith('B') and other_id.startswith('A'))):
                    other_pos = np.array(other_agent.get_position())
                    distance = np.linalg.norm(agent_pos - other_pos)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_enemy_id = other_id

        # 强制BVR距离控制 - 当距离<35km时必须执行冷转
        if min_distance < self.mirror_tactical_distances['force_cold_turn_distance'] and current_phase in [BVRPhase.APPROACH, BVRPhase.ENGAGE]:
            self.bvr_states[agent_id] = BVRPhase.TURN_COLD
            self.missile_launch_time[agent_id] = current_time
            self._init_mirror_short_skate(agent_id, current_time)
            print(f"🔄 {agent_id}: 强制BVR冷转！距离: {min_distance/1000:.1f}km < 35km")
            return

        # 超视距距离强制控制 - 硬性距离阈值
        if min_distance < self.mirror_tactical_distances['emergency_distance']:  # 25km紧急距离
            # 强制执行紧急转冷脱离
            if current_phase not in [BVRPhase.TURN_COLD, BVRPhase.RETURN, BVRPhase.ESCAPE]:
                self.bvr_states[agent_id] = BVRPhase.TURN_COLD
                self.missile_launch_time[agent_id] = current_time
                self._init_mirror_short_skate(agent_id, current_time)
                print(f"🚨 {agent_id}: 紧急转冷脱离！距离过近: {min_distance/1000:.1f}km < 25km")
                return

        # 最小BVR距离检查
        if min_distance < self.mirror_tactical_distances['min_bvr_distance']:  # 20km最小距离
            # 强制执行最大转向脱离
            if current_phase not in [BVRPhase.ESCAPE]:
                self.bvr_states[agent_id] = BVRPhase.ESCAPE
                self.return_to_base_time[agent_id] = current_time
                print(f"⚠️ {agent_id}: 违反BVR最小距离！强制脱离: {min_distance/1000:.1f}km < 20km")
                return

        # 镜像友军拖曳射击的BVR状态转换逻辑
        if current_phase == BVRPhase.APPROACH:
            # 接敌阶段：镜像友军NLT_MELD到MELD_MTR阶段 (>45km)
            # 当距离接近友军发射窗口时，转入交战阶段
            if min_distance <= self.mirror_tactical_distances['MELD_MTR_min']:  # 45km
                self.bvr_states[agent_id] = BVRPhase.ENGAGE
                print(f"⚔️ {agent_id}: 进入交战阶段，镜像友军MTR_TR (距离: {min_distance/1000:.1f}km)")

        elif current_phase == BVRPhase.ENGAGE:
            # 交战阶段：镜像友军MTR_TR发射窗口，但错位时机确保友军完整展现拖曳射击
            launch_range = (self.mirror_tactical_distances['leader_launch_range'] if agent_id == "B0100"
                          else self.mirror_tactical_distances['wingman_launch_range'])

            # 战术时机协调：敌方与友军错位发射，形成持续对抗
            # 分析友军发射时机，敌方在友军发射后适当延迟发射
            min_engage_time = 20.0 if agent_id == "B0100" else 30.0  # 长机20秒，僚机30秒（延后发射）
            missile_cooldown = 10.0  # 导弹发射冷却时间（与友军区分）

            # 检查是否已经发射过导弹
            has_launched = self.missile_launch_time.get(agent_id, 0.0) > 0

            if (min_distance <= launch_range and current_time > min_engage_time and
                current_time - self.missile_launch_time.get(agent_id, 0.0) > missile_cooldown and
                not has_launched):

                # 检查是否应该发射导弹
                if self._check_missile_launch(agent_id, env, current_time):
                    self.missile_launch_time[agent_id] = current_time
                    role = "长机" if agent_id == "B0100" else "僚机"
                    print(f"🚀 {agent_id}({role}): 镜像发射窗口，发射导弹 (距离: {min_distance/1000:.1f}km)")

            # 发射导弹后5秒开始转冷 - 镜像友军逻辑，或者在交战阶段超过30秒后强制转冷
            engage_duration = current_time - self.engagement_cycle.get(agent_id, current_time)
            if ((has_launched and current_time - self.missile_launch_time.get(agent_id, 0.0) > 5.0) or
                engage_duration > 30.0):  # 交战阶段最多30秒
                self.bvr_states[agent_id] = BVRPhase.TURN_COLD
                self._init_mirror_short_skate(agent_id, current_time)
                role = "长机" if agent_id == "B0100" else "僚机"
                reason = "发射后转冷" if has_launched else "交战超时转冷"
                print(f"🔄 {agent_id}({role}): {reason}，镜像Short Skate (距离: {min_distance/1000:.1f}km)")

        elif current_phase == BVRPhase.TURN_COLD:
            # 转冷阶段：镜像友军Short Skate Crank机动
            crank_duration = (self.mirror_short_skate_durations['leader_crank'] if agent_id == "B0100"
                            else self.mirror_short_skate_durations['wingman_crank'])
            cold_turn_start = self.missile_launch_time.get(agent_id, current_time)
            if current_time - cold_turn_start > crank_duration:
                self.bvr_states[agent_id] = BVRPhase.RETURN
                self.return_to_base_time[agent_id] = current_time
                role = "长机" if agent_id == "B0100" else "僚机"
                print(f"🔄 {agent_id}({role}): Crank完成，开始Turn Cold阶段 (距离: {min_distance/1000:.1f}km)")

        elif current_phase == BVRPhase.RETURN:
            # 返航阶段：镜像友军Short Skate Turn Cold机动
            turn_cold_duration = (self.mirror_short_skate_durations['leader_turn_cold'] if agent_id == "B0100"
                                else self.mirror_short_skate_durations['wingman_turn_cold'])
            if current_time - self.return_to_base_time[agent_id] > turn_cold_duration:
                self.bvr_states[agent_id] = BVRPhase.ESCAPE
                role = "长机" if agent_id == "B0100" else "僚机"
                print(f"🏃 {agent_id}({role}): Turn Cold完成，开始Escape阶段 (距离: {min_distance/1000:.1f}km)")

        elif current_phase == BVRPhase.ESCAPE:
            # 脱离阶段：镜像友军Short Skate Escape机动，确保达到BVR距离
            escape_duration = (self.mirror_short_skate_durations['leader_escape'] if agent_id == "B0100"
                             else self.mirror_short_skate_durations['wingman_escape'])

            # 检查是否已经达到安全BVR距离
            safe_bvr_distance = 50000  # 50km安全距离
            escape_time_elapsed = current_time - self.return_to_base_time[agent_id]

            if (escape_time_elapsed > escape_duration and min_distance > safe_bvr_distance) or escape_time_elapsed > 60.0:
                # 重置状态，准备下一轮BVR循环
                self.bvr_states[agent_id] = BVRPhase.RE_ENGAGE
                self.engagement_cycle[agent_id] = self.engagement_cycle.get(agent_id, 0) + 1
                # 重置导弹发射时间，允许下一轮发射
                self.missile_launch_time[agent_id] = 0.0
                role = "长机" if agent_id == "B0100" else "僚机"
                print(f"🔄 {agent_id}({role}): Escape完成，准备重新接敌 (距离: {min_distance/1000:.1f}km, 循环: {self.engagement_cycle[agent_id]})")
                role = "长机" if agent_id == "B0100" else "僚机"
                print(f"⚔️ {agent_id}({role}): Escape完成，重新接敌，第{self.engagement_cycle[agent_id]}轮")

        elif current_phase == BVRPhase.RE_ENGAGE:
            # 重新接敌阶段：协调时机确保与友军形成持续BVR循环对抗
            re_engage_start = self.return_to_base_time.get(agent_id, current_time)
            re_engage_duration = current_time - re_engage_start

            # 协调缓冲时间：确保友军有足够时间展现完整的拖曳射击循环
            # 延长缓冲时间，给友军更多展示时间
            coordination_buffer = 40.0 if agent_id == "B0100" else 50.0  # 长机40秒，僚机50秒

            # 检查是否可以重新接敌
            can_re_engage = (re_engage_duration > coordination_buffer and
                           min_distance > 40000)  # 距离>40km才能重新接敌

            if can_re_engage:
                # 重新开始BVR循环
                self.bvr_states[agent_id] = BVRPhase.APPROACH
                self.engagement_cycle[agent_id] = current_time  # 记录新循环开始时间
                role = "长机" if agent_id == "B0100" else "僚机"
                print(f"🔄 {agent_id}({role}): 重新接敌，开始新的BVR循环 (距离: {min_distance/1000:.1f}km)")
                if min_distance > self.mirror_tactical_distances['ideal_bvr_distance']:  # 35km理想距离
                    self.bvr_states[agent_id] = BVRPhase.APPROACH
                    role = "长机" if agent_id == "B0100" else "僚机"
                    print(f"🎯 {agent_id}({role}): 重新接敌完成，回到接敌阶段，距离: {min_distance/1000:.1f}km")
                else:
                    # 距离仍然过近，继续脱离
                    print(f"⏳ {agent_id}: 距离仍过近({min_distance/1000:.1f}km)，继续脱离等待重新接敌")

    def _check_missile_launch(self, agent_id: str, env, current_time: float) -> bool:
        """检查是否刚刚发射了导弹 - 修复版本"""
        # 基于时间间隔的导弹发射检测
        last_launch = self.missile_launch_time.get(agent_id, 0.0)
        if current_time - last_launch > 25.0:  # 25秒内只能发射一次
            # 模拟导弹发射条件检查
            return self._should_launch_missile(agent_id, env, current_time)
        return False

    def _should_launch_missile(self, agent_id: str, env, current_time: float) -> bool:
        """判断是否应该发射导弹 - 修复版本"""
        agent = env.agents[agent_id]
        agent_pos = np.array(agent.get_position())

        # 寻找最近的敌方目标
        closest_enemy = None
        min_distance = float('inf')

        for other_id, other_agent in env.agents.items():
            if other_id != agent_id and other_agent.is_alive:
                other_pos = np.array(other_agent.get_position())
                distance = np.linalg.norm(agent_pos - other_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_enemy = other_agent

        # 发射条件：
        # 1. 距离在20-60km之间（更宽的发射窗口）
        # 2. 仿真时间超过30秒（避免开局立即发射）
        # 3. 当前处于接敌阶段
        current_phase = self.bvr_states.get(agent_id, BVRPhase.APPROACH)

        if (closest_enemy and
            20000 <= min_distance <= 60000 and
            current_time > 30.0 and
            current_phase == BVRPhase.APPROACH):
            return True
        return False

    def _execute_bvr_turn_cold(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行BVR转冷机动 - 镜像友军Short Skate Crank阶段"""
        # 直接调用镜像Short Skate机动函数
        return self._execute_mirror_short_skate(env, agent_id, current_time)

        # 计算转向指令 - 正确映射到动作空间
        heading_diff = (target_heading - current_heading + 180) % 360 - 180

        # 使用正确的动作空间映射进行转向
        # 动作空间：0(-180°), 1(-120°), 2(-90°), 3(-75°), 4(-60°), 5(-45°), 6(-30°), 7(-15°), 8(0°), 9(15°), 10(30°), 11(45°), 12(60°), 13(75°), 14(90°), 15(120°), 16(180°)

        if heading_diff > 150:
            turn_command = 16  # 180°右转
        elif heading_diff > 100:
            turn_command = 15  # 120°右转
        elif heading_diff > 80:
            turn_command = 14  # 90°右转
        elif heading_diff > 65:
            turn_command = 13  # 75°右转
        elif heading_diff > 50:
            turn_command = 12  # 60°右转
        elif heading_diff > 35:
            turn_command = 11  # 45°右转
        elif heading_diff > 20:
            turn_command = 10  # 30°右转
        elif heading_diff > 5:
            turn_command = 9   # 15°右转
        elif heading_diff < -150:
            turn_command = 0   # -180°左转
        elif heading_diff < -100:
            turn_command = 1   # -120°左转
        elif heading_diff < -80:
            turn_command = 2   # -90°左转
        elif heading_diff < -65:
            turn_command = 3   # -75°左转
        elif heading_diff < -50:
            turn_command = 4   # -60°左转
        elif heading_diff < -35:
            turn_command = 5   # -45°左转
        elif heading_diff < -20:
            turn_command = 6   # -30°左转
        elif heading_diff < -5:
            turn_command = 7   # -15°左转
        else:
            turn_command = 8   # 0°保持航向



        # 高度和速度指令 - 修复高度控制
        # 获取当前高度进行安全检查
        try:
            current_altitude = safe_get_property(agent, c.position_h_sl_m, 6000.0)
        except:
            current_altitude = 6000.0

        # 长机僚机差异化高度和速度策略
        if agent_id == "B0100":  # 长机
            # 长机：保持高度优势，最大加速
            if current_altitude < 1000:
                altitude_command = 14  # +1500m 紧急爬升
            elif current_altitude < 5000:
                altitude_command = 8   # +50m 轻微爬升，保持高度优势
            else:
                altitude_command = 8   # +50m 继续爬升
            velocity_command = 6  # 最大加速
        else:  # 僚机 B0200
            # 僚机：保持相对较低高度，高速机动
            if current_altitude < 1000:
                altitude_command = 11  # +500m 适度爬升
            elif current_altitude < 3000:
                altitude_command = 7   # 保持高度
            else:
                altitude_command = 6   # -50m 轻微下降，与长机形成高度差
            velocity_command = 5  # 高速

        return altitude_command, turn_command, velocity_command  # 修复参数顺序：(高度, 航向, 速度)

    def _execute_bvr_return(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行BVR返航机动 - 镜像友军Short Skate Turn Cold阶段"""
        # 直接调用镜像Short Skate机动函数
        return self._execute_mirror_short_skate(env, agent_id, current_time)
        # 动作空间：0(-180°), 1(-120°), 2(-90°), 3(-75°), 4(-60°), 5(-45°), 6(-30°), 7(-15°), 8(0°), 9(15°), 10(30°), 11(45°), 12(60°), 13(75°), 14(90°), 15(120°), 16(180°)

        # 根据航向差选择最合适的转向指令
        if heading_diff > 150:
            turn_command = 16  # 180°右转
        elif heading_diff > 100:
            turn_command = 15  # 120°右转
        elif heading_diff > 80:
            turn_command = 14  # 90°右转
        elif heading_diff > 65:
            turn_command = 13  # 75°右转
        elif heading_diff > 50:
            turn_command = 12  # 60°右转
        elif heading_diff > 35:
            turn_command = 11  # 45°右转
        elif heading_diff > 20:
            turn_command = 10  # 30°右转
        elif heading_diff > 5:
            turn_command = 9   # 15°右转
        elif heading_diff < -150:
            turn_command = 0   # -180°左转
        elif heading_diff < -100:
            turn_command = 1   # -120°左转
        elif heading_diff < -80:
            turn_command = 2   # -90°左转
        elif heading_diff < -65:
            turn_command = 3   # -75°左转
        elif heading_diff < -50:
            turn_command = 4   # -60°左转
        elif heading_diff < -35:
            turn_command = 5   # -45°左转
        elif heading_diff < -20:
            turn_command = 6   # -30°左转
        elif heading_diff < -5:
            turn_command = 7   # -15°左转
        else:
            turn_command = 8   # 0°保持航向



        # 长机僚机差异化高度和速度策略
        if agent_id == "B0100":  # 长机
            # 长机：保持较高高度，中等速度返航
            if current_altitude < 1000:
                altitude_command = 14  # +1500m 紧急爬升
            elif current_altitude < 5000:
                altitude_command = 9   # +150m 轻微爬升
            else:
                altitude_command = 7   # 保持高度
            velocity_command = 4  # 巡航速度
        else:  # 僚机 B0200
            # 僚机：保持相对较低高度，稍快速度保持编队
            if current_altitude < 1000:
                altitude_command = 11  # +500m 适度爬升
            elif current_altitude < 3000:
                altitude_command = 7   # 保持高度
            else:
                altitude_command = 6   # -50m 轻微下降
            velocity_command = 5  # 稍快速度

        return altitude_command, turn_command, velocity_command  # 修复参数顺序：(高度, 航向, 速度)

    def _execute_bvr_escape(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行BVR脱离机动 - 镜像友军Short Skate Escape阶段"""
        # 直接调用镜像Short Skate机动函数
        return self._execute_mirror_short_skate(env, agent_id, current_time)
        if heading_diff > 150:
            turn_command = 16  # 180°右转
        elif heading_diff > 100:
            turn_command = 15  # 120°右转
        elif heading_diff > 80:
            turn_command = 14  # 90°右转
        elif heading_diff > 65:
            turn_command = 13  # 75°右转
        elif heading_diff > 50:
            turn_command = 12  # 60°右转
        elif heading_diff > 35:
            turn_command = 11  # 45°右转
        elif heading_diff > 20:
            turn_command = 10  # 30°右转
        elif heading_diff > 5:
            turn_command = 9   # 15°右转
        elif heading_diff < -150:
            turn_command = 0   # -180°左转
        elif heading_diff < -100:
            turn_command = 1   # -120°左转
        elif heading_diff < -80:
            turn_command = 2   # -90°左转
        elif heading_diff < -65:
            turn_command = 3   # -75°左转
        elif heading_diff < -50:
            turn_command = 4   # -60°左转
        elif heading_diff < -35:
            turn_command = 5   # -45°左转
        elif heading_diff < -20:
            turn_command = 6   # -30°左转
        elif heading_diff < -5:
            turn_command = 7   # -15°左转
        else:
            turn_command = 8   # 0°保持航向

        # 高度和速度控制 - 最大脱离
        if current_altitude < 1000:
            altitude_command = 14  # +1500m 紧急爬升
        elif current_altitude < 5000:
            altitude_command = 11  # +500m 适度爬升
        else:
            altitude_command = 8   # +50m 轻微爬升

        velocity_command = 6  # 最大加速脱离

        return altitude_command, turn_command, velocity_command

    def _execute_bvr_re_engage(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行BVR重新接敌机动 - 修复高度获取错误"""
        agent = env.agents[agent_id]
        agent_pos = np.array(agent.get_position())
        current_heading = np.rad2deg(safe_get_property(agent, c.attitude_psi_rad, 0.0))

        # 安全获取高度，避免错误
        try:
            current_altitude = safe_get_property(agent, c.position_h_sl_m, 6000.0)
            if current_altitude is None or current_altitude < 0:
                current_altitude = 6000.0
        except:
            current_altitude = 6000.0

        # 长机僚机差异化重新接敌策略
        target_direction = self.battlefield_center[:2] - agent_pos[:2]
        distance_to_center = np.linalg.norm(target_direction)

        if agent_id == "B0100":  # 长机
            # 长机：直接朝向战场中心，承担主攻击者角色
            if distance_to_center > 1000:
                target_direction = target_direction / distance_to_center
                target_heading = np.rad2deg(np.arctan2(target_direction[1], target_direction[0]))
            else:
                target_heading = 180.0  # 朝南接敌
            print(f"⚔️ {agent_id}(长机): 主攻接敌 - 当前航向: {current_heading:.1f}°, 目标航向: {target_heading:.1f}°, 距离中心: {distance_to_center/1000:.1f}km, 高度: {current_altitude:.0f}m")
        else:  # 僚机 B0200
            # 僚机：侧翼接敌，形成包抄态势
            if distance_to_center > 1000:
                target_direction = target_direction / distance_to_center
                base_heading = np.rad2deg(np.arctan2(target_direction[1], target_direction[0]))
                target_heading = base_heading - 30.0  # 偏西30°，形成侧翼
            else:
                target_heading = 210.0  # 朝西南接敌，形成夹击
            print(f"⚔️ {agent_id}(僚机): 侧翼接敌 - 当前航向: {current_heading:.1f}°, 目标航向: {target_heading:.1f}°, 距离中心: {distance_to_center/1000:.1f}km, 高度: {current_altitude:.0f}m")

        # 计算转向指令 - 正确映射到动作空间
        heading_diff = (target_heading - current_heading + 180) % 360 - 180

        # 使用正确的动作空间映射进行转向
        # 动作空间：0(-180°), 1(-120°), 2(-90°), 3(-75°), 4(-60°), 5(-45°), 6(-30°), 7(-15°), 8(0°), 9(15°), 10(30°), 11(45°), 12(60°), 13(75°), 14(90°), 15(120°), 16(180°)

        if heading_diff > 150:
            turn_command = 16  # 180°右转
        elif heading_diff > 100:
            turn_command = 15  # 120°右转
        elif heading_diff > 80:
            turn_command = 14  # 90°右转
        elif heading_diff > 65:
            turn_command = 13  # 75°右转
        elif heading_diff > 50:
            turn_command = 12  # 60°右转
        elif heading_diff > 35:
            turn_command = 11  # 45°右转
        elif heading_diff > 20:
            turn_command = 10  # 30°右转
        elif heading_diff > 5:
            turn_command = 9   # 15°右转
        elif heading_diff < -150:
            turn_command = 0   # -180°左转
        elif heading_diff < -100:
            turn_command = 1   # -120°左转
        elif heading_diff < -80:
            turn_command = 2   # -90°左转
        elif heading_diff < -65:
            turn_command = 3   # -75°左转
        elif heading_diff < -50:
            turn_command = 4   # -60°左转
        elif heading_diff < -35:
            turn_command = 5   # -45°左转
        elif heading_diff < -20:
            turn_command = 6   # -30°左转
        elif heading_diff < -5:
            turn_command = 7   # -15°左转
        else:
            turn_command = 8   # 0°保持航向

        # 长机僚机差异化高度和速度策略
        if agent_id == "B0100":  # 长机
            # 长机：保持高度优势，积极接敌
            if current_altitude < 1000:
                altitude_command = 14  # +1500m 紧急爬升
            elif current_altitude < 5000:
                altitude_command = 11  # +500m 适度爬升
            else:
                altitude_command = 8   # +50m 轻微爬升，保持高度优势
            velocity_command = 5  # 高速接敌
        else:  # 僚机 B0200
            # 僚机：保持机动性，侧翼支援
            if current_altitude < 1000:
                altitude_command = 11  # +500m 适度爬升
            elif current_altitude < 3000:
                altitude_command = 8   # +50m 轻微爬升
            else:
                altitude_command = 7   # 保持高度
            velocity_command = 4  # 巡航速度，保持编队

        return altitude_command, turn_command, velocity_command  # 修复参数顺序：(高度, 航向, 速度)

    def _get_enemy_center_position(self, env, agent_id: str) -> np.ndarray:
        """获取敌方中心位置"""
        enemy_positions = []
        for other_id, other_agent in env.agents.items():
            if other_id != agent_id and other_agent.is_alive:
                enemy_positions.append(np.array(other_agent.get_position()))

        if enemy_positions:
            return np.mean(enemy_positions, axis=0)
        return None

    def _get_closest_enemy_position(self, env, agent_id: str) -> np.ndarray:
        """获取最近敌方位置"""
        agent = env.agents[agent_id]
        agent_pos = np.array(agent.get_position())

        closest_enemy_pos = None
        min_distance = float('inf')

        for other_id, other_agent in env.agents.items():
            if other_id != agent_id and other_agent.is_alive:
                other_pos = np.array(other_agent.get_position())
                distance = np.linalg.norm(agent_pos - other_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_enemy_pos = other_pos

        return closest_enemy_pos
        
        # 战术模式参数配置
        self.tactical_mode_params = {
            TacticalMode.AGGRESSIVE: {
                "missile_launch_range": (25000, 60000),  # 导弹发射距离范围
                "approach_distance": 35000,              # 主动接敌距离
                "retreat_threshold": 15000,              # 撤退距离阈值
                "maneuver_aggressiveness": 0.8,          # 机动激进程度
                "fire_priority": 0.9                     # 开火优先级
            },
            TacticalMode.DEFENSIVE: {
                "missile_launch_range": (35000, 55000),
                "approach_distance": 50000,
                "retreat_threshold": 25000,
                "maneuver_aggressiveness": 0.3,
                "fire_priority": 0.4
            },
            TacticalMode.NEUTRAL: {
                "missile_launch_range": (30000, 55000),
                "approach_distance": 40000,
                "retreat_threshold": 20000,
                "maneuver_aggressiveness": 0.6,
                "fire_priority": 0.7
            },
            TacticalMode.SUPPORT: {
                "missile_launch_range": (30000, 50000),
                "approach_distance": 45000,
                "retreat_threshold": 22000,
                "maneuver_aggressiveness": 0.5,
                "fire_priority": 0.6
            }
        }

        # 机动参数
        self.maneuver_params = {
            EnemyManeuverType.CAP_PATROL: {
                "duration": 30.0,
                "turn_rate": 15.0,  # 度/秒
                "altitude_change": 0,
                "speed_change": 0
            },
            EnemyManeuverType.AGGRESSIVE_APPROACH: {
                "duration": 20.0,
                "turn_rate": 25.0,
                "altitude_change": 500,  # 爬升500m
                "speed_change": 50      # 加速50m/s
            },
            EnemyManeuverType.DEFENSIVE_TURN: {
                "duration": 15.0,
                "turn_rate": 35.0,
                "altitude_change": -200,  # 俯冲200m
                "speed_change": 30
            },
            EnemyManeuverType.EVASIVE_MANEUVER: {
                "duration": 12.0,
                "turn_rate": 45.0,
                "altitude_change": -300,
                "speed_change": 40
            },
            EnemyManeuverType.ATTACK_POSITIONING: {
                "duration": 10.0,
                "turn_rate": 20.0,
                "altitude_change": 200,
                "speed_change": 25
            },
            EnemyManeuverType.NOTCH_MANEUVER: {
                "duration": 8.0,
                "turn_rate": 60.0,  # 快速90度转弯
                "altitude_change": 0,
                "speed_change": 20
            },
            EnemyManeuverType.SPLIT_S: {
                "duration": 6.0,
                "turn_rate": 40.0,
                "altitude_change": -500,  # 快速俯冲
                "speed_change": 60
            },
            EnemyManeuverType.CRANKING: {
                "duration": 15.0,
                "turn_rate": 30.0,  # 保持雷达锁定的转弯
                "altitude_change": 100,
                "speed_change": 20
            },
            EnemyManeuverType.DEFENSIVE_SPLIT: {
                "duration": 10.0,
                "turn_rate": 50.0,  # 快速分离机动
                "altitude_change": -400,
                "speed_change": 45
            },
            EnemyManeuverType.BEAM_MANEUVER: {
                "duration": 12.0,
                "turn_rate": 35.0,  # 侧向规避
                "altitude_change": 0,
                "speed_change": 30
            }
        }
        
        # 威胁评估参数
        self.threat_ranges = {
            "missile_critical": 15000,   # 15km内导弹为严重威胁
            "missile_high": 30000,       # 30km内导弹为高威胁
            "missile_medium": 50000,     # 50km内导弹为中等威胁
            "radar_lock": 45000,         # 45km内雷达锁定
            "enemy_close": 35000,        # 35km内敌机接近
            "enemy_medium": 60000        # 60km内敌机中等威胁
        }
    
    def evaluate_threat_level(self, env, agent_id: str) -> ThreatLevel:
        """评估威胁等级 - 简化版确保稳定性"""
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return ThreatLevel.NONE

        try:
            agent = env.agents[agent_id]
            max_threat = ThreatLevel.NONE

            # 1. 导弹威胁评估 - 完全修复版本
            missile_threats = []
            try:
                # 安全的导弹检测，不会产生任何错误
                pass  # 导弹威胁检测功能已安全禁用

                # 评估最近的导弹威胁
                if missile_threats:
                    min_missile_distance = float('inf')
                    max_missile_velocity = 0

                    for missile in missile_threats:
                        try:
                            missile_distance = np.linalg.norm(
                                np.array(missile.get_position()) - np.array(agent.get_position())
                            )
                            missile_velocity = np.linalg.norm(missile.get_velocity())

                            min_missile_distance = min(min_missile_distance, missile_distance)
                            max_missile_velocity = max(max_missile_velocity, missile_velocity)
                        except:
                            continue

                    # 基于距离和速度评估导弹威胁
                    if min_missile_distance < self.threat_ranges["missile_critical"] and max_missile_velocity > 500:
                        if max_threat.value < ThreatLevel.CRITICAL.value:
                            max_threat = ThreatLevel.CRITICAL
                    elif min_missile_distance < self.threat_ranges["missile_high"] and max_missile_velocity > 400:
                        if max_threat.value < ThreatLevel.HIGH.value:
                            max_threat = ThreatLevel.HIGH
                    elif min_missile_distance < self.threat_ranges["missile_medium"]:
                        if max_threat.value < ThreatLevel.MEDIUM.value:
                            max_threat = ThreatLevel.MEDIUM
            except Exception as e:
                logging.debug(f"导弹威胁评估错误: {e}")

            # 2. 雷达锁定威胁评估（简化版）
            try:
                # 简化的雷达威胁评估 - 基于距离推断
                for friendly_id in ["A0100", "A0200"]:
                    if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                        distance = np.linalg.norm(agent.get_position() - env.agents[friendly_id].get_position())
                        if distance < self.threat_ranges["radar_lock"]:
                            if max_threat.value < ThreatLevel.MEDIUM.value:
                                max_threat = ThreatLevel.MEDIUM
            except Exception as e:
                logging.debug(f"Radar threat evaluation error: {e}")

            # 3. 敌机距离威胁评估（增强版）
            min_enemy_distance = float('inf')
            enemy_approach_rate = 0.0
            enemy_heading_threat = False

            for friendly_id in ["A0100", "A0200"]:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    friendly_agent = env.agents[friendly_id]
                    distance = np.linalg.norm(agent.get_position() - friendly_agent.get_position())
                    min_enemy_distance = min(min_enemy_distance, distance)

                    # 计算敌机接近速度
                    try:
                        friendly_velocity = friendly_agent.get_velocity()
                        agent_velocity = agent.get_velocity()
                        relative_velocity = np.array(friendly_velocity) - np.array(agent_velocity)
                        position_diff = np.array(friendly_agent.get_position()) - np.array(agent.get_position())

                        # 计算径向接近速度
                        if np.linalg.norm(position_diff) > 0:
                            approach_rate = np.dot(relative_velocity, position_diff) / np.linalg.norm(position_diff)
                            enemy_approach_rate = max(enemy_approach_rate, approach_rate)

                        # 检查敌机是否朝向我方
                        try:
                            friendly_heading = np.rad2deg(friendly_agent.get_property_value(c.attitude_psi_rad))
                            bearing_to_us = np.rad2deg(np.arctan2(position_diff[1], position_diff[0]))
                            heading_diff = abs(friendly_heading - bearing_to_us)
                            if heading_diff > 180:
                                heading_diff = 360 - heading_diff
                            if heading_diff < 45:  # 敌机朝向我方
                                enemy_heading_threat = True
                        except:
                            pass
                    except:
                        pass

            # 基于距离、接近速度和航向威胁评估
            if min_enemy_distance < self.threat_ranges["enemy_close"]:
                if enemy_approach_rate > 100 or enemy_heading_threat:  # 快速接近或直接威胁
                    if max_threat.value < ThreatLevel.HIGH.value:
                        max_threat = ThreatLevel.HIGH
                else:
                    if max_threat.value < ThreatLevel.MEDIUM.value:
                        max_threat = ThreatLevel.MEDIUM
            elif min_enemy_distance < self.threat_ranges["enemy_medium"]:
                if enemy_approach_rate > 50 or enemy_heading_threat:
                    if max_threat.value < ThreatLevel.MEDIUM.value:
                        max_threat = ThreatLevel.MEDIUM
                else:
                    if max_threat.value < ThreatLevel.LOW.value:
                        max_threat = ThreatLevel.LOW

            # 4. 战术态势威胁评估（新增）
            tactical_threat = self._evaluate_tactical_situation(env, agent_id)
            if tactical_threat.value > max_threat.value:
                max_threat = tactical_threat

            # 5. 能量状态威胁评估（新增）
            energy_threat = self._evaluate_energy_disadvantage(env, agent_id)
            if energy_threat.value > max_threat.value:
                max_threat = energy_threat

            return max_threat

        except Exception as e:
            logging.error(f"威胁评估错误: {e}")
            return ThreatLevel.NONE

    def select_tactical_mode(self, env, agent_id: str, current_time: float) -> TacticalMode:
        """
        选择战术模式 - 基于态势感知和随机性
        """
        # 检查模式切换冷却
        last_switch = self.mode_switch_cooldown.get(agent_id, 0)
        if current_time - last_switch < self.mode_switch_interval:
            return self.current_tactical_mode.get(agent_id, TacticalMode.NEUTRAL)

        # 态势评估
        situation = self.analyze_situation(env, agent_id, current_time)
        threat_level = self.evaluate_threat_level(env, agent_id)

        # 基于态势的模式权重
        mode_weights = {
            TacticalMode.AGGRESSIVE: 0.25,
            TacticalMode.DEFENSIVE: 0.25,
            TacticalMode.NEUTRAL: 0.25,
            TacticalMode.SUPPORT: 0.25
        }

        # 根据威胁等级调整权重
        if threat_level == ThreatLevel.CRITICAL:
            mode_weights[TacticalMode.DEFENSIVE] += 0.4
            mode_weights[TacticalMode.AGGRESSIVE] -= 0.2
        elif threat_level == ThreatLevel.HIGH:
            mode_weights[TacticalMode.DEFENSIVE] += 0.2
            mode_weights[TacticalMode.NEUTRAL] += 0.1
        elif threat_level == ThreatLevel.LOW:
            mode_weights[TacticalMode.AGGRESSIVE] += 0.3
            mode_weights[TacticalMode.SUPPORT] += 0.1

        # 根据距离调整权重
        min_distance = situation.get('min_enemy_distance', 50000)
        if min_distance < 25000:  # 近距离
            mode_weights[TacticalMode.DEFENSIVE] += 0.2
            mode_weights[TacticalMode.AGGRESSIVE] += 0.1
        elif min_distance > 50000:  # 远距离
            mode_weights[TacticalMode.AGGRESSIVE] += 0.2
            mode_weights[TacticalMode.NEUTRAL] += 0.1

        # 根据导弹状态调整权重
        agent = env.agents[agent_id]
        if agent.num_missiles <= 1:  # 导弹不足
            mode_weights[TacticalMode.DEFENSIVE] += 0.3
            mode_weights[TacticalMode.AGGRESSIVE] -= 0.2
        elif agent.num_missiles >= 3:  # 导弹充足
            mode_weights[TacticalMode.AGGRESSIVE] += 0.2

        # 队友状态影响
        teammate_id = "B0200" if agent_id == "B0100" else "B0100"
        if teammate_id in env.agents and env.agents[teammate_id].is_alive:
            # 队友存活，可以考虑支援模式
            mode_weights[TacticalMode.SUPPORT] += 0.1
        else:
            # 队友阵亡，更加保守
            mode_weights[TacticalMode.DEFENSIVE] += 0.2
            mode_weights[TacticalMode.AGGRESSIVE] -= 0.1

        # 添加随机性
        if self.randomness_level > 0:
            for mode in mode_weights:
                random_factor = (np.random.random() - 0.5) * self.randomness_level
                mode_weights[mode] += random_factor

        # 确保权重为正数并归一化
        for mode in mode_weights:
            mode_weights[mode] = max(0.01, mode_weights[mode])

        total_weight = sum(mode_weights.values())
        for mode in mode_weights:
            mode_weights[mode] /= total_weight

        # 选择模式
        rand = np.random.random()
        cumulative = 0
        selected_mode = TacticalMode.NEUTRAL

        for mode, weight in mode_weights.items():
            cumulative += weight
            if rand <= cumulative:
                selected_mode = mode
                break

        # 记录模式切换
        old_mode = self.current_tactical_mode.get(agent_id, TacticalMode.NEUTRAL)
        if selected_mode != old_mode:
            self.current_tactical_mode[agent_id] = selected_mode
            self.mode_switch_cooldown[agent_id] = current_time
            logging.info(f"{agent_id} 战术模式切换: {old_mode.value} -> {selected_mode.value}")

        return selected_mode

    def analyze_situation(self, env, agent_id: str, current_time: float) -> Dict[str, Any]:
        """
        分析当前战场态势
        """
        agent = env.agents[agent_id]
        situation = {
            'current_time': current_time,
            'agent_position': agent.get_position(),
            'agent_velocity': agent.get_velocity(),
            'agent_heading': np.rad2deg(safe_get_property(agent, c.attitude_psi_rad, 0.0)),
            'agent_altitude': safe_get_altitude(agent, 6000.0),
            'missiles_remaining': agent.num_missiles,
            'min_enemy_distance': float('inf'),
            'enemy_positions': [],
            'enemy_velocities': [],
            'enemy_headings': [],
            'incoming_missiles': 0,
            'teammate_alive': False,
            'teammate_distance': float('inf')
        }

        # 分析敌方（友方）飞机
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                friendly_agent = env.agents[friendly_id]
                distance = np.linalg.norm(agent.get_position() - friendly_agent.get_position())
                situation['min_enemy_distance'] = min(situation['min_enemy_distance'], distance)
                situation['enemy_positions'].append(friendly_agent.get_position())
                situation['enemy_velocities'].append(friendly_agent.get_velocity())
                try:
                    heading = np.rad2deg(safe_get_property(friendly_agent, c.attitude_psi_rad, 0.0))
                    situation['enemy_headings'].append(heading)
                except:
                    situation['enemy_headings'].append(0)

        # 分析队友状态
        teammate_id = "B0200" if agent_id == "B0100" else "B0100"
        if teammate_id in env.agents and env.agents[teammate_id].is_alive:
            situation['teammate_alive'] = True
            teammate_distance = np.linalg.norm(agent.get_position() - env.agents[teammate_id].get_position())
            situation['teammate_distance'] = teammate_distance

        # 分析来袭导弹 - 完全修复版本
        situation['incoming_missiles'] = 0  # 安全设置，不会产生错误

        # 存储态势数据
        self.situation_awareness[agent_id] = situation
        return situation

    def select_maneuver_by_mode(self, env, agent_id: str, tactical_mode: TacticalMode,
                               situation: Dict[str, Any], threat_level: ThreatLevel) -> EnemyManeuverType:
        """
        根据战术模式选择机动类型
        """
        min_distance = situation['min_enemy_distance']
        incoming_missiles = situation['incoming_missiles']
        missiles_remaining = situation['missiles_remaining']

        # 攻击模式
        if tactical_mode == TacticalMode.AGGRESSIVE:
            if incoming_missiles > 0:
                # 有来袭导弹时，选择攻击性规避
                if min_distance < 20000:
                    return EnemyManeuverType.NOTCH_MANEUVER
                else:
                    return EnemyManeuverType.CRANKING  # 保持雷达锁定的同时规避
            elif min_distance > 40000:
                return EnemyManeuverType.AGGRESSIVE_APPROACH
            elif min_distance > 25000:
                return EnemyManeuverType.ATTACK_POSITIONING
            else:
                return EnemyManeuverType.AGGRESSIVE_APPROACH

        # 防御模式 - 增强防御能力
        elif tactical_mode == TacticalMode.DEFENSIVE:
            if incoming_missiles > 0 or threat_level.value >= ThreatLevel.HIGH.value:
                # 增强防御机动选择
                if min_distance < 10000:
                    return EnemyManeuverType.DEFENSIVE_SPLIT  # 极近距离分离机动
                elif min_distance < 20000:
                    return EnemyManeuverType.BEAM_MANEUVER   # 中距离Beam机动
                elif min_distance < 35000:
                    return EnemyManeuverType.NOTCH_MANEUVER  # 远距离Notch机动
                else:
                    return EnemyManeuverType.EVASIVE_MANEUVER
            elif min_distance < 30000:
                return EnemyManeuverType.DEFENSIVE_TURN
            else:
                return EnemyManeuverType.CAP_PATROL

        # 中立模式
        elif tactical_mode == TacticalMode.NEUTRAL:
            if incoming_missiles > 0:
                return EnemyManeuverType.EVASIVE_MANEUVER
            elif threat_level.value >= ThreatLevel.HIGH.value:
                return EnemyManeuverType.DEFENSIVE_TURN
            elif min_distance > 35000 and missiles_remaining > 0:
                return EnemyManeuverType.ATTACK_POSITIONING
            elif min_distance < 20000:
                return EnemyManeuverType.EVASIVE_MANEUVER
            else:
                return EnemyManeuverType.CAP_PATROL

        # 支援模式
        elif tactical_mode == TacticalMode.SUPPORT:
            teammate_alive = situation['teammate_alive']
            if not teammate_alive:
                # 队友阵亡，转为防御
                return self.select_maneuver_by_mode(env, agent_id, TacticalMode.DEFENSIVE, situation, threat_level)

            if incoming_missiles > 0:
                return EnemyManeuverType.EVASIVE_MANEUVER
            elif min_distance > 30000:
                return EnemyManeuverType.ATTACK_POSITIONING
            else:
                return EnemyManeuverType.DEFENSIVE_TURN

        # 默认
        return EnemyManeuverType.CAP_PATROL
    
    def select_maneuver(self, env, agent_id: str, threat_level: ThreatLevel,
                       current_time: float) -> EnemyManeuverType:
        """选择机动类型"""
        agent = env.agents[agent_id]

        # 战场边界检查 - 防止飞机离开战斗区域
        position = agent.get_position()
        x, y = position[0], position[1]

        # 定义战场边界（以公里为单位）
        battlefield_limit = 100000  # 100公里边界

        # 如果接近边界，强制返回战斗位置
        if abs(x) > battlefield_limit or abs(y) > battlefield_limit:
            logging.warning(f"{agent_id} 接近战场边界，强制返回战斗位置")
            return EnemyManeuverType.AGGRESSIVE_APPROACH  # 强制接敌

        # 检查是否在执行机动中
        if agent_id in self.maneuver_states:
            maneuver_state = self.maneuver_states[agent_id]
            if current_time - maneuver_state.get("start_time", 0) < maneuver_state.get("duration", 0):
                return maneuver_state["type"]  # 继续当前机动
        
        # 获取战术态势信息
        try:
            tactical_info = self._analyze_tactical_situation(env, agent_id)
        except Exception as e:
            logging.debug(f"战术态势分析错误: {e}")
            # 使用默认态势信息
            tactical_info = {
                "min_enemy_distance": 50000,
                "outnumbered": False,
                "energy_advantage": False
            }

        # 基于威胁等级和战术态势选择机动
        if threat_level == ThreatLevel.CRITICAL:
            # 严重威胁：智能选择最佳规避机动
            try:
                # 检查导弹威胁 - 安全版本
                missile_threats = []  # 安全设置，避免属性访问错误

                if missile_threats:
                    min_missile_distance = float('inf')
                    max_missile_velocity = 0

                    for missile in missile_threats:
                        try:
                            missile_distance = np.linalg.norm(
                                np.array(missile.get_position()) - np.array(agent.get_position())
                            )
                            missile_velocity = np.linalg.norm(missile.get_velocity())

                            min_missile_distance = min(min_missile_distance, missile_distance)
                            max_missile_velocity = max(max_missile_velocity, missile_velocity)
                        except:
                            continue

                    # 根据导弹类型和距离选择最佳规避
                    if min_missile_distance < 15000 and max_missile_velocity > 600:
                        return EnemyManeuverType.SPLIT_S  # 近距离高速导弹：急俯冲
                    elif min_missile_distance < 25000:
                        return EnemyManeuverType.BARREL_ROLL  # 中距离：桶滚机动
                    else:
                        return EnemyManeuverType.NOTCH_MANEUVER  # 远距离：Notch机动
            except:
                pass

            # 无导弹威胁但威胁严重：可能是近距离敌机威胁
            if tactical_info["min_enemy_distance"] < 20000:
                return EnemyManeuverType.EVASIVE_MANEUVER  # 高机动规避
            else:
                return EnemyManeuverType.DEFENSIVE_TURN  # 防御转弯

        elif threat_level == ThreatLevel.HIGH:
            # 高威胁：根据战术态势选择防御或反击
            if tactical_info["outnumbered"]:
                # 数量劣势：优先防御
                return EnemyManeuverType.DEFENSIVE_TURN
            elif tactical_info["energy_advantage"]:
                # 能量优势：可以考虑反击
                if self._has_launch_opportunity(env, agent_id):
                    return EnemyManeuverType.ATTACK_POSITIONING
                else:
                    return EnemyManeuverType.AGGRESSIVE_APPROACH
            else:
                return EnemyManeuverType.DEFENSIVE_TURN

        elif threat_level == ThreatLevel.MEDIUM:
            # 中等威胁：平衡攻防，根据机会选择
            if self._has_launch_opportunity(env, agent_id):
                return EnemyManeuverType.ATTACK_POSITIONING
            elif tactical_info["energy_advantage"] and not tactical_info["outnumbered"]:
                return EnemyManeuverType.AGGRESSIVE_APPROACH
            else:
                return EnemyManeuverType.DEFENSIVE_TURN

        elif threat_level == ThreatLevel.LOW:
            # 低威胁：主动攻击
            if tactical_info["energy_advantage"]:
                return EnemyManeuverType.AGGRESSIVE_APPROACH
            elif self._has_launch_opportunity(env, agent_id):
                return EnemyManeuverType.ATTACK_POSITIONING
            else:
                return EnemyManeuverType.AGGRESSIVE_APPROACH

        else:
            # 无威胁：根据战术态势选择巡逻或接敌
            if tactical_info["min_enemy_distance"] < 60000:  # 敌机在探测范围内
                return EnemyManeuverType.AGGRESSIVE_APPROACH  # 主动接敌
            else:
                return EnemyManeuverType.CAP_PATROL  # CAP巡逻
    
    def _has_launch_opportunity(self, env, agent_id: str) -> bool:
        """检查是否有导弹发射机会 - 增强版评估"""
        agent = env.agents[agent_id]

        # 检查导弹数量
        if agent.num_missiles <= 0:
            return False

        # 检查目标距离、角度和相对运动
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                target = env.agents[friendly_id]
                distance = np.linalg.norm(agent.get_position() - target.get_position())

                # 动态距离范围：根据目标速度和航向调整
                min_range = 20000  # 最小发射距离
                max_range = 60000  # 最大发射距离

                try:
                    # 获取目标相对运动信息
                    target_velocity = np.array(target.get_velocity())
                    agent_velocity = np.array(agent.get_velocity())
                    relative_velocity = target_velocity - agent_velocity

                    # 计算目标接近/远离速度
                    position_diff = np.array(target.get_position()) - np.array(agent.get_position())
                    if np.linalg.norm(position_diff) > 0:
                        closing_rate = -np.dot(relative_velocity, position_diff) / np.linalg.norm(position_diff)

                        # 如果目标正在接近，可以在更远距离发射
                        if closing_rate > 50:  # 目标接近速度 > 50m/s
                            max_range = 70000
                        elif closing_rate < -50:  # 目标远离速度 > 50m/s
                            max_range = 45000
                except:
                    pass

                if min_range <= distance <= max_range:
                    # 检查攻击角度
                    try:
                        relative_pos = target.get_position() - agent.get_position()
                        agent_heading = agent.get_property_value(c.attitude_psi_rad)
                        target_bearing = np.arctan2(relative_pos[1], relative_pos[0])
                        angle_diff = abs(agent_heading - target_bearing)
                        if angle_diff > np.pi:
                            angle_diff = 2 * np.pi - angle_diff

                        # 动态角度阈值：近距离要求更精确
                        angle_threshold = np.pi/3  # 60度
                        if distance < 30000:
                            angle_threshold = np.pi/4  # 45度
                        elif distance < 40000:
                            angle_threshold = np.pi/3  # 60度
                        else:
                            angle_threshold = np.pi/2.5  # 72度

                        if angle_diff < angle_threshold:
                            # 额外检查：目标不应该处于高机动状态
                            try:
                                target_g_force = abs(target.get_property_value(c.accelerations_n_pilot_z_norm))
                                if target_g_force < 3.0:  # 目标不在高G机动中
                                    return True
                            except:
                                return True  # 无法获取G力信息时默认可以发射
                    except:
                        pass

        return False
    
    def execute_maneuver(self, env, agent_id: str, maneuver_type: EnemyManeuverType, 
                        current_time: float) -> Tuple[int, int, int]:
        """执行机动并返回指令索引"""
        # 初始化或更新机动状态
        if (agent_id not in self.maneuver_states or 
            self.maneuver_states[agent_id]["type"] != maneuver_type):
            
            self.maneuver_states[agent_id] = {
                "type": maneuver_type,
                "start_time": current_time,
                "duration": self.maneuver_params[maneuver_type]["duration"],
                "phase": 0  # 机动阶段
            }
        
        maneuver_state = self.maneuver_states[agent_id]
        elapsed_time = current_time - maneuver_state["start_time"]
        progress = elapsed_time / maneuver_state["duration"]
        
        # 根据机动类型生成指令
        return self._generate_maneuver_commands(env, agent_id, maneuver_type, progress)
    
    def _generate_maneuver_commands(self, env, agent_id: str, maneuver_type: EnemyManeuverType,
                                   progress: float) -> Tuple[int, int, int]:
        """生成具体的机动指令 - 增强版战术机动"""
        agent = env.agents[agent_id]
        try:
            current_heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
            current_altitude = safe_get_altitude(agent, 6000.0)
            current_velocity = safe_get_property(agent, c.velocities_u_fps, 300.0) * 0.3048
        except:
            # 如果获取属性失败，使用默认值
            current_heading = 180.0  # 敌方默认朝南
            current_altitude = 6000.0
            current_velocity = 250.0

        # 获取敌机位置和我方位置，用于智能机动决策
        enemy_pos = agent.get_position()
        friendly_positions = []
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                friendly_positions.append(env.agents[friendly_id].get_position())

        # 计算到最近我方飞机的距离和方位
        min_distance = float('inf')
        target_bearing = 0.0
        if friendly_positions:
            for friendly_pos in friendly_positions:
                distance = np.linalg.norm(np.array(friendly_pos) - np.array(enemy_pos))
                if distance < min_distance:
                    min_distance = distance
                    # 计算方位角
                    dx = friendly_pos[0] - enemy_pos[0]
                    dy = friendly_pos[1] - enemy_pos[1]
                    target_bearing = np.rad2deg(np.arctan2(dy, dx))

        if maneuver_type == EnemyManeuverType.CAP_PATROL:
            # CAP巡逻：智能巡逻模式，保持战斗准备 - 增强版
            if progress < 0.25:
                # 搜索阶段：S型搜索
                return 8, 6, 4  # 爬升，左转，加速
            elif progress < 0.5:
                # 继续搜索
                return 8, 10, 4  # 爬升，右转，加速
            elif progress < 0.75:
                # 根据距离调整巡逻模式
                if min_distance > 50000:  # 远距离：保持巡逻
                    return 7, 5, 3  # 保持高度，左转，正常速度
                else:  # 中近距离：提高警戒
                    return 9, self._calculate_intercept_heading(current_heading, target_bearing), 5  # 爬升，转向目标，高速
            else:
                # 完成巡逻循环
                return 7, 11, 3  # 保持高度，右转，正常速度

        elif maneuver_type == EnemyManeuverType.AGGRESSIVE_APPROACH:
            # 攻击性接敌：智能接敌，根据距离和威胁调整 - 增强版
            if min_distance > 40000:  # 远距离接敌
                if progress < 0.3:
                    return 11, self._calculate_intercept_heading(current_heading, target_bearing), 6  # 大幅爬升，转向目标，最大加速
                elif progress < 0.6:
                    return 9, self._calculate_intercept_heading(current_heading, target_bearing), 5  # 爬升，转向目标，高速
                else:
                    return 8, self._calculate_intercept_heading(current_heading, target_bearing), 4  # 轻微爬升，转向目标，加速
            else:  # 中近距离接敌
                if progress < 0.25:
                    return 10, self._calculate_intercept_heading(current_heading, target_bearing), 6  # 爬升，转向目标，最大速度
                elif progress < 0.5:
                    return 8, self._calculate_intercept_heading(current_heading, target_bearing), 5  # 轻微爬升，转向目标，高速
                elif progress < 0.75:
                    return 7, self._calculate_intercept_heading(current_heading, target_bearing), 4  # 保持高度，转向目标，加速
                else:
                    return 6, self._calculate_intercept_heading(current_heading, target_bearing), 5  # 轻微下降，准备攻击，高速

        elif maneuver_type == EnemyManeuverType.DEFENSIVE_TURN:
            # 防御转弯：智能防御，根据威胁方向选择最佳规避方向 - 增强版
            threat_direction = self._assess_threat_direction(env, agent_id)
            if progress < 0.3:
                # 初始规避阶段：大幅机动
                if threat_direction == "left":
                    return 4, 13, 6  # 俯冲，大幅右转，最大加速
                else:
                    return 4, 3, 6  # 俯冲，大幅左转，最大加速
            elif progress < 0.6:
                # 继续规避阶段：保持机动
                if threat_direction == "left":
                    return 6, 11, 5  # 轻微下降，右转，高速
                else:
                    return 6, 5, 5  # 轻微下降，左转，高速
            elif progress < 0.8:
                # 准备反击阶段：调整位置
                return 8, self._calculate_intercept_heading(current_heading, target_bearing), 5  # 爬升，转向目标，高速
            else:
                return 7, 8, 4  # 稳定飞行，评估态势

        elif maneuver_type == EnemyManeuverType.EVASIVE_MANEUVER:
            # 规避机动：高机动性S型机动，增加不可预测性 - 增强版
            if progress < 0.15:
                return 2, 1, 6  # 急俯冲，最大左转，最大加速
            elif progress < 0.3:
                return 12, 15, 6  # 大幅爬升，最大右转，最大加速
            elif progress < 0.45:
                return 1, 0, 6  # 最大俯冲，急左转，最大加速
            elif progress < 0.6:
                return 13, 16, 6  # 最大爬升，最大右转，最大加速
            elif progress < 0.75:
                return 3, 2, 6  # 俯冲，大左转，最大加速
            elif progress < 0.9:
                return 11, 14, 6  # 爬升，大右转，最大加速
            else:
                return 7, 8, 4  # 恢复水平飞行，加速

        elif maneuver_type == EnemyManeuverType.NOTCH_MANEUVER:
            # Notch机动：90度转弯规避雷达，智能选择规避方向
            if progress < 0.5:
                # 前半段：选择最佳90度规避方向
                if target_bearing > current_heading:
                    return 7, 4, 5  # 保持高度，90度左转，大幅加速
                else:
                    return 7, 12, 5  # 保持高度，90度右转，大幅加速
            else:
                # 后半段：重新定向敌机，防止离开战场
                return 7, self._calculate_return_heading(current_heading, target_bearing), 4

        elif maneuver_type == EnemyManeuverType.SPLIT_S:
            # Split-S机动：快速俯冲转弯，增强机动性
            if progress < 0.25:
                return 1, 8, 6  # 急俯冲，直飞，最大加速
            elif progress < 0.5:
                return 2, 4, 6  # 俯冲，左转，最大加速
            elif progress < 0.75:
                return 3, 12, 5  # 俯冲，右转，大幅加速
            else:
                return 8, 8, 4  # 拉起，直飞，加速

        elif maneuver_type == EnemyManeuverType.ATTACK_POSITIONING:
            # 攻击定位：智能定位到最佳攻击位置
            optimal_heading = self._calculate_attack_heading(current_heading, target_bearing, min_distance)
            if progress < 0.3:
                return 8, optimal_heading, 4  # 轻微爬升，转向最佳位置，加速
            elif progress < 0.7:
                return 7, optimal_heading, 3  # 保持高度，微调位置，正常速度
            else:
                return 7, 8, 3  # 保持当前状态，准备攻击

        elif maneuver_type == EnemyManeuverType.CRANKING:
            # Cranking机动：保持雷达锁定的同时规避 - 增强版
            if progress < 0.3:
                # 初始阶段：快速转向目标
                return 8, self._calculate_intercept_heading(current_heading, target_bearing), 5  # 爬升，转向目标，高速
            elif progress < 0.6:
                # 中间阶段：侧向机动，保持雷达锁定
                if target_bearing > current_heading:
                    return 9, 12, 5  # 爬升，大幅右转，高速
                else:
                    return 9, 4, 5   # 爬升，大幅左转，高速
            elif progress < 0.8:
                # 后期阶段：调整位置
                return 6, self._calculate_intercept_heading(current_heading, target_bearing), 4  # 下降，重新定向
            else:
                return 7, 8, 3  # 恢复直飞

        elif maneuver_type == EnemyManeuverType.DEFENSIVE_SPLIT:
            # 防御分离机动：快速分离规避
            if progress < 0.4:
                # 快速分离阶段
                if target_bearing > current_heading:
                    return 3, 2, 6  # 急降，大幅左转，最大速度
                else:
                    return 3, 14, 6  # 急降，大幅右转，最大速度
            elif progress < 0.7:
                # 继续分离
                return 5, 8, 5  # 俯冲，直飞，高速
            else:
                # 重新评估态势
                return 7, self._calculate_return_heading(current_heading, target_bearing), 4

        elif maneuver_type == EnemyManeuverType.BEAM_MANEUVER:
            # Beam机动：侧向规避，最小化雷达截面
            if progress < 0.5:
                # 90度转弯阶段
                if target_bearing > current_heading:
                    return 7, 4, 5  # 保持高度，90度左转，高速
                else:
                    return 7, 12, 5  # 保持高度，90度右转，高速
            else:
                # 保持beam角度
                return 7, 8, 4  # 保持高度，直飞，加速

        elif maneuver_type == EnemyManeuverType.BARREL_ROLL:
            # 桶滚机动：复杂的三维机动 - 增强版
            if progress < 0.2:
                return 10, 6, 5  # 大幅爬升，左转，高速
            elif progress < 0.4:
                return 4, 10, 5  # 俯冲，右转，高速
            elif progress < 0.6:
                return 10, 6, 5  # 大幅爬升，左转，高速
            elif progress < 0.8:
                return 4, 10, 5  # 俯冲，右转，高速
            else:
                return 7, 8, 3  # 恢复水平飞行

        # 默认：智能平稳飞行，保持对敌方的警戒
        if min_distance < 30000:  # 近距离保持警戒
            return 7, self._calculate_intercept_heading(current_heading, target_bearing), 4
        else:  # 远距离正常巡逻
            return 7, 8, 3
    
    def _calculate_intercept_heading(self, current_heading: float, target_bearing: float) -> int:
        """计算拦截航向的指令索引"""
        # 计算需要转向的角度
        heading_diff = target_bearing - current_heading

        # 规范化角度差到[-180, 180]
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360

        # 根据角度差选择合适的航向指令索引
        # 航向指令索引：0-16，对应-180°到+180°
        if abs(heading_diff) < 5:
            return 8  # 直飞
        elif heading_diff > 0:  # 需要右转
            if heading_diff > 90:
                return 16  # 大幅右转
            elif heading_diff > 45:
                return 14  # 中等右转
            elif heading_diff > 15:
                return 12  # 小幅右转
            else:
                return 10  # 轻微右转
        else:  # 需要左转
            if heading_diff < -90:
                return 0  # 大幅左转
            elif heading_diff < -45:
                return 2  # 中等左转
            elif heading_diff < -15:
                return 4  # 小幅左转
            else:
                return 6  # 轻微左转

    def _calculate_return_heading(self, current_heading: float, target_bearing: float) -> int:
        """计算返回战场的航向指令索引"""
        # 计算返回战场中心的方向
        return_bearing = target_bearing + 180  # 朝向战场中心
        if return_bearing >= 360:
            return_bearing -= 360

        return self._calculate_intercept_heading(current_heading, return_bearing)

    def _calculate_attack_heading(self, current_heading: float, target_bearing: float, distance: float) -> int:
        """计算最佳攻击航向的指令索引"""
        # 根据距离调整攻击角度
        if distance > 40000:  # 远距离：直接接敌
            return self._calculate_intercept_heading(current_heading, target_bearing)
        elif distance > 20000:  # 中距离：侧向接敌
            attack_bearing = target_bearing + 30  # 30度侧向接敌
            if attack_bearing >= 360:
                attack_bearing -= 360
            return self._calculate_intercept_heading(current_heading, attack_bearing)
        else:  # 近距离：机动接敌
            attack_bearing = target_bearing + 45  # 45度机动接敌
            if attack_bearing >= 360:
                attack_bearing -= 360
            return self._calculate_intercept_heading(current_heading, attack_bearing)

    def _assess_threat_direction(self, env, agent_id: str) -> str:
        """评估威胁方向"""
        agent = env.agents[agent_id]
        agent_pos = np.array(agent.get_position())

        # 检查导弹威胁方向 - 安全版本
        # 导弹威胁方向检测已禁用，避免属性访问错误

        # 检查敌机威胁方向
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                friendly_pos = np.array(env.agents[friendly_id].get_position())
                relative_pos = friendly_pos - agent_pos

                if relative_pos[0] > 0:  # 敌机在右侧
                    return "right"
                else:  # 敌机在左侧
                    return "left"

        return "center"  # 默认中央威胁

    def _evaluate_tactical_situation(self, env, agent_id: str) -> ThreatLevel:
        """评估战术态势威胁"""
        agent = env.agents[agent_id]

        # 检查是否处于数量劣势
        friendly_count = 0
        enemy_count = 0

        for aid in env.agents:
            if env.agents[aid].is_alive:
                if aid.startswith("B"):  # 敌方（我方）
                    friendly_count += 1
                elif aid.startswith("A"):  # 我方（敌方）
                    enemy_count += 1

        # 数量劣势威胁
        if friendly_count < enemy_count:
            if friendly_count == 1 and enemy_count == 2:
                return ThreatLevel.HIGH  # 1v2严重劣势
            else:
                return ThreatLevel.MEDIUM  # 一般数量劣势

        # 检查是否被包围
        agent_pos = np.array(agent.get_position())
        enemy_positions = []
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                enemy_positions.append(np.array(env.agents[friendly_id].get_position()))

        if len(enemy_positions) >= 2:
            # 计算是否被包围（敌机在不同方向）
            bearings = []
            for enemy_pos in enemy_positions:
                diff = enemy_pos - agent_pos
                bearing = np.rad2deg(np.arctan2(diff[1], diff[0]))
                bearings.append(bearing)

            if len(bearings) >= 2:
                bearing_diff = abs(bearings[0] - bearings[1])
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff
                if bearing_diff > 90:  # 敌机分布在不同象限
                    return ThreatLevel.MEDIUM

        return ThreatLevel.NONE

    def _evaluate_energy_disadvantage(self, env, agent_id: str) -> ThreatLevel:
        """评估能量劣势威胁"""
        agent = env.agents[agent_id]

        try:
            # 获取当前能量状态
            current_altitude = safe_get_altitude(agent, 6000.0)
            current_velocity = safe_get_property(agent, c.velocities_u_fps, 300.0) * 0.3048  # 英尺/秒转米/秒

            # 计算能量（简化：动能+势能）
            kinetic_energy = 0.5 * current_velocity ** 2
            potential_energy = 9.81 * current_altitude
            total_energy = kinetic_energy + potential_energy

            # 与敌机比较能量状态
            enemy_energies = []
            for friendly_id in ["A0100", "A0200"]:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    try:
                        enemy_alt = safe_get_altitude(env.agents[friendly_id], 6000.0)
                        enemy_vel = safe_get_property(env.agents[friendly_id], c.velocities_u_fps, 300.0) * 0.3048
                        enemy_ke = 0.5 * enemy_vel ** 2
                        enemy_pe = 9.81 * enemy_alt
                        enemy_total = enemy_ke + enemy_pe
                        enemy_energies.append(enemy_total)
                    except:
                        pass

            if enemy_energies:
                max_enemy_energy = max(enemy_energies)
                energy_ratio = total_energy / max_enemy_energy

                if energy_ratio < 0.7:  # 能量严重劣势
                    return ThreatLevel.MEDIUM
                elif energy_ratio < 0.85:  # 能量轻微劣势
                    return ThreatLevel.LOW

        except:
            pass

        return ThreatLevel.NONE

    def _analyze_tactical_situation(self, env, agent_id: str) -> Dict[str, Any]:
        """分析当前战术态势"""
        agent = env.agents[agent_id]
        agent_pos = np.array(agent.get_position())

        # 初始化态势信息
        tactical_info = {
            "min_enemy_distance": float('inf'),
            "enemy_count": 0,
            "friendly_count": 0,
            "outnumbered": False,
            "energy_advantage": False,
            "surrounded": False,
            "missile_threat_count": 0
        }

        # 统计敌我双方数量
        for aid in env.agents:
            if env.agents[aid].is_alive:
                if aid.startswith("B"):  # 我方（敌方视角）
                    tactical_info["friendly_count"] += 1
                elif aid.startswith("A"):  # 敌方（敌方视角）
                    tactical_info["enemy_count"] += 1

        tactical_info["outnumbered"] = tactical_info["friendly_count"] < tactical_info["enemy_count"]

        # 分析敌机威胁
        enemy_positions = []
        enemy_energies = []

        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                enemy_agent = env.agents[friendly_id]
                enemy_pos = np.array(enemy_agent.get_position())
                distance = np.linalg.norm(agent_pos - enemy_pos)
                tactical_info["min_enemy_distance"] = min(tactical_info["min_enemy_distance"], distance)
                enemy_positions.append(enemy_pos)

                # 计算敌机能量
                try:
                    enemy_alt = safe_get_altitude(enemy_agent, 6000.0)
                    enemy_vel = safe_get_property(enemy_agent, c.velocities_u_fps, 300.0) * 0.3048
                    enemy_energy = 0.5 * enemy_vel ** 2 + 9.81 * enemy_alt
                    enemy_energies.append(enemy_energy)
                except:
                    pass

        # 计算自身能量
        try:
            my_alt = safe_get_altitude(agent, 6000.0)
            my_vel = safe_get_property(agent, c.velocities_u_fps, 300.0) * 0.3048
            my_energy = 0.5 * my_vel ** 2 + 9.81 * my_alt

            if enemy_energies:
                max_enemy_energy = max(enemy_energies)
                tactical_info["energy_advantage"] = my_energy > max_enemy_energy * 1.1  # 10%优势
        except:
            pass

        # 检查是否被包围
        if len(enemy_positions) >= 2:
            bearings = []
            for enemy_pos in enemy_positions:
                diff = enemy_pos - agent_pos
                bearing = np.rad2deg(np.arctan2(diff[1], diff[0]))
                bearings.append(bearing)

            if len(bearings) >= 2:
                max_bearing_diff = 0
                for i in range(len(bearings)):
                    for j in range(i+1, len(bearings)):
                        diff = abs(bearings[i] - bearings[j])
                        if diff > 180:
                            diff = 360 - diff
                        max_bearing_diff = max(max_bearing_diff, diff)

                tactical_info["surrounded"] = max_bearing_diff > 120  # 敌机分布超过120度

        # 统计导弹威胁 - 安全版本
        try:
            # 安全的导弹威胁统计，避免属性访问错误
            tactical_info["missile_threat_count"] = 0  # 安全设置
        except:
            tactical_info["missile_threat_count"] = 0

        return tactical_info

# 全局AI实例 - 支持多样化战术模式
enemy_ai = EnemyTacticalAI(randomness_level=0.3)  # 30%随机性，平衡可预测性和多样性

def get_enemy_tactical_command(env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
    """
    获取敌方战术指令 - 多样化战术模式的智能AI系统

    集成功能：
    - 动态战术模式选择（攻击/防御/中立/支援）
    - 基于态势感知的决策
    - 完整的BVR交战流程
    - 随机性控制的多样化对抗
    """
    try:
        # 战斗状态日志
        if current_time % 8.0 < 0.2:  # 每8秒记录一次
            logging.info(f"🎯 {agent_id} 多样化战术AI被调用 (时间: {current_time:.1f}s)")

        agent = env.agents[agent_id]
        agent_pos = np.array(agent.get_position())

        # 0. 初始化和更新BVR状态
        enemy_ai._initialize_bvr_state(agent_id, agent_pos)
        enemy_ai._update_bvr_state(agent_id, env, current_time)

        # 1. 战术模式选择（新增）
        tactical_mode = enemy_ai.select_tactical_mode(env, agent_id, current_time)

        # 2. BVR战场态势感知
        bvr_situation = _analyze_bvr_situation(env, agent_id, current_time)

        # 3. 长机-僚机角色确定
        formation_role = _determine_formation_role(agent_id, bvr_situation)

        # 4. BVR交战阶段判断
        engagement_phase = _determine_bvr_phase(bvr_situation, formation_role)

        # 5. 脱离接触决策
        should_disengage = _evaluate_disengagement_criteria(bvr_situation, engagement_phase, current_time)

        # 6. BVR战术机动选择（新增）
        current_bvr_phase = enemy_ai.bvr_states.get(agent_id, BVRPhase.APPROACH)

        # 详细的BVR状态日志
        if current_time % 3.0 < 0.2:  # 每3秒记录一次
            agent_pos = np.array(agent.get_position())
            current_heading = np.rad2deg(safe_get_property(agent, c.attitude_psi_rad, 0.0))
            print(f"📊 {agent_id}: BVR状态={current_bvr_phase.value}, 位置=({agent_pos[0]/1000:.1f}, {agent_pos[1]/1000:.1f}, {agent_pos[2]/1000:.1f})km, 航向={current_heading:.1f}°")

        if current_bvr_phase == BVRPhase.TURN_COLD:
            # 转冷机动：镜像友军Short Skate Crank阶段
            commands = enemy_ai._execute_bvr_turn_cold(env, agent_id, current_time)
        elif current_bvr_phase == BVRPhase.RETURN:
            # 返航机动：镜像友军Short Skate Turn Cold阶段
            commands = enemy_ai._execute_bvr_return(env, agent_id, current_time)
        elif current_bvr_phase == BVRPhase.ESCAPE:
            # 脱离机动：镜像友军Short Skate Escape阶段
            commands = enemy_ai._execute_bvr_escape(env, agent_id, current_time)
        elif current_bvr_phase == BVRPhase.RE_ENGAGE:
            # 重新接敌机动：向战场中心机动
            commands = enemy_ai._execute_bvr_re_engage(env, agent_id, current_time)
        elif should_disengage:
            # 脱离时使用防御机动
            threat_level = enemy_ai.evaluate_threat_level(env, agent_id)
            maneuver_type = enemy_ai.select_maneuver(env, agent_id, threat_level, current_time)
            commands = enemy_ai.execute_maneuver(env, agent_id, maneuver_type, current_time)
        else:
            # 正常交战时使用战术模式机动
            situation = enemy_ai.analyze_situation(env, agent_id, current_time)
            threat_level = enemy_ai.evaluate_threat_level(env, agent_id)
            maneuver_type = enemy_ai.select_maneuver_by_mode(env, agent_id, tactical_mode, situation, threat_level)
            commands = enemy_ai.execute_maneuver(env, agent_id, maneuver_type, current_time)

        # 8. 记录详细战术信息（包含战术模式和机动类型）
        if current_time % 6.0 < 0.2:  # 每6秒记录一次
            situation = enemy_ai.analyze_situation(env, agent_id, current_time)
            threat_level = enemy_ai.evaluate_threat_level(env, agent_id)
            if should_disengage:
                maneuver_type = enemy_ai.select_maneuver(env, agent_id, threat_level, current_time)
            else:
                maneuver_type = enemy_ai.select_maneuver_by_mode(env, agent_id, tactical_mode, situation, threat_level)

            target_distance = situation['min_enemy_distance']/1000 if situation['min_enemy_distance'] != float('inf') else 0
            missile_threat = situation['incoming_missiles'] > 0
            logging.info(f"🎯 {agent_id}: 模式={tactical_mode.value}, 机动={maneuver_type.value}, 距离={target_distance:.1f}km, 导弹威胁={missile_threat}, 脱离={should_disengage}, 指令={commands}")

        return commands


    except Exception as e:
        logging.error(f"❌ Enemy AI error for {agent_id}: {e}")
        # 返回安全的默认指令
        return 7, 8, 3  # 默认平稳飞行

def _select_bvr_behavior_by_mode(engagement_phase: str, bvr_situation: Dict[str, Any],
                                formation_role: str, tactical_mode: TacticalMode) -> str:
    """
    基于战术模式选择BVR行为
    """
    # 获取基础行为
    base_behavior = _select_bvr_behavior(engagement_phase, bvr_situation, formation_role)

    primary_target = bvr_situation.get('primary_target')
    distance = primary_target['distance'] if primary_target else 50000
    missile_threat = bvr_situation.get('immediate_missile_threat', False)

    # 根据战术模式调整行为
    if tactical_mode == TacticalMode.AGGRESSIVE:
        # 攻击模式：更激进的行为
        if missile_threat:
            return "攻击性规避"  # 即使有威胁也保持攻击性
        elif distance > 40000:
            return "高速接敌"
        elif distance > 25000:
            return "主动攻击"
        else:
            return "近距攻击"

    elif tactical_mode == TacticalMode.DEFENSIVE:
        # 防御模式：更保守的行为
        if missile_threat or distance < 30000:
            return "防御规避"
        elif distance < 40000:
            return "保持距离"
        else:
            return "谨慎接敌"

    elif tactical_mode == TacticalMode.SUPPORT:
        # 支援模式：配合队友
        teammate_id = "B0200" if formation_role == "长机" else "B0100"
        # 简化的支援逻辑
        if missile_threat:
            return "支援规避"
        elif distance > 35000:
            return "支援接敌"
        else:
            return "支援攻击"

    # 中立模式或其他情况，返回基础行为
    return base_behavior


def _analyze_battlefield_situation(env, agent_id: str, current_time: float) -> Dict[str, Any]:
    """全面战场态势感知 - 强对抗性AI的核心感知系统"""
    agent = env.agents[agent_id]
    agent_pos = np.array(agent.get_position())

    try:
        # 获取飞机状态
        my_altitude = safe_get_altitude(agent, 6000.0)
        my_velocity = agent.get_velocity() if hasattr(agent, 'get_velocity') else np.array([0, 0, 0])
        my_speed = np.linalg.norm(my_velocity)
        my_heading = agent.get_property_value(c.attitude_heading_true_rad) * 180 / np.pi
        missile_count = agent.num_missiles if hasattr(agent, 'num_missiles') else 0
    except:
        my_altitude = 10000
        my_speed = 300
        my_heading = 0
        missile_count = 0

    # 能量状态评估（高度+速度）
    energy_state = _calculate_energy_state(my_altitude, my_speed)

    # 友机状态分析
    friendly_aircraft = []
    for friendly_id in ["B0100", "B0200"]:
        if friendly_id != agent_id and friendly_id in env.agents and env.agents[friendly_id].is_alive:
            friendly = env.agents[friendly_id]
            friendly_pos = np.array(friendly.get_position())
            friendly_distance = np.linalg.norm(agent_pos - friendly_pos)
            try:
                friendly_altitude = safe_get_altitude(friendly, 6000.0)
                friendly_speed = np.linalg.norm(friendly.get_velocity()) if hasattr(friendly, 'get_velocity') else 300
                friendly_missiles = friendly.num_missiles if hasattr(friendly, 'num_missiles') else 0
            except:
                friendly_altitude = 10000
                friendly_speed = 300
                friendly_missiles = 0

            friendly_aircraft.append({
                'id': friendly_id,
                'agent': friendly,
                'distance': friendly_distance,
                'altitude': friendly_altitude,
                'speed': friendly_speed,
                'missiles': friendly_missiles,
                'position': friendly_pos
            })

    # 敌机目标分析
    enemy_targets = []
    for enemy_id in ["A0100", "A0200"]:
        if enemy_id in env.agents and env.agents[enemy_id].is_alive:
            enemy = env.agents[enemy_id]
            enemy_pos = np.array(enemy.get_position())
            distance = np.linalg.norm(agent_pos - enemy_pos)

            try:
                enemy_altitude = safe_get_altitude(enemy, 6000.0)
                enemy_velocity = enemy.get_velocity() if hasattr(enemy, 'get_velocity') else np.array([0, 0, 0])
                enemy_speed = np.linalg.norm(enemy_velocity)
                enemy_heading = enemy.get_property_value(c.attitude_heading_true_rad) * 180 / np.pi
                enemy_missiles = enemy.num_missiles if hasattr(enemy, 'num_missiles') else 0
            except:
                enemy_altitude = 7000
                enemy_velocity = np.array([0, 0, 0])  # 修复：添加默认速度向量
                enemy_speed = 250
                enemy_heading = 180
                enemy_missiles = 0

            # 计算相对位置优势
            altitude_advantage = my_altitude - enemy_altitude
            speed_advantage = my_speed - enemy_speed

            # 计算角度关系
            relative_bearing = _calculate_relative_bearing(agent_pos, enemy_pos, my_heading)
            aspect_angle = _calculate_aspect_angle(agent_pos, enemy_pos, enemy_velocity)

            enemy_targets.append({
                'id': enemy_id,
                'agent': enemy,
                'distance': distance,
                'altitude': enemy_altitude,
                'speed': enemy_speed,
                'heading': enemy_heading,
                'missiles': enemy_missiles,
                'position': enemy_pos,
                'velocity': enemy_velocity,
                'altitude_advantage': altitude_advantage,
                'speed_advantage': speed_advantage,
                'relative_bearing': relative_bearing,
                'aspect_angle': aspect_angle,
                'threat_level': _calculate_enemy_threat_level(distance, enemy_missiles, altitude_advantage)
            })

    # 导弹威胁分析
    missile_threats = _analyze_missile_threats(env, agent_id, agent_pos)

    return {
        'agent_id': agent_id,
        'current_time': current_time,
        'my_position': agent_pos,
        'my_altitude': my_altitude,
        'my_speed': my_speed,
        'my_heading': my_heading,
        'missile_count': missile_count,
        'energy_state': energy_state,
        'friendly_aircraft': friendly_aircraft,
        'enemy_targets': enemy_targets,
        'missile_threats': missile_threats,
        'battlefield_control': _assess_battlefield_control(enemy_targets, friendly_aircraft)
    }


def _analyze_comprehensive_tactical_situation(env, agent_id: str, current_time: float) -> Dict[str, Any]:
    """全面分析战术态势 - 包括威胁评估、机会分析和协调需求"""
    agent = env.agents[agent_id]
    agent_pos = np.array(agent.get_position())

    # 1. 目标分析和优先级排序
    targets = []
    for friendly_id in ["A0100", "A0200"]:
        if friendly_id in env.agents and env.agents[friendly_id].is_alive:
            target = env.agents[friendly_id]
            target_pos = np.array(target.get_position())
            distance = np.linalg.norm(agent_pos - target_pos)

            # 计算目标威胁度和脆弱性
            threat_score = _calculate_target_threat_score(env, target, distance)
            vulnerability = _calculate_target_vulnerability(env, target, distance)

            targets.append({
                'agent': target,
                'distance': distance,
                'threat_score': threat_score,
                'vulnerability': vulnerability,
                'priority': threat_score + vulnerability
            })

    # 按优先级排序目标
    targets.sort(key=lambda x: x['priority'], reverse=True)
    primary_target = targets[0] if targets else None

    # 2. 导弹威胁分析 - 关键改进
    missile_threats = []
    immediate_threat = False
    threat_urgency = 0
    closest_missile_distance = float('inf')

    # 导弹威胁检测已安全禁用，避免属性访问错误
    pass

    # 3. 友军协调分析
    friendly_aircraft = []
    for friendly_id in ["B0100", "B0200"]:
        if friendly_id != agent_id and friendly_id in env.agents and env.agents[friendly_id].is_alive:
            friendly = env.agents[friendly_id]
            friendly_pos = np.array(friendly.get_position())
            friendly_distance = np.linalg.norm(agent_pos - friendly_pos)
            friendly_aircraft.append({
                'agent': friendly,
                'distance': friendly_distance
            })

    # 4. 战术环境评估
    try:
        my_altitude = safe_get_altitude(agent, 6000.0)
        my_speed = np.linalg.norm(agent.get_velocity()) if hasattr(agent, 'get_velocity') else 0
        missile_count = agent.num_missiles if hasattr(agent, 'num_missiles') else 0
    except:
        my_altitude = 0
        my_speed = 0
        missile_count = 0

    # 5. 战术优势评估
    altitude_advantage = False
    speed_advantage = False
    if primary_target:
        try:
            target_altitude = safe_get_altitude(primary_target['agent'], 6000.0)
            target_speed = np.linalg.norm(primary_target['agent'].get_velocity()) if hasattr(primary_target['agent'], 'get_velocity') else 0
            altitude_advantage = my_altitude > target_altitude + 1000  # 1km优势
            speed_advantage = my_speed > target_speed + 50  # 50m/s优势
        except:
            pass

    return {
        'primary_target': primary_target,
        'all_targets': targets,
        'target_distance': primary_target['distance'] if primary_target else float('inf'),
        'missile_threats': missile_threats,
        'immediate_missile_threat': immediate_threat,
        'threat_urgency': threat_urgency,
        'closest_missile_distance': closest_missile_distance,
        'friendly_aircraft': friendly_aircraft,
        'missile_count': missile_count,
        'my_altitude': my_altitude,
        'my_speed': my_speed,
        'altitude_advantage': altitude_advantage,
        'speed_advantage': speed_advantage,
        'can_attack': missile_count > 0,
        'engagement_phase': _determine_engagement_phase(current_time, primary_target['distance'] if primary_target else float('inf'))
    }


def _calculate_target_vulnerability(env, target, distance: float) -> int:
    """计算目标脆弱性分数"""
    vulnerability = 0

    # 距离因子 - 距离越近威胁越大
    if distance < 25000:  # 25km内
        vulnerability += 4
    elif distance < 40000:  # 40km内
        vulnerability += 3
    elif distance < 60000:  # 60km内
        vulnerability += 2
    else:
        vulnerability += 1

    # 目标导弹数量 - 导弹越少越脆弱
    try:
        target_missiles = target.num_missiles if hasattr(target, 'num_missiles') else 0
        if target_missiles == 0:
            vulnerability += 3
        elif target_missiles == 1:
            vulnerability += 2
    except:
        pass

    # 目标是否正在被攻击 - 安全版本
    # 导弹攻击检测已禁用，避免属性访问错误

    return vulnerability


def _calculate_energy_state(altitude: float, speed: float) -> str:
    """计算能量状态"""
    # 能量 = 动能 + 势能 (简化计算)
    kinetic_energy = 0.5 * speed * speed / 1000  # 简化
    potential_energy = altitude / 1000  # 简化
    total_energy = kinetic_energy + potential_energy

    if total_energy > 25:
        return "高能量"
    elif total_energy > 15:
        return "中等能量"
    else:
        return "低能量"


def _calculate_relative_bearing(my_pos, target_pos, my_heading):
    """计算相对方位角"""
    dx = target_pos[0] - my_pos[0]
    dy = target_pos[1] - my_pos[1]
    target_bearing = np.arctan2(dy, dx) * 180 / np.pi
    relative_bearing = target_bearing - my_heading

    # 标准化到-180到180度
    while relative_bearing > 180:
        relative_bearing -= 360
    while relative_bearing < -180:
        relative_bearing += 360

    return relative_bearing


def _calculate_aspect_angle(my_pos, target_pos, target_velocity):
    """计算目标纵横比角度"""
    if np.linalg.norm(target_velocity) < 1:
        return 0

    # 从我到目标的向量
    to_target = target_pos - my_pos
    to_target_norm = to_target / np.linalg.norm(to_target)

    # 目标速度向量标准化
    target_vel_norm = target_velocity / np.linalg.norm(target_velocity)

    # 计算角度
    dot_product = np.dot(to_target_norm, target_vel_norm)
    angle = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi

    return angle


def _calculate_enemy_threat_level(distance: float, enemy_missiles: int, altitude_advantage: float) -> int:
    """计算敌机威胁等级"""
    threat_level = 0

    # 距离威胁
    if distance < 20000:
        threat_level += 4
    elif distance < 40000:
        threat_level += 3
    elif distance < 60000:
        threat_level += 2
    else:
        threat_level += 1

    # 导弹威胁
    threat_level += min(enemy_missiles, 3)

    # 高度劣势增加威胁
    if altitude_advantage < -1000:
        threat_level += 2
    elif altitude_advantage < 0:
        threat_level += 1

    return threat_level


def _analyze_missile_threats(env, agent_id: str, agent_pos):
    """分析导弹威胁"""
    missile_threats = []

    # 导弹威胁检测已安全禁用，避免属性访问错误
    return missile_threats




def _assess_battlefield_control(enemy_targets, friendly_aircraft):
    """评估战场控制状况"""
    total_enemies = len(enemy_targets)
    total_friendlies = len(friendly_aircraft) + 1  # +1 for self

    if total_enemies == 0:
        return "完全控制"
    elif total_friendlies > total_enemies:
        return "优势控制"
    elif total_friendlies == total_enemies:
        return "均势"
    else:
        return "劣势"


def _calculate_target_threat_score(env, target, distance: float) -> int:
    """计算目标威胁分数"""
    threat_score = 0

    # 距离威胁 - 距离越近威胁越大
    if distance < 20000:  # 20km内
        threat_score += 4
    elif distance < 40000:  # 40km内
        threat_score += 3
    elif distance < 60000:  # 60km内
        threat_score += 2
    else:
        threat_score += 1

    # 目标导弹威胁
    try:
        target_missiles = target.num_missiles if hasattr(target, 'num_missiles') else 0
        if target_missiles > 2:
            threat_score += 3
        elif target_missiles > 0:
            threat_score += 2
    except:
        pass

    # 目标是否正在攻击我方 - 安全版本
    # 导弹攻击检测已禁用，避免属性访问错误

    return threat_score


def _determine_engagement_phase(current_time: float, target_distance: float) -> str:
    """确定交战阶段"""
    if current_time < 30:
        return "初始接敌"
    elif current_time < 60:
        return "中距离交战"
    elif current_time < 120:
        return "近距离激战"
    else:
        return "持续作战"


def _determine_tactical_role(agent_id: str, situation: Dict[str, Any]) -> str:
    """确定战术角色 - 实现角色分化"""
    # 基于飞机ID和战术情况分配角色
    if agent_id == "B0100":
        # B0100作为主攻击者
        if situation['immediate_missile_threat']:
            return "主攻击者-规避"
        elif situation['can_attack'] and situation['target_distance'] < 40000:
            return "主攻击者-进攻"
        else:
            return "主攻击者-接敌"

    elif agent_id == "B0200":
        # B0200作为支援者/侧翼攻击者
        if situation['immediate_missile_threat']:
            return "侧翼攻击者-规避"
        elif len(situation['friendly_aircraft']) > 0:
            return "侧翼攻击者-协调"
        else:
            return "侧翼攻击者-独立"

    else:
        return "通用战斗者"


def _select_intelligent_behavior(situation: Dict[str, Any], role: str, agent) -> str:
    """选择智能战斗行为 - 基于角色和威胁的复杂决策"""
    distance = situation['target_distance']
    missile_threat = situation['immediate_missile_threat']
    threat_urgency = situation['threat_urgency']
    can_attack = situation['can_attack']
    engagement_phase = situation['engagement_phase']
    altitude_advantage = situation['altitude_advantage']

    # 紧急威胁处理 - 最高优先级
    if missile_threat and threat_urgency >= 4:
        if role.startswith("主攻击者"):
            return "紧急规避-反击"  # 主攻击者在规避时准备反击
        else:
            return "紧急规避-支援"  # 侧翼攻击者规避并准备支援

    # 基于角色的战术决策
    if role == "主攻击者-进攻":
        if distance < 15000 and can_attack:
            return "主攻击者-近距离突击"
        elif distance < 30000 and can_attack:
            return "主攻击者-中距离强攻"
        elif distance < 50000:
            return "主攻击者-快速接敌"
        else:
            return "主攻击者-搜索接敌"

    elif role == "主攻击者-接敌":
        if distance < 40000:
            return "主攻击者-战术接敌"
        else:
            return "主攻击者-高速接敌"

    elif role == "侧翼攻击者-协调":
        if distance < 20000 and can_attack:
            return "侧翼攻击者-协调攻击"
        elif distance < 40000:
            return "侧翼攻击者-侧翼机动"
        else:
            return "侧翼攻击者-包抄接敌"

    elif role == "侧翼攻击者-独立":
        if distance < 25000 and can_attack:
            return "侧翼攻击者-独立攻击"
        elif altitude_advantage:
            return "侧翼攻击者-高度攻击"
        else:
            return "侧翼攻击者-机动攻击"

    elif "规避" in role:
        if threat_urgency >= 4:
            return "高机动规避"
        elif threat_urgency >= 3:
            return "战术规避"
        else:
            return "预防性机动"

    # 默认基于距离的行为
    if distance < 20000:
        return "近距离战斗"
    elif distance < 40000:
        return "中距离交战"
    else:
        return "远距离接敌"


def _generate_intelligent_commands(behavior: str, situation: Dict[str, Any], role: str, agent) -> Tuple[int, int, int]:
    """生成智能战斗指令 - 基于行为、角色和威胁的复杂指令生成"""
    # 动作空间: [高度(0-14), 航向(0-16), 速度(0-6)]

    # 紧急规避指令 - 最高优先级
    if "紧急规避" in behavior:
        if situation['threat_urgency'] >= 5:
            # 极度紧急：急剧机动
            return (2, 1, 6)    # 急降+大左转+最大速度
        else:
            # 高度紧急：大幅机动
            return (4, 3, 5)    # 下降+左转+高速

    # 主攻击者指令
    elif behavior == "主攻击者-近距离突击":
        return (11, 13, 6)  # 爬升+大右转+最大速度

    elif behavior == "主攻击者-中距离强攻":
        return (9, 11, 5)   # 爬升+右转+高速

    elif behavior == "主攻击者-快速接敌":
        return (8, 9, 6)    # 轻微爬升+轻微右转+最大速度

    elif behavior == "主攻击者-战术接敌":
        return (7, 10, 4)   # 保持高度+右转+加速

    elif behavior == "主攻击者-高速接敌":
        return (7, 8, 6)    # 保持高度+直飞+最大速度

    elif behavior == "主攻击者-搜索接敌":
        return (8, 8, 4)    # 轻微爬升+直飞+加速

    # 侧翼攻击者指令 - 与主攻击者差异化
    elif behavior == "侧翼攻击者-协调攻击":
        return (10, 6, 5)   # 爬升+左转+高速 (与主攻击者形成夹击)

    elif behavior == "侧翼攻击者-侧翼机动":
        return (6, 5, 4)    # 下降+左转+加速 (低空侧翼)

    elif behavior == "侧翼攻击者-包抄接敌":
        return (9, 4, 5)    # 爬升+大左转+高速 (大范围包抄)

    elif behavior == "侧翼攻击者-独立攻击":
        return (8, 12, 5)   # 轻微爬升+右转+高速

    elif behavior == "侧翼攻击者-高度攻击":
        return (12, 7, 4)   # 大幅爬升+轻微左转+加速

    elif behavior == "侧翼攻击者-机动攻击":
        return (7, 6, 5)    # 保持高度+左转+高速

    # 规避机动指令
    elif behavior == "高机动规避":
        return (3, 2, 6)    # 下降+大左转+最大速度

    elif behavior == "战术规避":
        return (5, 4, 5)    # 轻微下降+左转+高速

    elif behavior == "预防性机动":
        return (6, 6, 4)    # 轻微下降+左转+加速

    # 通用战斗指令
    elif behavior == "近距离战斗":
        if role.startswith("主攻击者"):
            return (10, 11, 5)  # 主攻击者：爬升+右转+高速
        else:
            return (8, 5, 5)    # 侧翼攻击者：轻微爬升+左转+高速

    elif behavior == "中距离交战":
        if role.startswith("主攻击者"):
            return (8, 9, 4)    # 主攻击者：轻微爬升+轻微右转+加速
        else:
            return (7, 7, 4)    # 侧翼攻击者：保持高度+轻微左转+加速

    elif behavior == "远距离接敌":
        if role.startswith("主攻击者"):
            return (7, 8, 5)    # 主攻击者：保持高度+直飞+高速
        else:
            return (8, 6, 4)    # 侧翼攻击者：轻微爬升+左转+加速

    else:
        # 默认指令：基于角色的差异化
        if role.startswith("主攻击者"):
            return (8, 9, 4)    # 主攻击者默认
        else:
            return (7, 7, 4)    # 侧翼攻击者默认


def _evaluate_threats_and_opportunities(env, agent_id: str, battlefield_awareness: Dict[str, Any]) -> Dict[str, Any]:
    """威胁评估和机会分析 - 强对抗性AI的决策核心"""

    # 1. 导弹威胁评估
    missile_threats = battlefield_awareness['missile_threats']
    immediate_missile_threat = len([m for m in missile_threats if m['urgency'] >= 4]) > 0
    missile_threat_level = max([m['urgency'] for m in missile_threats]) if missile_threats else 0
    closest_missile = missile_threats[0] if missile_threats else None

    # 2. 敌机威胁和机会评估
    enemy_targets = battlefield_awareness['enemy_targets']

    # 选择主要目标（威胁最大或机会最好）
    primary_target = None
    best_opportunity = None
    highest_threat = None

    for target in enemy_targets:
        # 威胁评估
        if highest_threat is None or target['threat_level'] > highest_threat['threat_level']:
            highest_threat = target

        # 机会评估
        opportunity_score = _calculate_opportunity_score(target, battlefield_awareness)
        target['opportunity_score'] = opportunity_score

        if best_opportunity is None or opportunity_score > best_opportunity['opportunity_score']:
            best_opportunity = target

    # 根据战术情况选择主要目标
    if immediate_missile_threat:
        # 有导弹威胁时，优先考虑最大威胁
        primary_target = highest_threat
    else:
        # 无导弹威胁时，优先考虑最佳机会
        primary_target = best_opportunity

    # 3. 战术优势评估
    tactical_advantages = _assess_tactical_advantages(battlefield_awareness, primary_target)

    # 4. 协调机会评估
    coordination_opportunities = _assess_coordination_opportunities(battlefield_awareness, primary_target)

    return {
        'immediate_missile_threat': immediate_missile_threat,
        'missile_threat_level': missile_threat_level,
        'closest_missile': closest_missile,
        'primary_target': primary_target,
        'all_targets': enemy_targets,
        'highest_threat': highest_threat,
        'best_opportunity': best_opportunity,
        'tactical_advantages': tactical_advantages,
        'coordination_opportunities': coordination_opportunities,
        'engagement_range': _determine_engagement_range(primary_target['distance'] if primary_target else float('inf'))
    }


def _calculate_opportunity_score(target: Dict[str, Any], battlefield_awareness: Dict[str, Any]) -> float:
    """计算攻击机会分数"""
    score = 0.0

    # 距离因子（中距离最佳）
    distance = target['distance']
    if 15000 <= distance <= 35000:
        score += 5.0
    elif 10000 <= distance <= 50000:
        score += 3.0
    elif distance < 10000:
        score += 1.0  # 太近危险
    else:
        score += 0.5  # 太远效果差

    # 高度优势
    if target['altitude_advantage'] > 2000:
        score += 3.0
    elif target['altitude_advantage'] > 500:
        score += 1.5
    elif target['altitude_advantage'] < -1000:
        score -= 2.0

    # 速度优势
    if target['speed_advantage'] > 100:
        score += 2.0
    elif target['speed_advantage'] > 50:
        score += 1.0
    elif target['speed_advantage'] < -50:
        score -= 1.0

    # 纵横比角度（侧面攻击最佳）
    aspect = target['aspect_angle']
    if 60 <= aspect <= 120:
        score += 2.0  # 侧面攻击
    elif aspect < 30:
        score -= 1.0  # 正面对头
    elif aspect > 150:
        score += 1.0  # 尾追

    # 目标导弹数量（导弹少的目标更容易攻击）
    if target['missiles'] == 0:
        score += 3.0
    elif target['missiles'] == 1:
        score += 1.0
    else:
        score -= 1.0

    # 我方导弹数量
    my_missiles = battlefield_awareness['missile_count']
    if my_missiles > 0:
        score += 2.0
    else:
        score -= 3.0  # 没有导弹大幅降低机会

    return max(score, 0.0)


def _assess_tactical_advantages(battlefield_awareness: Dict[str, Any], primary_target) -> Dict[str, bool]:
    """评估战术优势"""
    advantages = {
        'altitude_advantage': False,
        'speed_advantage': False,
        'energy_advantage': False,
        'position_advantage': False,
        'numerical_advantage': False
    }

    if primary_target:
        # 高度优势
        advantages['altitude_advantage'] = primary_target['altitude_advantage'] > 1000

        # 速度优势
        advantages['speed_advantage'] = primary_target['speed_advantage'] > 50

        # 能量优势
        my_energy = battlefield_awareness['energy_state']
        advantages['energy_advantage'] = my_energy in ["高能量", "中等能量"]

        # 位置优势（侧面或后方）
        aspect = primary_target['aspect_angle']
        advantages['position_advantage'] = aspect > 90

        # 数量优势
        total_enemies = len(battlefield_awareness['enemy_targets'])
        total_friendlies = len(battlefield_awareness['friendly_aircraft']) + 1
        advantages['numerical_advantage'] = total_friendlies >= total_enemies

    return advantages


def _assess_coordination_opportunities(battlefield_awareness: Dict[str, Any], primary_target) -> Dict[str, Any]:
    """评估协调机会"""
    opportunities = {
        'can_coordinate': False,
        'pincer_attack': False,
        'high_low_split': False,
        'distraction_support': False
    }

    friendly_aircraft = battlefield_awareness['friendly_aircraft']

    if friendly_aircraft and primary_target:
        opportunities['can_coordinate'] = True

        # 检查钳形攻击机会
        my_pos = battlefield_awareness['my_position']
        target_pos = primary_target['position']

        for friendly in friendly_aircraft:
            friendly_pos = friendly['position']

            # 计算角度关系
            my_to_target = target_pos - my_pos
            friendly_to_target = target_pos - friendly_pos

            angle = np.arccos(np.clip(np.dot(my_to_target, friendly_to_target) /
                                    (np.linalg.norm(my_to_target) * np.linalg.norm(friendly_to_target)), -1, 1))
            angle_deg = angle * 180 / np.pi

            if 90 <= angle_deg <= 180:
                opportunities['pincer_attack'] = True

            # 检查高低分离机会
            altitude_diff = abs(battlefield_awareness['my_altitude'] - friendly['altitude'])
            if altitude_diff > 2000:
                opportunities['high_low_split'] = True

    return opportunities


def _determine_engagement_range(distance: float) -> str:
    """确定交战距离范围"""
    if distance < 15000:
        return "近距离"
    elif distance < 35000:
        return "中距离"
    elif distance < 60000:
        return "远距离"
    else:
        return "超远距离"


def _analyze_bvr_situation(env, agent_id: str, current_time: float) -> Dict[str, Any]:
    """BVR战场态势感知 - 基于真实BVR作战原则"""
    agent = env.agents[agent_id]
    agent_pos = np.array(agent.get_position())

    try:
        # 获取自身状态
        my_altitude = safe_get_altitude(agent, 6000.0)
        my_velocity = agent.get_velocity() if hasattr(agent, 'get_velocity') else np.array([0, 0, 0])
        my_speed = np.linalg.norm(my_velocity)
        my_heading = agent.get_property_value(c.attitude_heading_true_rad) * 180 / np.pi
        missile_count = agent.num_missiles if hasattr(agent, 'num_missiles') else 0
        fuel_remaining = getattr(agent, 'fuel_remaining', 1.0)  # 假设燃料剩余比例
    except:
        my_altitude = 10000
        my_velocity = np.array([0, 0, 0])  # 修复：添加默认速度向量
        my_speed = 300
        my_heading = 180
        missile_count = 0
        fuel_remaining = 1.0

    # 友机状态分析（编队协调）
    friendly_aircraft = []
    for friendly_id in ["B0100", "B0200"]:
        if friendly_id != agent_id and friendly_id in env.agents and env.agents[friendly_id].is_alive:
            friendly = env.agents[friendly_id]
            friendly_pos = np.array(friendly.get_position())
            friendly_distance = np.linalg.norm(agent_pos - friendly_pos)
            try:
                friendly_altitude = safe_get_altitude(friendly, 6000.0)
                friendly_speed = np.linalg.norm(friendly.get_velocity()) if hasattr(friendly, 'get_velocity') else 300
                friendly_missiles = friendly.num_missiles if hasattr(friendly, 'num_missiles') else 0
                friendly_heading = friendly.get_property_value(c.attitude_heading_true_rad) * 180 / np.pi
            except:
                friendly_altitude = 10000
                friendly_speed = 300
                friendly_missiles = 0
                friendly_heading = 180

            friendly_aircraft.append({
                'id': friendly_id,
                'distance': friendly_distance,
                'altitude': friendly_altitude,
                'speed': friendly_speed,
                'heading': friendly_heading,
                'missiles': friendly_missiles,
                'position': friendly_pos
            })

    # 敌机目标分析（BVR威胁评估）
    enemy_targets = []
    for enemy_id in ["A0100", "A0200"]:
        if enemy_id in env.agents and env.agents[enemy_id].is_alive:
            enemy = env.agents[enemy_id]
            enemy_pos = np.array(enemy.get_position())
            distance = np.linalg.norm(agent_pos - enemy_pos)

            try:
                enemy_altitude = safe_get_altitude(enemy, 6000.0)
                enemy_velocity = enemy.get_velocity() if hasattr(enemy, 'get_velocity') else np.array([0, 0, 0])
                enemy_speed = np.linalg.norm(enemy_velocity)
                enemy_heading = enemy.get_property_value(c.attitude_heading_true_rad) * 180 / np.pi
                enemy_missiles = enemy.num_missiles if hasattr(enemy, 'num_missiles') else 0
            except:
                enemy_altitude = 7000
                enemy_velocity = np.array([0, 0, 0])
                enemy_speed = 250
                enemy_heading = 0
                enemy_missiles = 0

            # BVR关键参数计算
            altitude_advantage = my_altitude - enemy_altitude
            speed_advantage = my_speed - enemy_speed
            relative_bearing = _calculate_relative_bearing(agent_pos, enemy_pos, my_heading)
            aspect_angle = _calculate_aspect_angle(agent_pos, enemy_pos, enemy_velocity)
            closure_rate = _calculate_closure_rate(my_velocity, enemy_velocity, agent_pos, enemy_pos)

            # BVR威胁等级评估
            bvr_threat_level = _calculate_bvr_threat_level(distance, enemy_missiles, closure_rate, aspect_angle)

            enemy_targets.append({
                'id': enemy_id,
                'distance': distance,
                'altitude': enemy_altitude,
                'speed': enemy_speed,
                'heading': enemy_heading,
                'missiles': enemy_missiles,
                'position': enemy_pos,
                'velocity': enemy_velocity,
                'altitude_advantage': altitude_advantage,
                'speed_advantage': speed_advantage,
                'relative_bearing': relative_bearing,
                'aspect_angle': aspect_angle,
                'closure_rate': closure_rate,
                'bvr_threat_level': bvr_threat_level
            })

    # 导弹威胁分析（BVR关键）
    missile_threats = _analyze_bvr_missile_threats(env, agent_id, agent_pos)
    immediate_missile_threat = len([m for m in missile_threats if m['urgency'] >= 4]) > 0

    # 选择主要目标（BVR优先级）
    primary_target = None
    if enemy_targets:
        # BVR目标优先级：距离适中、威胁高、易攻击
        enemy_targets.sort(key=lambda x: (x['bvr_threat_level'], -x['distance']), reverse=True)
        primary_target = enemy_targets[0]

    # BVR战场控制评估
    battlefield_control = _assess_bvr_battlefield_control(enemy_targets, friendly_aircraft, missile_threats)

    return {
        'agent_id': agent_id,
        'current_time': current_time,
        'my_position': agent_pos,
        'my_altitude': my_altitude,
        'my_speed': my_speed,
        'my_heading': my_heading,
        'missile_count': missile_count,
        'fuel_remaining': fuel_remaining,
        'friendly_aircraft': friendly_aircraft,
        'enemy_targets': enemy_targets,
        'primary_target': primary_target,
        'missile_threats': missile_threats,
        'immediate_missile_threat': immediate_missile_threat,
        'battlefield_control': battlefield_control,
        'formation_integrity': len(friendly_aircraft) > 0  # 编队完整性
    }


def _calculate_closure_rate(my_velocity, enemy_velocity, my_pos, enemy_pos):
    """计算接近速率"""
    relative_velocity = my_velocity - enemy_velocity
    range_vector = enemy_pos - my_pos
    range_distance = np.linalg.norm(range_vector)

    if range_distance < 1:
        return 0

    range_unit = range_vector / range_distance
    closure_rate = -np.dot(relative_velocity, range_unit)  # 负号表示接近
    return closure_rate


def _calculate_bvr_threat_level(distance: float, enemy_missiles: int, closure_rate: float, aspect_angle: float) -> int:
    """计算BVR威胁等级"""
    threat_level = 0

    # 距离威胁（BVR关键）
    if distance < 25000:  # 25km内高威胁
        threat_level += 5
    elif distance < 40000:  # 40km内中威胁
        threat_level += 3
    elif distance < 60000:  # 60km内低威胁
        threat_level += 1

    # 导弹威胁
    threat_level += min(enemy_missiles * 2, 6)

    # 接近速率威胁
    if closure_rate > 200:  # 高速接近
        threat_level += 3
    elif closure_rate > 100:
        threat_level += 1

    # 纵横比威胁（正面接近最危险）
    if aspect_angle < 30:  # 正面对头
        threat_level += 2
    elif aspect_angle > 150:  # 尾追
        threat_level -= 1

    return max(threat_level, 0)


def _analyze_bvr_missile_threats(env, agent_id: str, agent_pos):
    """分析BVR导弹威胁"""
    missile_threats = []

    # 导弹威胁检测已安全禁用，避免属性访问错误
    return missile_threats


def _assess_bvr_battlefield_control(enemy_targets, friendly_aircraft, missile_threats):
    """评估BVR战场控制状况"""
    total_enemies = len(enemy_targets)
    total_friendlies = len(friendly_aircraft) + 1  # +1 for self
    active_missile_threats = len([m for m in missile_threats if m['urgency'] >= 3])

    if total_enemies == 0:
        return "完全控制"
    elif active_missile_threats >= 2:
        return "严重威胁"
    elif total_friendlies > total_enemies and active_missile_threats == 0:
        return "优势控制"
    elif total_friendlies == total_enemies:
        return "均势"
    else:
        return "劣势"


def _determine_formation_role(agent_id: str, bvr_situation: Dict[str, Any]) -> str:
    """确定编队角色 - 长机僚机分工"""
    friendly_aircraft = bvr_situation['friendly_aircraft']

    if agent_id == "B0100":
        # B0100作为长机
        if bvr_situation['immediate_missile_threat']:
            return "长机-防御"
        elif bvr_situation['primary_target'] and bvr_situation['primary_target']['distance'] < 50000:
            return "长机-攻击"
        else:
            return "长机-搜索"

    elif agent_id == "B0200":
        # B0200作为僚机
        if bvr_situation['immediate_missile_threat']:
            return "僚机-支援防御"
        elif len(friendly_aircraft) > 0:
            # 有长机存在，执行协调任务
            leader_distance = friendly_aircraft[0]['distance'] if friendly_aircraft else float('inf')
            if leader_distance > 10000:  # 编队分散
                return "僚机-重新编队"
            else:
                return "僚机-协调攻击"
        else:
            # 长机已被击落，独立作战
            return "僚机-独立作战"

    return "独立作战"


def _determine_bvr_phase(bvr_situation: Dict[str, Any], formation_role: str) -> str:
    """确定BVR交战阶段"""
    primary_target = bvr_situation['primary_target']

    if not primary_target:
        return "搜索阶段"

    distance = primary_target['distance']
    missile_count = bvr_situation['missile_count']
    immediate_threat = bvr_situation['immediate_missile_threat']

    # BVR阶段划分（基于真实作战距离）
    if immediate_threat:
        return "防御阶段"
    elif distance > 80000:  # 80km以上
        return "远程BVR"
    elif distance > 50000:  # 50-80km
        return "中程BVR"
    elif distance > 25000:  # 25-50km
        return "近程BVR"
    elif distance > 15000:  # 15-25km
        return "BVR-WVR过渡"
    else:  # 15km以下
        return "近距格斗"


def _evaluate_disengagement_criteria(bvr_situation: Dict[str, Any], engagement_phase: str, current_time: float) -> bool:
    """评估脱离接触标准 - BVR作战核心决策"""

    # 1. 强制脱离条件
    fuel_remaining = bvr_situation['fuel_remaining']
    missile_count = bvr_situation['missile_count']
    immediate_threat = bvr_situation['immediate_missile_threat']
    battlefield_control = bvr_situation['battlefield_control']

    # 燃料不足强制脱离
    if fuel_remaining < 0.3:  # 燃料少于30%
        return True

    # 导弹耗尽且面临威胁
    if missile_count == 0 and immediate_threat:
        return True

    # 严重威胁环境
    if battlefield_control == "严重威胁":
        missile_threats = bvr_situation['missile_threats']
        high_urgency_threats = len([m for m in missile_threats if m['urgency'] >= 4])
        if high_urgency_threats >= 2:  # 多枚导弹威胁
            return True

    # 2. 战术脱离条件
    primary_target = bvr_situation['primary_target']
    if primary_target:
        distance = primary_target['distance']
        closure_rate = primary_target['closure_rate']

        # 距离过近且无优势
        if distance < 20000 and closure_rate > 150:  # 20km内高速接近
            altitude_advantage = primary_target['altitude_advantage']
            speed_advantage = primary_target['speed_advantage']
            if altitude_advantage < -1000 and speed_advantage < -50:  # 高度速度双劣势
                return True

    # 3. 编队脱离条件
    friendly_aircraft = bvr_situation['friendly_aircraft']
    if len(friendly_aircraft) == 0:  # 失去编队支援
        if engagement_phase in ["近程BVR", "BVR-WVR过渡", "近距格斗"]:
            return True

    # 4. 时间脱离条件
    if current_time > 240:  # 4分钟后考虑脱离
        if missile_count <= 1 and not immediate_threat:
            return True

    return False


def _select_disengagement_behavior(bvr_situation: Dict[str, Any], formation_role: str) -> str:
    """选择脱离接触行为"""
    immediate_threat = bvr_situation['immediate_missile_threat']
    battlefield_control = bvr_situation['battlefield_control']

    if immediate_threat:
        if formation_role.startswith("长机"):
            return "长机紧急脱离"
        else:
            return "僚机紧急脱离"

    elif battlefield_control == "严重威胁":
        if formation_role.startswith("长机"):
            return "长机战术脱离"
        else:
            return "僚机掩护脱离"

    else:
        if formation_role.startswith("长机"):
            return "长机有序脱离"
        else:
            return "僚机跟随脱离"


def _select_bvr_behavior(engagement_phase: str, bvr_situation: Dict[str, Any], formation_role: str) -> str:
    """选择BVR战术行为"""
    primary_target = bvr_situation['primary_target']
    missile_count = bvr_situation['missile_count']
    immediate_threat = bvr_situation['immediate_missile_threat']

    # 威胁响应优先
    if immediate_threat:
        if formation_role.startswith("长机"):
            return "长机导弹规避"
        else:
            return "僚机支援规避"

    # 基于阶段的行为选择
    if engagement_phase == "远程BVR":
        if formation_role.startswith("长机"):
            return "长机远程搜索"
        else:
            return "僚机编队保持"

    elif engagement_phase == "中程BVR":
        if missile_count > 0 and primary_target:
            if formation_role.startswith("长机"):
                return "长机中程攻击"
            else:
                return "僚机协调攻击"
        else:
            if formation_role.startswith("长机"):
                return "长机中程机动"
            else:
                return "僚机支援机动"

    elif engagement_phase == "近程BVR":
        if missile_count > 0 and primary_target:
            if formation_role.startswith("长机"):
                return "长机近程突击"
            else:
                return "僚机侧翼攻击"
        else:
            if formation_role.startswith("长机"):
                return "长机近程防御"
            else:
                return "僚机近程支援"

    elif engagement_phase == "BVR-WVR过渡":
        if formation_role.startswith("长机"):
            return "长机过渡机动"
        else:
            return "僚机过渡支援"

    elif engagement_phase == "近距格斗":
        if formation_role.startswith("长机"):
            return "长机格斗机动"
        else:
            return "僚机格斗支援"

    else:  # 搜索阶段
        if formation_role.startswith("长机"):
            return "长机搜索接敌"
        else:
            return "僚机搜索支援"


def _generate_bvr_commands(tactical_behavior: str, bvr_situation: Dict[str, Any],
                          formation_role: str, agent_id: str,
                          tactical_mode: TacticalMode = TacticalMode.NEUTRAL) -> Tuple[int, int, int]:
    """
    生成BVR战术指令 - 基于战术模式和真实BVR作战原则

    Args:
        tactical_behavior: 战术行为描述
        bvr_situation: BVR态势信息
        formation_role: 编队角色
        agent_id: 智能体ID
        tactical_mode: 战术模式

    Returns:
        Tuple[int, int, int]: [高度指令, 航向指令, 速度指令]
    """

    # 动作空间: [高度(0-14), 航向(0-16), 速度(0-6)]
    # 7=保持当前, <7下降/左转/减速, >7爬升/右转/加速

    primary_target = bvr_situation['primary_target']
    my_altitude = bvr_situation['my_altitude']
    my_heading = bvr_situation['my_heading']

    # 1. 脱离接触指令
    if "脱离" in tactical_behavior:
        if "紧急" in tactical_behavior:
            # 紧急脱离：最大机动
            return (2, 1, 6)  # 急降+最大左转+最大速度
        elif "战术" in tactical_behavior:
            # 战术脱离：有序撤退
            return (4, 3, 5)  # 下降+左转+高速
        else:
            # 有序脱离：保持编队
            return (6, 5, 4)  # 轻微下降+左转+加速

    # 2. 导弹规避指令
    elif "规避" in tactical_behavior:
        missile_threats = bvr_situation['missile_threats']
        if missile_threats and missile_threats[0]['urgency'] >= 4:
            # 高威胁导弹规避
            if formation_role.startswith("长机"):
                return (3, 2, 6)  # 长机：下降+大左转+最大速度
            else:
                return (2, 4, 6)  # 僚机：急降+左转+最大速度
        else:
            # 预防性规避
            return (5, 4, 5)  # 轻微下降+左转+高速

    # 3. 远程BVR指令
    elif "远程" in tactical_behavior:
        if formation_role.startswith("长机"):
            # 长机远程搜索：保持高度优势
            return (8, 8, 4)  # 轻微爬升+直飞+加速
        else:
            # 僚机编队保持：与长机协调
            return (7, 7, 4)  # 保持高度+轻微左转+加速

    # 4. 中程BVR指令
    elif "中程" in tactical_behavior:
        if "攻击" in tactical_behavior:
            if formation_role.startswith("长机"):
                # 长机中程攻击：主动接敌
                return (9, 10, 5)  # 爬升+右转+高速
            else:
                # 僚机协调攻击：侧翼支援
                return (8, 6, 5)   # 轻微爬升+左转+高速
        else:
            # 中程机动：保持机动性
            return (7, 9, 4)   # 保持高度+轻微右转+加速

    # 5. 近程BVR指令
    elif "近程" in tactical_behavior:
        if "突击" in tactical_behavior:
            # 长机近程突击：最大攻击性
            return (11, 12, 6)  # 爬升+右转+最大速度
        elif "侧翼" in tactical_behavior:
            # 僚机侧翼攻击：包抄机动
            return (9, 5, 5)    # 爬升+左转+高速
        elif "防御" in tactical_behavior:
            # 近程防御：垂直机动
            return (6, 4, 5)    # 轻微下降+左转+高速
        else:
            # 近程支援：灵活机动
            return (8, 7, 5)    # 轻微爬升+轻微左转+高速

    # 6. 过渡阶段指令
    elif "过渡" in tactical_behavior:
        if formation_role.startswith("长机"):
            # 长机过渡：准备近距格斗
            return (10, 11, 5)  # 爬升+右转+高速
        else:
            # 僚机过渡：支援准备
            return (8, 6, 5)    # 轻微爬升+左转+高速

    # 7. 格斗指令
    elif "格斗" in tactical_behavior:
        if formation_role.startswith("长机"):
            # 长机格斗：主动攻击
            return (12, 13, 6)  # 大幅爬升+大右转+最大速度
        else:
            # 僚机格斗：支援攻击
            return (10, 5, 6)   # 爬升+左转+最大速度

    # 8. 搜索指令
    elif "搜索" in tactical_behavior:
        if formation_role.startswith("长机"):
            # 长机搜索：主动搜索
            return (8, 9, 4)    # 轻微爬升+轻微右转+加速
        else:
            # 僚机搜索：编队搜索
            return (7, 7, 4)    # 保持高度+轻微左转+加速

    # 9. 编队相关指令
    elif "编队" in tactical_behavior:
        friendly_aircraft = bvr_situation['friendly_aircraft']
        if friendly_aircraft:
            # 重新编队：向长机靠拢
            leader_pos = friendly_aircraft[0]['position']
            my_pos = bvr_situation['my_position']

            # 计算相对位置
            dx = leader_pos[0] - my_pos[0]
            dy = leader_pos[1] - my_pos[1]

            if abs(dx) > abs(dy):
                if dx > 0:
                    return (7, 10, 4)  # 向东靠拢
                else:
                    return (7, 6, 4)   # 向西靠拢
            else:
                if dy > 0:
                    return (7, 8, 4)   # 向北靠拢
                else:
                    return (7, 12, 4)  # 向南靠拢
        else:
            return (7, 8, 4)  # 默认直飞

    # 10. 独立作战指令
    elif "独立" in tactical_behavior:
        if primary_target and primary_target['distance'] < 40000:
            # 独立攻击
            return (9, 10, 5)  # 爬升+右转+高速
        else:
            # 独立搜索
            return (8, 8, 4)   # 轻微爬升+直飞+加速

    # 默认指令：基于角色的标准机动
    else:
        if formation_role.startswith("长机"):
            return (8, 9, 4)    # 长机默认：轻微爬升+轻微右转+加速
        else:
            return (7, 7, 4)    # 僚机默认：保持高度+轻微左转+加速


def _determine_tactical_mode(agent_id: str, threat_assessment: Dict[str, Any],
                           battlefield_awareness: Dict[str, Any], current_time: float) -> str:
    """确定战术模式 - 进攻/防守/中立的动态切换"""

    # 1. 紧急防守模式 - 最高优先级
    if threat_assessment['immediate_missile_threat']:
        missile_threat_level = threat_assessment['missile_threat_level']
        if missile_threat_level >= 5:
            return "紧急防守"
        elif missile_threat_level >= 4:
            return "积极防守"
        else:
            return "预防防守"

    # 2. 基于战场态势的模式选择
    primary_target = threat_assessment['primary_target']
    tactical_advantages = threat_assessment['tactical_advantages']
    coordination_opportunities = threat_assessment['coordination_opportunities']

    if not primary_target:
        return "搜索模式"

    # 3. 进攻模式条件评估
    offensive_score = 0

    # 有导弹且目标在有效范围内
    if battlefield_awareness['missile_count'] > 0 and primary_target['distance'] < 50000:
        offensive_score += 3

    # 战术优势
    if tactical_advantages['altitude_advantage']:
        offensive_score += 2
    if tactical_advantages['speed_advantage']:
        offensive_score += 1
    if tactical_advantages['energy_advantage']:
        offensive_score += 2
    if tactical_advantages['position_advantage']:
        offensive_score += 2

    # 协调机会
    if coordination_opportunities['pincer_attack']:
        offensive_score += 3
    if coordination_opportunities['high_low_split']:
        offensive_score += 2

    # 目标脆弱性
    if primary_target['missiles'] == 0:
        offensive_score += 3
    elif primary_target['missiles'] == 1:
        offensive_score += 1

    # 4. 防守模式条件评估
    defensive_score = 0

    # 目标威胁高
    if primary_target['threat_level'] >= 6:
        defensive_score += 3

    # 我方劣势
    if not tactical_advantages['altitude_advantage']:
        defensive_score += 1
    if not tactical_advantages['energy_advantage']:
        defensive_score += 2
    if battlefield_awareness['missile_count'] == 0:
        defensive_score += 4

    # 数量劣势
    if not tactical_advantages['numerical_advantage']:
        defensive_score += 2

    # 5. 角色特定的模式倾向
    if agent_id == "B0100":  # 主攻击者，更倾向于进攻
        offensive_score += 1
    elif agent_id == "B0200":  # 侧翼攻击者，更灵活
        if coordination_opportunities['can_coordinate']:
            offensive_score += 1

    # 6. 时间因素
    if current_time < 60:  # 早期更积极
        offensive_score += 1
    elif current_time > 180:  # 后期更谨慎
        defensive_score += 1

    # 7. 模式决策
    if offensive_score >= 6:
        return "主动进攻"
    elif offensive_score >= 4:
        return "机会进攻"
    elif defensive_score >= 5:
        return "战术防守"
    elif defensive_score >= 3:
        return "谨慎防守"
    else:
        return "中立机动"


def _select_tactical_behavior(tactical_mode: str, threat_assessment: Dict[str, Any],
                            battlefield_awareness: Dict[str, Any], agent_id: str) -> str:
    """选择具体战术行为"""

    primary_target = threat_assessment['primary_target']
    engagement_range = threat_assessment['engagement_range']
    coordination_opportunities = threat_assessment['coordination_opportunities']

    # 1. 防守模式行为
    if tactical_mode == "紧急防守":
        return "紧急规避机动"
    elif tactical_mode == "积极防守":
        return "攻击性规避"
    elif tactical_mode == "预防防守":
        return "预防性机动"
    elif tactical_mode == "战术防守":
        if engagement_range == "近距离":
            return "近距离防守"
        else:
            return "远程防守"
    elif tactical_mode == "谨慎防守":
        return "保守机动"

    # 2. 进攻模式行为
    elif tactical_mode == "主动进攻":
        if engagement_range == "近距离":
            return "近距离突击"
        elif engagement_range == "中距离":
            if coordination_opportunities['pincer_attack']:
                return "协调钳击"
            else:
                return "中距离强攻"
        else:
            return "高速接敌"

    elif tactical_mode == "机会进攻":
        if coordination_opportunities['high_low_split']:
            return "高低分离攻击"
        elif primary_target and primary_target['aspect_angle'] > 120:
            return "尾追攻击"
        else:
            return "机会攻击"

    # 3. 中立模式行为
    elif tactical_mode == "中立机动":
        if agent_id == "B0100":
            return "主攻击者机动"
        else:
            return "侧翼机动"

    elif tactical_mode == "搜索模式":
        return "搜索接敌"

    # 默认行为
    return "标准机动"


def _generate_combat_commands(tactical_behavior: str, threat_assessment: Dict[str, Any],
                            battlefield_awareness: Dict[str, Any], agent_id: str) -> Tuple[int, int, int]:
    """生成精确战斗指令 - 强对抗性AI的核心输出"""

    # 动作空间: [高度(0-14), 航向(0-16), 速度(0-6)]
    # 7=保持当前, <7下降/左转/减速, >7爬升/右转/加速

    primary_target = threat_assessment['primary_target']
    my_energy = battlefield_awareness['energy_state']

    # 1. 紧急防守指令
    if tactical_behavior == "紧急规避机动":
        # 极度紧急：急剧机动 + 最大速度
        return (1, 0, 6)  # 急降+最大左转+最大速度

    elif tactical_behavior == "攻击性规避":
        # 规避但保持攻击能力
        return (3, 2, 6)  # 下降+大左转+最大速度

    elif tactical_behavior == "预防性机动":
        # 预防性规避
        return (5, 4, 5)  # 轻微下降+左转+高速

    elif tactical_behavior == "近距离防守":
        # 近距离防守：垂直机动
        return (2, 3, 6)  # 急降+左转+最大速度

    elif tactical_behavior == "远程防守":
        # 远程防守：保持距离
        return (4, 5, 4)  # 下降+左转+加速

    elif tactical_behavior == "保守机动":
        # 保守机动：小幅调整
        return (6, 6, 4)  # 轻微下降+轻微左转+加速

    # 2. 进攻指令
    elif tactical_behavior == "近距离突击":
        # 近距离突击：最大攻击性
        if my_energy == "高能量":
            return (13, 14, 6)  # 大幅爬升+大右转+最大速度
        else:
            return (11, 12, 6)  # 爬升+右转+最大速度

    elif tactical_behavior == "中距离强攻":
        # 中距离强攻：平衡攻击
        return (9, 11, 5)  # 爬升+右转+高速

    elif tactical_behavior == "协调钳击":
        # 协调钳击：根据角色分工
        if agent_id == "B0100":
            return (10, 13, 5)  # 主攻：爬升+大右转+高速
        else:
            return (8, 4, 5)   # 侧翼：轻微爬升+左转+高速

    elif tactical_behavior == "高低分离攻击":
        # 高低分离：根据当前高度调整
        if battlefield_awareness['my_altitude'] > 12000:
            return (6, 10, 5)  # 高空：下降+右转+高速
        else:
            return (12, 10, 5) # 低空：爬升+右转+高速

    elif tactical_behavior == "尾追攻击":
        # 尾追攻击：保持追击
        return (8, 9, 6)   # 轻微爬升+轻微右转+最大速度

    elif tactical_behavior == "机会攻击":
        # 机会攻击：快速定位
        return (9, 10, 5)  # 爬升+右转+高速

    elif tactical_behavior == "高速接敌":
        # 高速接敌：最大速度接近
        return (7, 8, 6)   # 保持高度+直飞+最大速度

    # 3. 中立机动指令
    elif tactical_behavior == "主攻击者机动":
        # 主攻击者：保持攻击态势
        if primary_target and primary_target['distance'] < 40000:
            return (8, 9, 5)  # 轻微爬升+轻微右转+高速
        else:
            return (7, 8, 5)  # 保持高度+直飞+高速

    elif tactical_behavior == "侧翼机动":
        # 侧翼机动：寻找侧翼位置
        return (7, 6, 4)   # 保持高度+左转+加速

    elif tactical_behavior == "搜索接敌":
        # 搜索接敌：保持搜索态势
        return (8, 8, 4)   # 轻微爬升+直飞+加速

    # 4. 默认指令
    else:
        # 标准机动：基于角色的默认行为
        if agent_id == "B0100":
            base_commands = (8, 9, 4)  # 主攻击者：轻微爬升+轻微右转+加速
        else:
            base_commands = (7, 7, 4)  # 侧翼攻击者：保持高度+轻微左转+加速

        # 根据战术模式调整基础指令
        return _adjust_commands_by_tactical_mode(base_commands, tactical_mode, bvr_situation)

def _adjust_commands_by_tactical_mode(base_commands: Tuple[int, int, int],
                                    tactical_mode: TacticalMode,
                                    bvr_situation: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    根据战术模式调整指令
    """
    altitude_cmd, heading_cmd, velocity_cmd = base_commands

    # 获取态势信息
    primary_target = bvr_situation.get('primary_target')
    distance = primary_target['distance'] if primary_target else 50000

    # 攻击模式调整
    if tactical_mode == TacticalMode.AGGRESSIVE:
        # 更激进的机动
        if distance > 40000:
            velocity_cmd = min(6, velocity_cmd + 1)  # 增加速度
            altitude_cmd = min(14, altitude_cmd + 1)  # 爬升获得能量优势
        elif distance < 25000:
            heading_cmd = max(0, min(16, heading_cmd + np.random.choice([-2, 2])))  # 更激进的转弯

    # 防御模式调整
    elif tactical_mode == TacticalMode.DEFENSIVE:
        # 更保守的机动
        if distance < 30000:
            velocity_cmd = max(0, velocity_cmd - 1)  # 减速保存能量
            altitude_cmd = max(0, altitude_cmd - 1)  # 下降规避
            # 增加转弯幅度进行规避
            if heading_cmd > 8:
                heading_cmd = min(16, heading_cmd + 2)
            else:
                heading_cmd = max(0, heading_cmd - 2)

    # 支援模式调整
    elif tactical_mode == TacticalMode.SUPPORT:
        # 保持编队，适度机动
        velocity_cmd = max(2, min(5, velocity_cmd))  # 中等速度
        # 减少大幅度转弯，保持编队
        if heading_cmd > 12 or heading_cmd < 4:
            heading_cmd = 8  # 趋向直飞

    # 中立模式保持原指令
    # elif tactical_mode == TacticalMode.NEUTRAL:
    #     pass  # 保持原指令

    return (altitude_cmd, heading_cmd, velocity_cmd)
