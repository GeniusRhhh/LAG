import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import py_trees
from py_trees.trees import BehaviourTree
from py_trees.composites import Sequence, Selector
from py_trees.behaviour import Behaviour
from ..core.catalog import Catalog as c
from ..utils.RadarModel import RadarModel


class ConditionNode(Behaviour):
    def __init__(self, name: str, condition_func: callable):
        super().__init__(name)
        self.condition_func = condition_func

    def update(self):
        return py_trees.common.Status.SUCCESS if self.condition_func() else py_trees.common.Status.FAILURE


class ActionNode(Behaviour):
    def __init__(self, name: str, action_func: callable):
        super().__init__(name)
        self.action_func = action_func

    def update(self):
        self.action_func()
        return py_trees.common.Status.SUCCESS


class EnhancedTacticalTemplate:
    """增强版战术模板系统，支持14种经典超视距空战战术"""

    # 10个作战阶段
    PHASES = [
        "contact_guidance",  # 接敌引导：远距离搜索接近
        "target_search",  # 目标搜索：雷达主动搜索
        "target_identification",  # 目标识别：敌我识别
        "threat_assessment",  # 威胁判断：评估威胁等级
        "target_allocation",  # 目标分配：协同分配目标
        "tactical_decision",  # 战术决策：选择攻击战术
        "missile_launch",  # 发射导弹：执行攻击
        "mid_guidance_defense",  # 中制导防御：导弹中段制导
        "terminal_guidance",  # 末制导：导弹末段攻击
        "effect_assessment"  # 效果评估：评估攻击结果
    ]

    # 战术距离定义
    TACTICAL_DISTANCES = {
        "detection_range": 80000,   # 80km：现代雷达有效探测距离
        "engagement_range": 60000,  # 60km：进入交战意图明确区域
        "launch_range": 40000,      # 40km：AIM-120有效射程
        "mar_range": 20000,         # 20km：最小规避距离(Minimum Abort Range)
        "wez_range": 32000,         # 32km：武器交战区(Weapon Engagement Zone)
        "rmax": 60000,              # 60km：导弹理论最大射程
        "rmin": 5000                # 5km：导弹最小武装距离
    }

    def __init__(self, is_enemy: bool = False, env=None, agent_id: str = None):
        if env is None or agent_id is None:
            raise ValueError("env and agent_id must be provided for TacticalTemplate initialization")

        self.is_enemy = is_enemy
        self.env = env
        self.agent_id = agent_id
        self.radar = RadarModel(max_range=120000, h_beamwidth=60, v_beamwidth=30)

        # 14种战术模板定义
        self.tactical_templates = self._init_tactical_templates()

        self.current_phase = self.PHASES[0]
        self.phase_start_time = 0
        self.behavior_tree = self.build_behavior_tree()
        self.tactic_cooldown = 0
        self.last_template_id = 0
        self.template_history = []

        # 战术状态记录
        self.missile_launched = False
        self.crank_active = False
        self.beam_active = False
        self.notch_active = False
        self.defensive_sequence = []
        self.defensive_action_logged = False
        # logging.info(f"Enhanced TacticalTemplate initialized for {agent_id}, initial phase: {self.current_phase}")

    def _init_tactical_templates(self) -> Dict[int, Dict[str, Any]]:
        """初始化14种战术模板"""
        return {
            0: {"name": "No_Template", "type": "direct_rl", "description": "直接强化学习控制"},

            # 单机规避机动
            1: {"name": "Crank", "type": "evasion",
                "description": "偏离正向30-60度维持雷达锁定",
                "params": {
                    "heading_offset_range": (30, 60),  # 偏航角度范围
                    "maintain_lock": True,
                    "velocity_cmd": 600,
                    "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                    "conditions": ["radar_lock", "missile_threat"]
                }},

            2: {"name": "Beam", "type": "evasion",
                "description": "横向机动消耗导弹动能",
                "params": {
                    "heading_offset": 90,  # 垂直敌机航向
                    "velocity_cmd": 550,
                    "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                    "doppler_optimization": True,
                    "conditions": ["incoming_missile", "doppler_escape"]
                }},

            3: {"name": "Notch", "type": "evasion",
                "description": "利用地面杂波遮蔽",
                "params": {
                    "heading_offset": 90,
                    "altitude_cmd": lambda s: -500 if s["enemy_distance"] > 20000 else -200,
                    "velocity_cmd": 500,
                    "ground_clutter": True,
                    "conditions": ["radar_warning", "altitude_sufficient"]
                }},

            # 进攻战术
            4: {"name": "Skate", "type": "attack",
                "description": "远距发射-机动-转冷-二次攻击",
                "params": {
                    "sequence": ["launch", "crank", "turn_cold", "re_engage", "launch2", "turn_cold"],
                    "velocity_cmd": 650,
                    "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                    "mar_consideration": True,
                    "conditions": ["bvr_range", "first_shot"]
                }},

            5: {"name": "Short_Skate", "type": "attack",
                "description": "发射后快速脱离",
                "params": {
                    "sequence": ["launch", "turn_cold"],
                    "velocity_cmd": 700,
                    "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                    "quick_escape": True,
                    "conditions": ["advantageous_position", "energy_management"]
                }},

            6: {"name": "Banzai", "type": "attack",
                "description": "发射后决策-交汇格斗",
                "params": {
                    "sequence": ["launch", "crank", "merge_decision"],
                    "velocity_cmd": 650,
                    "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                    "merge_threshold": 15000,
                    "conditions": ["wez_entry", "energy_advantage"]
                }},

            7: {"name": "Simple_F_Pole", "type": "attack",
                "description": "保持锁定最大化F-pole",
                "params": {
                    "maintain_lock": True,
                    "velocity_cmd": 600,
                    "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                    "f_pole_optimization": True,
                    "conditions": ["stable_track", "fox3_guidance"]
                }},

            8: {"name": "Advanced_F_Pole", "type": "dynamic_attack",
                "description": "动态F-pole优化",
                "params": {
                    "dynamic_crank": True,
                    "threat_assessment": True,
                    "velocity_cmd": 650,
                    "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                    "intelligent_escape": True,
                    "conditions": ["multi_threat", "advanced_guidance"]
                }},

            # 协同战术 (2v2)
            9: {"name": "Pincer", "type": "cooperative",
                "description": "钳形夹击",
                "params": {
                    "formation_angle": (30, 60),  # 夹角范围
                    "sync_timing": True,
                    "velocity_cmd": 600,
                    "altitude_cmd": lambda s: self._cooperative_altitude_cmd(s),
                    "coordinate_launch": True,
                    "conditions": ["dual_aircraft", "flanking_position"]
                }},

            10: {"name": "Defensive_Split", "type": "cooperative",
                 "description": "防御分割后钳形",
                 "params": {
                     "split_direction": "bilateral",
                     "separation_distance": 10000,
                     "velocity_cmd": 600,
                     "altitude_cmd": lambda s: self._cooperative_altitude_cmd(s),
                     "regroup_condition": "threat_neutralized",
                     "conditions": ["under_attack", "formation_integrity"]
                 }},

            11: {"name": "High_Low", "type": "cooperative",
                 "description": "高低搭配",
                 "params": {
                     "altitude_separation": 3000,  # 高度差
                     "velocity_cmd": 600,
                     "altitude_cmd": lambda s: self._high_low_altitude_cmd(s),
                     "coverage_optimization": True,
                     "conditions": ["complementary_coverage", "altitude_advantage"]
                 }},

            12: {"name": "Engaging_Trail", "type": "cooperative",
                 "description": "前后交战",
                 "params": {
                     "trail_distance": 8000,  # 纵队间距
                     "follow_leader": True,
                     "velocity_cmd": 600,
                     "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                     "backup_engagement": True,
                     "conditions": ["lead_wingman", "sequential_attack"]
                 }},

            13: {"name": "Loose_Deuce", "type": "cooperative",
                 "description": "疏散双机盘旋",
                 "params": {
                     "orbit_radius": 5000,
                     "altitude_separation": 1000,
                     "velocity_cmd": 550,
                     "altitude_cmd": lambda s: self._loose_deuce_altitude_cmd(s),
                     "continuous_coverage": True,
                     "conditions": ["area_denial", "persistent_threat"]
                 }},

            14: {"name": "Defensive_Sequence", "type": "complex_evasion",
                 "description": "连续防御机动序列",
                 "params": {
                     "sequence": ["beam", "notch", "f_pole", "crank", "turn_cold", "drag_split"],
                     "adaptive_timing": True,
                     "velocity_cmd": lambda s: self._defensive_velocity_cmd(s),
                     "altitude_cmd": lambda s: self._defensive_altitude_cmd(s),
                     "threat_priority": True,
                     "conditions": ["multiple_threats", "defensive_posture"]
                 }}
        }

    def _adaptive_altitude_cmd(self, state: Dict[str, Any]) -> float:
        """自适应高度指令"""
        current_alt = state.get("current_altitude", 5000)
        enemy_dist = state.get("enemy_distance", 50000)
        has_warning = state.get("has_warning", False)

        # 基础目标高度 (米)
        if enemy_dist > 80000:
            target_alt = 9000  # 高空巡航
        elif enemy_dist > 50000:
            target_alt = 8000  # 中高空交战
        else:
            target_alt = 7000  # 中空格斗

        # 威胁响应
        if has_warning and current_alt > 3000:
            target_alt = max(current_alt - 1000, 2000)  # 下降规避

        alt_diff = target_alt - current_alt
        return np.clip(alt_diff, -800, 800)

    def _cooperative_altitude_cmd(self, state: Dict[str, Any]) -> float:
        """协同战术高度指令"""
        is_leader = state.get("is_leader", False)
        base_cmd = self._adaptive_altitude_cmd(state)

        if is_leader:
            return base_cmd + 200  # 长机稍高
        else:
            return base_cmd - 200  # 僚机稍低

    def _high_low_altitude_cmd(self, state: Dict[str, Any]) -> float:
        """高低搭配高度指令"""
        is_leader = state.get("is_leader", False)
        current_alt = state.get("current_altitude", 5000)

        if is_leader:  # 高机
            target_alt = 12000
        else:  # 低机
            target_alt = 6000

        alt_diff = target_alt - current_alt
        return np.clip(alt_diff, -1000, 1000)

    def _loose_deuce_altitude_cmd(self, state: Dict[str, Any]) -> float:
        """疏散双机高度指令"""
        is_leader = state.get("is_leader", False)
        base_cmd = self._adaptive_altitude_cmd(state)

        # 通过高度差避免碰撞
        offset = 500 if is_leader else -500
        return base_cmd + offset

    def _defensive_altitude_cmd(self, state: Dict[str, Any]) -> float:
        """防御机动高度指令"""
        missile_dist = state.get("missile_distance", np.inf)
        current_alt = state.get("current_altitude", 5000)

        if missile_dist < 20000:
            # 紧急下降规避
            return -1000 if current_alt > 3000 else 0
        else:
            return self._adaptive_altitude_cmd(state)

    def _defensive_velocity_cmd(self, state: Dict[str, Any]) -> float:
        """防御机动速度指令"""
        missile_dist = state.get("missile_distance", np.inf)

        if missile_dist < 15000:
            return 750  # 最大速度规避
        elif missile_dist < 30000:
            return 650  # 高速机动
        else:
            return 600  # 正常速度

    def get_tactical_action(self, template_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """获取战术动作指令"""
        if template_id not in self.tactical_templates:
            logging.warning(f"Unknown template_id: {template_id}, using default")
            template_id = 7  # 默认使用Simple_F_Pole

        template = self.tactical_templates[template_id]
        self.last_template_id = template_id
        self.template_history.append((template_id, getattr(self.env, 'current_step', 0)))

        # 生成基础动作
        action = self._generate_base_action(template, state)

        # 特殊战术逻辑处理
        action = self._apply_tactical_logic(template_id, template, state, action)

        logging.debug(f"Agent {self.agent_id} tactical action: template={template['name']}, "
                      f"heading_cmd={action.get('heading_cmd', 0):.3f}, "
                      f"altitude_cmd={action.get('altitude_cmd', 0):.1f}, "
                      f"velocity_cmd={action.get('velocity_cmd', 600):.1f}")

        return action

    def _generate_base_action(self, template: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """生成基础动作指令"""
        params = template["params"]

        # 基础参数
        action = {
            "maneuver": template["name"],
            "heading_cmd": 0.0,
            "altitude_cmd": 0.0,
            "velocity_cmd": params.get("velocity_cmd", 600),
            "shoot": False,
            "template_type": template["type"]
        }

        # 高度指令
        if callable(params.get("altitude_cmd")):
            action["altitude_cmd"] = params["altitude_cmd"](state)
        else:
            action["altitude_cmd"] = params.get("altitude_cmd", 0)

        # 速度指令
        if callable(params.get("velocity_cmd")):
            action["velocity_cmd"] = params["velocity_cmd"](state)
        else:
            action["velocity_cmd"] = params.get("velocity_cmd", 600)

        return action

    def _apply_tactical_logic(self, template_id: int, template: Dict[str, Any],
                              state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """应用具体战术逻辑"""
        enemy_distance = state.get("enemy_distance", 50000)
        enemy_angle_off = state.get("enemy_angle_off", 0)
        radar_lock = state.get("radar_lock", False)
        has_warning = state.get("has_warning", False)
        missile_distance = state.get("missile_distance", np.inf)

        if template_id == 1:  # Crank
            action.update(self._crank_logic(state))
        elif template_id == 2:  # Beam
            action.update(self._beam_logic(state))
        elif template_id == 3:  # Notch
            action.update(self._notch_logic(state))
        elif template_id == 4:  # Skate
            action.update(self._skate_logic(state))
        elif template_id == 5:  # Short_Skate
            action.update(self._short_skate_logic(state))
        elif template_id == 6:  # Banzai
            action.update(self._banzai_logic(state))
        elif template_id == 7:  # Simple_F_Pole
            action.update(self._simple_f_pole_logic(state))
        elif template_id == 8:  # Advanced_F_Pole
            action.update(self._advanced_f_pole_logic(state))
        elif template_id == 9:  # Pincer
            action.update(self._pincer_logic(state))
        elif template_id == 10:  # Defensive_Split
            action.update(self._defensive_split_logic(state))
        elif template_id == 11:  # High_Low
            action.update(self._high_low_logic(state))
        elif template_id == 12:  # Engaging_Trail
            action.update(self._engaging_trail_logic(state))
        elif template_id == 13:  # Loose_Deuce
            action.update(self._loose_deuce_logic(state))
        elif template_id == 14:  # Defensive_Sequence
            action.update(self._defensive_sequence_logic(state))

        return action

    def _crank_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Crank机动逻辑：偏离30-60度维持雷达锁定"""
        enemy_angle_off = state.get("enemy_angle_off", 0)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)

        # 根据威胁程度调整偏航角
        if missile_distance < 25000:
            crank_angle = np.radians(45)  # 较大偏航
        else:
            crank_angle = np.radians(30)  # 标准偏航

        # 选择偏航方向
        if enemy_angle_off > 0:
            heading_cmd = -crank_angle  # 向左偏
        else:
            heading_cmd = crank_angle  # 向右偏

        return {
            "heading_cmd": heading_cmd,
            "maintain_lock": True,
            "shoot": radar_lock and state.get("enemy_distance", 50000) < 35000
        }

    def _beam_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Beam机动逻辑：90度横向机动"""
        enemy_angle_off = state.get("enemy_angle_off", 0)

        # 确保90度横向
        if abs(enemy_angle_off) < 85:
            if enemy_angle_off >= 0:
                heading_cmd = np.radians(90 - enemy_angle_off)
            else:
                heading_cmd = np.radians(-90 - enemy_angle_off)
        else:
            heading_cmd = 0  # 已接近垂直

        return {
            "heading_cmd": heading_cmd,
            "doppler_minimize": True,
            "shoot": False  # Beam时不射击
        }

    def _notch_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Notch机动逻辑：利用地面杂波"""
        current_alt = state.get("current_altitude", 5000)
        beam_action = self._beam_logic(state)

        # 在Beam基础上加入高度机动
        if current_alt > 3000:
            beam_action["altitude_cmd"] = -600  # 下降进入杂波区
        else:
            beam_action["altitude_cmd"] = 0

        beam_action["ground_clutter"] = True
        return beam_action

    def _skate_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Skate机动逻辑：复杂攻击序列"""
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)

        # 简化的状态机
        if enemy_distance > 40000 and radar_lock:
            # 第一次发射
            return {"heading_cmd": 0, "shoot": True, "phase": "launch1"}
        elif enemy_distance > 25000:
            # Crank机动
            return self._crank_logic(state)
        elif enemy_distance > 15000:
            # Turn Cold
            return {"heading_cmd": np.pi, "shoot": False, "phase": "turn_cold"}
        else:
            # 重新接敌
            return {"heading_cmd": 0, "shoot": radar_lock, "phase": "re_engage"}

    def _short_skate_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Short Skate逻辑：发射后快速脱离"""
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)

        if enemy_distance > 35000 and radar_lock:
            return {"heading_cmd": 0, "shoot": True, "phase": "launch"}
        else:
            return {"heading_cmd": np.pi, "shoot": False, "phase": "escape"}

    def _banzai_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Banzai逻辑：发射后决策"""
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)

        if enemy_distance > 30000:
            return {"heading_cmd": 0, "shoot": radar_lock}
        elif enemy_distance > 15000:
            return self._crank_logic(state)
        else:
            # 决定是否交汇
            return {"heading_cmd": 0, "shoot": False, "merge_decision": True}

    def _simple_f_pole_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple F-Pole逻辑：保持锁定"""
        return {
            "heading_cmd": 0,  # 保持正向
            "maintain_lock": True,
            "shoot": state.get("radar_lock", False) and state.get("enemy_distance", 50000) < 40000
        }

    def _advanced_f_pole_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced F-Pole逻辑：动态优化"""
        base_action = self._simple_f_pole_logic(state)

        # 威胁评估
        if state.get("has_warning", False):
            # 结合Crank机动
            crank_action = self._crank_logic(state)
            base_action["heading_cmd"] = crank_action["heading_cmd"] * 0.5  # 减小偏航

        return base_action

    # 钳形夹击
    def _pincer_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        is_leader = state.get("is_leader", False)
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        # 计算理想夹角(45度)
        if is_leader:
            target_angle = np.radians(22.5)  # 减小夹角
        else:
            target_angle = np.radians(-22.5)
        heading_correction = target_angle
        return {
            "heading_cmd": np.clip(heading_correction, -np.radians(30), np.radians(30)),
            "coordinate_attack": True,
            "shoot": radar_lock and 40000 <= enemy_distance <= 80000
        }

    def _defensive_split_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Defensive Split逻辑：防御分割"""
        is_leader = state.get("is_leader", False)

        # 分别向两侧机动
        if is_leader:
            heading_cmd = np.radians(45)  # 向右
        else:
            heading_cmd = np.radians(-45)  # 向左

        return {
            "heading_cmd": heading_cmd,
            "separation_maneuver": True,
            "shoot": False  # 分离时不射击
        }

    def _high_low_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """High-Low逻辑：高低搭配"""
        return {
            "heading_cmd": 0,  # 保持航向
            "altitude_optimization": True,
            "shoot": state.get("radar_lock", False)
        }

    def _engaging_trail_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Engaging Trail逻辑：前后交战"""
        is_leader = state.get("is_leader", False)

        if is_leader:
            return {"heading_cmd": 0, "shoot": state.get("radar_lock", False), "role": "lead"}
        else:
            return {"heading_cmd": 0, "shoot": False, "role": "follow", "trail_distance": 8000}

    def _loose_deuce_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Loose Deuce逻辑：疏散双机"""
        return {
            "heading_cmd": 0,
            "orbit_pattern": True,
            "shoot": state.get("radar_lock", False)
        }

    def _defensive_sequence_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Defensive Sequence逻辑：连续防御"""
        missile_distance = state.get("missile_distance", np.inf)

        if missile_distance < 15000:
            return self._beam_logic(state)
        elif missile_distance < 25000:
            return self._notch_logic(state)
        elif missile_distance < 35000:
            return self._crank_logic(state)
        else:
            return {"heading_cmd": np.pi, "shoot": False}  # Turn Cold

    def build_behavior_tree(self):
        """构建行为树"""
        root = Selector("Root", memory=True)

        # 威胁响应优先级最高
        threat_response = Sequence("Threat_Response", memory=True)
        threat_condition = ConditionNode("Missile_Threat",
                                         lambda: self._check_immediate_threat())
        threat_action = ActionNode("Execute_Defense",
                                   lambda: self._execute_defensive_action())
        threat_response.add_children([threat_condition, threat_action])
        root.add_child(threat_response)

        # 各阶段处理
        for phase in self.PHASES:
            phase_node = Sequence(f"Phase_{phase}", memory=True)
            condition_node = ConditionNode(f"{phase}_conditions",
                                           lambda p=phase: self._check_phase_conditions(p))
            action_node = ActionNode(f"{phase}_actions",
                                     lambda p=phase: self._execute_phase_actions(p))
            phase_node.add_children([condition_node, action_node])
            root.add_child(phase_node)

        behavior_tree = BehaviourTree(root)
        return behavior_tree

    def _check_immediate_threat(self) -> bool:
        """检查紧急威胁"""
        if not hasattr(self.env, 'agents') or self.agent_id not in self.env.agents:
            return False

        agent = self.env.agents[self.agent_id]
        missile_sim = agent.check_missile_warning()

        if missile_sim:
            distance = np.linalg.norm(missile_sim.get_position() - agent.get_position())
            return distance < 30000
        return False

    def _execute_defensive_action(self):
        """执行防御动作"""
        self.current_phase = "mid_guidance_defense"
        if not self.defensive_action_logged:
            logging.info(f"Agent {self.agent_id} executing defensive action")
            self.defensive_action_logged = True

    def _check_phase_conditions(self, phase: str) -> bool:
        """检查阶段条件"""
        if not hasattr(self.env, 'agents') or self.agent_id not in self.env.agents:
            return False

        agent = self.env.agents[self.agent_id]
        if not agent.is_alive:
            return False

        # 简化的阶段条件检查
        enemies = agent.enemies
        if not enemies:
            return False

        enemy_distance = min([np.linalg.norm(enemy.get_position() - agent.get_position())
                              for enemy in enemies if enemy.is_alive], default=np.inf)

        if phase == "contact_guidance":
            return enemy_distance > self.TACTICAL_DISTANCES["detection_range"]
        elif phase == "target_search":
            return enemy_distance <= self.TACTICAL_DISTANCES["detection_range"]
        elif phase == "target_identification":
            return enemy_distance <= self.TACTICAL_DISTANCES["engagement_range"]
        elif phase == "threat_assessment":
            return enemy_distance <= self.TACTICAL_DISTANCES["wez_range"]
        elif phase == "target_allocation":
            return enemy_distance <= self.TACTICAL_DISTANCES["launch_range"]
        elif phase == "tactical_decision":
            return enemy_distance <= self.TACTICAL_DISTANCES["launch_range"]
        elif phase == "missile_launch":
            return enemy_distance <= self.TACTICAL_DISTANCES["launch_range"]
        elif phase == "mid_guidance_defense":
            return enemy_distance <= self.TACTICAL_DISTANCES["mar_range"]
        elif phase == "terminal_guidance":
            return enemy_distance <= self.TACTICAL_DISTANCES["rmin"]
        elif phase == "effect_assessment":
            return True

        return False

    def _execute_phase_actions(self, phase: str):
        """执行阶段动作"""
        self.current_phase = phase
        logging.debug(f"Agent {self.agent_id} executing phase: {phase}")

    def update_phase(self, state: Dict[str, Any]) -> str:
        """更新相位"""
        if self.tactic_cooldown > 0:
            self.tactic_cooldown -= 1
            return self.current_phase

        prev_phase = self.current_phase
        # 如果不再处于防御阶段，重置日志标志
        if not self._check_immediate_threat():
            self.defensive_action_logged = False
        try:
            self.behavior_tree.tick()
        except Exception as e:
            logging.error(f"Behavior tree tick error for {self.agent_id}: {e}")

        if prev_phase != self.current_phase:
            logging.info(f"Agent {self.agent_id} phase changed: {prev_phase} -> {self.current_phase}")

        return self.current_phase

    def get_radar_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """获取雷达状态"""
        try:
            radar_state = self.radar.get_radar_state(state, self.env, self.agent_id)
            return radar_state
        except Exception as e:
            logging.error(f"Radar state error for {self.agent_id}: {e}")
            return {"radar_lock": False, "has_warning": False, "snr": 0.0}

    def get_template_info(self, template_id: int) -> Dict[str, Any]:
        """获取战术模板信息"""
        if template_id in self.tactical_templates:
            template = self.tactical_templates[template_id]
            return {
                "name": template["name"],
                "type": template["type"],
                "description": template["description"]
            }
        return {"name": "Unknown", "type": "unknown", "description": "Unknown template"}


class TacticalTemplate(EnhancedTacticalTemplate):
    """保持向后兼容的战术模板类"""

    def __init__(self, is_enemy: bool = False, env=None, agent_id: str = None):
        super().__init__(is_enemy, env, agent_id)
        logging.warning("Using deprecated TacticalTemplate class, consider upgrading to EnhancedTacticalTemplate")

    def get_action(self, template_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """向后兼容的动作获取方法"""
        return self.get_tactical_action(template_id, state)