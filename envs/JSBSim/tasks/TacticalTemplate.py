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
        "engagement_range": 65000,  # 60km：进入交战意图明确区域
        "launch_range": 45000,      # 40km：AIM-120有效射程
        "mar_range": 20000,         # 20km：最小规避距离(Minimum Abort Range)
        "wez_range": 35000,         # 32km：武器交战区(Weapon Engagement Zone)
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
        self._skate_state = None
        self._short_skate_state = None
        self._banzai_state = None
        self._defense_sequence_state = None
        self._orbit_state = None

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
        """修复后的Crank机动逻辑 - 更合理的触发条件"""

        enemy_distance = state.get("enemy_distance", 50000)
        enemy_angle_off = state.get("enemy_angle_off", 0)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)
        missiles_incoming = state.get("missiles_incoming", [])
        has_missile_threat = len([m for m in missiles_incoming if hasattr(m, 'is_alive') and m.is_alive]) > 0

        # 【关键修复】: 更宽松和实用的触发条件
        should_crank = (
                enemy_distance < 65000 or  # 扩大触发距离到65km
                missile_distance < 40000 or  # 扩大导弹威胁距离
                has_missile_threat or  # 任何导弹威胁
                (enemy_distance < 70000 and radar_lock)  # 或者被锁定且距离<70km
        )

        if not should_crank:
            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 600,
                "shoot": False,
                "maneuver_active": False
            }

        # 威胁等级评估
        if missile_distance < 25000 or has_missile_threat:
            crank_angle = np.radians(55)  # 高威胁：大角度机动
            velocity_boost = 75
            priority = "HIGH"
        elif enemy_distance < 45000:
            crank_angle = np.radians(40)  # 中威胁：中等角度
            velocity_boost = 50
            priority = "MEDIUM"
        else:
            crank_angle = np.radians(30)  # 低威胁：小角度
            velocity_boost = 25
            priority = "LOW"

        # 机动方向选择（基于敌机相对位置）
        if enemy_angle_off > 0:
            heading_cmd = -crank_angle  # 向左规避
        else:
            heading_cmd = crank_angle  # 向右规避

        # 射击条件评估
        shoot_ok = (
                radar_lock and
                30000 < enemy_distance < 50000 and  # 射击窗口
                abs(heading_cmd) < np.radians(50) and  # 角度容限
                not has_missile_threat  # 无导弹威胁时才射击
        )

        logging.info(f"Crank执行: 距离={enemy_distance:.0f}m, 角度={np.degrees(heading_cmd):.1f}°, "
                     f"威胁={priority}, 射击={shoot_ok}")

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": 0,
            "velocity_cmd": 600 + velocity_boost,
            "maintain_lock": True,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "crank_angle": abs(heading_cmd),  # 用于质量评估
            "threat_priority": priority
        }

    # 4. 立即修改Beam机动逻辑
    def _beam_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Beam机动逻辑 - 90度横向机动消耗导弹动能"""

        enemy_angle_off = state.get("enemy_angle_off", 0)
        missile_distance = state.get("missile_distance", np.inf)
        enemy_distance = state.get("enemy_distance", 50000)
        has_missile_threat = len(
            [m for m in state.get("missiles_incoming", []) if hasattr(m, 'is_alive') and m.is_alive]) > 0

        # 触发条件
        should_beam = (
                missile_distance < 30000 or
                has_missile_threat or
                (enemy_distance < 25000 and abs(enemy_angle_off) < 45)  # 近距离正面威胁
        )

        if not should_beam:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        # 计算90度横向机动
        target_angle = 90.0  # 目标90度
        current_angle = abs(enemy_angle_off)

        if current_angle < 85:  # 还没到90度
            angle_deficit = target_angle - current_angle
            if enemy_angle_off >= 0:
                heading_cmd = np.radians(min(angle_deficit, 60))  # 向左转，最大60度
            else:
                heading_cmd = np.radians(-min(angle_deficit, 60))  # 向右转
        else:
            heading_cmd = 0  # 已接近垂直，保持

        # 威胁等级调整
        if missile_distance < 15000:
            velocity_cmd = 700  # 最高速度规避
            altitude_cmd = 150  # 轻微爬升
            beam_intensity = "MAXIMUM"
        elif missile_distance < 25000:
            velocity_cmd = 650  # 高速规避
            altitude_cmd = 100
            beam_intensity = "HIGH"
        else:
            velocity_cmd = 620
            altitude_cmd = 0
            beam_intensity = "STANDARD"

        logging.debug(
            f"Beam执行: 目标角度=90°, 当前={enemy_angle_off:.1f}°, 转向={np.degrees(heading_cmd):.1f}°, 强度={beam_intensity}")

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": altitude_cmd,
            "velocity_cmd": velocity_cmd,
            "doppler_minimize": True,
            "shoot": False,  # Beam时不射击
            "maneuver_active": True,
            "beam_angle": abs(enemy_angle_off),
            "beam_intensity": beam_intensity
        }

    def _notch_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Notch机动逻辑 - 地面杂波遮蔽"""

        current_alt = state.get("current_altitude", 5000)
        missile_distance = state.get("missile_distance", np.inf)
        enemy_distance = state.get("enemy_distance", 50000)
        has_missile_threat = len(
            [m for m in state.get("missiles_incoming", []) if hasattr(m, 'is_alive') and m.is_alive]) > 0

        # 触发条件
        should_notch = (
                missile_distance < 35000 or
                has_missile_threat or
                (enemy_distance < 20000)  # 近距离必须用Notch
        )

        if not should_notch:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        # 首先执行Beam机动
        beam_action = self._beam_logic(state)

        # 高度管理 - Notch的关键
        if missile_distance < 25000:
            # 紧急情况：快速下降到杂波区
            if current_alt > 2500:
                target_altitude = 1500  # 目标1500米
                altitude_cmd = max(-1000, target_altitude - current_alt)
                notch_phase = "emergency_descent"
            else:
                altitude_cmd = -300  # 已在低空，继续下降
                notch_phase = "in_clutter"
        elif missile_distance < 40000:
            # 预防性下降
            if current_alt > 3500:
                target_altitude = 2500
                altitude_cmd = max(-600, target_altitude - current_alt)
                notch_phase = "preventive_descent"
            else:
                altitude_cmd = -200
                notch_phase = "clutter_level"
        else:
            altitude_cmd = 0
            notch_phase = "normal"

        # 速度管理
        if current_alt < 2000:
            velocity_cmd = 580  # 低空减速，保持控制
        elif altitude_cmd < -500:
            velocity_cmd = 650  # 下降时可以加速
        else:
            velocity_cmd = 620

        # 计算杂波效果
        clutter_effectiveness = 0.0
        if current_alt < 1800:
            clutter_effectiveness = 0.85  # 85%杂波遮蔽
        elif current_alt < 2500:
            clutter_effectiveness = 0.65  # 65%效果
        elif current_alt < 3500:
            clutter_effectiveness = 0.35  # 35%效果

        logging.debug(
            f"Notch执行: 高度={current_alt:.0f}m, 下降={altitude_cmd:.0f}m, 杂波效果={clutter_effectiveness:.2f}, 阶段={notch_phase}")

        # 继承Beam的横向机动，添加高度控制
        notch_action = beam_action.copy()
        notch_action.update({
            "altitude_cmd": altitude_cmd,
            "velocity_cmd": velocity_cmd,
            "ground_clutter": True,
            "clutter_effectiveness": clutter_effectiveness,
            "notch_phase": notch_phase,
            "target_altitude": 1500 if missile_distance < 25000 else current_alt
        })

        return notch_action

    def _skate_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Skate机动逻辑 - 复杂攻击序列"""

        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)

        # 使用内部状态跟踪Skate阶段
        if not hasattr(self, '_skate_state'):
            self._skate_state = {"phase": "approach", "phase_timer": 0, "missiles_fired": 0}

        self._skate_state["phase_timer"] += 1

        # 阶段1：远距离发射
        if (self._skate_state["phase"] == "approach" and
                enemy_distance > 45000 and radar_lock and
                self._skate_state["missiles_fired"] == 0):

            self._skate_state["phase"] = "first_launch"
            self._skate_state["missiles_fired"] = 1

            logging.debug(f"Skate阶段1: 远距发射, 距离={enemy_distance:.0f}m")

            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 620,
                "shoot": True,
                "maneuver_active": True,
                "skate_phase": "first_launch"
            }

        # 阶段2：Crank机动
        elif (self._skate_state["phase"] == "first_launch" and
              self._skate_state["phase_timer"] > 50):  # 发射后等待50步

            if enemy_distance > 25000:
                self._skate_state["phase"] = "crank"
                crank_action = self._crank_logic(state)
                crank_action["skate_phase"] = "crank"

                logging.debug(f"Skate阶段2: Crank机动, 距离={enemy_distance:.0f}m")
                return crank_action
            else:
                self._skate_state["phase"] = "turn_cold"

        # 阶段3：转冷脱离
        elif (self._skate_state["phase"] == "crank" and
              (missile_distance < 20000 or enemy_distance < 25000)):

            self._skate_state["phase"] = "turn_cold"
            self._skate_state["phase_timer"] = 0

            logging.debug(f"Skate阶段3: 转冷脱离, 导弹距离={missile_distance:.0f}m")

            return {
                "heading_cmd": np.pi,  # 180度转向
                "altitude_cmd": 0,
                "velocity_cmd": 700,  # 最高速度脱离
                "shoot": False,
                "maneuver_active": True,
                "skate_phase": "turn_cold"
            }

        # 阶段4：重新接敌
        elif (self._skate_state["phase"] == "turn_cold" and
              self._skate_state["phase_timer"] > 200 and  # 脱离200步
              missile_distance > 40000):

            self._skate_state["phase"] = "re_engage"

            logging.debug(f"Skate阶段4: 重新接敌, 距离={enemy_distance:.0f}m")

            return {
                "heading_cmd": 0,  # 转向敌机
                "altitude_cmd": 0,
                "velocity_cmd": 650,
                "shoot": radar_lock and 35000 < enemy_distance < 50000,
                "maneuver_active": True,
                "skate_phase": "re_engage"
            }

        # 阶段5：第二次发射+最终脱离
        elif (self._skate_state["phase"] == "re_engage" and
              radar_lock and 35000 < enemy_distance < 50000 and
              self._skate_state["missiles_fired"] < 2):

            self._skate_state["missiles_fired"] = 2
            self._skate_state["phase"] = "final_escape"

            logging.debug(f"Skate阶段5: 第二次发射, 距离={enemy_distance:.0f}m")

            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 620,
                "shoot": True,
                "maneuver_active": True,
                "skate_phase": "second_launch"
            }

        # 默认：保持当前机动或结束
        else:
            current_phase = self._skate_state.get("phase", "complete")
            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 600,
                "shoot": False,
                "maneuver_active": current_phase != "complete",
                "skate_phase": current_phase
            }

    def _short_skate_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Short Skate机动逻辑 - 发射后快速脱离"""

        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)

        # 简化的两阶段Skate
        if not hasattr(self, '_short_skate_state'):
            self._short_skate_state = {"phase": "approach", "fired": False, "timer": 0}

        self._short_skate_state["timer"] += 1

        # 阶段1：发射
        if (not self._short_skate_state["fired"] and
                enemy_distance > 40000 and radar_lock):

            self._short_skate_state["fired"] = True
            self._short_skate_state["phase"] = "launch"

            logging.debug(f"Short Skate发射: 距离={enemy_distance:.0f}m")

            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 640,
                "shoot": True,
                "maneuver_active": True,
                "short_skate_phase": "launch"
            }

        # 阶段2：立即脱离
        elif (self._short_skate_state["fired"] and
              self._short_skate_state["timer"] > 30):  # 发射后30步开始脱离

            self._short_skate_state["phase"] = "escape"

            # 根据威胁程度调整脱离强度
            if missile_distance < 30000:
                escape_intensity = "EMERGENCY"
                heading_cmd = np.pi  # 180度掉头
                velocity_cmd = 720  # 最高速度
            else:
                escape_intensity = "STANDARD"
                heading_cmd = np.pi * 0.8  # 144度转向
                velocity_cmd = 680

            logging.debug(f"Short Skate脱离: 强度={escape_intensity}, 导弹距离={missile_distance:.0f}m")

            return {
                "heading_cmd": heading_cmd,
                "altitude_cmd": 0,
                "velocity_cmd": velocity_cmd,
                "shoot": False,
                "maneuver_active": True,
                "short_skate_phase": "escape",
                "escape_intensity": escape_intensity
            }

        # 默认接近阶段
        else:
            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 630,
                "shoot": False,
                "maneuver_active": True,
                "short_skate_phase": "approach"
            }

    def _banzai_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Banzai机动逻辑 - 发射后决策(L&D)"""

        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)

        # Banzai状态跟踪
        if not hasattr(self, '_banzai_state'):
            self._banzai_state = {"phase": "approach", "fired": False, "decision_made": False}

        # 阶段1：MAR外发射
        if (not self._banzai_state["fired"] and
                enemy_distance > 25000 and radar_lock):  # MAR外（25km外）

            self._banzai_state["fired"] = True
            self._banzai_state["phase"] = "post_launch"

            logging.debug(f"Banzai发射: 距离={enemy_distance:.0f}m (MAR外)")

            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 620,
                "shoot": True,
                "maneuver_active": True,
                "banzai_phase": "launch"
            }

        # 阶段2：发射后Crank机动
        elif (self._banzai_state["fired"] and
              not self._banzai_state["decision_made"] and
              enemy_distance > 20000):

            # 执行Crank机动防御敌方导弹
            crank_action = self._crank_logic(state)
            crank_action.update({
                "banzai_phase": "post_launch_crank",
                "awaiting_decision": True
            })

            logging.debug(f"Banzai Crank: 距离={enemy_distance:.0f}m, 等待决策")
            return crank_action

        # 阶段3：决策点 - 敌机存活则交汇
        elif (self._banzai_state["fired"] and
              enemy_distance <= 20000):  # 进入决策距离

            self._banzai_state["decision_made"] = True

            # 检查敌机是否被击中（简化判断）
            enemy_alive = True  # 实际应该检查敌机状态

            if enemy_alive:
                # 决定交汇格斗
                self._banzai_state["phase"] = "merge"

                logging.debug(f"Banzai决策: 敌机存活，准备交汇格斗")

                return {
                    "heading_cmd": 0,  # 直接朝向敌机
                    "altitude_cmd": 0,
                    "velocity_cmd": 650,
                    "shoot": False,  # 准备近距格斗
                    "maneuver_active": True,
                    "banzai_phase": "merge_commit",
                    "merge_decision": "commit"
                }
            else:
                # 敌机被击中，脱离
                return {
                    "heading_cmd": np.pi,
                    "altitude_cmd": 200,
                    "velocity_cmd": 600,
                    "shoot": False,
                    "maneuver_active": False,
                    "banzai_phase": "target_destroyed"
                }

        # 默认
        else:
            return {
                "heading_cmd": 0,
                "altitude_cmd": 0,
                "velocity_cmd": 620,
                "shoot": False,
                "maneuver_active": True,
                "banzai_phase": "approach"
            }

    def _simple_f_pole_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple F-Pole机动逻辑 - 保持锁定最大化F-pole"""

        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        missile_distance = state.get("missile_distance", np.inf)

        # 触发条件
        should_f_pole = (
                enemy_distance > 30000 and  # 适合中远距离
                radar_lock  # 需要雷达锁定
        )

        if not should_f_pole:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        # F-Pole的核心：保持正向锁定，最大化命中时距离
        # 根据距离调整策略
        if enemy_distance > 60000:
            # 远距离：接近
            heading_cmd = 0
            velocity_cmd = 650
            f_pole_strategy = "approach"
        elif enemy_distance > 40000:
            # 中距离：保持锁定，准备发射
            heading_cmd = 0
            velocity_cmd = 620
            f_pole_strategy = "maintain"
        else:
            # 近距离：轻微拉开距离
            heading_cmd = np.radians(15)  # 轻微偏离
            velocity_cmd = 600
            f_pole_strategy = "extend"

        # 射击条件
        shoot_ok = (
                radar_lock and
                35000 < enemy_distance < 55000 and  # 理想发射窗口
                missile_distance > 40000  # 无紧急威胁
        )

        logging.debug(f"Simple F-Pole: 距离={enemy_distance:.0f}m, 策略={f_pole_strategy}, 射击={shoot_ok}")

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": 0,
            "velocity_cmd": velocity_cmd,
            "maintain_lock": True,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "f_pole_strategy": f_pole_strategy,
            "optimal_f_pole": True
        }

    def _advanced_f_pole_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced F-Pole机动逻辑 - 动态态势判断"""

        # 基础F-Pole
        base_action = self._simple_f_pole_logic(state)

        enemy_distance = state.get("enemy_distance", 50000)
        has_warning = state.get("has_warning", False)
        missile_distance = state.get("missile_distance", np.inf)

        # 威胁分析和智能调整
        if has_warning or missile_distance < 35000:
            # 有威胁时结合Crank机动
            crank_action = self._crank_logic(state)

            # 混合策略：保持F-Pole优势的同时规避威胁
            heading_cmd = crank_action["heading_cmd"] * 0.6  # 减小偏航角度
            velocity_cmd = max(base_action["velocity_cmd"], crank_action["velocity_cmd"])

            advanced_strategy = "threat_aware_f_pole"

            logging.debug(f"Advanced F-Pole: 威胁感知模式, 距离={enemy_distance:.0f}m")

        elif enemy_distance < 30000:
            # 近距离动态调整
            heading_cmd = np.radians(25)  # 增大偏离角
            velocity_cmd = 580  # 减速控制
            advanced_strategy = "close_range_management"

        else:
            # 标准F-Pole
            heading_cmd = base_action["heading_cmd"]
            velocity_cmd = base_action["velocity_cmd"]
            advanced_strategy = "standard_f_pole"

        # 更新base_action
        base_action.update({
            "heading_cmd": heading_cmd,
            "velocity_cmd": velocity_cmd,
            "advanced_strategy": advanced_strategy,
            "threat_adaptive": has_warning
        })

        return base_action

    def _pincer_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Pincer机动逻辑 - 钳形夹击(2v2)"""

        is_leader = state.get("is_leader", False)
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)

        # 触发条件：适合中距离协同攻击
        should_pincer = (
                40000 < enemy_distance < 80000 and
                radar_lock  # 需要锁定目标
        )

        if not should_pincer:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        # 钳形夹角控制：30-60度
        target_separation_angle = np.radians(45)  # 理想45度夹角

        if is_leader:
            # 长机：向右侧机动
            heading_cmd = np.radians(22.5)  # 向右偏离22.5度
            role = "primary_attacker"
        else:
            # 僚机：向左侧机动
            heading_cmd = np.radians(-22.5)  # 向左偏离22.5度
            role = "secondary_attacker"

        # 距离相关的速度调整
        if enemy_distance > 65000:
            velocity_cmd = 680  # 加速接近
            pincer_phase = "approach"
        elif enemy_distance > 45000:
            velocity_cmd = 640  # 保持速度
            pincer_phase = "execute"
        else:
            velocity_cmd = 600  # 减速控制
            pincer_phase = "close_range"

        # 同步射击窗口
        shoot_ok = (
                radar_lock and
                45000 < enemy_distance < 70000 and
                pincer_phase == "execute"
        )

        logging.debug(f"Pincer机动: {role}, 距离={enemy_distance:.0f}m, 阶段={pincer_phase}")

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": 0,
            "velocity_cmd": velocity_cmd,
            "coordinate_attack": True,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "pincer_role": role,
            "pincer_phase": pincer_phase,
            "separation_angle": target_separation_angle
        }

    def _defensive_split_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Defensive Split机动逻辑 - 防御分割(2v2)"""

        is_leader = state.get("is_leader", False)
        enemy_distance = state.get("enemy_distance", 50000)
        has_warning = state.get("has_warning", False)
        missile_distance = state.get("missile_distance", np.inf)

        # 触发条件：受到威胁或被追击
        should_split = (
                has_warning or
                missile_distance < 40000 or
                enemy_distance < 30000  # 敌机过于接近
        )

        if not should_split:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        # 分离强度根据威胁程度调整
        if missile_distance < 20000:
            # 紧急分离
            separation_angle = np.radians(60)  # 大角度分离
            velocity_cmd = 720  # 最高速度
            split_intensity = "EMERGENCY"
        elif missile_distance < 35000:
            # 标准分离
            separation_angle = np.radians(45)
            velocity_cmd = 680
            split_intensity = "STANDARD"
        else:
            # 预防性分离
            separation_angle = np.radians(30)
            velocity_cmd = 640
            split_intensity = "PREVENTIVE"

        # 分离方向
        if is_leader:
            heading_cmd = separation_angle  # 长机向右
            split_role = "primary_evader"
        else:
            heading_cmd = -separation_angle  # 僚机向左
            split_role = "secondary_evader"

        # 高度分离（增加3D机动）
        if is_leader:
            altitude_cmd = 200  # 长机爬升
        else:
            altitude_cmd = -200  # 僚机下降

        logging.debug(f"Defensive Split: {split_role}, 强度={split_intensity}, 导弹距离={missile_distance:.0f}m")

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": altitude_cmd,
            "velocity_cmd": velocity_cmd,
            "separation_maneuver": True,
            "shoot": False,  # 分离时不射击
            "maneuver_active": True,
            "split_role": split_role,
            "split_intensity": split_intensity,
            "coordination_required": True
        }

    def _high_low_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """High-Low机动逻辑 - 高低搭配(2v2)"""

        is_leader = state.get("is_leader", False)
        current_alt = state.get("current_altitude", 5000)
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)

        # 触发条件：适合中远距离协同
        should_high_low = (
                enemy_distance > 35000 and
                radar_lock
        )

        if not should_high_low:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        # 高低搭配的高度目标
        if is_leader:
            # 高机：占据高度优势
            target_altitude = 9000  # 9km高空
            role = "high_fighter"
            tactical_advantage = "altitude_energy"
        else:
            # 低机：低空隐蔽接近
            target_altitude = 3000  # 3km低空
            role = "low_fighter"
            tactical_advantage = "stealth_approach"

        # 高度控制
        altitude_diff = target_altitude - current_alt
        if abs(altitude_diff) > 500:
            altitude_cmd = np.clip(altitude_diff, -800, 800)
            high_low_phase = "positioning"
        else:
            altitude_cmd = 0
            high_low_phase = "maintained"

        # 速度和航向协调
        if high_low_phase == "positioning":
            velocity_cmd = 620  # 调整阶段保持稳定
            heading_cmd = 0
        else:
            # 保持阵型，适当分散
            if is_leader:
                heading_cmd = np.radians(10)  # 高机右偏
                velocity_cmd = 640
            else:
                heading_cmd = np.radians(-10)  # 低机左偏
                velocity_cmd = 660  # 低机稍快

        # 射击条件：高低机错开射击
        if is_leader:
            shoot_ok = radar_lock and 45000 < enemy_distance < 65000
        else:
            shoot_ok = radar_lock and 35000 < enemy_distance < 55000

        logging.debug(f"High-Low: {role}, 目标高度={target_altitude}m, 当前={current_alt:.0f}m, 阶段={high_low_phase}")

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": altitude_cmd,
            "velocity_cmd": velocity_cmd,
            "altitude_optimization": True,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "high_low_role": role,
            "high_low_phase": high_low_phase,
            "tactical_advantage": tactical_advantage
        }

    def _engaging_trail_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Engaging Trail机动逻辑 - 前后交战(2v2)"""

        is_leader = state.get("is_leader", False)
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)

        # 触发条件
        should_trail = (
                enemy_distance > 25000 and
                radar_lock
        )

        if not should_trail:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        if is_leader:
            # 长机：主攻击者
            heading_cmd = 0  # 直接接敌

            if enemy_distance > 40000:
                velocity_cmd = 660  # 接近速度
                trail_phase = "approach"
                shoot_ok = False
            elif enemy_distance > 30000:
                velocity_cmd = 640  # 攻击速度
                trail_phase = "attack"
                shoot_ok = radar_lock
            else:
                # 准备脱离，让僚机接手
                velocity_cmd = 580
                trail_phase = "disengage"
                shoot_ok = False

            role = "lead_attacker"

        else:
            # 僚机：跟随在后5-10海里
            ideal_trail_distance = 8000  # 8km跟随距离
            current_trail_distance = enemy_distance - 8000  # 简化计算

            if current_trail_distance < 6000:
                # 太近，减速
                velocity_cmd = 580
                heading_cmd = np.radians(5)  # 略微偏离
            elif current_trail_distance > 10000:
                # 太远，加速
                velocity_cmd = 680
                heading_cmd = 0
            else:
                # 保持理想距离
                velocity_cmd = 620
                heading_cmd = 0

            # 僚机射击时机：长机脱离后
            if enemy_distance < 35000:
                trail_phase = "wingman_attack"
                shoot_ok = radar_lock
            else:
                trail_phase = "following"
                shoot_ok = False

            role = "trail_attacker"

        logging.debug(f"Engaging Trail: {role}, 距离={enemy_distance:.0f}m, 阶段={trail_phase}")

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": 0,
            "velocity_cmd": velocity_cmd,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "trail_role": role,
            "trail_phase": trail_phase,
            "trail_distance": 8000 if not is_leader else None
        }

    def _loose_deuce_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Loose Deuce机动逻辑 - 疏散双机盘旋(2vN)"""

        is_leader = state.get("is_leader", False)
        enemy_distance = state.get("enemy_distance", 50000)
        radar_lock = state.get("radar_lock", False)
        current_alt = state.get("current_altitude", 5000)

        # 触发条件：多威胁环境
        should_loose_deuce = (
                enemy_distance > 20000  # 适合中远距离
        )

        if not should_loose_deuce:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        # 盘旋轨道参数
        if not hasattr(self, '_orbit_state'):
            self._orbit_state = {"angle": 0, "timer": 0}

        self._orbit_state["timer"] += 1
        self._orbit_state["angle"] += 0.02  # 缓慢盘旋

        if self._orbit_state["angle"] > 2 * np.pi:
            self._orbit_state["angle"] = 0

        # 高度分离避免碰撞
        if is_leader:
            target_altitude = 7000  # 长机高空
            orbit_radius = 5000  # 5km盘旋半径
            role = "high_orbit"
        else:
            target_altitude = 4000  # 僚机低空
            orbit_radius = 6000  # 6km盘旋半径（稍大避开）
            role = "low_orbit"

        # 高度控制
        altitude_diff = target_altitude - current_alt
        altitude_cmd = np.clip(altitude_diff, -400, 400)

        # 盘旋机动
        orbit_angle = self._orbit_state["angle"]
        if is_leader:
            heading_cmd = np.sin(orbit_angle) * 0.3  # 盘旋幅度
        else:
            heading_cmd = np.sin(orbit_angle + np.pi) * 0.3  # 相位差π

        # 速度管理
        if enemy_distance > 50000:
            velocity_cmd = 640  # 远距离保持能量
        elif enemy_distance > 30000:
            velocity_cmd = 620  # 中距离标准速度
        else:
            velocity_cmd = 660  # 近距离提速规避

        # 射击窗口
        shoot_ok = (
                radar_lock and
                30000 < enemy_distance < 60000 and
                abs(heading_cmd) < np.radians(20)  # 盘旋时射击窗口
        )

        logging.debug(f"Loose Deuce: {role}, 盘旋角={np.degrees(orbit_angle):.1f}°, 高度={current_alt:.0f}m")

        return {
            "heading_cmd": heading_cmd,
            "altitude_cmd": altitude_cmd,
            "velocity_cmd": velocity_cmd,
            "orbit_pattern": True,
            "shoot": shoot_ok,
            "maneuver_active": True,
            "orbit_role": role,
            "orbit_angle": orbit_angle,
            "mutual_support": True
        }

    def _defensive_sequence_logic(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Defensive Sequence机动逻辑 - 连续防御机动"""

        missile_distance = state.get("missile_distance", np.inf)
        enemy_distance = state.get("enemy_distance", 50000)
        has_warning = state.get("has_warning", False)

        # 防御序列状态机
        if not hasattr(self, '_defense_sequence_state'):
            self._defense_sequence_state = {"stage": 1, "timer": 0}

        self._defense_sequence_state["timer"] += 1

        # 触发条件：有导弹威胁
        should_defend = (
                has_warning or
                missile_distance < 40000
        )

        if not should_defend:
            return {"heading_cmd": 0, "altitude_cmd": 0, "velocity_cmd": 600, "shoot": False, "maneuver_active": False}

        # 六阶段防御序列
        stage = self._defense_sequence_state["stage"]
        timer = self._defense_sequence_state["timer"]

        # 阶段1：Beam机动 - 建立横向态势
        if stage == 1:
            if timer > 100 or missile_distance < 30000:  # 条件满足进入下一阶段
                self._defense_sequence_state["stage"] = 2
                self._defense_sequence_state["timer"] = 0

            beam_action = self._beam_logic(state)
            beam_action.update({
                "defense_stage": 1,
                "defense_phase": "beam_establish"
            })

            logging.debug(f"防御序列阶段1: Beam机动, 导弹距离={missile_distance:.0f}m")
            return beam_action

        # 阶段2：Notch机动 - 进入杂波盲区
        elif stage == 2:
            if timer > 150 or missile_distance < 25000:
                self._defense_sequence_state["stage"] = 3
                self._defense_sequence_state["timer"] = 0

            notch_action = self._notch_logic(state)
            notch_action.update({
                "defense_stage": 2,
                "defense_phase": "notch_clutter"
            })

            logging.debug(f"防御序列阶段2: Notch机动, 导弹距离={missile_distance:.0f}m")
            return notch_action

        # 阶段3：F-Pole拉开距离
        elif stage == 3:
            if timer > 100 or missile_distance < 20000:
                self._defense_sequence_state["stage"] = 4
                self._defense_sequence_state["timer"] = 0

            f_pole_action = self._simple_f_pole_logic(state)
            f_pole_action.update({
                "defense_stage": 3,
                "defense_phase": "f_pole_extend",
                "heading_cmd": np.radians(180),  # 拉开距离
                "velocity_cmd": 680,
                "shoot": False
            })

            logging.debug(f"防御序列阶段3: F-Pole拉距, 导弹距离={missile_distance:.0f}m")
            return f_pole_action

        # 阶段4：Crank机动 - 维持锁定规避
        elif stage == 4:
            if timer > 120 or missile_distance < 15000:
                self._defense_sequence_state["stage"] = 5
                self._defense_sequence_state["timer"] = 0

            crank_action = self._crank_logic(state)
            crank_action.update({
                "defense_stage": 4,
                "defense_phase": "crank_evade"
            })

            logging.debug(f"防御序列阶段4: Crank规避, 导弹距离={missile_distance:.0f}m")
            return crank_action

        # 阶段5：Turn Cold - 背离脱离
        elif stage == 5:
            if timer > 200 or missile_distance > 35000:
                self._defense_sequence_state["stage"] = 6
                self._defense_sequence_state["timer"] = 0

            logging.debug(f"防御序列阶段5: Turn Cold, 导弹距离={missile_distance:.0f}m")

            return {
                "heading_cmd": np.pi,  # 180度掉头
                "altitude_cmd": 0,
                "velocity_cmd": 720,  # 最高速度脱离
                "shoot": False,
                "maneuver_active": True,
                "defense_stage": 5,
                "defense_phase": "turn_cold"
            }

        # 阶段6：Drag + Split - 分离逃逸或反打
        else:  # stage == 6
            if timer > 150:
                # 重置防御序列
                self._defense_sequence_state = {"stage": 1, "timer": 0}

            # 根据威胁情况决定
            if missile_distance > 40000:
                # 威胁解除，可以反打
                logging.debug(f"防御序列阶段6: 威胁解除，准备反打")

                return {
                    "heading_cmd": 0,  # 转向敌机
                    "altitude_cmd": 200,
                    "velocity_cmd": 650,
                    "shoot": state.get("radar_lock", False),
                    "maneuver_active": True,
                    "defense_stage": 6,
                    "defense_phase": "counter_attack"
                }
            else:
                # 继续分离逃逸
                logging.debug(f"防御序列阶段6: 继续分离逃逸")

                return {
                    "heading_cmd": np.radians(135),  # 135度分离
                    "altitude_cmd": -300,  # 下降逃逸
                    "velocity_cmd": 700,
                    "shoot": False,
                    "maneuver_active": True,
                    "defense_stage": 6,
                    "defense_phase": "drag_split"
                }
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

    def analyze_maneuver_effectiveness(self, template_id: int, state_before: Dict, state_after: Dict) -> Dict:
        """分析机动效果"""
        effectiveness = {
            "distance_change": state_after["enemy_distance"] - state_before["enemy_distance"],
            "altitude_change": state_after["current_altitude"] - state_before["current_altitude"],
            "threat_reduction": 0,
            "position_improvement": 0,
            "energy_cost": 0
        }

        # 威胁减少评估
        if state_before.get("has_warning", False) and not state_after.get("has_warning", False):
            effectiveness["threat_reduction"] = 1.0
        elif state_after["missile_distance"] > state_before["missile_distance"]:
            effectiveness["threat_reduction"] = 0.5

        # 位置改善评估
        if template_id in [1, 7, 8]:  # 攻击类模板
            if state_after.get("radar_lock", False) and 35000 <= state_after["enemy_distance"] <= 60000:
                effectiveness["position_improvement"] = 1.0
        elif template_id in [2, 3, 14]:  # 防御类模板
            if abs(state_after["enemy_angle_off"]) > 80:  # 成功形成横向态势
                effectiveness["position_improvement"] = 1.0

        # 能量消耗评估（基于高度和速度变化）
        altitude_loss = max(0, state_before["current_altitude"] - state_after["current_altitude"])
        effectiveness["energy_cost"] = altitude_loss / 1000.0  # 归一化

        return effectiveness
class TacticalTemplate(EnhancedTacticalTemplate):
    """保持向后兼容的战术模板类"""

    def __init__(self, is_enemy: bool = False, env=None, agent_id: str = None):
        super().__init__(is_enemy, env, agent_id)
        logging.warning("Using deprecated TacticalTemplate class, consider upgrading to EnhancedTacticalTemplate")

    def get_action(self, template_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """向后兼容的动作获取方法"""
        return self.get_tactical_action(template_id, state)