import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import py_trees
from py_trees.trees import BehaviourTree
from py_trees.composites import Sequence, Selector
from py_trees.behaviour import Behaviour
from ..core.catalog import Catalog as c
from ..utils.RadarModel import RadarModel
from .RefinedTacticalManeuvers import RefinedTacticalManeuvers

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
    """Enhanced tactical template system supporting 14 classic BVR air combat tactics"""

    PHASES = [
        "contact_guidance", "target_search", "target_identification", "threat_assessment",
        "target_allocation", "tactical_decision", "missile_launch", "mid_guidance_defense",
        "terminal_guidance", "effect_assessment"
    ]

    TACTICAL_DISTANCES = {
        "detection_range": 80000, "engagement_range": 65000, "launch_range": 45000,
        "mar_range": 20000, "wez_range": 35000, "rmax": 60000, "rmin": 5000
    }

    def __init__(self, is_enemy: bool = False, env=None, agent_id: str = None):
        if env is None or agent_id is None:
            raise ValueError("env and agent_id must be provided for TacticalTemplate initialization")
        self.maneuver_states = {}
        self.is_enemy = is_enemy
        self.env = env
        self.agent_id = agent_id
        self.radar = RadarModel(max_range=120000, h_beamwidth=60, v_beamwidth=30)
        self.maneuvers = RefinedTacticalManeuvers()  # Instantiate RefinedTacticalManeuvers

        self.tactical_templates = self._init_tactical_templates()
        self.current_phase = self.PHASES[0]
        self.phase_start_time = 0
        self.behavior_tree = self.build_behavior_tree()
        self.tactic_cooldown = 0
        self.last_template_id = 0
        self.template_history = []

        self.missile_launched = False
        self.crank_active = False
        self.beam_active = False
        self.notch_active = False
        self.defensive_sequence = []
        self.defensive_action_logged = False
        self._skate_state = None
        self._short_skate_state = None
        self._banzai_state = None
        self._defense_sequence_state = None
        self._orbit_state = None

    def _init_tactical_templates(self) -> Dict[int, Dict[str, Any]]:
        """Initialize 14 tactical templates"""
        return {
            0: {"name": "No_Template", "type": "direct_rl", "description": "Direct RL control"},
            1: {"name": "Crank", "type": "evasion", "description": "Offset 30-60° maintaining radar lock",
                "params": {"heading_offset_range": (30, 60), "maintain_lock": True, "velocity_cmd": 600,
                           "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                           "conditions": ["radar_lock", "missile_threat"]}},
            2: {"name": "Beam", "type": "evasion", "description": "Lateral maneuver to bleed missile energy",
                "params": {"heading_offset": 90, "velocity_cmd": 550,
                           "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                           "doppler_optimization": True, "conditions": ["incoming_missile", "doppler_escape"]}},
            3: {"name": "Notch", "type": "evasion", "description": "Use ground clutter for cover",
                "params": {"heading_offset": 90,
                           "altitude_cmd": lambda s: -500 if s["enemy_distance"] > 20000 else -200,
                           "velocity_cmd": 500, "ground_clutter": True,
                           "conditions": ["radar_warning", "altitude_sufficient"]}},
            4: {"name": "Skate", "type": "attack", "description": "Long-range launch, maneuver, turn cold, re-attack",
                "params": {"sequence": ["launch", "crank", "turn_cold", "re_engage", "launch2", "turn_cold"],
                           "velocity_cmd": 650, "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s),
                           "mar_consideration": True, "conditions": ["bvr_range", "first_shot"]}},
            5: {"name": "Short_Skate", "type": "attack", "description": "Launch and quick escape",
                "params": {"sequence": ["launch", "turn_cold"], "velocity_cmd": 700,
                           "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s), "quick_escape": True,
                           "conditions": ["advantageous_position", "energy_management"]}},
            6: {"name": "Banzai", "type": "attack", "description": "Launch then decide-merge",
                "params": {"sequence": ["launch", "crank", "merge_decision"], "velocity_cmd": 650,
                           "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s), "merge_threshold": 15000,
                           "conditions": ["wez_entry", "energy_advantage"]}},
            7: {"name": "Simple_F_Pole", "type": "attack", "description": "Maintain lock to maximize F-pole",
                "params": {"maintain_lock": True, "velocity_cmd": 600,
                           "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s), "f_pole_optimization": True,
                           "conditions": ["stable_track", "fox3_guidance"]}},
            8: {"name": "Advanced_F_Pole", "type": "dynamic_attack", "description": "Dynamic F-pole optimization",
                "params": {"dynamic_crank": True, "threat_assessment": True, "velocity_cmd": 650,
                           "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s), "intelligent_escape": True,
                           "conditions": ["multi_threat", "advanced_guidance"]}},
            9: {"name": "Pincer", "type": "cooperative", "description": "Pincer attack",
                "params": {"formation_angle": (30, 60), "sync_timing": True, "velocity_cmd": 600,
                           "altitude_cmd": lambda s: self._cooperative_altitude_cmd(s), "coordinate_launch": True,
                           "conditions": ["dual_aircraft", "flanking_position"]}},
            10: {"name": "Defensive_Split", "type": "cooperative", "description": "Defensive split then pincer",
                 "params": {"split_direction": "bilateral", "separation_distance": 10000, "velocity_cmd": 600,
                            "altitude_cmd": lambda s: self._cooperative_altitude_cmd(s),
                            "regroup_condition": "threat_neutralized", "conditions": ["under_attack", "formation_integrity"]}},
            11: {"name": "High_Low", "type": "cooperative", "description": "High-low pairing",
                 "params": {"altitude_separation": 3000, "velocity_cmd": 600,
                            "altitude_cmd": lambda s: self._high_low_altitude_cmd(s), "coverage_optimization": True,
                            "conditions": ["complementary_coverage", "altitude_advantage"]}},
            12: {"name": "Engaging_Trail", "type": "cooperative", "description": "Lead-trail engagement",
                 "params": {"trail_distance": 8000, "follow_leader": True, "velocity_cmd": 600,
                            "altitude_cmd": lambda s: self._adaptive_altitude_cmd(s), "backup_engagement": True,
                            "conditions": ["lead_wingman", "sequential_attack"]}},
            13: {"name": "Loose_Deuce", "type": "cooperative", "description": "Loose deuce orbiting",
                 "params": {"orbit_radius": 5000, "altitude_separation": 1000, "velocity_cmd": 550,
                            "altitude_cmd": lambda s: self._loose_deuce_altitude_cmd(s), "continuous_coverage": True,
                            "conditions": ["area_denial", "persistent_threat"]}},
            14: {"name": "Defensive_Sequence", "type": "complex_evasion", "description": "Continuous defensive sequence",
                 "params": {"sequence": ["beam", "notch", "f_pole", "crank", "turn_cold", "drag_split"],
                            "adaptive_timing": True, "velocity_cmd": lambda s: self._defensive_velocity_cmd(s),
                            "altitude_cmd": lambda s: self._defensive_altitude_cmd(s), "threat_priority": True,
                            "conditions": ["multiple_threats", "defensive_posture"]}}
        }

    def _adaptive_altitude_cmd(self, state: Dict[str, Any]) -> float:
        """Adaptive altitude command"""
        current_alt = state.get("current_altitude", 5000)
        enemy_dist = state.get("enemy_distance", 50000)
        has_warning = state.get("has_warning", False)

        if enemy_dist > 80000:
            target_alt = 9000
        elif enemy_dist > 50000:
            target_alt = 8000
        else:
            target_alt = 7000

        if has_warning and current_alt > 3000:
            target_alt = max(current_alt - 1000, 2000)

        alt_diff = target_alt - current_alt
        return np.clip(alt_diff, -800, 800)

    def _cooperative_altitude_cmd(self, state: Dict[str, Any]) -> float:
        """Cooperative altitude command"""
        is_leader = state.get("is_leader", False)
        base_cmd = self._adaptive_altitude_cmd(state)
        return base_cmd + 200 if is_leader else base_cmd - 200

    def _high_low_altitude_cmd(self, state: Dict[str, Any]) -> float:
        """High-low altitude command"""
        is_leader = state.get("is_leader", False)
        current_alt = state.get("current_altitude", 5000)
        target_alt = 12000 if is_leader else 6000
        alt_diff = target_alt - current_alt
        return np.clip(alt_diff, -1000, 1000)

    def _loose_deuce_altitude_cmd(self, state: Dict[str, Any]) -> float:
        """Loose deuce altitude command"""
        is_leader = state.get("is_leader", False)
        base_cmd = self._adaptive_altitude_cmd(state)
        offset = 500 if is_leader else -500
        return base_cmd + offset

    def _defensive_altitude_cmd(self, state: Dict[str, Any]) -> float:
        """Defensive altitude command"""
        missile_dist = state.get("missile_distance", np.inf)
        current_alt = state.get("current_altitude", 5000)
        return -1000 if missile_dist < 20000 and current_alt > 3000 else self._adaptive_altitude_cmd(state)

    def _defensive_velocity_cmd(self, state: Dict[str, Any]) -> float:
        """Defensive velocity command"""
        missile_dist = state.get("missile_distance", np.inf)
        if missile_dist < 15000:
            return 750
        elif missile_dist < 30000:
            return 650
        return 600

    def get_tactical_action(self, template_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get tactical action command"""
        if template_id not in self.tactical_templates:
            logging.warning(f"Unknown template_id: {template_id}, using default")
            template_id = 7  # Default to Simple_F_Pole

        template = self.tactical_templates[template_id]
        self.last_template_id = template_id
        self.template_history.append((template_id, getattr(self.env, 'current_step', 0)))

        # Generate base action
        action = self._generate_base_action(template, state)

        # Apply maneuver logic via RefinedTacticalManeuvers
        maneuver_action = self.maneuvers.get_maneuver_action(template_id, state)
        action.update(maneuver_action)

        logging.debug(f"Agent {self.agent_id} tactical action: template={template['name']}, "
                      f"heading_cmd={action.get('heading_cmd', 0):.3f}, "
                      f"altitude_cmd={action.get('altitude_cmd', 0):.1f}, "
                      f"velocity_cmd={action.get('velocity_cmd', 600):.1f}")
        return action

    def _generate_base_action(self, template: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base action command"""
        params = template["params"]
        action = {
            "maneuver": template["name"],
            "heading_cmd": 0.0,
            "altitude_cmd": 0.0,
            "velocity_cmd": params.get("velocity_cmd", 600),
            "shoot": False,
            "template_type": template["type"]
        }

        if callable(params.get("altitude_cmd")):
            action["altitude_cmd"] = params["altitude_cmd"](state)
        else:
            action["altitude_cmd"] = params.get("altitude_cmd", 0)

        if callable(params.get("velocity_cmd")):
            action["velocity_cmd"] = params["velocity_cmd"](state)
        else:
            action["velocity_cmd"] = params.get("velocity_cmd", 600)

        return action

    def build_behavior_tree(self):
        """Build behavior tree"""
        root = Selector("Root", memory=True)
        threat_response = Sequence("Threat_Response", memory=True)
        threat_condition = ConditionNode("Missile_Threat", lambda: self._check_immediate_threat())
        threat_action = ActionNode("Execute_Defense", lambda: self._execute_defensive_action())
        threat_response.add_children([threat_condition, threat_action])
        root.add_child(threat_response)

        for phase in self.PHASES:
            phase_node = Sequence(f"Phase_{phase}", memory=True)
            condition_node = ConditionNode(f"{phase}_conditions", lambda p=phase: self._check_phase_conditions(p))
            action_node = ActionNode(f"{phase}_actions", lambda p=phase: self._execute_phase_actions(p))
            phase_node.add_children([condition_node, action_node])
            root.add_child(phase_node)

        return BehaviourTree(root)

    def _check_immediate_threat(self) -> bool:
        """Check for immediate threat"""
        if not hasattr(self.env, 'agents') or self.agent_id not in self.env.agents:
            return False
        agent = self.env.agents[self.agent_id]
        missile_sim = agent.check_missile_warning()
        if missile_sim:
            distance = np.linalg.norm(missile_sim.get_position() - agent.get_position())
            return distance < 30000
        return False

    def _execute_defensive_action(self):
        """Execute defensive action"""
        self.current_phase = "mid_guidance_defense"
        if not self.defensive_action_logged:
            logging.info(f"Agent {self.agent_id} executing defensive action")
            self.defensive_action_logged = True

    def _check_phase_conditions(self, phase: str) -> bool:
        """Check phase conditions"""
        if not hasattr(self.env, 'agents') or self.agent_id not in self.env.agents:
            return False
        agent = self.env.agents[self.agent_id]
        if not agent.is_alive:
            return False
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
        """Execute phase actions"""
        self.current_phase = phase
        logging.debug(f"Agent {self.agent_id} executing phase: {phase}")

    def update_phase(self, state: Dict[str, Any]) -> str:
        """Update phase"""
        if self.tactic_cooldown > 0:
            self.tactic_cooldown -= 1
            return self.current_phase
        prev_phase = self.current_phase
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
        """Get radar state"""
        try:
            radar_state = self.radar.get_radar_state(state, self.env, self.agent_id)
            return radar_state
        except Exception as e:
            logging.error(f"Radar state error for {self.agent_id}: {e}")
            return {"radar_lock": False, "has_warning": False, "snr": 0.0}

    def get_template_info(self, template_id: int) -> Dict[str, Any]:
        """Get tactical template info"""
        if template_id in self.tactical_templates:
            template = self.tactical_templates[template_id]
            return {"name": template["name"], "type": template["type"], "description": template["description"]}
        return {"name": "Unknown", "type": "unknown", "description": "Unknown template"}

    def analyze_maneuver_effectiveness(self, template_id: int, state_before: Dict, state_after: Dict) -> Dict:
        """Analyze maneuver effectiveness"""
        effectiveness = {
            "distance_change": state_after["enemy_distance"] - state_before["enemy_distance"],
            "altitude_change": state_after["current_altitude"] - state_before["current_altitude"],
            "threat_reduction": 0,
            "position_improvement": 0,
            "energy_cost": 0
        }
        if state_before.get("has_warning", False) and not state_after.get("has_warning", False):
            effectiveness["threat_reduction"] = 1.0
        elif state_after["missile_distance"] > state_before["missile_distance"]:
            effectiveness["threat_reduction"] = 0.5
        if template_id in [1, 7, 8]:
            if state_after.get("radar_lock", False) and 35000 <= state_after["enemy_distance"] <= 60000:
                effectiveness["position_improvement"] = 1.0
        elif template_id in [2, 3, 14]:
            if abs(state_after["enemy_angle_off"]) > 80:
                effectiveness["position_improvement"] = 1.0
        altitude_loss = max(0, state_before["current_altitude"] - state_after["current_altitude"])
        effectiveness["energy_cost"] = altitude_loss / 1000.0
        return effectiveness


class TacticalTemplate(EnhancedTacticalTemplate):
    """Backward-compatible tactical template class"""
    def __init__(self, is_enemy: bool = False, env=None, agent_id: str = None):
        super().__init__(is_enemy, env, agent_id)
        logging.warning("Using deprecated TacticalTemplate class, consider upgrading to EnhancedTacticalTemplate")

    def get_action(self, template_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compatible action method"""
        return self.get_tactical_action(template_id, state)