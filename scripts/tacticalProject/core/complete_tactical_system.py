"""
Complete tactical system wrapper.

This module keeps the public API used by TacticalTask while routing enemy
intent recognition through SituationAlgorithmSwitcher.
"""

import logging
from typing import Dict, Optional

import numpy as np

from .decision_table import DecisionTable
from .situation_algorithm_switcher import SituationAlgorithmConfig, get_situation_algorithm_switcher
from .tactical_selector_algorithm import TacticalSelectorAlgorithm
from .threat_evaluator_complete import CompleteThreatEvaluator


class CompleteTacticalSystem:
    """High-level tactical selector facade."""

    def __init__(self, my_intent: str = "CONSERVATIVE_CLEAR", situation_algorithm: Optional[str] = None):
        self.my_intent = my_intent
        self.current_env = None
        self.current_tactic = None
        self.current_roles = {}

        self.threat_evaluator = CompleteThreatEvaluator()
        self.algorithm_switcher = get_situation_algorithm_switcher()
        self.intent_recognizer = self.algorithm_switcher.algorithm_instance
        self.decision_table = DecisionTable()
        self.tactical_selector = TacticalSelectorAlgorithm(self.threat_evaluator, self.decision_table)

        if situation_algorithm:
            if str(situation_algorithm).strip().lower() != "bvr_intent_new":
                logging.warning("Intent algorithm %s requested; forcing bvr_intent_new", situation_algorithm)
            self.algorithm_switcher.set_algorithm(SituationAlgorithmConfig.BVR_INTENT_NEW)

        algo_status = self.algorithm_switcher.get_algorithm_status()
        logging.info("CompleteTacticalSystem initialised (friendly intent: %s)", my_intent)
        logging.info(
            "Intent algorithm: %s - %s",
            algo_status["current_algorithm"],
            algo_status["note"][algo_status["current_algorithm"]],
        )

    def select_tactic(self, control_distance, my_aircraft_list, enemy_aircraft_list, env):
        try:
            self.current_env = env
            situation = self._evaluate_situation(my_aircraft_list, enemy_aircraft_list)
            enemy_intent = self._recognize_enemy_intent(my_aircraft_list, enemy_aircraft_list, env)

            selected_tactic = self.tactical_selector.select_tactic(
                control_distance=control_distance,
                my_intent=self.my_intent,
                enemy_intent=enemy_intent,
                situation=situation,
                my_aircraft=my_aircraft_list,
                enemy_aircraft=enemy_aircraft_list,
                env=env,
                current_tactic=self.current_tactic,
            )

            roles = self._assign_roles(selected_tactic, my_aircraft_list, enemy_aircraft_list, env)
            self.current_tactic = selected_tactic
            self.current_roles = roles
            logging.info("Tactic selected: %s | roles=%s", selected_tactic, roles)
            return selected_tactic, roles
        except Exception as exc:
            logging.error("Tactic selection failed: %s", exc)
            return "SIDE_BY_SIDE", {"lead": "left", "wingman": "right"}

    def _evaluate_situation(self, my_aircraft_list, enemy_aircraft_list):
        try:
            my_alive = sum(1 for ac in my_aircraft_list if ac and ac.is_alive)
            enemy_alive = sum(1 for ac in enemy_aircraft_list if ac and ac.is_alive)

            if my_alive > enemy_alive:
                return "ADVANTAGE"
            if my_alive < enemy_alive:
                return "DISADVANTAGE"

            if my_aircraft_list and enemy_aircraft_list:
                my_lead = my_aircraft_list[0] if my_aircraft_list[0].is_alive else my_aircraft_list[1]
                enemy_lead = enemy_aircraft_list[0] if enemy_aircraft_list[0].is_alive else enemy_aircraft_list[1]
                my_alt = my_lead.get_position()[2]
                enemy_alt = enemy_lead.get_position()[2]

                if my_alt - enemy_alt > 1000:
                    return "ADVANTAGE"
                if enemy_alt - my_alt > 1000:
                    return "DISADVANTAGE"

            return "NEUTRAL"
        except Exception as exc:
            logging.error("Situation evaluation failed: %s", exc)
            return "NEUTRAL"

    def _recognize_enemy_intent(self, my_aircraft_list, enemy_aircraft_list, env=None):
        try:
            if not my_aircraft_list or not enemy_aircraft_list:
                return "NEUTRAL"

            env = env or self.current_env
            my_lead = my_aircraft_list[0] if my_aircraft_list[0].is_alive else my_aircraft_list[1]
            enemy_lead = enemy_aircraft_list[0] if enemy_aircraft_list[0].is_alive else enemy_aircraft_list[1]

            if env is not None:
                return self.algorithm_switcher.recognize_intent(enemy_lead, my_lead, env)

            return "NEUTRAL"
        except Exception as exc:
            logging.error("Enemy intent recognition failed: %s", exc)
            return "NEUTRAL"

    def switch_situation_algorithm(self, new_algorithm: str):
        try:
            old_algorithm = self.algorithm_switcher.get_current_algorithm()
            self.algorithm_switcher.switch_algorithm(new_algorithm)
            self.intent_recognizer = self.algorithm_switcher.algorithm_instance
            logging.info("Intent algorithm switched: %s -> %s", old_algorithm, new_algorithm)
        except Exception as exc:
            logging.error("Intent algorithm switch failed: %s", exc)

    def get_situation_algorithm_status(self) -> Dict:
        return self.algorithm_switcher.get_algorithm_status()

    def set_current_env(self, env):
        self.current_env = env

    def _assign_roles(self, tactic, my_aircraft_list, enemy_aircraft_list, env):
        return {
            "lead": "A0100",
            "wingman": "A0200",
            "lead_target": "B0100",
            "wingman_target": "B0200",
            "tactic": tactic,
        }

    def get_threat_level(self, my_aircraft, enemy_aircraft_list, env):
        try:
            if not enemy_aircraft_list:
                return 0.0

            max_threat = 0.0
            for enemy_ac in enemy_aircraft_list:
                if enemy_ac and enemy_ac.is_alive:
                    threat = self.threat_evaluator.calculate_total_threat(my_aircraft, enemy_ac, env)
                    max_threat = max(max_threat, threat)
            return max_threat
        except Exception as exc:
            logging.error("Threat level calculation failed: %s", exc)
            return 0.5

    def check_emergency_interrupt(self, my_aircraft, enemy_aircraft_list, env):
        try:
            if self.threat_evaluator.detect_incoming_missiles(my_aircraft, env):
                return True, "MISSILE"

            rwr_level = self.threat_evaluator.calculate_rwr_level(my_aircraft, enemy_aircraft_list)
            if rwr_level >= 4:
                return True, "RWR"

            for enemy_ac in enemy_aircraft_list or []:
                if enemy_ac and enemy_ac.is_alive:
                    my_pos = my_aircraft.get_position()
                    enemy_pos = enemy_ac.get_position()
                    distance = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))
                    if distance <= 40000:
                        return True, "MAR"

            return False, None
        except Exception as exc:
            logging.error("Emergency interrupt check failed: %s", exc)
            return False, None
