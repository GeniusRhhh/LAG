"""Enemy tactical AI compatibility wrapper.

This module restores the tactical enemy logic by delegating to the unified
enemy tactical AI implementation while preserving the NewEnemyManeuverAI
interface expected by CAP.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

try:
    from unified_enemy_tactical_ai_impl import (
        ActionType,
        EnemyTacticalPhase,
        TacticalMode,
        UnifiedEnemyTacticalAI,
    )
except Exception:
    try:
        from unified_enemy_tactical_ai import (
            ActionType,
            EnemyTacticalPhase,
            TacticalMode,
            UnifiedEnemyTacticalAI,
        )
    except Exception:
        UnifiedEnemyTacticalAI = None
        ActionType = None
        TacticalMode = None
        EnemyTacticalPhase = None


class NewEnemyManeuverAI:
    """Compatibility wrapper around the unified enemy tactical AI."""

    def __init__(self, project_name: str = "cap_tactical_project"):
        self.project_name = project_name
        self.logger = logging.getLogger(self.__class__.__name__)
        if UnifiedEnemyTacticalAI is None:
            raise ImportError("UnifiedEnemyTacticalAI is unavailable")
        self._impl = UnifiedEnemyTacticalAI()

    @staticmethod
    def _fmt(value) -> str:
        return np.array2string(
            np.asarray(value),
            precision=6,
            separator=",",
            threshold=np.inf,
            max_line_width=100000,
        )

    def _current_time(self, env, current_time: Optional[float]) -> float:
        if current_time is not None:
            return float(current_time)
        return float(getattr(env, "current_step", 0)) * float(getattr(env, "time_interval", 0.2))

    def get_enemy_command_indices(self, env, agent_id: str, current_time: float = None, task=None) -> Tuple[int, int, int]:
        return self.get_enemy_command(env, agent_id, current_time, task)

    def get_tactical_command_indices(self, env, agent_id: str) -> Tuple[int, int, int]:
        return self.get_enemy_command_indices(env, agent_id)

    def _get_enemy_command_indices(self, env, agent_id: str) -> Tuple[int, int, int]:
        return self.get_enemy_command_indices(env, agent_id)

    def get_enemy_command(self, env, agent_id: str, current_time: float = None, task=None) -> Tuple[int, int, int]:
        current_time = self._current_time(env, current_time)
        deep_trace = os.environ.get("CAP_B0100_DEEP_TRACE", "1").strip().lower() in ("1", "true", "yes", "on")

        if agent_id == "B0100" and deep_trace:
            try:
                aircraft = getattr(env, "agents", {}).get(agent_id)
                if aircraft is not None and getattr(aircraft, "is_alive", False):
                    current_alt = float(aircraft.get_property_value(getattr(__import__("envs.JSBSim.core.catalog", fromlist=["Catalog"]).Catalog, "position_h_sl_m")))
                    current_vc = float(aircraft.get_property_value(getattr(__import__("envs.JSBSim.core.catalog", fromlist=["Catalog"]).Catalog, "velocities_vc_mps")))
                    current_hdg = float(np.degrees(aircraft.get_property_value(getattr(__import__("envs.JSBSim.core.catalog", fromlist=["Catalog"]).Catalog, "attitude_heading_true_rad"))) ) % 360.0
                    current_pos = aircraft.get_position()
                    phase_before = getattr(self._impl, "current_phase", {}).get(agent_id, "UNKNOWN")
                    action_before = getattr(self._impl, "current_action", {}).get(agent_id, "UNKNOWN")
                    mode_before = getattr(self._impl, "tactical_mode", {}).get(agent_id, "UNKNOWN")
                    self.logger.warning(
                        "[B0100][T+%07.1fs][new_enemy_maneuver_ai][before_delegate][state] phase=%s action=%s mode=%s alt=%.3fm vc=%.3fmps hdg=%.3fdeg pos=%s",
                        current_time,
                        getattr(phase_before, "value", phase_before),
                        getattr(action_before, "value", action_before),
                        getattr(mode_before, "value", mode_before),
                        current_alt,
                        current_vc,
                        current_hdg,
                        self._fmt(current_pos),
                    )
            except Exception as exc:
                self.logger.warning("[B0100][T+%07.1fs][new_enemy_maneuver_ai][before_delegate][state] failed=%s", current_time, exc)

        cmd = self._impl.get_enemy_command(env, agent_id, current_time, task)

        if agent_id == "B0100" and deep_trace:
            try:
                aircraft = getattr(env, "agents", {}).get(agent_id)
                if aircraft is not None and getattr(aircraft, "is_alive", False):
                    current_alt = float(aircraft.get_property_value(getattr(__import__("envs.JSBSim.core.catalog", fromlist=["Catalog"]).Catalog, "position_h_sl_m")))
                    current_vc = float(aircraft.get_property_value(getattr(__import__("envs.JSBSim.core.catalog", fromlist=["Catalog"]).Catalog, "velocities_vc_mps")))
                    current_hdg = float(np.degrees(aircraft.get_property_value(getattr(__import__("envs.JSBSim.core.catalog", fromlist=["Catalog"]).Catalog, "attitude_heading_true_rad"))) ) % 360.0
                    current_pos = aircraft.get_position()
                    phase_after = getattr(self._impl, "current_phase", {}).get(agent_id, "UNKNOWN")
                    action_after = getattr(self._impl, "current_action", {}).get(agent_id, "UNKNOWN")
                    mode_after = getattr(self._impl, "tactical_mode", {}).get(agent_id, "UNKNOWN")
                    self.logger.warning(
                        "[B0100][T+%07.1fs][new_enemy_maneuver_ai][after_delegate][state] phase=%s action=%s mode=%s cmd=%s alt=%.3fm vc=%.3fmps hdg=%.3fdeg pos=%s",
                        current_time,
                        getattr(phase_after, "value", phase_after),
                        getattr(action_after, "value", action_after),
                        getattr(mode_after, "value", mode_after),
                        self._fmt(cmd),
                        current_alt,
                        current_vc,
                        current_hdg,
                        self._fmt(current_pos),
                    )
            except Exception as exc:
                self.logger.warning("[B0100][T+%07.1fs][new_enemy_maneuver_ai][after_delegate][state] failed=%s", current_time, exc)

        return cmd

    def reset_agent(self, agent_id: str):
        if hasattr(self._impl, "reset_agent"):
            self._impl.reset_agent(agent_id)

    def reset_all_agents(self):
        if hasattr(self._impl, "reset_all_agents"):
            self._impl.reset_all_agents()

    def update_tactical_phase(self, env, agent_id: str):
        if hasattr(self._impl, "update_tactical_phase"):
            try:
                return self._impl.update_tactical_phase(env, agent_id)
            except TypeError:
                return self._impl.update_tactical_phase(env, agent_id, None)
        return "UNKNOWN"

    def get_action_annotation(self, agent_id: str) -> str:
        if hasattr(self._impl, "get_action_annotation"):
            return self._impl.get_action_annotation(agent_id)
        return ""

    def get_action_annotation_for_csv(self, agent_id: str):
        if hasattr(self._impl, "get_action_annotation_for_csv"):
            return self._impl.get_action_annotation_for_csv(agent_id)
        return {}

    def __getattr__(self, item):
        return getattr(self._impl, item)


__all__ = ["NewEnemyManeuverAI"]
