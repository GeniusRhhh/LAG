"""Route all project intent recognition through bvr_intent_new."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _as_vec3(value: Any) -> np.ndarray:
    arr = np.asarray(value if value is not None else [0.0, 0.0, 0.0], dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size))
    return arr[:3]


def _heading_to_yaw_rad(heading_deg: Optional[float], velocity: np.ndarray) -> float:
    if heading_deg is not None:
        try:
            return math.radians(float(heading_deg))
        except Exception:
            pass
    if np.linalg.norm(velocity[:2]) > 1e-6:
        return math.atan2(float(velocity[1]), float(velocity[0]))
    return 0.0


@dataclass
class _DictAircraft:
    agent_id: str
    position: np.ndarray
    velocity: np.ndarray
    yaw_rad: float
    radar_mode: str = "search"
    is_alive: bool = True

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_rpy(self):
        return np.asarray([0.0, 0.0, self.yaw_rad], dtype=np.float64)


@dataclass
class _ShimEnv:
    agents: Dict[str, Any]
    current_step: int = 0
    time_interval: float = 0.2


def _state_to_aircraft(state: Dict[str, Any], agent_id: str) -> _DictAircraft:
    position = _as_vec3(state.get("position"))
    velocity = _as_vec3(state.get("velocity"))
    heading = state.get("heading")
    radar_mode = state.get("radar_mode") or state.get("radar_status") or "search"
    is_alive = bool(state.get("is_alive", True))
    return _DictAircraft(
        agent_id=str(agent_id),
        position=position,
        velocity=velocity,
        yaw_rad=_heading_to_yaw_rad(heading, velocity),
        radar_mode=str(radar_mode),
        is_alive=is_alive,
    )


def _coerce_aircraft(value: Any, agent_id: str) -> Any:
    if value is None:
        return None
    if hasattr(value, "get_position") and hasattr(value, "get_velocity") and hasattr(value, "get_rpy"):
        if getattr(value, "agent_id", None) is None:
            try:
                value.agent_id = str(agent_id)
            except Exception:
                pass
        if getattr(value, "is_alive", None) is None:
            try:
                value.is_alive = True
            except Exception:
                pass
        return value
    if isinstance(value, dict):
        return _state_to_aircraft(value, agent_id)
    state = {
        "position": getattr(value, "position", None),
        "velocity": getattr(value, "velocity", None),
        "heading": getattr(value, "heading", None),
        "radar_mode": getattr(value, "radar_mode", None),
        "is_alive": getattr(value, "is_alive", True),
    }
    return _state_to_aircraft(state, agent_id)


def infer_project_intent(
    my_state_or_aircraft: Any,
    enemy_state_or_aircraft: Any,
    env: Any = None,
    *,
    my_agent_id: str = "A0100",
    enemy_agent_id: str = "B0100",
    extra_agents: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Infer intent through SituationAlgorithmSwitcher -> bvr_intent_new."""

    my_aircraft = _coerce_aircraft(my_state_or_aircraft, my_agent_id)
    enemy_aircraft = _coerce_aircraft(enemy_state_or_aircraft, enemy_agent_id)
    if my_aircraft is None or enemy_aircraft is None:
        return "NEUTRAL", {"used_model": False, "reason": "missing_aircraft"}
    if not getattr(my_aircraft, "is_alive", True) or not getattr(enemy_aircraft, "is_alive", True):
        return "NEUTRAL", {"used_model": False, "reason": "inactive_aircraft"}

    runtime_env = env
    if runtime_env is None or getattr(runtime_env, "agents", None) is None:
        agents: Dict[str, Any] = {
            str(getattr(my_aircraft, "agent_id", my_agent_id)): my_aircraft,
            str(getattr(enemy_aircraft, "agent_id", enemy_agent_id)): enemy_aircraft,
        }
        for extra_id, extra_value in (extra_agents or {}).items():
            agents[str(extra_id)] = _coerce_aircraft(extra_value, str(extra_id))
        runtime_env = _ShimEnv(
            agents=agents,
            current_step=int(getattr(env, "current_step", 0) or 0),
            time_interval=float(getattr(env, "time_interval", 0.2) or 0.2),
        )

    from core.situation_algorithm_switcher import get_situation_algorithm_switcher

    switcher = get_situation_algorithm_switcher()
    intent = str(switcher.recognize_intent(enemy_aircraft, my_aircraft, runtime_env) or "NEUTRAL")
    meta = switcher.bvr_intent_new.get_last_infer_info(enemy_aircraft) or {}
    return intent, dict(meta)


def project_intent_to_legacy_text(intent: str) -> str:
    mapping = {
        "ATTACK": "attack",
        "NEUTRAL": "neutral",
        "RETREAT": "disengage",
    }
    return mapping.get(str(intent), "neutral")


def project_intent_to_core_enum_name(intent: str) -> str:
    mapping = {
        "ATTACK": "ATTACK",
        "NEUTRAL": "RECONNAISSANCE",
        "RETREAT": "ESCAPE",
    }
    return mapping.get(str(intent), "RECONNAISSANCE")
