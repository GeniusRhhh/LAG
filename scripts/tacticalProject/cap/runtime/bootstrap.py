from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RuntimeContext:
    current_dir: str
    project_root: str
    tactical_dir: str


def _clear_related_module_cache(module_keywords: Iterable[str]) -> None:
    keywords = tuple(k.lower() for k in module_keywords)
    for module_name in list(sys.modules.keys()):
        lowered_name = module_name.lower()
        if any(keyword in lowered_name for keyword in keywords):
            del sys.modules[module_name]


def _resolve_paths(entry_file: str) -> RuntimeContext:
    current_dir = os.path.dirname(os.path.abspath(entry_file))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    tactical_dir = os.path.abspath(os.path.join(current_dir, ".."))
    return RuntimeContext(
        current_dir=current_dir,
        project_root=project_root,
        tactical_dir=tactical_dir,
    )


def _ensure_python_paths(context: RuntimeContext) -> None:
    for path in (context.project_root, context.tactical_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


def _configure_environment(my_aircraft_type: str, enemy_aircraft_type: str) -> None:
    os.environ["FRIEND_BASELINE_MODEL"] = "SU27" if my_aircraft_type == "su27sk" else "F16"
    os.environ["ENEMY_BASELINE_MODEL"] = "SU27" if enemy_aircraft_type == "su27sk" else "F16"
    os.environ.setdefault("CAP_ROOTCAUSE_TRACE", "0")
    os.environ.setdefault("CAP_B0100_DEEP_TRACE", "0")
    os.environ.setdefault("CAP_ALL_B_DEEP_TRACE", "0")
    os.environ.setdefault("CAP_CONTROL_DEBUG", "0")
    os.environ.setdefault("CAP_DEBUG_PRINT", "0")
    os.environ["CAP_ENEMY_F16_NATIVE_ENABLED"] = "0"
    # Force the default runtime onto the enemy's own unified tactical AI path.
    os.environ["CAP_NEW_ENEMY_MANEUVER_AI_ENABLED"] = "0"
    # Default to the enemy's own raw UnifiedEnemyTacticalAI path instead of pair-aware overlays.
    os.environ["CAP_ENEMY_USE_PAIRAWARE_UNIFIED_ENABLED"] = "0"
    # Enable the pair-based wave controller on top of the enemy's own tactical AI.
    os.environ["ENEMY_DISABLE_WAVE_MODE"] = "0"
    # Force enemy-friendly bridge off in the default runtime.
    os.environ["CAP_ENEMY_FRIENDLY_BRIDGE_ENABLED"] = "0"
    # Keep enemy on the original baseline low-level path.
    os.environ["CAP_ENEMY_RULE_LOWLEVEL_ENABLED"] = "0"
    # Wave controller owns the distance-discipline loop; keep the direct RTB hack off.
    os.environ.setdefault("CAP_ENEMY_DIRECT_RTB_ENABLED", "0")
    os.environ.setdefault("CAP_ENEMY_DIRECT_RTB_DISTANCE_KM", "46")
    os.environ.setdefault("ENEMY_TURNBACK_DISTANCE_KM", "44")
    os.environ.setdefault("ENEMY_HARD_STANDOFF_DISTANCE_KM", "37")
    os.environ.setdefault("ENEMY_REATTACK_DISTANCE_KM", "64")
    os.environ.setdefault("ENEMY_REGROUP_RELEASE_DISTANCE_KM", "62")
    os.environ.setdefault("ENEMY_PULLBACK_DISTANCE_KM", "28")
    os.environ.setdefault("ENEMY_REGROUP_TIMEOUT_S", "13")
    os.environ.setdefault("ENEMY_REGROUP_MIN_HOLD_S", "8")
    os.environ.setdefault("ENEMY_REGROUP_DEPART_MARGIN_KM", "16")
    os.environ.setdefault("ENEMY_TURN_PHASE_S", "5")
    os.environ.setdefault("ENEMY_REATTACK_HOLD_S", "2")
    os.environ.setdefault("ENEMY_TURN_SOUTH_MIN_HOLD_S", "6")
    os.environ.setdefault("ENEMY_TURN_SOUTH_DEPART_MARGIN_KM", "6")
    os.environ.setdefault("ENEMY_RECOVERY_HOLD_S", "5")
    os.environ.setdefault("ENEMY_RTB_ENABLED", "0")
    os.environ.setdefault("ENEMY_SECOND_ATTACK_PROB", "0")
    # Experimental patch: allow one controlled enemy simulator recreate when
    # sustained energy loss is detected.
    os.environ["CAP_ENEMY_SIM_RECREATE_ENABLED"] = "1"
    os.environ.setdefault("CAP_ENEMY_SIM_RECREATE_MAX_COUNT", "8")
    os.environ.setdefault("CAP_ENEMY_SIM_RECREATE_WINDOW", "12")
    os.environ.setdefault("CAP_ENEMY_SIM_RECREATE_COOLDOWN_STEPS", "220")
    os.environ["CAP_FRIENDLY_SIM_RECREATE_ENABLED"] = "1"
    os.environ.setdefault("CAP_FRIENDLY_SIM_RECREATE_MAX_COUNT", "4")
    os.environ.setdefault("CAP_FRIENDLY_SIM_RECREATE_WINDOW", "12")
    os.environ.setdefault("CAP_FRIENDLY_SIM_RECREATE_COOLDOWN_STEPS", "260")


def bootstrap_runtime(
    entry_file: str,
    my_aircraft_type: str,
    enemy_aircraft_type: str,
    module_keywords: Iterable[str] = ("patrol", "tactical", "cap"),
) -> RuntimeContext:
    _clear_related_module_cache(module_keywords)
    context = _resolve_paths(entry_file)
    _ensure_python_paths(context)
    _configure_environment(my_aircraft_type, enemy_aircraft_type)
    return context
