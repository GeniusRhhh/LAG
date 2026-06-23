from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


CURRENT_DIR = Path(__file__).resolve().parent
CAP_DIR = CURRENT_DIR.parent
TACTICAL_DIR = CAP_DIR.parent
PROJECT_ROOT = TACTICAL_DIR.parent.parent

for candidate in (PROJECT_ROOT, TACTICAL_DIR, CAP_DIR, CURRENT_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


try:
    from runtime.bootstrap import bootstrap_runtime
except ImportError:
    from cap.runtime.bootstrap import bootstrap_runtime


_RUNTIME_CONTEXT = bootstrap_runtime(
    entry_file=str(CAP_DIR / "run_cap_simulation.py"),
    my_aircraft_type="su27sk",
    enemy_aircraft_type="f16",
)


from cap.cap_task import CAPTask
from cap.faor_manager import FAORManager, RiskZone
from cap.picture import RiskZone as PictureRiskZone
from envs.JSBSim.core.catalog import Catalog as c

try:
    from acmi_writer import AcmiRecorder
except ImportError:
    from cap.acmi_writer import AcmiRecorder

try:
    from run_helpers import (
        build_step_actions,
        create_patrol_task,
        detect_patrol_display,
        log_final_document_summary,
        log_final_guidance_summary,
        log_heading_changes,
        log_structured_battle_events,
        maybe_log_guidance_snapshot,
        prepare_patrol_config,
    )
except ImportError:
    from cap.run_helpers import (
        build_step_actions,
        create_patrol_task,
        detect_patrol_display,
        log_final_document_summary,
        log_final_guidance_summary,
        log_heading_changes,
        log_structured_battle_events,
        maybe_log_guidance_snapshot,
        prepare_patrol_config,
    )

try:
    from runtime.engine import CAPEnv
except ImportError:
    from cap.runtime.engine import CAPEnv
try:
    from run_logging import set_sim_log_clock
except ImportError:
    from cap.run_logging import set_sim_log_clock
from run_detection_verification_real import (  # noqa: E402
    A0100_LAT,
    A0100_LON,
    DEG_TO_KM,
    apply_enemy_maneuver,
    generate_scenario_config,
    setup_logging,
)

from ch6_validation_scenarios import (  # noqa: E402
    Ch6ScenarioSpec,
    get_scenario_spec,
    iter_scenario_specs,
)

try:
    import generate_ch6_validation_assets as ch6_asset_generator  # noqa: E402
except Exception:
    ch6_asset_generator = None


OUTPUT_BASE_DIR = TACTICAL_DIR / "cap_results" / "Chapter6_validation"
FRIENDLY_AGENT_IDS = ("A0100", "A0200", "A0300", "A0400")
ENEMY_AGENT_IDS = ("B0100", "B0200", "B0300", "B0400")
ENEMY_INTENT_GROUPS = {
    "enemy_left": ("B0100", "B0200"),
    "enemy_right": ("B0300", "B0400"),
}
THREAT_PRIORITY = {"critical": 4, "high": 3, "medium": 2, "low": 1}
THREAT_LEVEL_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
CONTROL_PHASE_ORDER = (
    "BEYOND_NLT",
    "NLT_MELD",
    "MELD_MTR",
    "MTR_LR",
    "LR_TR",
    "TR_DOR",
    "DOR_DR",
    "DR_MAR",
    "BEYOND_MAR",
)
CONTROL_PHASE_DESCRIPTIONS = {
    "BEYOND_NLT": "尚未进入 NLT 节点的远距接敌段",
    "NLT_MELD": "由 NLT 向 MELD 压缩的控制距离段",
    "MELD_MTR": "由 MELD 向 MTR 压缩的控制距离段",
    "MTR_LR": "进入由 MTR 向 LR 推进的阶段",
    "LR_TR": "进入由 LR 向 TR 推进的阶段",
    "TR_DOR": "首轮发射后转入脱离准备段",
    "DOR_DR": "首轮交战后进入再次压缩准备段",
    "DR_MAR": "进入临界规避约束更强的阶段",
    "BEYOND_MAR": "后期交战收口前的重新整理段",
}
INTENT_CLASS_ORDER = ("侦察", "防御", "撤退", "攻击")
INTENT_CLASS_TO_CODE = {
    "攻击": "attack",
    "防御": "defense",
    "撤退": "retreat",
    "侦察": "reconnaissance",
}
INTENT_ALIAS_MAP = {
    "攻击": "攻击",
    "协同": "攻击",
    "attack": "攻击",
    "ATTACK": "攻击",
    "penetrate": "攻击",
    "PENETRATE": "攻击",
    "防御": "防御",
    "规避": "防御",
    "defense": "防御",
    "DEFENSE": "防御",
    "defensive": "防御",
    "DEFENSIVE": "防御",
    "evade": "防御",
    "EVADE": "防御",
    "撤退": "撤退",
    "retreat": "撤退",
    "RETREAT": "撤退",
    "侦察": "侦察",
    "中立": "侦察",
    "neutral": "侦察",
    "NEUTRAL": "侦察",
    "recon": "侦察",
    "RECON": "侦察",
    "reconnaissance": "侦察",
    "RECONNAISSANCE": "侦察",
}


INTENT_ALIAS_MAP.update(
    {
        "search": INTENT_ALIAS_MAP.get("recon", ""),
        "SEARCH": INTENT_ALIAS_MAP.get("RECON", INTENT_ALIAS_MAP.get("recon", "")),
    }
)

INTENT_ALIAS_MAP.update(
    {
        "attack_maneuver": INTENT_ALIAS_MAP.get("attack", ""),
        "ATTACK_MANEUVER": INTENT_ALIAS_MAP.get("ATTACK", INTENT_ALIAS_MAP.get("attack", "")),
        "aggressive_approach": INTENT_ALIAS_MAP.get("attack", ""),
        "AGGRESSIVE_APPROACH": INTENT_ALIAS_MAP.get("ATTACK", INTENT_ALIAS_MAP.get("attack", "")),
        "lock_on": INTENT_ALIAS_MAP.get("attack", ""),
        "LOCK_ON": INTENT_ALIAS_MAP.get("ATTACK", INTENT_ALIAS_MAP.get("attack", "")),
        "defensive_positioning": INTENT_ALIAS_MAP.get("defense", ""),
        "DEFENSIVE_POSITIONING": INTENT_ALIAS_MAP.get("DEFENSE", INTENT_ALIAS_MAP.get("defense", "")),
        "evasive_maneuver": INTENT_ALIAS_MAP.get("defense", ""),
        "EVASIVE_MANEUVER": INTENT_ALIAS_MAP.get("DEFENSE", INTENT_ALIAS_MAP.get("defense", "")),
        "dive_escape": INTENT_ALIAS_MAP.get("defense", ""),
        "DIVE_ESCAPE": INTENT_ALIAS_MAP.get("DEFENSE", INTENT_ALIAS_MAP.get("defense", "")),
        "climb_escape": INTENT_ALIAS_MAP.get("defense", ""),
        "CLIMB_ESCAPE": INTENT_ALIAS_MAP.get("DEFENSE", INTENT_ALIAS_MAP.get("defense", "")),
        "escape": INTENT_ALIAS_MAP.get("retreat", ""),
        "ESCAPE": INTENT_ALIAS_MAP.get("RETREAT", INTENT_ALIAS_MAP.get("retreat", "")),
        "neutral_flight": INTENT_ALIAS_MAP.get("recon", ""),
        "NEUTRAL_FLIGHT": INTENT_ALIAS_MAP.get("RECON", INTENT_ALIAS_MAP.get("recon", "")),
        "formation_maintain": INTENT_ALIAS_MAP.get("recon", ""),
        "FORMATION_MAINTAIN": INTENT_ALIAS_MAP.get("RECON", INTENT_ALIAS_MAP.get("recon", "")),
    }
)


def _apply_outer_compat_patches() -> None:
    try:
        import cap.cap_task_impl as cap_task_impl_module
        from cap import cap_task_action_helpers

        if not hasattr(cap_task_impl_module, "_capact"):
            cap_task_impl_module._capact = cap_task_action_helpers
    except Exception:
        pass


def _refresh_cap_debug_flags_from_env() -> None:
    try:
        import cap.cap_task_refactor_helpers as refactor_helpers

        refactor_helpers._B0100_DEEP_TRACE = os.environ.get("CAP_B0100_DEEP_TRACE", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        refactor_helpers._ALL_B_DEEP_TRACE = os.environ.get("CAP_ALL_B_DEEP_TRACE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    except Exception:
        pass


class ScenarioAwareAwacsProxy:
    def __init__(self, base_awacs, scenario):
        self._base = base_awacs
        self._scenario = scenario
        self._last_time = 0.0
        self._last_distance = 999.0
        self._last_available = True

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _calc_distance(self, friendly_positions: Optional[List[np.ndarray]]) -> float:
        if not friendly_positions:
            return self._last_distance
        ground_truth = getattr(self._base, "_ground_truth", {}) or {}
        if not ground_truth:
            return self._last_distance
        center = np.mean(np.asarray(friendly_positions, dtype=float), axis=0)
        distances = []
        for item in ground_truth.values():
            pos = np.asarray(item.get("position", [0.0, 0.0, 0.0]), dtype=float)
            distances.append(float(np.linalg.norm(pos - center)))
        return min(distances) if distances else self._last_distance

    def update(self, current_time: float, friendly_positions: List[np.ndarray] = None):
        self._last_time = float(current_time)
        self._last_distance = float(self._calc_distance(friendly_positions))
        tracks = self._base.update(current_time, friendly_positions=friendly_positions)
        self._last_available = bool(
            self._scenario.is_awacs_available(self._last_time, self._last_distance)
        )
        return tracks if self._last_available else {}

    def get_tracks(self):
        if not self._last_available:
            return {}
        return self._base.get_tracks()


@contextmanager
def _temporary_env(overrides: Dict[str, str]):
    sentinel = object()
    backup = {key: os.environ.get(key, sentinel) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, old_value in backup.items():
            if old_value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _battlefield_xy(task: CAPTask, env, agent_id: str) -> Tuple[float, float]:
    try:
        x_km, y_km = task._get_battlefield_pos(env, agent_id)
        return float(x_km), float(y_km)
    except Exception:
        lon, lat, _ = env.agents[agent_id].get_geodetic()
        x_km = (lon - A0100_LON) * DEG_TO_KM + 75.0
        y_km = (lat - A0100_LAT) * DEG_TO_KM
        return float(x_km), float(y_km)


def _truth_positions(task: CAPTask, env, prefix: str) -> Dict[str, Tuple[float, float, float]]:
    result: Dict[str, Tuple[float, float, float]] = {}
    for agent_id, agent in env.agents.items():
        if not str(agent_id).startswith(prefix) or not getattr(agent, "is_alive", False):
            continue
        x_km, y_km = _battlefield_xy(task, env, agent_id)
        try:
            _, _, alt_m = agent.get_geodetic()
            z_km = float(alt_m) / 1000.0
        except Exception:
            z_km = 0.0
        result[str(agent_id)] = (x_km, y_km, z_km)
    return result


def _collect_team_flight_metrics(env, prefix: str) -> Dict[str, Optional[float]]:
    altitudes_m: List[float] = []
    speeds_mps: List[float] = []
    climb_rates_mps: List[float] = []
    for agent_id, agent in env.agents.items():
        if not str(agent_id).startswith(prefix) or not getattr(agent, "is_alive", False):
            continue
        try:
            altitude_m = float(agent.get_property_value(c.position_h_sl_m))
        except Exception:
            try:
                _, _, altitude_m = agent.get_geodetic()
                altitude_m = float(altitude_m)
            except Exception:
                altitude_m = float("nan")
        try:
            speed_mps = float(agent.get_property_value(c.velocities_vc_mps))
        except Exception:
            speed_mps = float("nan")
        try:
            climb_rate_mps = -float(agent.get_property_value(c.velocities_v_down_mps))
        except Exception:
            climb_rate_mps = float("nan")

        if np.isfinite(altitude_m):
            altitudes_m.append(float(altitude_m))
        if np.isfinite(speed_mps):
            speeds_mps.append(float(speed_mps))
        if np.isfinite(climb_rate_mps):
            climb_rates_mps.append(float(climb_rate_mps))

    max_descent_rate_mps = max((-value for value in climb_rates_mps), default=float("nan"))
    return {
        "count": float(len(altitudes_m)),
        "min_altitude_m": min(altitudes_m) if altitudes_m else None,
        "mean_altitude_m": float(np.mean(altitudes_m)) if altitudes_m else None,
        "min_speed_mps": min(speeds_mps) if speeds_mps else None,
        "mean_speed_mps": float(np.mean(speeds_mps)) if speeds_mps else None,
        "max_descent_rate_mps": float(max_descent_rate_mps) if np.isfinite(max_descent_rate_mps) else None,
    }


def _collect_recreate_snapshot(patrol_task: CAPTask) -> Dict[str, object]:
    def _extract_counts(store_name: str) -> Tuple[int, Dict[str, int]]:
        store = getattr(patrol_task, store_name, None)
        if not isinstance(store, dict):
            return 0, {}
        by_agent: Dict[str, int] = {}
        total = 0
        for agent_id, payload in store.items():
            if not isinstance(payload, dict):
                continue
            try:
                count = int(payload.get("count", 0) or 0)
            except Exception:
                count = 0
            if count <= 0:
                continue
            by_agent[str(agent_id)] = count
            total += count
        return total, by_agent

    friendly_total, friendly_by_agent = _extract_counts("_friendly_sim_recreate_state")
    enemy_total, enemy_by_agent = _extract_counts("_enemy_sim_recreate_state")
    return {
        "friendly_sim_recreate_total": int(friendly_total),
        "enemy_sim_recreate_total": int(enemy_total),
        "friendly_sim_recreate_by_agent_json": json.dumps(friendly_by_agent, ensure_ascii=False, sort_keys=True),
        "enemy_sim_recreate_by_agent_json": json.dumps(enemy_by_agent, ensure_ascii=False, sort_keys=True),
    }


def _normalize_tactic_name(value: object) -> str:
    return str(value or "").strip().upper()


def _matches_expected_tactic(value: str, expected_keywords: Iterable[str]) -> bool:
    norm = _normalize_tactic_name(value)
    if not norm:
        return False
    return any(keyword in norm for keyword in expected_keywords)


def _friendly_missiles_left(env) -> int:
    remaining = 0
    for agent_id, agent in env.agents.items():
        if str(agent_id).startswith("A") and getattr(agent, "is_alive", False):
            remaining += int(getattr(agent, "num_missiles", 0) or 0)
    return remaining


def _enemy_missiles_left(env) -> int:
    remaining = 0
    for agent_id, agent in env.agents.items():
        if str(agent_id).startswith("B") and getattr(agent, "is_alive", False):
            remaining += int(getattr(agent, "num_missiles", 0) or 0)
    return remaining


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _norm_label(value: object, default: str = "UNKNOWN") -> str:
    text = str(value or "").strip()
    return text if text else default


def _enum_text(value: object, default: str = "") -> str:
    return _norm_label(getattr(value, "value", value), default=default)


def _normalize_phase_label(value: object, default: str = "") -> str:
    text = _norm_label(value, default=default)
    if not text:
        return default
    return text.split(".")[-1]


def _normalize_intent_class(pred_label: object, raw_intent: object = "") -> str:
    for candidate in (pred_label, raw_intent):
        text = str(candidate or "").strip()
        if not text:
            continue
        mapped = INTENT_ALIAS_MAP.get(text)
        if mapped:
            return mapped
        lowered = text.lower()
        mapped = INTENT_ALIAS_MAP.get(lowered)
        if mapped:
            return mapped
        uppered = text.upper()
        mapped = INTENT_ALIAS_MAP.get(uppered)
        if mapped:
            return mapped
    return ""


def _intent_class_code(intent_class: object) -> str:
    return INTENT_CLASS_TO_CODE.get(str(intent_class or "").strip(), "")


def _safe_float(value: object, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _series_float(rows: List[Dict[str, object]], key: str, default: float = float("nan")) -> List[float]:
    return [_safe_float(row.get(key), default) for row in rows]


def _series_int(rows: List[Dict[str, object]], key: str, default: int = 0) -> List[int]:
    return [_safe_int(row.get(key), default) for row in rows]


def _guidance_metric_series(
    timeline_rows: List[Dict[str, object]],
    *,
    base_key: str,
    prefer_unique: bool = True,
) -> List[int]:
    if prefer_unique:
        unique_key = f"{base_key}_unique_count"
        if any(unique_key in row for row in timeline_rows):
            return _series_int(timeline_rows, unique_key)
    if any(base_key in row for row in timeline_rows):
        return _series_int(timeline_rows, base_key)
    raw_key = f"{base_key}_raw_count"
    if any(raw_key in row for row in timeline_rows):
        return _series_int(timeline_rows, raw_key)
    return [0 for _ in timeline_rows]


def _intent_entry_from_analysis(analysis: object, *, ready: bool) -> Dict[str, object]:
    intent_type = _enum_text(getattr(analysis, "intent", None), default="").lower()
    label_text = str(getattr(analysis, "pred_label", "") or "")
    intent_class = _normalize_intent_class(label_text, intent_type)
    name_text = intent_class
    threat_text = _enum_text(getattr(analysis, "threat_level", None), default="").lower()
    distance_km = _safe_float(getattr(analysis, "distance_km", 0.0), 0.0)
    confidence = _safe_float(getattr(analysis, "confidence", 0.0), 0.0)
    return {
        "track_id": str(getattr(analysis, "track_id", "") or ""),
        "intent_name": name_text,
        "intent_label": intent_class,
        "intent_raw_label": label_text,
        "intent_class": intent_class,
        "intent_class_code": _intent_class_code(intent_class),
        "intent_type": intent_type,
        "intent_threat": threat_text,
        "intent_confidence": confidence,
        "intent_model_ready": int(bool(ready)),
        "intent_distance_km": distance_km,
        "intent_present": 1,
    }


def _intent_blank_entry(track_id: str = "") -> Dict[str, object]:
    return {
        "track_id": str(track_id or ""),
        "intent_name": "",
        "intent_label": "",
        "intent_raw_label": "",
        "intent_class": "",
        "intent_class_code": "",
        "intent_type": "",
        "intent_threat": "",
        "intent_confidence": 0.0,
        "intent_model_ready": 0,
        "intent_distance_km": 0.0,
        "intent_present": 0,
    }


def _collect_intent_truth_snapshot(patrol_task: CAPTask) -> Dict[str, object]:
    snapshot: Dict[str, object] = {
        "intent_truth_any_count": 0,
        "intent_truth_match_count": 0,
        "intent_truth_ready_match_count": 0,
    }
    enemy_adapter = getattr(patrol_task, "enemy_adapter", None)
    for enemy_id in ENEMY_AGENT_IDS:
        truth_class = ""
        truth_code = ""
        if enemy_adapter is not None and hasattr(enemy_adapter, "get_action_annotation_for_csv"):
            try:
                annotation = enemy_adapter.get_action_annotation_for_csv(enemy_id) or {}
            except Exception:
                annotation = {}
            raw_truth = str(
                annotation.get("Action_Intent_DatasetTruth", "")
                or annotation.get("Action_Intent_Coarse_DatasetTruth", "")
                or annotation.get("Action_Intent_Coarse", "")
                or annotation.get("Action_Intent", "")
                or ""
            )
            truth_class = _normalize_intent_class(raw_truth, raw_truth)
            truth_code = _intent_class_code(truth_class)
        snapshot[f"{enemy_id}_intent_truth_class"] = truth_class
        snapshot[f"{enemy_id}_intent_truth_code"] = truth_code
    return snapshot


def _intent_group_entry(group_name: str, track_ids: Iterable[str], track_entries: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    members = [dict(track_entries.get(track_id) or _intent_blank_entry(track_id)) for track_id in track_ids]
    present_members = [member for member in members if int(member.get("intent_present", 0) or 0) > 0]
    if not present_members:
        return {
            "group_name": group_name,
            "dominant_target": "",
            "dominant_name": "",
            "dominant_class": "",
            "dominant_class_code": "",
            "dominant_type": "",
            "dominant_threat": "",
            "dominant_confidence": 0.0,
            "contact_count": 0,
            "classified_count": 0,
            "unclassified_count": 0,
            "ready_count": 0,
            "attack_count": 0,
            "defense_count": 0,
            "retreat_count": 0,
            "recon_count": 0,
            "high_count": 0,
            "critical_count": 0,
        }

    present_members.sort(
        key=lambda item: (
            1 if str(item.get("intent_class", "") or "").strip() else 0,
            THREAT_PRIORITY.get(str(item.get("intent_threat", "") or "").lower(), 0),
            float(item.get("intent_confidence", 0.0) or 0.0),
            -float(item.get("intent_distance_km", 0.0) or 0.0),
        ),
        reverse=True,
    )
    dominant = present_members[0]
    class_counter = Counter(str(member.get("intent_class_code", "") or "").lower() for member in present_members)
    high_count = sum(1 for member in present_members if str(member.get("intent_threat", "") or "").lower() in {"high", "critical"})
    critical_count = sum(
        1 for member in present_members if str(member.get("intent_threat", "") or "").lower() == "critical"
    )
    return {
        "group_name": group_name,
        "dominant_target": str(dominant.get("track_id", "") or ""),
        "dominant_name": str(dominant.get("intent_name", "") or ""),
        "dominant_class": str(dominant.get("intent_class", "") or ""),
        "dominant_class_code": str(dominant.get("intent_class_code", "") or ""),
        "dominant_type": str(dominant.get("intent_type", "") or ""),
        "dominant_threat": str(dominant.get("intent_threat", "") or ""),
        "dominant_confidence": float(dominant.get("intent_confidence", 0.0) or 0.0),
        "contact_count": int(len(present_members)),
        "classified_count": int(sum(1 for member in present_members if str(member.get("intent_class", "") or "").strip())),
        "unclassified_count": int(sum(1 for member in present_members if not str(member.get("intent_class", "") or "").strip())),
        "ready_count": int(
            sum(
                1
                for member in present_members
                if int(member.get("intent_model_ready", 0) or 0) > 0
                and str(member.get("intent_class", "") or "").strip()
            )
        ),
        "attack_count": int(class_counter.get("attack", 0)),
        "defense_count": int(class_counter.get("defense", 0)),
        "retreat_count": int(class_counter.get("retreat", 0)),
        "recon_count": int(class_counter.get("reconnaissance", 0)),
        "high_count": int(high_count),
        "critical_count": int(critical_count),
    }


def _count_switches_nonempty(timeline_rows: List[Dict[str, object]], key: str) -> int:
    count = 0
    prev_value = ""
    seen = False
    for row in timeline_rows:
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        if not seen:
            prev_value = value
            seen = True
            continue
        if value != prev_value:
            count += 1
            prev_value = value
    return count


def _duration_where(timeline_rows: List[Dict[str, object]], durations: List[float], predicate) -> float:
    total = 0.0
    for row, dt in zip(timeline_rows, durations):
        try:
            if predicate(row):
                total += float(dt)
        except Exception:
            continue
    return float(total)


def _plot_multi_level_lines(
    ax,
    times: List[float],
    series_map: Dict[str, List[object]],
    title: str,
    level_order: Dict[str, int],
) -> None:
    for label, values in series_map.items():
        y_values = []
        for value in values:
            text = _norm_label(value, default="").lower()
            y_values.append(level_order.get(text, np.nan) if text else np.nan)
        ax.step(times, y_values, where="post", linewidth=1.5, label=label)
    ordered = sorted(level_order.items(), key=lambda item: item[1])
    ax.set_title(title)
    ax.set_yticks([level for _, level in ordered])
    ax.set_yticklabels([label for label, _ in ordered])
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_categorical(
    ax,
    times: List[float],
    values: List[str],
    title: str,
    *,
    preferred_order: Optional[Iterable[str]] = None,
) -> None:
    labels: List[str] = []
    preferred = [str(item) for item in list(preferred_order or []) if str(item or "").strip()]
    for label in preferred:
        if any(_norm_label(item, default="") == label for item in values) and label not in labels:
            labels.append(label)
    for item in values:
        norm = _norm_label(item, default="")
        if norm and norm not in labels:
            labels.append(norm)
    mapping = {label: idx for idx, label in enumerate(labels)}
    y_values = [mapping.get(_norm_label(item, default=""), np.nan) for item in values]
    ax.step(times, y_values, where="post", linewidth=1.8)
    ax.set_title(title)
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels(labels)
    ax.grid(True, alpha=0.3)


def _plot_level_timeline(
    ax,
    times: List[float],
    values: List[object],
    title: str,
    level_order: Dict[str, int],
) -> None:
    ordered = sorted(level_order.items(), key=lambda item: item[1])
    labels = [item[0] for item in ordered]
    mapping = {str(label).lower(): int(level) for label, level in ordered}
    y_values = []
    for item in values:
        text = _norm_label(item, default="").lower()
        y_values.append(mapping.get(text, np.nan) if text else np.nan)
    ax.step(times, y_values, where="post", linewidth=1.8)
    ax.set_title(title)
    ax.set_yticks([level_order[label] for label in labels])
    ax.set_yticklabels(labels)
    ax.grid(True, alpha=0.3)


def _running_ratio_series(numerator: List[int], denominator: List[int]) -> List[float]:
    values: List[float] = []
    for num, den in zip(numerator, denominator):
        values.append(float(num) / float(den) if den > 0 else 0.0)
    return values


def _phase_entry_rows_from_timeline(
    timeline_rows: List[Dict[str, object]],
    route_label: str,
    phase_key: str,
) -> List[Dict[str, object]]:
    seen: Dict[str, Dict[str, object]] = {}
    for row in timeline_rows:
        phase = _normalize_phase_label(row.get(phase_key), default="")
        if not phase or phase in seen:
            continue
        time_s = float(row.get("time_s", 0.0) or 0.0)
        distance_km = _safe_float(row.get("nearest_enemy_to_friendly_km"), float("nan"))
        seen[phase] = {
            "route": route_label,
            "phase": phase,
            "first_time_s": time_s,
            "distance_km": None if not np.isfinite(distance_km) else float(distance_km),
            "description": CONTROL_PHASE_DESCRIPTIONS.get(phase, ""),
        }

    rows = [seen[phase] for phase in CONTROL_PHASE_ORDER if phase in seen]
    extras = [
        row
        for phase, row in seen.items()
        if phase not in CONTROL_PHASE_ORDER
    ]
    extras.sort(key=lambda item: float(item.get("first_time_s", 0.0) or 0.0))
    rows.extend(extras)
    return rows


def _control_distance_segments(node_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    route_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in node_rows:
        route_rows[str(row.get("route", "") or "")].append(row)

    for route_label, rows in route_rows.items():
        rows = sorted(rows, key=lambda item: float(item.get("first_time_s", 0.0) or 0.0))
        for start_row, end_row in zip(rows, rows[1:]):
            start_time = float(start_row.get("first_time_s", 0.0) or 0.0)
            end_time = float(end_row.get("first_time_s", 0.0) or 0.0)
            start_distance = _safe_float(start_row.get("distance_km"), float("nan"))
            end_distance = _safe_float(end_row.get("distance_km"), float("nan"))
            compression_km = (
                float(start_distance - end_distance)
                if np.isfinite(start_distance) and np.isfinite(end_distance)
                else None
            )
            duration_s = max(0.0, float(end_time - start_time))
            segments.append(
                {
                    "route": route_label,
                    "from_phase": str(start_row.get("phase", "") or ""),
                    "to_phase": str(end_row.get("phase", "") or ""),
                    "start_time_s": start_time,
                    "end_time_s": end_time,
                    "start_distance_km": None if not np.isfinite(start_distance) else float(start_distance),
                    "end_distance_km": None if not np.isfinite(end_distance) else float(end_distance),
                    "compression_km": compression_km,
                    "duration_s": duration_s,
                    "compression_rate_kmps": (
                        float(compression_km / duration_s) if compression_km is not None and duration_s > 0 else None
                    ),
                }
            )
    return segments


def _collect_control_distance_metrics(timeline_rows: List[Dict[str, object]]) -> Dict[str, object]:
    node_rows = []
    node_rows.extend(_phase_entry_rows_from_timeline(timeline_rows, "left", "left_phase"))
    node_rows.extend(_phase_entry_rows_from_timeline(timeline_rows, "right", "right_phase"))
    segment_rows = _control_distance_segments(node_rows)

    left_time_map = {
        str(row.get("phase", "") or ""): row.get("first_time_s")
        for row in node_rows
        if str(row.get("route", "") or "") == "left"
    }
    right_time_map = {
        str(row.get("phase", "") or ""): row.get("first_time_s")
        for row in node_rows
        if str(row.get("route", "") or "") == "right"
    }
    left_distance_map = {
        str(row.get("phase", "") or ""): row.get("distance_km")
        for row in node_rows
        if str(row.get("route", "") or "") == "left"
    }
    right_distance_map = {
        str(row.get("phase", "") or ""): row.get("distance_km")
        for row in node_rows
        if str(row.get("route", "") or "") == "right"
    }

    return {
        "control_distance_nodes": node_rows,
        "control_distance_segments": segment_rows,
        "left_phase_first_entry_time_s": left_time_map,
        "right_phase_first_entry_time_s": right_time_map,
        "left_phase_first_entry_distance_km": left_distance_map,
        "right_phase_first_entry_distance_km": right_distance_map,
    }


def _parse_zone_series(rows: List[Dict[str, object]], enemy_id: str) -> List[str]:
    result: List[str] = []
    for row in rows:
        try:
            payload = json.loads(str(row.get("zone_by_enemy_json", "{}") or "{}"))
        except Exception:
            payload = {}
        result.append(_norm_label(payload.get(enemy_id), default="OUTSIDE"))
    return result


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _generate_plots(output_dir: Path, timeline_rows: List[Dict[str, object]], summary: Optional[Dict[str, object]] = None) -> None:
    if not HAS_MPL or not timeline_rows:
        return

    times = _series_float(timeline_rows, "time_s", 0.0)
    bullseye_series = _series_float(timeline_rows, "nearest_enemy_bullseye_km")
    friendly_distance_series = _series_float(timeline_rows, "nearest_enemy_to_friendly_km")

    fig, axes = plt.subplots(6, 1, figsize=(13.5, 18.0), sharex=True)
    axes[0].step(times, _series_int(timeline_rows, "truth_low_count"), where="post", label="Truth low", linewidth=2.0)
    axes[0].step(times, _series_int(timeline_rows, "truth_medium_count"), where="post", label="Truth medium", linewidth=2.0)
    axes[0].step(times, _series_int(timeline_rows, "truth_high_count"), where="post", label="Truth high", linewidth=2.0)
    if any("picture_low_count" in row for row in timeline_rows):
        axes[0].plot(times, _series_int(timeline_rows, "picture_low_count"), label="Picture low", linewidth=1.2, alpha=0.9)
        axes[0].plot(
            times,
            _series_int(timeline_rows, "picture_medium_count"),
            label="Picture medium",
            linewidth=1.2,
            alpha=0.9,
        )
        axes[0].plot(times, _series_int(timeline_rows, "picture_high_count"), label="Picture high", linewidth=1.2, alpha=0.9)
    axes[0].set_title("Risk-Zone Occupancy: truth vs decision picture")
    axes[0].legend(ncol=3, fontsize=9)
    axes[0].grid(True, alpha=0.3)

    _plot_categorical(axes[1], times, [row.get("enemy_script_phase_name", "") for row in timeline_rows], "Enemy scene phase")

    axes[2].step(
        times,
        _series_int(timeline_rows, "enemy_script_awacs_available"),
        where="post",
        label="AWACS available",
        linewidth=1.8,
    )
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_title("Enemy information condition and trigger distance")
    awacs_ax = axes[2].twinx()
    awacs_ax.plot(
        times,
        _series_float(timeline_rows, "enemy_script_reference_distance_km"),
        label="Reference distance (km)",
        linewidth=1.3,
        color="#C0504D",
    )
    lines_a, labels_a = axes[2].get_legend_handles_labels()
    lines_b, labels_b = awacs_ax.get_legend_handles_labels()
    axes[2].legend(lines_a + lines_b, labels_a + labels_b, fontsize=9, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    _plot_categorical(axes[3], times, [row.get("left_tactic", "") for row in timeline_rows], "Left pair tactic")
    _plot_categorical(axes[4], times, [row.get("right_tactic", "") for row in timeline_rows], "Right pair tactic")

    alive_ax = axes[5]
    alive_ax.step(times, _series_int(timeline_rows, "friendly_alive"), where="post", label="Friendly alive", linewidth=1.9)
    alive_ax.step(times, _series_int(timeline_rows, "enemy_alive"), where="post", label="Enemy alive", linewidth=1.9)
    alive_ax.set_ylabel("Alive count")
    alive_ax.grid(True, alpha=0.3)
    dist_ax = alive_ax.twinx()
    dist_ax.plot(times, bullseye_series, label="Nearest enemy to bullseye", linewidth=1.3, color="#C0504D")
    dist_ax.plot(times, friendly_distance_series, label="Nearest enemy to friendly", linewidth=1.3, color="#4F81BD")
    alive_ax.set_title("Survivability and range compression")
    alive_ax.set_xlabel("Time (s)")
    lines_a, labels_a = alive_ax.get_legend_handles_labels()
    lines_b, labels_b = dist_ax.get_legend_handles_labels()
    alive_ax.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right", fontsize=9)
    _save_figure(fig, output_dir / "overview.png")

    fig, axes = plt.subplots(4, 1, figsize=(13.5, 14.5), sharex=True)
    axes[0].plot(times, _series_int(timeline_rows, "awacs_track_count"), label="AWACS tracks", linewidth=1.8)
    axes[0].plot(times, _series_int(timeline_rows, "radar_track_count"), label="Radar tracks", linewidth=1.8)
    axes[0].plot(times, _series_int(timeline_rows, "stable_ready_count"), label="Stable ready", linewidth=1.8)
    if any("scan_target_count" in row for row in timeline_rows):
        axes[0].plot(times, _series_int(timeline_rows, "scan_target_count"), label="Scan targets", linewidth=1.4)
    axes[0].set_title("Detection, tracking, and ready-state buildup")
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].grid(True, alpha=0.3)

    gate_pass = _guidance_metric_series(timeline_rows, base_key="gate_pass", prefer_unique=True)
    gate_block = _guidance_metric_series(timeline_rows, base_key="gate_block", prefer_unique=True)
    gate_total = [p + b for p, b in zip(gate_pass, gate_block)]
    axes[1].step(times, gate_pass, where="post", label="Gate pass (unique)", linewidth=1.9)
    axes[1].step(times, gate_block, where="post", label="Gate block (unique)", linewidth=1.9)
    gate_rate_ax = axes[1].twinx()
    gate_rate_ax.plot(
        times,
        _running_ratio_series(gate_pass, gate_total),
        label="Gate pass rate (unique)",
        linewidth=1.2,
        color="#7F7F7F",
    )
    axes[1].set_title("Prelaunch gate accumulation and pass rate (deduplicated)")
    lines_a, labels_a = axes[1].get_legend_handles_labels()
    lines_b, labels_b = gate_rate_ax.get_legend_handles_labels()
    axes[1].legend(lines_a + lines_b, labels_a + labels_b, loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    relay_attempt = _guidance_metric_series(timeline_rows, base_key="relay_attempt", prefer_unique=True)
    relay_success = _guidance_metric_series(timeline_rows, base_key="relay_success", prefer_unique=True)
    axes[2].step(times, relay_attempt, where="post", label="Relay attempt (unique)", linewidth=1.9)
    axes[2].step(times, relay_success, where="post", label="Relay success (unique)", linewidth=1.9)
    relay_rate_ax = axes[2].twinx()
    relay_rate_ax.plot(
        times,
        _running_ratio_series(relay_success, relay_attempt),
        label="Relay success rate (unique)",
        linewidth=1.2,
        color="#7F7F7F",
    )
    axes[2].set_title("Relay guidance accumulation and success rate (deduplicated)")
    lines_a, labels_a = axes[2].get_legend_handles_labels()
    lines_b, labels_b = relay_rate_ax.get_legend_handles_labels()
    axes[2].legend(lines_a + lines_b, labels_a + labels_b, loc="upper left", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    axes[3].step(times, _series_int(timeline_rows, "missile_launch_count"), where="post", label="Launch total", linewidth=1.8)
    axes[3].step(times, _series_int(timeline_rows, "missile_outcome_count"), where="post", label="Outcome total", linewidth=1.8)
    axes[3].plot(times, _series_int(timeline_rows, "active_guided_missile_peak"), label="Active guided peak", linewidth=1.3)
    axes[3].plot(times, _series_int(timeline_rows, "friendly_missiles_left"), label="Friendly missiles left", linewidth=1.3)
    axes[3].set_title("Missile-chain progression")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=2, fontsize=9)
    axes[3].grid(True, alpha=0.3)
    _save_figure(fig, output_dir / "guidance.png")

    _generate_detailed_plots(output_dir / "figures", timeline_rows, summary=summary)


def _generate_detailed_plots(
    figures_dir: Path,
    timeline_rows: List[Dict[str, object]],
    *,
    summary: Optional[Dict[str, object]] = None,
) -> None:
    if not HAS_MPL or not timeline_rows:
        return

    summary = dict(summary or {})
    times = _series_float(timeline_rows, "time_s", 0.0)
    threat_levels = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    intent_levels = dict(THREAT_LEVEL_ORDER)

    fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True)
    axes[0].step(times, _series_int(timeline_rows, "truth_low_count"), where="post", label="Truth low", linewidth=1.8)
    axes[0].step(times, _series_int(timeline_rows, "truth_medium_count"), where="post", label="Truth medium", linewidth=1.8)
    axes[0].step(times, _series_int(timeline_rows, "truth_high_count"), where="post", label="Truth high", linewidth=1.8)
    axes[0].set_title("Truth risk-zone counts")
    axes[0].legend(ncol=3, fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].step(times, _series_int(timeline_rows, "picture_low_count"), where="post", label="Picture low", linewidth=1.8)
    axes[1].step(times, _series_int(timeline_rows, "picture_medium_count"), where="post", label="Picture medium", linewidth=1.8)
    axes[1].step(times, _series_int(timeline_rows, "picture_high_count"), where="post", label="Picture high", linewidth=1.8)
    axes[1].set_title("Decision-picture risk-zone counts")
    axes[1].legend(ncol=3, fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, _series_int(timeline_rows, "picture_total_count"), label="Picture threats", linewidth=1.5)
    axes[2].plot(times, _series_int(timeline_rows, "mission_total_threats"), label="Mission-evaluator threats", linewidth=1.5)
    axes[2].set_title("Perceived hostile counts")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times, _series_float(timeline_rows, "nearest_enemy_bullseye_km"), label="Enemy to bullseye (km)", linewidth=1.4)
    axes[3].plot(times, _series_float(timeline_rows, "nearest_enemy_to_friendly_km"), label="Enemy to formation (km)", linewidth=1.4)
    axes[3].set_title("Geometric compression")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)
    _save_figure(fig, figures_dir / "fig01_risk_picture_geometry.png")

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    axes[0].plot(times, _series_int(timeline_rows, "awacs_track_count"), label="AWACS", linewidth=1.7)
    axes[0].plot(times, _series_int(timeline_rows, "radar_track_count"), label="Radar unique", linewidth=1.7)
    axes[0].plot(times, _series_int(timeline_rows, "stable_ready_count"), label="Stable ready", linewidth=1.7)
    axes[0].plot(times, _series_int(timeline_rows, "stable_tracking_target_count"), label="Stable targets", linewidth=1.3)
    axes[0].set_title("Tracking quality buildup")
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].grid(True, alpha=0.3)

    _plot_categorical(axes[1], times, [row.get("detect_mode", "") for row in timeline_rows], "Detection mode")
    axes[2].plot(times, _series_float(timeline_rows, "scan_coverage_total_deg", 0.0), label="Coverage total (deg)", linewidth=1.6)
    axes[2].plot(times, _series_float(timeline_rows, "scan_overlap_deg", 0.0), label="Coverage overlap (deg)", linewidth=1.6)
    axes[2].plot(times, _series_float(timeline_rows, "confidence_radius_km", 0.0), label="Confidence radius (km)", linewidth=1.2)
    axes[2].set_title("Detection geometry")
    axes[2].legend(ncol=2, fontsize=9)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times, _series_int(timeline_rows, "radar_track_count"), label="Radar unique tracks", linewidth=1.6)
    axes[3].plot(times, _series_int(timeline_rows, "scan_target_count"), label="Unique scan targets", linewidth=1.2)
    axes[3].plot(times, _series_int(timeline_rows, "intent_contact_count"), label="Intent contacts", linewidth=1.6)
    axes[3].plot(times, _series_int(timeline_rows, "intent_ready_count"), label="Intent model-ready contacts", linewidth=1.3)
    axes[3].set_title("Detection-to-intent pipeline load")
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(times, _series_int(timeline_rows, "scan_assignment_count"), label="Scan assignments", linewidth=1.6)
    axes[4].plot(times, _series_int(timeline_rows, "assignment_target_count"), label="Tactic-assigned targets", linewidth=1.6)
    axes[4].plot(times, _series_int(timeline_rows, "active_relay_flag"), label="Active relay", linewidth=1.2)
    axes[4].set_title("Assignment and relay activation")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(fontsize=9)
    axes[4].grid(True, alpha=0.3)
    _save_figure(fig, figures_dir / "fig02_detection_tracking_pipeline.png")

    if summary:
        first_detect_map = dict(summary.get("radar_first_detect_time_by_target_s", {}) or {})
        detect_times = sorted(
            float(value)
            for value in first_detect_map.values()
            if value not in (None, "") and np.isfinite(float(value))
        )
        cumulative_detected: List[int] = []
        detect_index = 0
        current_count = 0
        for time_s in times:
            while detect_index < len(detect_times) and float(time_s) >= float(detect_times[detect_index]) - 1e-9:
                current_count += 1
                detect_index += 1
            cumulative_detected.append(current_count)

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        axes[0].step(times, cumulative_detected, where="post", label="Cumulative first-detected targets", linewidth=1.8)
        axes[0].plot(times, _series_int(timeline_rows, "radar_track_count"), label="Radar currently tracked targets", linewidth=1.4)
        axes[0].set_title("Cooperative detection closure")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, _series_int(timeline_rows, "stable_ready_count"), label="Stable ready", linewidth=1.7)
        axes[1].plot(times, _series_int(timeline_rows, "stable_tracking_target_count"), label="Stable tracked targets", linewidth=1.7)
        axes[1].plot(times, _series_int(timeline_rows, "assignment_target_count"), label="Assigned targets", linewidth=1.2)
        axes[1].set_title("Cooperative tracking buildup")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        axes[2].step(
            times,
            _guidance_metric_series(timeline_rows, base_key="gate_pass", prefer_unique=True),
            where="post",
            label="Gate pass (unique)",
            linewidth=1.7,
        )
        axes[2].step(
            times,
            _guidance_metric_series(timeline_rows, base_key="relay_success", prefer_unique=True),
            where="post",
            label="Relay success (unique)",
            linewidth=1.7,
        )
        axes[2].plot(times, _series_int(timeline_rows, "active_guided_missile_peak"), label="Active guided peak", linewidth=1.3)
        axes[2].set_title("Gate release and relay-guidance closure (deduplicated)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)
        _save_figure(fig, figures_dir / "fig02b_cooperative_chain_timeline.png")

    fig, axes = plt.subplots(8, 1, figsize=(15, 22), sharex=True)
    _plot_categorical(axes[0], times, [row.get("cap_state", "") for row in timeline_rows], "CAP state")
    _plot_categorical(axes[1], times, [row.get("enemy_script_phase_name", "") for row in timeline_rows], "Enemy scene phase")
    axes[2].step(
        times,
        _series_int(timeline_rows, "enemy_script_awacs_available"),
        where="post",
        label="AWACS available",
        linewidth=1.7,
    )
    enemy_awacs_ax = axes[2].twinx()
    enemy_awacs_ax.plot(
        times,
        _series_float(timeline_rows, "enemy_script_reference_distance_km"),
        label="Reference distance (km)",
        linewidth=1.2,
        color="#C0504D",
    )
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_title("Enemy information condition and trigger distance")
    lines_a, labels_a = axes[2].get_legend_handles_labels()
    lines_b, labels_b = enemy_awacs_ax.get_legend_handles_labels()
    axes[2].legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper right")
    axes[2].grid(True, alpha=0.3)
    _plot_level_timeline(axes[3], times, [row.get("mission_threat_level", "") for row in timeline_rows], "Mission threat level", threat_levels)
    _plot_categorical(axes[4], times, [row.get("left_tactic", "") for row in timeline_rows], "Left pair tactic")
    _plot_categorical(axes[5], times, [row.get("right_tactic", "") for row in timeline_rows], "Right pair tactic")
    _plot_categorical(
        axes[6],
        times,
        [_normalize_phase_label(row.get("left_phase", ""), default="") for row in timeline_rows],
        "Left pair phase",
        preferred_order=CONTROL_PHASE_ORDER,
    )
    _plot_categorical(
        axes[7],
        times,
        [_normalize_phase_label(row.get("right_phase", ""), default="") for row in timeline_rows],
        "Right pair phase",
        preferred_order=CONTROL_PHASE_ORDER,
    )
    axes[7].set_xlabel("Time (s)")
    _save_figure(fig, figures_dir / "fig03_state_tactic_phase_timeline.png")

    if summary:
        node_rows = list(summary.get("control_distance_nodes", []) or [])
        route_nodes = {
            "left": [row for row in node_rows if str(row.get("route", "") or "") == "left"],
            "right": [row for row in node_rows if str(row.get("route", "") or "") == "right"],
        }
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        for axis, route_label, color in (
            (axes[0], "left", "#2F6B9A"),
            (axes[1], "right", "#D95F02"),
        ):
            axis.plot(times, _series_float(timeline_rows, "nearest_enemy_to_friendly_km"), color="#4C78A8", linewidth=1.5)
            for row in route_nodes.get(route_label, []):
                time_s = _safe_float(row.get("first_time_s"), float("nan"))
                distance_km = _safe_float(row.get("distance_km"), float("nan"))
                if not np.isfinite(time_s) or not np.isfinite(distance_km):
                    continue
                axis.scatter([time_s], [distance_km], color=color, s=32, zorder=4)
                axis.axvline(time_s, color=color, alpha=0.18, linewidth=0.9)
                axis.annotate(
                    str(row.get("phase", "") or ""),
                    xy=(time_s, distance_km),
                    xytext=(4, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )
            axis.set_title(f"{route_label.capitalize()} pair control-distance nodes")
            axis.set_ylabel("Nearest enemy-to-friendly (km)")
            axis.grid(True, alpha=0.3)
        axes[1].set_xlabel("Time (s)")
        _save_figure(fig, figures_dir / "fig03b_control_distance_timeline.png")

    fig, axes = plt.subplots(6, 1, figsize=(15, 20), sharex=True)
    _plot_level_timeline(axes[0], times, [row.get("intent_threat", "") for row in timeline_rows], "Highest intent threat", intent_levels)
    _plot_categorical(
        axes[1],
        times,
        [row.get("intent_class", "") for row in timeline_rows],
        "Highest classified intent",
        preferred_order=INTENT_CLASS_ORDER,
    )
    axes[2].step(times, _series_int(timeline_rows, "intent_contact_count"), where="post", label="Intent contacts", linewidth=1.7)
    axes[2].step(times, _series_int(timeline_rows, "intent_classified_count"), where="post", label="Classified targets", linewidth=1.7)
    axes[2].step(times, _series_int(timeline_rows, "intent_ready_count"), where="post", label="Ready classified targets", linewidth=1.7)
    axes[2].step(times, _series_int(timeline_rows, "intent_high_count"), where="post", label="High-threat targets", linewidth=1.5)
    axes[2].step(
        times,
        _series_int(timeline_rows, "intent_critical_count"),
        where="post",
        label="Critical-threat targets",
        linewidth=1.5,
    )
    axes[2].set_title("Intent-analysis coverage and high-threat target count")
    axes[2].legend(ncol=2, fontsize=9)
    axes[2].grid(True, alpha=0.3)

    axes[3].step(times, _series_int(timeline_rows, "intent_attack_count"), where="post", label="Attack", linewidth=1.7)
    axes[3].step(times, _series_int(timeline_rows, "intent_defense_count"), where="post", label="Defense", linewidth=1.7)
    axes[3].step(times, _series_int(timeline_rows, "intent_retreat_count"), where="post", label="Retreat", linewidth=1.7)
    axes[3].step(times, _series_int(timeline_rows, "intent_recon_count"), where="post", label="Reconnaissance", linewidth=1.3)
    axes[3].set_title("Intent-type target count")
    axes[3].legend(ncol=4, fontsize=9)
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(times, _series_float(timeline_rows, "intent_confidence", 0.0), label="Highest confidence", linewidth=1.7)
    axes[4].plot(times, _series_int(timeline_rows, "intent_model_ready"), label="Highest ready", linewidth=1.2)
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_title("Highest-threat confidence and readiness")
    axes[4].legend(fontsize=9)
    axes[4].grid(True, alpha=0.3)

    _plot_categorical(
        axes[5],
        times,
        [row.get("enemy_left_intent_class", "") for row in timeline_rows],
        "Left enemy pair dominant intent",
        preferred_order=INTENT_CLASS_ORDER,
    )
    group_ax = axes[5].twinx()
    group_ax.step(
        times,
        _series_int(timeline_rows, "enemy_left_intent_ready_count"),
        where="post",
        label="Left ready count",
        linewidth=1.1,
        color="#C0504D",
    )
    group_ax.step(
        times,
        _series_int(timeline_rows, "enemy_right_intent_ready_count"),
        where="post",
        label="Right ready count",
        linewidth=1.1,
        color="#4F81BD",
    )
    group_ax.set_ylim(-0.1, 2.1)
    group_ax.legend(fontsize=8, loc="upper right")
    axes[5].set_xlabel("Time (s)")
    _save_figure(fig, figures_dir / "fig04_intent_analysis_trace.png")

    fig, axes = plt.subplots(6, 1, figsize=(15, 20), sharex=True)
    _plot_categorical(axes[0], times, [row.get("enemy_left_intent_class", "") for row in timeline_rows], "Left enemy pair dominant intent", preferred_order=INTENT_CLASS_ORDER)
    _plot_categorical(axes[1], times, [row.get("enemy_right_intent_class", "") for row in timeline_rows], "Right enemy pair dominant intent", preferred_order=INTENT_CLASS_ORDER)
    _plot_categorical(axes[2], times, [row.get("B0100_intent_class", "") for row in timeline_rows], "B0100 intent", preferred_order=INTENT_CLASS_ORDER)
    _plot_categorical(axes[3], times, [row.get("B0200_intent_class", "") for row in timeline_rows], "B0200 intent", preferred_order=INTENT_CLASS_ORDER)
    _plot_categorical(axes[4], times, [row.get("B0300_intent_class", "") for row in timeline_rows], "B0300 intent", preferred_order=INTENT_CLASS_ORDER)
    _plot_categorical(axes[5], times, [row.get("B0400_intent_class", "") for row in timeline_rows], "B0400 intent", preferred_order=INTENT_CLASS_ORDER)
    axes[5].set_xlabel("Time (s)")
    _save_figure(fig, figures_dir / "fig04b_intent_target_timeline.png")

    fig, axes = plt.subplots(3, 1, figsize=(15, 13.5), sharex=True)
    _plot_multi_level_lines(
        axes[0],
        times,
        {
            "B0100": [row.get("B0100_intent_threat", "") for row in timeline_rows],
            "B0200": [row.get("B0200_intent_threat", "") for row in timeline_rows],
        },
        "Left enemy pair threat timeline",
        intent_levels,
    )
    _plot_multi_level_lines(
        axes[1],
        times,
        {
            "B0300": [row.get("B0300_intent_threat", "") for row in timeline_rows],
            "B0400": [row.get("B0400_intent_threat", "") for row in timeline_rows],
        },
        "Right enemy pair threat timeline",
        intent_levels,
    )
    axes[2].plot(times, _series_float(timeline_rows, "B0100_intent_confidence", 0.0), label="B0100", linewidth=1.4)
    axes[2].plot(times, _series_float(timeline_rows, "B0200_intent_confidence", 0.0), label="B0200", linewidth=1.4)
    axes[2].plot(times, _series_float(timeline_rows, "B0300_intent_confidence", 0.0), label="B0300", linewidth=1.4)
    axes[2].plot(times, _series_float(timeline_rows, "B0400_intent_confidence", 0.0), label="B0400", linewidth=1.4)
    axes[2].set_title("Per-target intent confidence")
    axes[2].legend(ncol=2, fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Time (s)")
    _save_figure(fig, figures_dir / "fig04c_intent_target_threat.png")

    fig, axes = plt.subplots(5, 1, figsize=(15, 17), sharex=True)
    gate_pass = _guidance_metric_series(timeline_rows, base_key="gate_pass", prefer_unique=True)
    gate_block = _guidance_metric_series(timeline_rows, base_key="gate_block", prefer_unique=True)
    relay_attempt = _guidance_metric_series(timeline_rows, base_key="relay_attempt", prefer_unique=True)
    relay_success = _guidance_metric_series(timeline_rows, base_key="relay_success", prefer_unique=True)
    gate_total = [p + b for p, b in zip(gate_pass, gate_block)]
    axes[0].step(times, gate_pass, where="post", label="Pass (unique)", linewidth=1.7)
    axes[0].step(times, gate_block, where="post", label="Block (unique)", linewidth=1.7)
    axes[0].plot(times, _running_ratio_series(gate_pass, gate_total), label="Pass rate (unique)", linewidth=1.2)
    axes[0].set_title("Prelaunch gate evolution (deduplicated)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].step(times, relay_attempt, where="post", label="Attempt (unique)", linewidth=1.7)
    axes[1].step(times, relay_success, where="post", label="Success (unique)", linewidth=1.7)
    axes[1].plot(times, _running_ratio_series(relay_success, relay_attempt), label="Success rate (unique)", linewidth=1.2)
    axes[1].set_title("Relay guidance evolution (deduplicated)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].step(times, _series_int(timeline_rows, "missile_launch_count"), where="post", label="Launches", linewidth=1.7)
    axes[2].step(times, _series_int(timeline_rows, "missile_outcome_count"), where="post", label="Outcomes", linewidth=1.7)
    axes[2].plot(times, _series_int(timeline_rows, "missile_state_guiding"), label="Guiding", linewidth=1.2)
    axes[2].plot(times, _series_int(timeline_rows, "missile_state_terminal"), label="Terminal", linewidth=1.2)
    axes[2].set_title("Missile-chain state counts")
    axes[2].legend(ncol=2, fontsize=9)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times, _series_int(timeline_rows, "active_guided_missile_peak"), label="Active guided peak", linewidth=1.6)
    axes[3].plot(times, _series_int(timeline_rows, "friendly_missiles_left"), label="Friendly missiles left", linewidth=1.6)
    axes[3].set_title("Guidance load and remaining inventory")
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(times, _series_int(timeline_rows, "enemy_kill_count"), label="Enemy kills", linewidth=1.7)
    axes[4].plot(times, _series_int(timeline_rows, "friendly_loss_count"), label="Friendly losses", linewidth=1.7)
    axes[4].set_title("Accumulated battle outcome")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(fontsize=9)
    axes[4].grid(True, alpha=0.3)
    _save_figure(fig, figures_dir / "fig05_engagement_chain.png")

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    axes[0].step(times, _series_int(timeline_rows, "friendly_alive"), where="post", label="Friendly alive", linewidth=1.8)
    axes[0].step(times, _series_int(timeline_rows, "enemy_alive"), where="post", label="Enemy alive", linewidth=1.8)
    axes[0].set_title("Alive aircraft count")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, _series_float(timeline_rows, "nearest_enemy_bullseye_km"), label="To bullseye", linewidth=1.6)
    axes[1].plot(times, _series_float(timeline_rows, "nearest_enemy_to_friendly_km"), label="To formation", linewidth=1.6)
    axes[1].set_title("Nearest-threat range")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    _plot_categorical(axes[2], times, [row.get("left_target", "") for row in timeline_rows], "Left pair target")
    _plot_categorical(axes[3], times, [row.get("right_target", "") for row in timeline_rows], "Right pair target")
    axes[3].set_xlabel("Time (s)")
    _save_figure(fig, figures_dir / "fig06_survival_distance_targeting.png")

    fig, ax = plt.subplots(figsize=(15, 5.5))
    enemy_ids = ["B0100", "B0200", "B0300", "B0400"]
    zone_order = {"OUTSIDE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    matrix = np.asarray(
        [
            [zone_order.get(_norm_label(zone, default="OUTSIDE").upper(), 0) for zone in _parse_zone_series(timeline_rows, enemy_id)]
            for enemy_id in enemy_ids
        ],
        dtype=float,
    )
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(list(range(len(enemy_ids))))
    ax.set_yticklabels(enemy_ids)
    ax.set_xticks(np.linspace(0, max(0, len(times) - 1), min(8, len(times)), dtype=int))
    ax.set_xticklabels([f"{times[idx]:.0f}" for idx in np.linspace(0, max(0, len(times) - 1), min(8, len(times)), dtype=int)])
    ax.set_xlabel("Time (s)")
    ax.set_title("Enemy-by-enemy truth risk-zone timeline")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["OUTSIDE", "LOW", "MEDIUM", "HIGH"])
    _save_figure(fig, figures_dir / "fig07_enemy_zone_heatmap.png")

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    _plot_categorical(axes[0], times, [row.get("A0100_scan_target", "") for row in timeline_rows], "A0100 scan target")
    _plot_categorical(axes[1], times, [row.get("A0200_scan_target", "") for row in timeline_rows], "A0200 scan target")
    _plot_categorical(axes[2], times, [row.get("A0300_scan_target", "") for row in timeline_rows], "A0300 scan target")
    _plot_categorical(axes[3], times, [row.get("A0400_scan_target", "") for row in timeline_rows], "A0400 scan target")
    axes[3].set_xlabel("Time (s)")
    _save_figure(fig, figures_dir / "fig08_scan_target_timeline.png")

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(times, _series_float(timeline_rows, "friendly_min_altitude_m"), label="Friendly min alt", linewidth=1.7)
    axes[0].plot(times, _series_float(timeline_rows, "enemy_min_altitude_m"), label="Enemy min alt", linewidth=1.7)
    axes[0].plot(times, _series_float(timeline_rows, "friendly_mean_altitude_m"), label="Friendly mean alt", linewidth=1.1, alpha=0.85)
    axes[0].plot(times, _series_float(timeline_rows, "enemy_mean_altitude_m"), label="Enemy mean alt", linewidth=1.1, alpha=0.85)
    axes[0].set_title("Altitude envelope")
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, _series_float(timeline_rows, "friendly_min_speed_mps"), label="Friendly min speed", linewidth=1.7)
    axes[1].plot(times, _series_float(timeline_rows, "enemy_min_speed_mps"), label="Enemy min speed", linewidth=1.7)
    axes[1].plot(times, _series_float(timeline_rows, "friendly_mean_speed_mps"), label="Friendly mean speed", linewidth=1.1, alpha=0.85)
    axes[1].plot(times, _series_float(timeline_rows, "enemy_mean_speed_mps"), label="Enemy mean speed", linewidth=1.1, alpha=0.85)
    axes[1].set_title("Speed envelope")
    axes[1].legend(ncol=2, fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        times,
        _series_float(timeline_rows, "friendly_max_descent_rate_mps"),
        label="Friendly max descent",
        linewidth=1.7,
    )
    axes[2].plot(
        times,
        _series_float(timeline_rows, "enemy_max_descent_rate_mps"),
        label="Enemy max descent",
        linewidth=1.7,
    )
    axes[2].set_title("Maximum descent rate envelope")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time (s)")
    _save_figure(fig, figures_dir / "fig09_flight_safety_envelope.png")


def _collect_picture_snapshot(patrol_task: CAPTask, current_time: float) -> Dict[str, object]:
    snapshot = {
        "picture_low_count": 0,
        "picture_medium_count": 0,
        "picture_high_count": 0,
        "picture_total_count": 0,
        "search_picture_total_count": 0,
    }
    if not hasattr(patrol_task, "_get_picture_zone_count") or not hasattr(patrol_task, "_get_picture_threats"):
        return snapshot
    try:
        snapshot["picture_low_count"] = int(
            patrol_task._get_picture_zone_count(PictureRiskZone.LOW, current_time=current_time, purpose="decision")
        )
        snapshot["picture_medium_count"] = int(
            patrol_task._get_picture_zone_count(PictureRiskZone.MEDIUM, current_time=current_time, purpose="decision")
        )
        snapshot["picture_high_count"] = int(
            patrol_task._get_picture_zone_count(PictureRiskZone.HIGH, current_time=current_time, purpose="decision")
        )
        snapshot["picture_total_count"] = len(
            patrol_task._get_picture_threats(current_time=current_time, purpose="decision") or []
        )
        snapshot["search_picture_total_count"] = len(
            patrol_task._get_picture_threats(current_time=current_time, purpose="search") or []
        )
    except Exception:
        return snapshot
    return snapshot


def _collect_intent_snapshot(patrol_task: CAPTask, env, current_time: float) -> Dict[str, object]:
    snapshot = {
        "intent_target": "",
        "intent_name": "",
        "intent_class": "",
        "intent_threat": "",
        "intent_confidence": 0.0,
        "intent_model_ready": 0,
        "intent_contact_count": 0,
        "intent_classified_count": 0,
        "intent_unclassified_count": 0,
        "intent_ready_count": 0,
        "intent_top2_json": "[]",
        "intent_top4_json": "[]",
        "intent_all_json": "{}",
        "intent_present_targets_json": "[]",
        "intent_attack_count": 0,
        "intent_defense_count": 0,
        "intent_retreat_count": 0,
        "intent_recon_count": 0,
        "intent_high_count": 0,
        "intent_critical_count": 0,
    }
    for enemy_id in ENEMY_AGENT_IDS:
        blank = _intent_blank_entry(enemy_id)
        for suffix, value in blank.items():
            if suffix == "track_id":
                continue
            snapshot[f"{enemy_id}_{suffix}"] = value
    for group_prefix in ENEMY_INTENT_GROUPS:
        snapshot[f"{group_prefix}_intent_target"] = ""
        snapshot[f"{group_prefix}_intent_name"] = ""
        snapshot[f"{group_prefix}_intent_class"] = ""
        snapshot[f"{group_prefix}_intent_class_code"] = ""
        snapshot[f"{group_prefix}_intent_type"] = ""
        snapshot[f"{group_prefix}_intent_threat"] = ""
        snapshot[f"{group_prefix}_intent_confidence"] = 0.0
        snapshot[f"{group_prefix}_intent_contact_count"] = 0
        snapshot[f"{group_prefix}_intent_classified_count"] = 0
        snapshot[f"{group_prefix}_intent_unclassified_count"] = 0
        snapshot[f"{group_prefix}_intent_ready_count"] = 0
        snapshot[f"{group_prefix}_intent_attack_count"] = 0
        snapshot[f"{group_prefix}_intent_defense_count"] = 0
        snapshot[f"{group_prefix}_intent_retreat_count"] = 0
        snapshot[f"{group_prefix}_intent_recon_count"] = 0
        snapshot[f"{group_prefix}_intent_high_count"] = 0
        snapshot[f"{group_prefix}_intent_critical_count"] = 0
    intent_adapter = getattr(patrol_task, "intent_adapter", None)
    if intent_adapter is None:
        return snapshot

    # This collector must be passive. Calling IntentAdapter._build_analysis() or
    # analyze_intent() here advances the bvr_intent_new buffers a second time per
    # step and changes the native run_cap_simulation engagement rhythm.
    analyses = list((getattr(intent_adapter, "_analyses", {}) or {}).values())
    if not analyses:
        return snapshot

    ready_fn = getattr(intent_adapter, "_is_model_ready", None)
    ready_count = 0
    ranked: List[Any] = []
    track_entries: Dict[str, Dict[str, object]] = {enemy_id: _intent_blank_entry(enemy_id) for enemy_id in ENEMY_AGENT_IDS}
    class_counter: Counter[str] = Counter()
    classified_count = 0
    high_count = 0
    critical_count = 0
    for analysis in analyses:
        ready = bool(callable(ready_fn) and ready_fn(getattr(analysis, "model_window", None)))
        entry = _intent_entry_from_analysis(analysis, ready=ready)
        track_id = str(entry.get("track_id", "") or "")
        if track_id in track_entries:
            track_entries[track_id] = entry
        intent_class = str(entry.get("intent_class", "") or "").strip()
        intent_code = str(entry.get("intent_class_code", "") or "").lower()
        if intent_class:
            classified_count += 1
        if ready and intent_class:
            ready_count += 1
        if intent_code:
            class_counter[intent_code] += 1
        threat_text = str(entry.get("intent_threat", "") or "").lower()
        if threat_text in {"high", "critical"}:
            high_count += 1
        if threat_text == "critical":
            critical_count += 1
        ranked.append(
            (
                1 if intent_class else 0,
                THREAT_PRIORITY.get(threat_text, 0),
                float(entry.get("intent_confidence", 0.0) or 0.0),
                -float(entry.get("intent_distance_km", 0.0) or 0.0),
                ready,
                analysis,
                entry,
            )
        )
    ranked.sort(reverse=True)
    highest_entry = dict(ranked[0][-1]) if ranked else _intent_blank_entry()
    top_ranked_entries = [dict(item[-1]) for item in ranked]
    top2 = []
    for entry in top_ranked_entries[:2]:
        top2.append(
            {
                "track_id": str(entry.get("track_id", "") or ""),
                "intent_class": str(entry.get("intent_class", "") or ""),
                "intent_code": str(entry.get("intent_class_code", "") or ""),
                "raw_intent": str(entry.get("intent_type", "") or ""),
                "raw_label": str(entry.get("intent_raw_label", "") or ""),
                "threat": str(entry.get("intent_threat", "") or ""),
                "confidence": float(entry.get("intent_confidence", 0.0) or 0.0),
                "distance_km": float(entry.get("intent_distance_km", 0.0) or 0.0),
                "model_ready": bool(entry.get("intent_model_ready", 0)),
            }
        )
    group_entries = {
        group_prefix: _intent_group_entry(group_prefix, track_ids, track_entries)
        for group_prefix, track_ids in ENEMY_INTENT_GROUPS.items()
    }
    snapshot.update(
        {
            "intent_target": str(highest_entry.get("track_id", "") or ""),
            "intent_name": str(highest_entry.get("intent_name", "") or ""),
            "intent_class": str(highest_entry.get("intent_class", "") or ""),
            "intent_threat": str(highest_entry.get("intent_threat", "") or ""),
            "intent_confidence": _safe_float(highest_entry.get("intent_confidence", 0.0), 0.0),
            "intent_model_ready": int(highest_entry.get("intent_model_ready", 0) or 0),
            "intent_contact_count": int(len(analyses)),
            "intent_classified_count": int(classified_count),
            "intent_unclassified_count": int(max(0, len(analyses) - classified_count)),
            "intent_ready_count": int(ready_count),
            "intent_top2_json": json.dumps(top2, ensure_ascii=False, sort_keys=True),
            "intent_top4_json": json.dumps(top_ranked_entries[:4], ensure_ascii=False, sort_keys=True),
            "intent_all_json": json.dumps(track_entries, ensure_ascii=False, sort_keys=True),
            "intent_present_targets_json": json.dumps(
                sorted(track_id for track_id, entry in track_entries.items() if int(entry.get("intent_present", 0) or 0) > 0),
                ensure_ascii=False,
            ),
            "intent_attack_count": int(class_counter.get("attack", 0)),
            "intent_defense_count": int(class_counter.get("defense", 0)),
            "intent_retreat_count": int(class_counter.get("retreat", 0)),
            "intent_recon_count": int(class_counter.get("reconnaissance", 0)),
            "intent_high_count": int(high_count),
            "intent_critical_count": int(critical_count),
        }
    )
    for enemy_id, entry in track_entries.items():
        for suffix, value in entry.items():
            if suffix == "track_id":
                continue
            snapshot[f"{enemy_id}_{suffix}"] = value
    for group_prefix, entry in group_entries.items():
        snapshot[f"{group_prefix}_intent_target"] = entry["dominant_target"]
        snapshot[f"{group_prefix}_intent_name"] = entry["dominant_name"]
        snapshot[f"{group_prefix}_intent_class"] = entry["dominant_class"]
        snapshot[f"{group_prefix}_intent_class_code"] = entry["dominant_class_code"]
        snapshot[f"{group_prefix}_intent_type"] = entry["dominant_type"]
        snapshot[f"{group_prefix}_intent_threat"] = entry["dominant_threat"]
        snapshot[f"{group_prefix}_intent_confidence"] = entry["dominant_confidence"]
        snapshot[f"{group_prefix}_intent_contact_count"] = entry["contact_count"]
        snapshot[f"{group_prefix}_intent_classified_count"] = entry["classified_count"]
        snapshot[f"{group_prefix}_intent_unclassified_count"] = entry["unclassified_count"]
        snapshot[f"{group_prefix}_intent_ready_count"] = entry["ready_count"]
        snapshot[f"{group_prefix}_intent_attack_count"] = entry["attack_count"]
        snapshot[f"{group_prefix}_intent_defense_count"] = entry["defense_count"]
        snapshot[f"{group_prefix}_intent_retreat_count"] = entry["retreat_count"]
        snapshot[f"{group_prefix}_intent_recon_count"] = entry["recon_count"]
        snapshot[f"{group_prefix}_intent_high_count"] = entry["high_count"]
        snapshot[f"{group_prefix}_intent_critical_count"] = entry["critical_count"]
    return snapshot


def _collect_scan_snapshot(patrol_task: CAPTask) -> Dict[str, object]:
    snapshot = {
        "scan_assignment_count": 0,
        "scan_target_count": 0,
        "scan_targets_json": "[]",
        "scan_coverage_total_deg": 0.0,
        "scan_overlap_deg": 0.0,
        "confidence_radius_km": 0.0,
    }
    for aid in ("A0100", "A0200", "A0300", "A0400"):
        snapshot[f"{aid}_scan_target"] = ""
        snapshot[f"{aid}_scan_center_deg"] = 0.0
        snapshot[f"{aid}_scan_range_deg"] = 0.0

    coop_detection = getattr(patrol_task, "coop_detection", None)
    if coop_detection is None:
        return snapshot

    targets = set()
    if hasattr(coop_detection, "get_coverage_report"):
        try:
            coverage = coop_detection.get_coverage_report() or {}
            snapshot["scan_coverage_total_deg"] = _safe_float(coverage.get("total_coverage"), 0.0)
            snapshot["scan_overlap_deg"] = _safe_float(coverage.get("overlap"), 0.0)
        except Exception:
            pass
    snapshot["confidence_radius_km"] = _safe_float(getattr(coop_detection, "confidence_radius", 0.0), 0.0)

    for aid in ("A0100", "A0200", "A0300", "A0400"):
        try:
            assignment = coop_detection.get_assignment(aid) if hasattr(coop_detection, "get_assignment") else None
        except Exception:
            assignment = None
        if assignment is None:
            continue
        snapshot["scan_assignment_count"] = int(snapshot["scan_assignment_count"]) + 1
        target_id = str(getattr(assignment, "target_id", "") or "")
        snapshot[f"{aid}_scan_target"] = target_id
        snapshot[f"{aid}_scan_center_deg"] = _safe_float(getattr(assignment, "azimuth_center", 0.0), 0.0)
        snapshot[f"{aid}_scan_range_deg"] = _safe_float(getattr(assignment, "azimuth_range", 0.0), 0.0)
        if target_id:
            targets.add(target_id)
    snapshot["scan_target_count"] = int(len(targets))
    snapshot["scan_targets_json"] = json.dumps(sorted(targets), ensure_ascii=False)
    return snapshot


def _collect_pair_assignment_snapshot(patrol_task: CAPTask) -> Dict[str, object]:
    snapshot = {
        "left_tactic": "UNKNOWN",
        "left_target": "",
        "left_shooter": "",
        "left_support": "",
        "left_stage_reason": "",
        "left_tactic_reason": "",
        "left_maneuver_reason": "",
        "left_parameter_reason": "",
        "left_decision_snapshot_json": "{}",
        "right_tactic": "UNKNOWN",
        "right_target": "",
        "right_shooter": "",
        "right_support": "",
        "right_stage_reason": "",
        "right_tactic_reason": "",
        "right_maneuver_reason": "",
        "right_parameter_reason": "",
        "right_decision_snapshot_json": "{}",
        "assignment_target_count": 0,
        "assignment_targets_json": "[]",
    }

    assignments = getattr(patrol_task, "_tactic_assignments_by_agent", {}) or {}
    unique_targets = sorted(
        {
            str(getattr(assignment, "target_id", "") or "")
            for assignment in assignments.values()
            if str(getattr(assignment, "target_id", "") or "")
        }
    )
    snapshot["assignment_target_count"] = int(len(unique_targets))
    snapshot["assignment_targets_json"] = json.dumps(unique_targets, ensure_ascii=False)

    def _resolve_pair(agent_id: str) -> Any:
        if hasattr(patrol_task, "_get_pair_tactic_assignment"):
            try:
                pair_assignment = patrol_task._get_pair_tactic_assignment(agent_id)
                if pair_assignment is not None:
                    return pair_assignment
            except Exception:
                pass
        return assignments.get(agent_id)

    left_assignment = _resolve_pair("A0100")
    right_assignment = _resolve_pair("A0300")
    if left_assignment is not None:
        snapshot["left_tactic"] = str(getattr(getattr(left_assignment, "tactic", None), "value", getattr(left_assignment, "tactic", "UNKNOWN")) or "UNKNOWN")
        snapshot["left_target"] = str(getattr(left_assignment, "target_id", "") or "")
        snapshot["left_shooter"] = str(getattr(left_assignment, "shooter_id", "") or "")
        snapshot["left_support"] = str(getattr(left_assignment, "support_id", "") or "")
        snapshot["left_stage_reason"] = str(getattr(left_assignment, "stage_reason", "") or "")
        snapshot["left_tactic_reason"] = str(getattr(left_assignment, "tactic_reason", "") or "")
        snapshot["left_maneuver_reason"] = str(getattr(left_assignment, "maneuver_reason", "") or "")
        snapshot["left_parameter_reason"] = str(getattr(left_assignment, "parameter_reason", "") or "")
        snapshot["left_decision_snapshot_json"] = json.dumps(
            getattr(left_assignment, "decision_snapshot", None) or {},
            ensure_ascii=False,
            sort_keys=True,
        )
    if right_assignment is not None:
        snapshot["right_tactic"] = str(getattr(getattr(right_assignment, "tactic", None), "value", getattr(right_assignment, "tactic", "UNKNOWN")) or "UNKNOWN")
        snapshot["right_target"] = str(getattr(right_assignment, "target_id", "") or "")
        snapshot["right_shooter"] = str(getattr(right_assignment, "shooter_id", "") or "")
        snapshot["right_support"] = str(getattr(right_assignment, "support_id", "") or "")
        snapshot["right_stage_reason"] = str(getattr(right_assignment, "stage_reason", "") or "")
        snapshot["right_tactic_reason"] = str(getattr(right_assignment, "tactic_reason", "") or "")
        snapshot["right_maneuver_reason"] = str(getattr(right_assignment, "maneuver_reason", "") or "")
        snapshot["right_parameter_reason"] = str(getattr(right_assignment, "parameter_reason", "") or "")
        snapshot["right_decision_snapshot_json"] = json.dumps(
            getattr(right_assignment, "decision_snapshot", None) or {},
            ensure_ascii=False,
            sort_keys=True,
        )
    return snapshot


def _collect_enemy_script_snapshot(
    spec: Ch6ScenarioSpec,
    patrol_task: CAPTask,
    env,
    current_time: float,
    current_distance: float,
) -> Dict[str, object]:
    snapshot = {
        "enemy_script_family": "",
        "enemy_script_phase_id": -1,
        "enemy_script_phase_name": "",
        "enemy_script_phase_trigger": "",
        "enemy_script_phase_objective": "",
        "enemy_script_phase_elapsed_s": 0.0,
        "enemy_script_reference_distance_km": current_distance,
        "enemy_script_next_phase_id": -1,
        "enemy_script_next_phase_name": "",
        "enemy_script_next_phase_trigger": "",
        "enemy_script_awacs_available": 1,
        "enemy_script_awacs_state": "AVAILABLE",
        "enemy_script_phase_offsets_json": "[]",
        "enemy_script_phase_altitudes_json": "[]",
        "enemy_left_group_phase": "",
        "enemy_right_group_phase": "",
        "enemy_left_wave_index": 0,
        "enemy_right_wave_index": 0,
        "enemy_left_pressure_tag": "",
        "enemy_right_pressure_tag": "",
    }
    scenario = getattr(spec, "scenario", None)
    if scenario is None:
        return snapshot

    enemy_adapter = getattr(patrol_task, "enemy_adapter", None)
    group_snapshots: Dict[str, Dict[str, object]] = {}
    if enemy_adapter is not None and hasattr(enemy_adapter, "get_group_snapshot"):
        for label, agent_id in (("PAIR_LEFT", "B0100"), ("PAIR_RIGHT", "B0300")):
            try:
                group_snapshot = enemy_adapter.get_group_snapshot(agent_id) or {}
            except Exception:
                group_snapshot = {}
            group_snapshots[label] = dict(group_snapshot)
        left_snapshot = group_snapshots.get("PAIR_LEFT", {})
        right_snapshot = group_snapshots.get("PAIR_RIGHT", {})
        snapshot.update(
            {
                "enemy_left_group_phase": str(left_snapshot.get("phase", "") or ""),
                "enemy_right_group_phase": str(right_snapshot.get("phase", "") or ""),
                "enemy_left_wave_index": _safe_int(left_snapshot.get("wave_index", 0), 0),
                "enemy_right_wave_index": _safe_int(right_snapshot.get("wave_index", 0), 0),
                "enemy_left_pressure_tag": str(left_snapshot.get("pressure_tag", "") or ""),
                "enemy_right_pressure_tag": str(right_snapshot.get("pressure_tag", "") or ""),
            }
        )

    if spec.enemy_control_mode != "scripted" and hasattr(scenario, "infer_validation_phase_snapshot"):
        try:
            enemy_positions = _truth_positions(patrol_task, env, "B")
            phase_snapshot = scenario.infer_validation_phase_snapshot(
                time=float(current_time),
                distance=float(current_distance),
                group_snapshots=group_snapshots,
                enemy_positions=enemy_positions,
            ) or {}
            snapshot.update(
                {
                    "enemy_script_family": str(phase_snapshot.get("enemy_script_family", "") or ""),
                    "enemy_script_phase_id": _safe_int(phase_snapshot.get("enemy_script_phase_id", -1), -1),
                    "enemy_script_phase_name": str(phase_snapshot.get("enemy_script_phase_name", "") or ""),
                    "enemy_script_phase_trigger": str(phase_snapshot.get("enemy_script_phase_trigger", "") or ""),
                    "enemy_script_phase_objective": str(phase_snapshot.get("enemy_script_phase_objective", "") or ""),
                    "enemy_script_phase_elapsed_s": float(phase_snapshot.get("enemy_script_phase_elapsed_s", 0.0) or 0.0),
                    "enemy_script_reference_distance_km": float(
                        phase_snapshot.get("enemy_script_reference_distance_km", current_distance) or current_distance
                    ),
                    "enemy_script_next_phase_id": _safe_int(phase_snapshot.get("enemy_script_next_phase_id", -1), -1),
                    "enemy_script_next_phase_name": str(phase_snapshot.get("enemy_script_next_phase_name", "") or ""),
                    "enemy_script_next_phase_trigger": str(phase_snapshot.get("enemy_script_next_phase_trigger", "") or ""),
                    "enemy_script_phase_offsets_json": json.dumps(
                        list(phase_snapshot.get("enemy_script_phase_offsets", []) or []),
                        ensure_ascii=False,
                    ),
                    "enemy_script_phase_altitudes_json": json.dumps(
                        list(phase_snapshot.get("enemy_script_phase_altitudes", []) or []),
                        ensure_ascii=False,
                    ),
                }
            )
        except Exception:
            pass
    elif hasattr(scenario, "get_phase_snapshot"):
        try:
            phase_snapshot = scenario.get_phase_snapshot(time=float(current_time), distance=float(current_distance)) or {}
            snapshot.update(
                {
                    "enemy_script_family": str(phase_snapshot.get("enemy_script_family", "") or ""),
                    "enemy_script_phase_id": _safe_int(phase_snapshot.get("enemy_script_phase_id", -1), -1),
                    "enemy_script_phase_name": str(phase_snapshot.get("enemy_script_phase_name", "") or ""),
                    "enemy_script_phase_trigger": str(phase_snapshot.get("enemy_script_phase_trigger", "") or ""),
                    "enemy_script_phase_objective": str(phase_snapshot.get("enemy_script_phase_objective", "") or ""),
                    "enemy_script_phase_elapsed_s": float(phase_snapshot.get("enemy_script_phase_elapsed_s", 0.0) or 0.0),
                    "enemy_script_reference_distance_km": float(
                        phase_snapshot.get("enemy_script_reference_distance_km", current_distance) or current_distance
                    ),
                    "enemy_script_next_phase_id": _safe_int(phase_snapshot.get("enemy_script_next_phase_id", -1), -1),
                    "enemy_script_next_phase_name": str(phase_snapshot.get("enemy_script_next_phase_name", "") or ""),
                    "enemy_script_next_phase_trigger": str(phase_snapshot.get("enemy_script_next_phase_trigger", "") or ""),
                    "enemy_script_phase_offsets_json": json.dumps(
                        list(phase_snapshot.get("enemy_script_phase_offsets", []) or []),
                        ensure_ascii=False,
                    ),
                    "enemy_script_phase_altitudes_json": json.dumps(
                        list(phase_snapshot.get("enemy_script_phase_altitudes", []) or []),
                        ensure_ascii=False,
                    ),
                }
            )
        except Exception:
            pass

    if hasattr(scenario, "is_awacs_available"):
        try:
            awacs_available = bool(scenario.is_awacs_available(float(current_time), float(current_distance)))
            snapshot["enemy_script_awacs_available"] = int(awacs_available)
            snapshot["enemy_script_awacs_state"] = "AVAILABLE" if awacs_available else "DENIED"
        except Exception:
            pass

    return snapshot


def _collect_step_row(
    spec: Ch6ScenarioSpec,
    patrol_task: CAPTask,
    env,
    time_s: float,
) -> Dict[str, object]:
    faor = getattr(patrol_task, "faor", None) or FAORManager()
    bullseye = faor.get_bullseye()
    enemy_truth = _truth_positions(patrol_task, env, "B")
    friendly_truth = _truth_positions(patrol_task, env, "A")
    friendly_flight = _collect_team_flight_metrics(env, "A")
    enemy_flight = _collect_team_flight_metrics(env, "B")
    recreate_snapshot = _collect_recreate_snapshot(patrol_task)

    low_count = medium_count = high_count = 0
    nearest_bullseye = float("inf")
    nearest_friendly = float("inf")
    friendly_center = None
    if friendly_truth:
        friendly_center = np.mean(np.asarray(list(friendly_truth.values()), dtype=float), axis=0)

    zone_by_enemy: Dict[str, str] = {}
    for enemy_id, (x_km, y_km, _) in enemy_truth.items():
        zone = faor.get_risk_zone(x_km, y_km)
        zone_name = zone.value if hasattr(zone, "value") else str(zone)
        zone_by_enemy[enemy_id] = zone_name
        if zone == RiskZone.LOW:
            low_count += 1
        elif zone == RiskZone.MEDIUM:
            medium_count += 1
        elif zone == RiskZone.HIGH:
            high_count += 1
        nearest_bullseye = min(nearest_bullseye, float(np.hypot(x_km - bullseye[0], y_km - bullseye[1])))
        if friendly_center is not None:
            nearest_friendly = min(
                nearest_friendly,
                float(np.linalg.norm(np.asarray([x_km, y_km, 0.0], dtype=float) - friendly_center)),
            )

    guidance_snapshot = {}
    if hasattr(patrol_task, "get_guidance_verification_snapshot"):
        guidance_snapshot = patrol_task.get_guidance_verification_snapshot() or {}

    bridge = getattr(patrol_task, "tactical_bridge", None)
    left_info = bridge.get_agent_tactic_info("A0100") if bridge is not None else {}
    right_info = bridge.get_agent_tactic_info("A0300") if bridge is not None else {}
    picture_snapshot = _collect_picture_snapshot(patrol_task, time_s)
    intent_snapshot = _collect_intent_snapshot(patrol_task, env, time_s)
    intent_truth_snapshot = _collect_intent_truth_snapshot(patrol_task)
    scan_snapshot = _collect_scan_snapshot(patrol_task)
    assignment_snapshot = _collect_pair_assignment_snapshot(patrol_task)
    enemy_script_snapshot = _collect_enemy_script_snapshot(
        spec,
        patrol_task,
        env,
        time_s,
        float(nearest_friendly) if np.isfinite(nearest_friendly) else 999.0,
    )

    awacs_tracks = patrol_task.awacs.get_tracks() if hasattr(patrol_task, "awacs") and patrol_task.awacs is not None else {}
    radar_track_ids = set()
    if hasattr(patrol_task, "cap_radar"):
        for aid in ("A0100", "A0200", "A0300", "A0400"):
            try:
                radar_track_ids.update((patrol_task.cap_radar.get_tracks(aid) or {}).keys())
            except Exception:
                continue

    missile_summary = {}
    missile_adapter = getattr(patrol_task, "missile_adapter", None)
    if missile_adapter is not None and hasattr(missile_adapter, "get_summary_snapshot"):
        missile_summary = missile_adapter.get_summary_snapshot() or {}
    missile_states = dict(missile_summary.get("state_counts", {}) or {})

    mission_summary = {}
    mission_evaluator = getattr(patrol_task, "mission_evaluator", None)
    if mission_evaluator is not None and hasattr(mission_evaluator, "get_summary"):
        mission_summary = mission_evaluator.get_summary() or {}

    cap_state = getattr(getattr(patrol_task, "cap_state_machine", None), "state", "")
    detect_mode = getattr(getattr(patrol_task, "coop_detection", None), "_mode", "")
    friendly_alive = len(friendly_truth)
    enemy_alive = len(enemy_truth)
    row = {
        "time_s": float(time_s),
        "scenario_id": spec.scenario_id,
        "cap_state": _enum_text(cap_state, default=""),
        "cap_state_reason": str(getattr(patrol_task, "_last_cap_state_reason", "") or ""),
        "detect_mode": _enum_text(detect_mode, default=""),
        "friendly_alive": friendly_alive,
        "enemy_alive": enemy_alive,
        "friendly_loss_count": max(0, 4 - friendly_alive),
        "enemy_kill_count": max(0, 4 - enemy_alive),
        "friendly_missiles_left": _friendly_missiles_left(env),
        "enemy_missiles_left": _enemy_missiles_left(env),
        "friendly_min_altitude_m": friendly_flight.get("min_altitude_m"),
        "friendly_mean_altitude_m": friendly_flight.get("mean_altitude_m"),
        "friendly_min_speed_mps": friendly_flight.get("min_speed_mps"),
        "friendly_mean_speed_mps": friendly_flight.get("mean_speed_mps"),
        "friendly_max_descent_rate_mps": friendly_flight.get("max_descent_rate_mps"),
        "enemy_min_altitude_m": enemy_flight.get("min_altitude_m"),
        "enemy_mean_altitude_m": enemy_flight.get("mean_altitude_m"),
        "enemy_min_speed_mps": enemy_flight.get("min_speed_mps"),
        "enemy_mean_speed_mps": enemy_flight.get("mean_speed_mps"),
        "enemy_max_descent_rate_mps": enemy_flight.get("max_descent_rate_mps"),
        "friendly_sim_recreate_total": int(recreate_snapshot.get("friendly_sim_recreate_total", 0) or 0),
        "enemy_sim_recreate_total": int(recreate_snapshot.get("enemy_sim_recreate_total", 0) or 0),
        "friendly_sim_recreate_by_agent_json": str(
            recreate_snapshot.get("friendly_sim_recreate_by_agent_json", "{}") or "{}"
        ),
        "enemy_sim_recreate_by_agent_json": str(
            recreate_snapshot.get("enemy_sim_recreate_by_agent_json", "{}") or "{}"
        ),
        "truth_low_count": low_count,
        "truth_medium_count": medium_count,
        "truth_high_count": high_count,
        "nearest_enemy_bullseye_km": None if not np.isfinite(nearest_bullseye) else float(nearest_bullseye),
        "nearest_enemy_to_friendly_km": None if not np.isfinite(nearest_friendly) else float(nearest_friendly),
        "awacs_track_count": len(awacs_tracks or {}),
        "radar_track_count": len(radar_track_ids),
        "stable_ready_count": len(guidance_snapshot.get("stable_tracking_ready_targets", []) or []),
        "stable_tracking_target_count": len(guidance_snapshot.get("stable_tracking_trackers", {}) or {}),
        "left_tactic": _normalize_tactic_name(assignment_snapshot.get("left_tactic") or (left_info or {}).get("tactic")),
        "right_tactic": _normalize_tactic_name(assignment_snapshot.get("right_tactic") or (right_info or {}).get("tactic")),
        "left_phase": str((left_info or {}).get("phase", "") or ""),
        "right_phase": str((right_info or {}).get("phase", "") or ""),
        "enemy_script_family": str(enemy_script_snapshot.get("enemy_script_family", "") or ""),
        "enemy_script_phase_id": _safe_int(enemy_script_snapshot.get("enemy_script_phase_id", -1), -1),
        "enemy_script_phase_name": str(enemy_script_snapshot.get("enemy_script_phase_name", "") or ""),
        "enemy_script_phase_trigger": str(enemy_script_snapshot.get("enemy_script_phase_trigger", "") or ""),
        "enemy_script_phase_objective": str(enemy_script_snapshot.get("enemy_script_phase_objective", "") or ""),
        "enemy_script_phase_elapsed_s": float(enemy_script_snapshot.get("enemy_script_phase_elapsed_s", 0.0) or 0.0),
        "enemy_script_reference_distance_km": float(
            enemy_script_snapshot.get("enemy_script_reference_distance_km", 999.0) or 999.0
        ),
        "enemy_script_next_phase_id": _safe_int(enemy_script_snapshot.get("enemy_script_next_phase_id", -1), -1),
        "enemy_script_next_phase_name": str(enemy_script_snapshot.get("enemy_script_next_phase_name", "") or ""),
        "enemy_script_next_phase_trigger": str(enemy_script_snapshot.get("enemy_script_next_phase_trigger", "") or ""),
        "enemy_script_awacs_available": int(enemy_script_snapshot.get("enemy_script_awacs_available", 1) or 0),
        "enemy_script_awacs_state": str(enemy_script_snapshot.get("enemy_script_awacs_state", "AVAILABLE") or "AVAILABLE"),
        "enemy_script_phase_offsets_json": str(enemy_script_snapshot.get("enemy_script_phase_offsets_json", "[]") or "[]"),
        "enemy_script_phase_altitudes_json": str(
            enemy_script_snapshot.get("enemy_script_phase_altitudes_json", "[]") or "[]"
        ),
        "enemy_left_group_phase": str(enemy_script_snapshot.get("enemy_left_group_phase", "") or ""),
        "enemy_right_group_phase": str(enemy_script_snapshot.get("enemy_right_group_phase", "") or ""),
        "enemy_left_wave_index": _safe_int(enemy_script_snapshot.get("enemy_left_wave_index", 0), 0),
        "enemy_right_wave_index": _safe_int(enemy_script_snapshot.get("enemy_right_wave_index", 0), 0),
        "enemy_left_pressure_tag": str(enemy_script_snapshot.get("enemy_left_pressure_tag", "") or ""),
        "enemy_right_pressure_tag": str(enemy_script_snapshot.get("enemy_right_pressure_tag", "") or ""),
        "intent_target": intent_snapshot["intent_target"],
        "intent_name": intent_snapshot["intent_name"],
        "intent_class": str(intent_snapshot.get("intent_class", "") or ""),
        "intent_threat": intent_snapshot["intent_threat"],
        "intent_confidence": intent_snapshot["intent_confidence"],
        "intent_model_ready": int(intent_snapshot["intent_model_ready"]),
        "intent_contact_count": int(intent_snapshot["intent_contact_count"]),
        "intent_classified_count": int(intent_snapshot.get("intent_classified_count", 0) or 0),
        "intent_unclassified_count": int(intent_snapshot.get("intent_unclassified_count", 0) or 0),
        "intent_ready_count": int(intent_snapshot["intent_ready_count"]),
        "intent_top2_json": str(intent_snapshot["intent_top2_json"]),
        "intent_top4_json": str(intent_snapshot.get("intent_top4_json", "[]") or "[]"),
        "intent_all_json": str(intent_snapshot.get("intent_all_json", "{}") or "{}"),
        "intent_present_targets_json": str(intent_snapshot.get("intent_present_targets_json", "[]") or "[]"),
        "intent_attack_count": int(intent_snapshot.get("intent_attack_count", 0) or 0),
        "intent_defense_count": int(intent_snapshot.get("intent_defense_count", 0) or 0),
        "intent_retreat_count": int(intent_snapshot.get("intent_retreat_count", 0) or 0),
        "intent_recon_count": int(intent_snapshot.get("intent_recon_count", 0) or 0),
        "intent_high_count": int(intent_snapshot.get("intent_high_count", 0) or 0),
        "intent_critical_count": int(intent_snapshot.get("intent_critical_count", 0) or 0),
        "intent_truth_any_count": 0,
        "intent_truth_match_count": 0,
        "intent_truth_ready_match_count": 0,
        "gate_pass_count": int(guidance_snapshot.get("prelaunch_gate_unique_pass_count", guidance_snapshot.get("prelaunch_gate_pass_count", 0))),
        "gate_block_count": int(guidance_snapshot.get("prelaunch_gate_unique_block_count", guidance_snapshot.get("prelaunch_gate_block_count", 0))),
        "gate_total_request_count": int(guidance_snapshot.get("prelaunch_gate_unique_total_requests", guidance_snapshot.get("prelaunch_gate_total_requests", 0))),
        "gate_ready_pass_count": int(guidance_snapshot.get("prelaunch_gate_ready_unique_pass_count", guidance_snapshot.get("prelaunch_gate_ready_pass_count", 0))),
        "gate_ready_block_count": int(guidance_snapshot.get("prelaunch_gate_ready_unique_block_count", guidance_snapshot.get("prelaunch_gate_ready_block_count", 0))),
        "gate_ready_total_request_count": int(guidance_snapshot.get("prelaunch_gate_ready_unique_total_requests", guidance_snapshot.get("prelaunch_gate_ready_total_requests", 0))),
        "gate_block_stable_not_ready_count": int(guidance_snapshot.get("prelaunch_gate_block_stable_not_ready_count", 0)),
        "gate_block_friendly_safe_count": int(guidance_snapshot.get("prelaunch_gate_block_friendly_safe_count", 0)),
        "gate_block_long_shot_count": int(guidance_snapshot.get("prelaunch_gate_block_long_shot_count", 0)),
        "gate_block_launch_exec_fail_count": int(guidance_snapshot.get("prelaunch_gate_block_launch_exec_fail_count", 0)),
        "gate_block_other_count": int(guidance_snapshot.get("prelaunch_gate_block_other_count", 0)),
        "relay_attempt_count": int(guidance_snapshot.get("relay_unique_attempt_count", guidance_snapshot.get("relay_attempt_count", 0))),
        "relay_success_count": int(guidance_snapshot.get("relay_unique_success_count", guidance_snapshot.get("relay_success_count", 0))),
        "relay_fail_count": int(guidance_snapshot.get("relay_fail_count", 0)),
        "gate_pass_raw_count": int(guidance_snapshot.get("prelaunch_gate_pass_count", 0)),
        "gate_block_raw_count": int(guidance_snapshot.get("prelaunch_gate_block_count", 0)),
        "gate_pass_unique_count": int(guidance_snapshot.get("prelaunch_gate_unique_pass_count", guidance_snapshot.get("prelaunch_gate_pass_count", 0))),
        "gate_block_unique_count": int(guidance_snapshot.get("prelaunch_gate_unique_block_count", guidance_snapshot.get("prelaunch_gate_block_count", 0))),
        "gate_total_request_raw_count": int(guidance_snapshot.get("prelaunch_gate_raw_total_requests", 0)),
        "gate_total_request_unique_count": int(guidance_snapshot.get("prelaunch_gate_unique_total_requests", guidance_snapshot.get("prelaunch_gate_total_requests", 0))),
        "gate_pass_rate_raw": float(guidance_snapshot.get("prelaunch_gate_raw_pass_rate", 0.0)),
        "gate_pass_rate_unique": float(guidance_snapshot.get("prelaunch_gate_unique_pass_rate", guidance_snapshot.get("prelaunch_gate_pass_rate", 0.0))),
        "gate_ready_pass_rate": float(guidance_snapshot.get("prelaunch_gate_ready_pass_rate", 0.0)),
        "gate_ready_pass_rate_raw": float(guidance_snapshot.get("prelaunch_gate_ready_pass_rate_raw", 0.0)),
        "gate_pass_per_launch_rate": float(guidance_snapshot.get("gate_pass_per_launch_rate", 0.0)),
        "gate_ready_pass_per_launch_rate": float(guidance_snapshot.get("gate_ready_pass_per_launch_rate", 0.0)),
        "relay_attempt_raw_count": int(guidance_snapshot.get("relay_attempt_count", 0)),
        "relay_success_raw_count": int(guidance_snapshot.get("relay_success_count", 0)),
        "relay_attempt_unique_count": int(guidance_snapshot.get("relay_unique_attempt_count", guidance_snapshot.get("relay_attempt_count", 0))),
        "relay_success_unique_count": int(guidance_snapshot.get("relay_unique_success_count", guidance_snapshot.get("relay_success_count", 0))),
        "relay_success_rate_raw": float(guidance_snapshot.get("relay_raw_success_rate", 0.0)),
        "relay_success_rate_unique": float(guidance_snapshot.get("relay_unique_success_rate", guidance_snapshot.get("relay_success_rate", 0.0))),
        "relay_success_per_launch_rate": float(guidance_snapshot.get("relay_success_per_launch_rate", 0.0)),
        "active_relay_flag": int(bool(guidance_snapshot.get("active_relay", False))),
        "active_guided_missile_peak": int(guidance_snapshot.get("active_guided_missile_peak", 0)),
        "missile_launch_count": int(missile_summary.get("launch_count", 0)),
        "missile_outcome_count": int(missile_summary.get("outcome_count", 0)),
        "missile_state_ready": int(missile_states.get("ready", 0)),
        "missile_state_launched": int(missile_states.get("launched", 0)),
        "missile_state_guiding": int(missile_states.get("guiding", 0)),
        "missile_state_terminal": int(missile_states.get("terminal", 0)),
        "missile_state_hit": int(missile_states.get("hit", 0)),
        "missile_state_miss": int(missile_states.get("miss", 0)),
        "mission_result": str(mission_summary.get("result", "") or ""),
        "mission_threat_level": str(mission_summary.get("threat_level", "") or ""),
        "mission_runtime_kills": int(mission_summary.get("kills", 0) or 0),
        "mission_high_zone_threats": int(mission_summary.get("high_zone_threats", 0) or 0),
        "mission_medium_zone_threats": int(mission_summary.get("medium_zone_threats", 0) or 0),
        "mission_total_threats": int(mission_summary.get("total_threats", 0) or 0),
        "mission_friendly_lost": int(mission_summary.get("friendly_lost", 0) or 0),
        "mission_missiles_fired": int(mission_summary.get("missiles_fired", 0) or 0),
        "mission_high_zone_breach_events": int(mission_summary.get("high_zone_breach_events", 0) or 0),
        "left_target": str(assignment_snapshot.get("left_target", "") or ""),
        "left_shooter": str(assignment_snapshot.get("left_shooter", "") or ""),
        "left_support": str(assignment_snapshot.get("left_support", "") or ""),
        "left_stage_reason": str(assignment_snapshot.get("left_stage_reason", "") or ""),
        "left_tactic_reason": str(assignment_snapshot.get("left_tactic_reason", "") or ""),
        "left_maneuver_reason": str(assignment_snapshot.get("left_maneuver_reason", "") or ""),
        "left_parameter_reason": str(assignment_snapshot.get("left_parameter_reason", "") or ""),
        "left_decision_snapshot_json": str(assignment_snapshot.get("left_decision_snapshot_json", "{}") or "{}"),
        "right_target": str(assignment_snapshot.get("right_target", "") or ""),
        "right_shooter": str(assignment_snapshot.get("right_shooter", "") or ""),
        "right_support": str(assignment_snapshot.get("right_support", "") or ""),
        "right_stage_reason": str(assignment_snapshot.get("right_stage_reason", "") or ""),
        "right_tactic_reason": str(assignment_snapshot.get("right_tactic_reason", "") or ""),
        "right_maneuver_reason": str(assignment_snapshot.get("right_maneuver_reason", "") or ""),
        "right_parameter_reason": str(assignment_snapshot.get("right_parameter_reason", "") or ""),
        "right_decision_snapshot_json": str(assignment_snapshot.get("right_decision_snapshot_json", "{}") or "{}"),
        "assignment_target_count": int(assignment_snapshot.get("assignment_target_count", 0) or 0),
        "assignment_targets_json": str(assignment_snapshot.get("assignment_targets_json", "[]") or "[]"),
        "picture_low_count": int(picture_snapshot.get("picture_low_count", 0) or 0),
        "picture_medium_count": int(picture_snapshot.get("picture_medium_count", 0) or 0),
        "picture_high_count": int(picture_snapshot.get("picture_high_count", 0) or 0),
        "picture_total_count": int(picture_snapshot.get("picture_total_count", 0) or 0),
        "search_picture_total_count": int(picture_snapshot.get("search_picture_total_count", 0) or 0),
        "scan_assignment_count": int(scan_snapshot.get("scan_assignment_count", 0) or 0),
        "scan_target_count": int(scan_snapshot.get("scan_target_count", 0) or 0),
        "scan_targets_json": str(scan_snapshot.get("scan_targets_json", "[]") or "[]"),
        "scan_coverage_total_deg": float(scan_snapshot.get("scan_coverage_total_deg", 0.0) or 0.0),
        "scan_overlap_deg": float(scan_snapshot.get("scan_overlap_deg", 0.0) or 0.0),
        "confidence_radius_km": float(scan_snapshot.get("confidence_radius_km", 0.0) or 0.0),
        "A0100_scan_target": str(scan_snapshot.get("A0100_scan_target", "") or ""),
        "A0200_scan_target": str(scan_snapshot.get("A0200_scan_target", "") or ""),
        "A0300_scan_target": str(scan_snapshot.get("A0300_scan_target", "") or ""),
        "A0400_scan_target": str(scan_snapshot.get("A0400_scan_target", "") or ""),
        "A0100_scan_center_deg": float(scan_snapshot.get("A0100_scan_center_deg", 0.0) or 0.0),
        "A0200_scan_center_deg": float(scan_snapshot.get("A0200_scan_center_deg", 0.0) or 0.0),
        "A0300_scan_center_deg": float(scan_snapshot.get("A0300_scan_center_deg", 0.0) or 0.0),
        "A0400_scan_center_deg": float(scan_snapshot.get("A0400_scan_center_deg", 0.0) or 0.0),
        "A0100_scan_range_deg": float(scan_snapshot.get("A0100_scan_range_deg", 0.0) or 0.0),
        "A0200_scan_range_deg": float(scan_snapshot.get("A0200_scan_range_deg", 0.0) or 0.0),
        "A0300_scan_range_deg": float(scan_snapshot.get("A0300_scan_range_deg", 0.0) or 0.0),
        "A0400_scan_range_deg": float(scan_snapshot.get("A0400_scan_range_deg", 0.0) or 0.0),
        "zone_by_enemy_json": json.dumps(zone_by_enemy, ensure_ascii=False, sort_keys=True),
        "enemy_truth_positions_json": _truth_positions_json(enemy_truth),
        "friendly_truth_positions_json": _truth_positions_json(friendly_truth),
        "guidance_snapshot_json": json.dumps(guidance_snapshot, ensure_ascii=False, sort_keys=True),
        "missile_by_shooter_json": json.dumps(missile_summary.get("by_shooter", {}) or {}, ensure_ascii=False, sort_keys=True),
    }
    for enemy_id in ENEMY_AGENT_IDS:
        row[f"{enemy_id}_intent_name"] = str(intent_snapshot.get(f"{enemy_id}_intent_name", "") or "")
        row[f"{enemy_id}_intent_label"] = str(intent_snapshot.get(f"{enemy_id}_intent_label", "") or "")
        row[f"{enemy_id}_intent_raw_label"] = str(intent_snapshot.get(f"{enemy_id}_intent_raw_label", "") or "")
        row[f"{enemy_id}_intent_class"] = str(intent_snapshot.get(f"{enemy_id}_intent_class", "") or "")
        row[f"{enemy_id}_intent_class_code"] = str(intent_snapshot.get(f"{enemy_id}_intent_class_code", "") or "")
        row[f"{enemy_id}_intent_type"] = str(intent_snapshot.get(f"{enemy_id}_intent_type", "") or "")
        row[f"{enemy_id}_intent_threat"] = str(intent_snapshot.get(f"{enemy_id}_intent_threat", "") or "")
        row[f"{enemy_id}_intent_confidence"] = float(intent_snapshot.get(f"{enemy_id}_intent_confidence", 0.0) or 0.0)
        row[f"{enemy_id}_intent_model_ready"] = int(intent_snapshot.get(f"{enemy_id}_intent_model_ready", 0) or 0)
        row[f"{enemy_id}_intent_distance_km"] = float(intent_snapshot.get(f"{enemy_id}_intent_distance_km", 0.0) or 0.0)
        row[f"{enemy_id}_intent_present"] = int(intent_snapshot.get(f"{enemy_id}_intent_present", 0) or 0)
        row[f"{enemy_id}_intent_truth_class"] = str(intent_truth_snapshot.get(f"{enemy_id}_intent_truth_class", "") or "")
        row[f"{enemy_id}_intent_truth_code"] = str(intent_truth_snapshot.get(f"{enemy_id}_intent_truth_code", "") or "")
        pred_class = str(row.get(f"{enemy_id}_intent_class", "") or "")
        truth_class = str(row.get(f"{enemy_id}_intent_truth_class", "") or "")
        pred_ready = int(row.get(f"{enemy_id}_intent_model_ready", 0) or 0) > 0
        if truth_class:
            row["intent_truth_any_count"] = int(row.get("intent_truth_any_count", 0) or 0) + 1
            if pred_class and pred_class == truth_class:
                row["intent_truth_match_count"] = int(row.get("intent_truth_match_count", 0) or 0) + 1
                if pred_ready:
                    row["intent_truth_ready_match_count"] = int(row.get("intent_truth_ready_match_count", 0) or 0) + 1
    for group_prefix in ENEMY_INTENT_GROUPS:
        row[f"{group_prefix}_intent_target"] = str(intent_snapshot.get(f"{group_prefix}_intent_target", "") or "")
        row[f"{group_prefix}_intent_name"] = str(intent_snapshot.get(f"{group_prefix}_intent_name", "") or "")
        row[f"{group_prefix}_intent_class"] = str(intent_snapshot.get(f"{group_prefix}_intent_class", "") or "")
        row[f"{group_prefix}_intent_class_code"] = str(intent_snapshot.get(f"{group_prefix}_intent_class_code", "") or "")
        row[f"{group_prefix}_intent_type"] = str(intent_snapshot.get(f"{group_prefix}_intent_type", "") or "")
        row[f"{group_prefix}_intent_threat"] = str(intent_snapshot.get(f"{group_prefix}_intent_threat", "") or "")
        row[f"{group_prefix}_intent_confidence"] = float(
            intent_snapshot.get(f"{group_prefix}_intent_confidence", 0.0) or 0.0
        )
        row[f"{group_prefix}_intent_contact_count"] = int(
            intent_snapshot.get(f"{group_prefix}_intent_contact_count", 0) or 0
        )
        row[f"{group_prefix}_intent_classified_count"] = int(
            intent_snapshot.get(f"{group_prefix}_intent_classified_count", 0) or 0
        )
        row[f"{group_prefix}_intent_unclassified_count"] = int(
            intent_snapshot.get(f"{group_prefix}_intent_unclassified_count", 0) or 0
        )
        row[f"{group_prefix}_intent_ready_count"] = int(
            intent_snapshot.get(f"{group_prefix}_intent_ready_count", 0) or 0
        )
        row[f"{group_prefix}_intent_attack_count"] = int(
            intent_snapshot.get(f"{group_prefix}_intent_attack_count", 0) or 0
        )
        row[f"{group_prefix}_intent_defense_count"] = int(
            intent_snapshot.get(f"{group_prefix}_intent_defense_count", 0) or 0
        )
        row[f"{group_prefix}_intent_retreat_count"] = int(
            intent_snapshot.get(f"{group_prefix}_intent_retreat_count", 0) or 0
        )
        row[f"{group_prefix}_intent_recon_count"] = int(
            intent_snapshot.get(f"{group_prefix}_intent_recon_count", 0) or 0
        )
        row[f"{group_prefix}_intent_high_count"] = int(
            intent_snapshot.get(f"{group_prefix}_intent_high_count", 0) or 0
        )
        row[f"{group_prefix}_intent_critical_count"] = int(
            intent_snapshot.get(f"{group_prefix}_intent_critical_count", 0) or 0
        )
    return row


def _update_events(
    events: List[Dict[str, object]],
    prev_state: Dict[str, object],
    row: Dict[str, object],
) -> None:
    time_s = float(row["time_s"])

    for label in ("cap_state", "detect_mode", "mission_threat_level", "mission_result", "enemy_script_awacs_state"):
        prev_value = prev_state.get(label)
        current_value = row.get(label)
        if prev_value is not None and prev_value != current_value:
            events.append({"time_s": time_s, "event_type": "state_change", "field": label, "value": current_value})
        prev_state[label] = current_value

    for label in ("left_tactic", "right_tactic", "left_phase", "right_phase"):
        prev_value = prev_state.get(label)
        current_value = row.get(label)
        if prev_value is not None and prev_value != current_value:
            events.append({"time_s": time_s, "event_type": "tactic_change", "field": label, "value": current_value})
        prev_state[label] = current_value

    for label in (
        "left_target",
        "right_target",
        "A0100_scan_target",
        "A0200_scan_target",
        "A0300_scan_target",
        "A0400_scan_target",
    ):
        prev_value = prev_state.get(label)
        current_value = row.get(label)
        if prev_value is not None and prev_value != current_value:
            events.append({"time_s": time_s, "event_type": "target_change", "field": label, "value": current_value})
        prev_state[label] = current_value

    for label in (
        "gate_pass_count",
        "gate_block_count",
        "relay_attempt_count",
        "relay_success_count",
        "missile_launch_count",
        "missile_outcome_count",
        "enemy_kill_count",
        "friendly_loss_count",
    ):
        prev_value = int(prev_state.get(label, 0))
        current_value = int(row.get(label, 0) or 0)
        if current_value > prev_value:
            events.append({"time_s": time_s, "event_type": "counter_increase", "field": label, "value": current_value})
        prev_state[label] = current_value

    intent_key = f"{row['intent_target']}|{row['intent_name']}"
    prev_intent = prev_state.get("intent_key")
    if row["intent_target"] and intent_key != prev_intent:
        events.append({"time_s": time_s, "event_type": "intent_change", "field": "highest_threat", "value": intent_key})
    prev_state["intent_key"] = intent_key

    prev_intent_threat = prev_state.get("intent_threat")
    current_intent_threat = row.get("intent_threat")
    if prev_intent_threat is not None and prev_intent_threat != current_intent_threat:
        events.append(
            {"time_s": time_s, "event_type": "intent_level_change", "field": "intent_threat", "value": current_intent_threat}
        )
    prev_state["intent_threat"] = current_intent_threat

    for enemy_id in ENEMY_AGENT_IDS:
        name_label = f"{enemy_id}_intent_name"
        prev_name = prev_state.get(name_label)
        current_name = row.get(name_label)
        if prev_name is not None and prev_name != current_name and str(current_name or "").strip():
            events.append({"time_s": time_s, "event_type": "intent_target_change", "field": name_label, "value": current_name})
        prev_state[name_label] = current_name

        threat_label = f"{enemy_id}_intent_threat"
        prev_threat = prev_state.get(threat_label)
        current_threat = row.get(threat_label)
        if prev_threat is not None and prev_threat != current_threat and str(current_threat or "").strip():
            events.append(
                {"time_s": time_s, "event_type": "intent_target_level_change", "field": threat_label, "value": current_threat}
            )
        prev_state[threat_label] = current_threat

    for group_prefix in ENEMY_INTENT_GROUPS:
        group_name_label = f"{group_prefix}_intent_name"
        prev_group_name = prev_state.get(group_name_label)
        current_group_name = row.get(group_name_label)
        if prev_group_name is not None and prev_group_name != current_group_name and str(current_group_name or "").strip():
            events.append(
                {"time_s": time_s, "event_type": "intent_group_change", "field": group_name_label, "value": current_group_name}
            )
        prev_state[group_name_label] = current_group_name

    prev_enemy_phase = prev_state.get("enemy_script_phase_name")
    current_enemy_phase = row.get("enemy_script_phase_name")
    if prev_enemy_phase is not None and prev_enemy_phase != current_enemy_phase:
        events.append(
            {
                "time_s": time_s,
                "event_type": "enemy_phase_change",
                "field": "enemy_script_phase_name",
                "value": current_enemy_phase,
            }
        )
    prev_state["enemy_script_phase_name"] = current_enemy_phase

    current_zones = json.loads(str(row["zone_by_enemy_json"]))
    prev_zones = prev_state.get("zones", {})
    for enemy_id, zone_name in current_zones.items():
        if prev_zones.get(enemy_id) != zone_name:
            events.append({"time_s": time_s, "event_type": "zone_transition", "field": enemy_id, "value": zone_name})
    prev_state["zones"] = current_zones


def _row_durations(timeline_rows: List[Dict[str, object]], default_dt: float = 0.2) -> List[float]:
    if not timeline_rows:
        return []
    durations: List[float] = []
    prev_time = 0.0
    for row in timeline_rows:
        current_time = float(row.get("time_s", 0.0) or 0.0)
        dt = current_time - prev_time
        if dt <= 0.0:
            dt = float(default_dt)
        durations.append(float(dt))
        prev_time = current_time
    return durations


def _first_time_when(timeline_rows: List[Dict[str, object]], key: str, predicate) -> Optional[float]:
    for row in timeline_rows:
        value = row.get(key)
        try:
            if predicate(value):
                return float(row.get("time_s", 0.0) or 0.0)
        except Exception:
            continue
    return None


def _first_time_matching_rows(timeline_rows: List[Dict[str, object]], predicate) -> Optional[float]:
    for row in timeline_rows:
        try:
            if predicate(row):
                return float(row.get("time_s", 0.0) or 0.0)
        except Exception:
            continue
    return None


def _count_switches(timeline_rows: List[Dict[str, object]], key: str) -> int:
    count = 0
    prev_value = None
    initialized = False
    for row in timeline_rows:
        value = str(row.get(key, "") or "")
        if not initialized:
            prev_value = value
            initialized = True
            continue
        if value != prev_value:
            count += 1
            prev_value = value
    return count


def _safe_min_float(timeline_rows: List[Dict[str, object]], key: str) -> Optional[float]:
    values: List[float] = []
    for row in timeline_rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return min(values) if values else None


def _safe_max_float(timeline_rows: List[Dict[str, object]], key: str) -> Optional[float]:
    values: List[float] = []
    for row in timeline_rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return max(values) if values else None


def _safe_mean_float(timeline_rows: List[Dict[str, object]], key: str) -> Optional[float]:
    values: List[float] = []
    for row in timeline_rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return float(sum(values) / len(values)) if values else None


def _truth_positions_json(positions: Dict[str, Tuple[float, float, float]]) -> str:
    payload: Dict[str, Dict[str, float]] = {}
    for agent_id, coords in dict(positions or {}).items():
        try:
            x_km, y_km, z_km = coords
        except Exception:
            continue
        payload[str(agent_id)] = {
            "x_km": float(x_km),
            "y_km": float(y_km),
            "z_km": float(z_km),
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _decode_truth_positions_json(payload: object) -> Dict[str, Dict[str, float]]:
    try:
        raw = json.loads(str(payload or "{}"))
    except Exception:
        return {}

    result: Dict[str, Dict[str, float]] = {}
    if not isinstance(raw, dict):
        return result

    for agent_id, item in raw.items():
        x_km = y_km = z_km = float("nan")
        if isinstance(item, dict):
            x_km = _safe_float(item.get("x_km"), float("nan"))
            y_km = _safe_float(item.get("y_km"), float("nan"))
            z_km = _safe_float(item.get("z_km"), float("nan"))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            x_km = _safe_float(item[0], float("nan"))
            y_km = _safe_float(item[1], float("nan"))
            z_km = _safe_float(item[2], float("nan"))

        if not np.isfinite(x_km) or not np.isfinite(y_km):
            continue
        result[str(agent_id)] = {
            "x_km": float(x_km),
            "y_km": float(y_km),
            "z_km": float(z_km) if np.isfinite(z_km) else 0.0,
        }
    return result


def _enemy_zone_profile_summary(
    timeline_rows: List[Dict[str, object]],
    faor: Optional[FAORManager] = None,
) -> List[Dict[str, object]]:
    faor = faor or FAORManager()
    high_line = float(getattr(faor, "high_y_max", 100.0))
    medium_line = float(getattr(faor, "medium_y_max", 200.0))
    x_min = float(getattr(getattr(faor, "boundary", None), "x_min", 0.0))
    x_max = float(getattr(getattr(faor, "boundary", None), "x_max", 200.0))
    profiles: Dict[str, Dict[str, object]] = {
        enemy_id: {
            "enemy_id": enemy_id,
            "deepest_zone": "OUTSIDE",
            "first_medium_time_s": None,
            "first_high_time_s": None,
            "medium_time_s": 0.0,
            "high_time_s": 0.0,
            "closest_to_high_line_km": None,
            "closest_to_medium_line_km": None,
            "high_penetration_km": 0.0,
            "medium_penetration_km": 0.0,
            "position_source": "zone_only",
        }
        for enemy_id in ENEMY_AGENT_IDS
    }
    zone_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "OUTSIDE": 0}
    if not timeline_rows:
        return list(profiles.values())

    for idx, row in enumerate(timeline_rows):
        time_s = _safe_float(row.get("time_s"), float("nan"))
        if idx + 1 < len(timeline_rows):
            next_time = _safe_float(timeline_rows[idx + 1].get("time_s"), time_s)
            duration_s = max(0.0, next_time - time_s) if np.isfinite(next_time) and np.isfinite(time_s) else 0.0
        else:
            duration_s = 0.0
        try:
            zone_map = json.loads(str(row.get("zone_by_enemy_json", "{}") or "{}"))
        except Exception:
            zone_map = {}
        truth_positions = _decode_truth_positions_json(row.get("enemy_truth_positions_json", "{}"))
        for enemy_id in ENEMY_AGENT_IDS:
            zone = str(zone_map.get(enemy_id, "OUTSIDE") or "OUTSIDE").upper()
            profile = profiles[enemy_id]
            if zone_rank.get(zone, 0) > zone_rank.get(str(profile["deepest_zone"]), 0):
                profile["deepest_zone"] = zone
            if zone == "MEDIUM":
                profile["medium_time_s"] = float(profile["medium_time_s"]) + float(duration_s)
                if profile["first_medium_time_s"] is None and np.isfinite(time_s):
                    profile["first_medium_time_s"] = float(time_s)
            elif zone == "HIGH":
                profile["high_time_s"] = float(profile["high_time_s"]) + float(duration_s)
                if profile["first_high_time_s"] is None and np.isfinite(time_s):
                    profile["first_high_time_s"] = float(time_s)
                if profile["first_medium_time_s"] is None and np.isfinite(time_s):
                    profile["first_medium_time_s"] = float(time_s)
            truth = truth_positions.get(enemy_id)
            if not truth:
                continue
            x_km = _safe_float(truth.get("x_km"), float("nan"))
            y_km = _safe_float(truth.get("y_km"), float("nan"))
            if not np.isfinite(x_km) or not np.isfinite(y_km):
                continue

            profile["position_source"] = "truth_xy"
            if not (x_min <= x_km <= x_max):
                continue

            high_closest = abs(y_km - high_line)
            medium_closest = abs(y_km - medium_line)
            prev_high = profile["closest_to_high_line_km"]
            prev_medium = profile["closest_to_medium_line_km"]
            if prev_high is None or float(high_closest) < float(prev_high):
                profile["closest_to_high_line_km"] = float(high_closest)
            if prev_medium is None or float(medium_closest) < float(prev_medium):
                profile["closest_to_medium_line_km"] = float(medium_closest)

            profile["high_penetration_km"] = max(float(profile["high_penetration_km"]), max(0.0, high_line - y_km))
            profile["medium_penetration_km"] = max(float(profile["medium_penetration_km"]), max(0.0, medium_line - y_km))
    return list(profiles.values())


def _collect_ch6_cooperative_detection_metrics(
    patrol_task: CAPTask,
    timeline_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    expected_targets = ("B0100", "B0200", "B0300", "B0400")

    activation_times_raw = getattr(patrol_task, "_radar_range_entry_times", {}) or {}
    activation_times: Dict[str, float] = {}
    for aid, value in dict(activation_times_raw).items():
        aid_str = str(aid or "")
        if not aid_str.startswith("A"):
            continue
        try:
            activation_times[aid_str] = float(value)
        except Exception:
            continue

    if activation_times:
        activation_values = list(activation_times.values())
        metrics["radar_activation_times_s"] = dict(sorted(activation_times.items(), key=lambda item: item[0]))
        metrics["radar_first_activation_time_s"] = float(min(activation_values))
        metrics["radar_last_activation_time_s"] = float(max(activation_values))
        metrics["radar_activation_mean_s"] = float(np.mean(activation_values))
        metrics["radar_activation_std_s"] = float(np.std(activation_values))

    total_target_count = float(len(expected_targets))
    radar_track_counts = [int(row.get("radar_track_count", 0) or 0) for row in timeline_rows]
    alive_enemy_counts = [max(1, int(row.get("enemy_alive", len(expected_targets)) or len(expected_targets))) for row in timeline_rows]
    if radar_track_counts and total_target_count > 0:
        metrics["radar_detection_coverage_ratio_all_time"] = float(np.mean(radar_track_counts) / total_target_count)
        dynamic_all_time = [
            min(float(track_count), float(alive_count)) / float(alive_count)
            for track_count, alive_count in zip(radar_track_counts, alive_enemy_counts)
            if alive_count > 0
        ]
        metrics["radar_detection_coverage_ratio_alive_targets_all_time"] = float(np.mean(dynamic_all_time)) if dynamic_all_time else 0.0
        metrics["radar_any_track_continuity_all_time"] = float(
            np.mean([1.0 if count > 0 else 0.0 for count in radar_track_counts])
        )

        first_activation_time = metrics.get("radar_first_activation_time_s")
        if first_activation_time is None:
            post_counts = list(radar_track_counts)
        else:
            post_counts = [
                radar_track_counts[index]
                for index, row in enumerate(timeline_rows)
                if float(row.get("time_s", 0.0) or 0.0) >= float(first_activation_time) - 1e-9
            ]
        if first_activation_time is None:
            post_alive_counts = list(alive_enemy_counts)
        else:
            post_alive_counts = [
                alive_enemy_counts[index]
                for index, row in enumerate(timeline_rows)
                if float(row.get("time_s", 0.0) or 0.0) >= float(first_activation_time) - 1e-9
            ]
        if post_counts:
            metrics["radar_detection_coverage_ratio_post_activation"] = float(np.mean(post_counts) / total_target_count)
            metrics["radar_full_coverage_continuity_post_activation"] = float(
                np.mean([1.0 if count >= len(expected_targets) else 0.0 for count in post_counts])
            )
            metrics["radar_any_track_continuity_post_activation"] = float(
                np.mean([1.0 if count > 0 else 0.0 for count in post_counts])
            )
            dynamic_post = [
                min(float(track_count), float(alive_count)) / float(alive_count)
                for track_count, alive_count in zip(post_counts, post_alive_counts)
                if alive_count > 0
            ]
            metrics["radar_detection_coverage_ratio_alive_targets_post_activation"] = float(np.mean(dynamic_post)) if dynamic_post else 0.0
            metrics["radar_full_coverage_continuity_alive_targets_post_activation"] = float(
                np.mean(
                    [
                        1.0 if int(track_count) >= int(alive_count) else 0.0
                        for track_count, alive_count in zip(post_counts, post_alive_counts)
                        if alive_count > 0
                    ]
                )
            ) if post_alive_counts else 0.0

    first_detect_raw = getattr(patrol_task, "_first_radar_detection_time", {}) or {}
    per_target_best: List[Dict[str, object]] = []
    per_target_earliest: Dict[str, float] = {}
    detected_target_count_by_agent: Dict[str, int] = {aid: 0 for aid in ("A0100", "A0200", "A0300", "A0400")}

    for tid, detections in dict(first_detect_raw).items():
        target_id = str(tid or "")
        if not target_id.startswith("B"):
            continue
        detection_map = dict(detections or {})
        if not detection_map:
            continue

        earliest_time_s: Optional[float] = None
        best_delay_s: Optional[float] = None
        best_agent_id = ""
        valid_detecting_agents = set()

        for aid, detected_time in detection_map.items():
            agent_id = str(aid or "")
            if not agent_id.startswith("A"):
                continue
            try:
                detect_time_s = float(detected_time)
            except Exception:
                continue
            valid_detecting_agents.add(agent_id)
            if earliest_time_s is None or detect_time_s < earliest_time_s:
                earliest_time_s = detect_time_s

            activation_time_s = activation_times.get(agent_id)
            if activation_time_s is None:
                continue
            delay_s = max(0.2, float(detect_time_s - activation_time_s))
            if best_delay_s is None or delay_s < best_delay_s:
                best_delay_s = delay_s
                best_agent_id = agent_id

        for agent_id in valid_detecting_agents:
            detected_target_count_by_agent[agent_id] = int(detected_target_count_by_agent.get(agent_id, 0) + 1)

        if earliest_time_s is not None:
            per_target_earliest[target_id] = float(earliest_time_s)
        if best_delay_s is not None and best_agent_id:
            per_target_best.append(
                {
                    "target": target_id,
                    "delay_s": float(best_delay_s),
                    "agent": best_agent_id,
                    "detect_time_s": float(per_target_earliest.get(target_id, 0.0)),
                }
            )

    if detected_target_count_by_agent:
        metrics["radar_detected_target_count_by_agent"] = dict(
            sorted(detected_target_count_by_agent.items(), key=lambda item: item[0])
        )

    if per_target_earliest:
        metrics["radar_first_detect_time_by_target_s"] = dict(sorted(per_target_earliest.items(), key=lambda item: item[0]))
        metrics["radar_detected_target_total"] = int(len(per_target_earliest))
        if len(per_target_earliest) == len(expected_targets):
            full_detect_time_s = float(max(per_target_earliest.values()))
            metrics["full_detect_time_s"] = full_detect_time_s
            first_activation_time = metrics.get("radar_first_activation_time_s")
            if first_activation_time is not None:
                metrics["full_detect_delay_s"] = max(0.0, float(full_detect_time_s - float(first_activation_time)))

    if per_target_best:
        per_target_best = sorted(per_target_best, key=lambda item: str(item.get("target", "")))
        delay_values = [float(item["delay_s"]) for item in per_target_best]
        metrics["per_target_best_first_detect"] = per_target_best
        metrics["avg_first_detect_delay_s"] = float(np.mean(delay_values))
        if len(delay_values) >= 2:
            metrics["first_detect_delay_std_s"] = float(np.std(delay_values))

    return metrics


def _last_row_value(timeline_rows: List[Dict[str, object]], key: str, default=None):
    if not timeline_rows:
        return default
    value = timeline_rows[-1].get(key, default)
    return default if value is None else value


def _zone_metrics(
    timeline_rows: List[Dict[str, object]],
    durations: List[float],
    key: str,
) -> Dict[str, object]:
    peak_count = 0
    occupied_time_s = 0.0
    breach_events = 0
    first_time_s: Optional[float] = None
    prev_positive = False

    for row, dt in zip(timeline_rows, durations):
        count = int(row.get(key, 0) or 0)
        if count > peak_count:
            peak_count = count
        positive = count > 0
        if positive:
            occupied_time_s += float(dt)
            if first_time_s is None:
                first_time_s = float(row.get("time_s", 0.0) or 0.0)
            if not prev_positive:
                breach_events += 1
        prev_positive = positive

    return {
        "peak_count": int(peak_count),
        "occupied_time_s": float(occupied_time_s),
        "breach_events": int(breach_events),
        "first_time_s": first_time_s,
    }


def _duration_breakdown(
    timeline_rows: List[Dict[str, object]],
    durations: List[float],
    key: str,
    *,
    default: str = "UNKNOWN",
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for row, dt in zip(timeline_rows, durations):
        label = _norm_label(row.get(key), default=default)
        result[label] = float(result.get(label, 0.0) + float(dt))
    return result


def _exact_match_ratio(timeline_rows: List[Dict[str, object]], left_key: str, right_key: str) -> float:
    if not timeline_rows:
        return 0.0
    matches = 0
    for row in timeline_rows:
        if _safe_int(row.get(left_key), -999) == _safe_int(row.get(right_key), -998):
            matches += 1
    return float(matches) / float(len(timeline_rows))


def _all_exact_match_ratio(
    timeline_rows: List[Dict[str, object]],
    key_pairs: Iterable[Tuple[str, str]],
) -> float:
    pairs = list(key_pairs)
    if not timeline_rows or not pairs:
        return 0.0
    matches = 0
    for row in timeline_rows:
        if all(_safe_int(row.get(left_key), -999) == _safe_int(row.get(right_key), -998) for left_key, right_key in pairs):
            matches += 1
    return float(matches) / float(len(timeline_rows))


def _collect_intent_target_metrics(
    timeline_rows: List[Dict[str, object]],
    durations: List[float],
) -> Dict[str, object]:
    target_metrics: Dict[str, Dict[str, object]] = {}
    group_metrics: Dict[str, Dict[str, object]] = {}

    for target_id in ENEMY_AGENT_IDS:
        present_key = f"{target_id}_intent_present"
        class_key = f"{target_id}_intent_class"
        truth_class_key = f"{target_id}_intent_truth_class"
        threat_key = f"{target_id}_intent_threat"
        confidence_key = f"{target_id}_intent_confidence"
        ready_key = f"{target_id}_intent_model_ready"

        present_time_s = _duration_where(
            timeline_rows,
            durations,
            lambda row, key=class_key: bool(str(row.get(key, "") or "").strip()),
        )
        ready_time_s = _duration_where(
            timeline_rows,
            durations,
            lambda row, ckey=class_key, rkey=ready_key: bool(str(row.get(ckey, "") or "").strip())
            and int(row.get(rkey, 0) or 0) > 0,
        )
        truth_time_s = _duration_where(
            timeline_rows,
            durations,
            lambda row, key=truth_class_key: bool(str(row.get(key, "") or "").strip()),
        )
        truth_match_time_s = _duration_where(
            timeline_rows,
            durations,
            lambda row, pkey=class_key, tkey=truth_class_key: (
                bool(str(row.get(tkey, "") or "").strip())
                and str(row.get(pkey, "") or "").strip() == str(row.get(tkey, "") or "").strip()
            ),
        )
        ready_truth_time_s = _duration_where(
            timeline_rows,
            durations,
            lambda row, tkey=truth_class_key, rkey=ready_key: (
                bool(str(row.get(tkey, "") or "").strip())
                and int(row.get(rkey, 0) or 0) > 0
            ),
        )
        ready_truth_match_time_s = _duration_where(
            timeline_rows,
            durations,
            lambda row, pkey=class_key, tkey=truth_class_key, rkey=ready_key: (
                bool(str(row.get(tkey, "") or "").strip())
                and int(row.get(rkey, 0) or 0) > 0
                and str(row.get(pkey, "") or "").strip() == str(row.get(tkey, "") or "").strip()
            ),
        )
        attack_time_s = _duration_where(
            timeline_rows, durations, lambda row, key=class_key: str(row.get(key, "") or "").strip() == "攻击"
        )
        defense_time_s = _duration_where(
            timeline_rows, durations, lambda row, key=class_key: str(row.get(key, "") or "").strip() == "防御"
        )
        retreat_time_s = _duration_where(
            timeline_rows, durations, lambda row, key=class_key: str(row.get(key, "") or "").strip() == "撤退"
        )
        recon_time_s = _duration_where(
            timeline_rows, durations, lambda row, key=class_key: str(row.get(key, "") or "").strip() == "侦察"
        )
        high_time_s = _duration_where(
            timeline_rows,
            durations,
            lambda row, key=threat_key: str(row.get(key, "") or "").strip().lower() in {"high", "critical"},
        )
        confidence_values = [
            _safe_float(row.get(confidence_key), float("nan"))
            for row in timeline_rows
            if str(row.get(class_key, "") or "").strip()
        ]
        confidence_values = [value for value in confidence_values if np.isfinite(value)]
        target_metrics[target_id] = {
            "target_id": target_id,
            "first_present_time_s": _first_time_when(
                timeline_rows,
                class_key,
                lambda value: bool(str(value or "").strip()),
            ),
            "first_truth_time_s": _first_time_when(
                timeline_rows,
                truth_class_key,
                lambda value: bool(str(value or "").strip()),
            ),
            "first_ready_time_s": _first_time_matching_rows(
                timeline_rows,
                lambda row, ckey=class_key, rkey=ready_key: bool(str(row.get(ckey, "") or "").strip())
                and int(row.get(rkey, 0) or 0) > 0,
            ),
            "first_attack_time_s": _first_time_when(
                timeline_rows,
                class_key,
                lambda value: str(value or "").strip() == "攻击",
            ),
            "first_defense_time_s": _first_time_when(
                timeline_rows,
                class_key,
                lambda value: str(value or "").strip() == "防御",
            ),
            "first_retreat_time_s": _first_time_when(
                timeline_rows,
                class_key,
                lambda value: str(value or "").strip() == "撤退",
            ),
            "first_recon_time_s": _first_time_when(
                timeline_rows,
                class_key,
                lambda value: str(value or "").strip() == "侦察",
            ),
            "first_high_time_s": _first_time_when(
                timeline_rows,
                threat_key,
                lambda value: str(value or "").strip().lower() in {"high", "critical"},
            ),
            "peak_threat": _max_label_by_order(timeline_rows, threat_key, THREAT_PRIORITY, default="NONE"),
            "label_switch_count": _count_switches_nonempty(timeline_rows, class_key),
            "threat_switch_count": _count_switches_nonempty(timeline_rows, threat_key),
            "present_time_s": float(present_time_s),
            "ready_time_s": float(ready_time_s),
            "ready_ratio_in_present": (float(ready_time_s) / float(present_time_s) if present_time_s > 0 else 0.0),
            "truth_time_s": float(truth_time_s),
            "truth_match_time_s": float(truth_match_time_s),
            "truth_accuracy": (float(truth_match_time_s) / float(truth_time_s) if truth_time_s > 0 else 0.0),
            "ready_truth_time_s": float(ready_truth_time_s),
            "ready_truth_match_time_s": float(ready_truth_match_time_s),
            "ready_truth_accuracy": (
                float(ready_truth_match_time_s) / float(ready_truth_time_s) if ready_truth_time_s > 0 else 0.0
            ),
            "attack_time_s": float(attack_time_s),
            "defense_time_s": float(defense_time_s),
            "retreat_time_s": float(retreat_time_s),
            "recon_time_s": float(recon_time_s),
            "high_time_s": float(high_time_s),
            "confidence_peak": max(confidence_values) if confidence_values else None,
            "confidence_mean_present": (
                float(sum(confidence_values) / len(confidence_values)) if confidence_values else None
            ),
            "end_intent_name": str(_last_row_value(timeline_rows, class_key, "") or ""),
            "end_truth_name": str(_last_row_value(timeline_rows, truth_class_key, "") or ""),
            "end_intent_threat": str(_last_row_value(timeline_rows, threat_key, "") or ""),
        }

    for group_prefix in ENEMY_INTENT_GROUPS:
        class_key = f"{group_prefix}_intent_class"
        threat_key = f"{group_prefix}_intent_threat"
        ready_count_key = f"{group_prefix}_intent_ready_count"
        contact_count_key = f"{group_prefix}_intent_contact_count"
        classified_count_key = f"{group_prefix}_intent_classified_count"
        attack_count_key = f"{group_prefix}_intent_attack_count"
        defense_count_key = f"{group_prefix}_intent_defense_count"
        retreat_count_key = f"{group_prefix}_intent_retreat_count"
        recon_count_key = f"{group_prefix}_intent_recon_count"
        high_count_key = f"{group_prefix}_intent_high_count"
        group_metrics[group_prefix] = {
            "group_name": group_prefix,
            "first_present_time_s": _first_time_when(
                timeline_rows, classified_count_key, lambda value: int(value or 0) > 0
            ),
            "first_ready_time_s": _first_time_when(
                timeline_rows, ready_count_key, lambda value: int(value or 0) > 0
            ),
            "first_attack_time_s": _first_time_when(
                timeline_rows, attack_count_key, lambda value: int(value or 0) > 0
            ),
            "first_defense_time_s": _first_time_when(
                timeline_rows, defense_count_key, lambda value: int(value or 0) > 0
            ),
            "first_retreat_time_s": _first_time_when(
                timeline_rows, retreat_count_key, lambda value: int(value or 0) > 0
            ),
            "first_recon_time_s": _first_time_when(
                timeline_rows, recon_count_key, lambda value: int(value or 0) > 0
            ),
            "first_high_time_s": _first_time_when(
                timeline_rows, high_count_key, lambda value: int(value or 0) > 0
            ),
            "dominant_switch_count": _count_switches_nonempty(timeline_rows, class_key),
            "peak_threat": _max_label_by_order(timeline_rows, threat_key, THREAT_PRIORITY, default="NONE"),
            "contact_time_s": _duration_where(
                timeline_rows, durations, lambda row, key=contact_count_key: int(row.get(key, 0) or 0) > 0
            ),
            "classified_time_s": _duration_where(
                timeline_rows, durations, lambda row, key=classified_count_key: int(row.get(key, 0) or 0) > 0
            ),
            "ready_time_s": _duration_where(
                timeline_rows, durations, lambda row, key=ready_count_key: int(row.get(key, 0) or 0) > 0
            ),
            "attack_time_s": _duration_where(
                timeline_rows, durations, lambda row, key=attack_count_key: int(row.get(key, 0) or 0) > 0
            ),
            "defense_time_s": _duration_where(
                timeline_rows, durations, lambda row, key=defense_count_key: int(row.get(key, 0) or 0) > 0
            ),
            "retreat_time_s": _duration_where(
                timeline_rows, durations, lambda row, key=retreat_count_key: int(row.get(key, 0) or 0) > 0
            ),
            "recon_time_s": _duration_where(
                timeline_rows, durations, lambda row, key=recon_count_key: int(row.get(key, 0) or 0) > 0
            ),
            "high_time_s": _duration_where(
                timeline_rows, durations, lambda row, key=high_count_key: int(row.get(key, 0) or 0) > 0
            ),
            "end_intent_name": str(_last_row_value(timeline_rows, class_key, "") or ""),
            "end_intent_threat": str(_last_row_value(timeline_rows, threat_key, "") or ""),
        }

    truth_time_total = float(sum(float(metrics.get("truth_time_s", 0.0) or 0.0) for metrics in target_metrics.values()))
    truth_match_time_total = float(
        sum(float(metrics.get("truth_match_time_s", 0.0) or 0.0) for metrics in target_metrics.values())
    )
    ready_truth_time_total = float(
        sum(float(metrics.get("ready_truth_time_s", 0.0) or 0.0) for metrics in target_metrics.values())
    )
    ready_truth_match_time_total = float(
        sum(float(metrics.get("ready_truth_match_time_s", 0.0) or 0.0) for metrics in target_metrics.values())
    )
    return {
        "intent_target_metrics": target_metrics,
        "intent_group_metrics": group_metrics,
        "intent_truth_time_s": truth_time_total,
        "intent_truth_match_time_s": truth_match_time_total,
        "intent_truth_ready_time_s": ready_truth_time_total,
        "intent_truth_ready_match_time_s": ready_truth_match_time_total,
        "intent_truth_accuracy": (truth_match_time_total / truth_time_total if truth_time_total > 0 else 0.0),
        "intent_truth_ready_accuracy": (
            ready_truth_match_time_total / ready_truth_time_total if ready_truth_time_total > 0 else 0.0
        ),
        "intent_total_label_switch_count": int(
            sum(int(metrics.get("label_switch_count", 0) or 0) for metrics in target_metrics.values())
        ),
        "intent_total_threat_switch_count": int(
            sum(int(metrics.get("threat_switch_count", 0) or 0) for metrics in target_metrics.values())
        ),
    }


def _max_label_by_order(
    timeline_rows: List[Dict[str, object]],
    key: str,
    order: Dict[str, int],
    *,
    default: str = "",
) -> str:
    best_label = default
    best_rank = -1
    order_map = {str(label).lower(): int(rank) for label, rank in order.items()}
    for row in timeline_rows:
        label = _norm_label(row.get(key), default=default)
        rank = order_map.get(label.lower(), -1)
        if rank > best_rank:
            best_rank = rank
            best_label = label
    return best_label


def _scenario_summary(
    spec: Ch6ScenarioSpec,
    patrol_task: CAPTask,
    timeline_rows: List[Dict[str, object]],
    event_rows: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    durations = _row_durations(timeline_rows)
    cooperative_detection_metrics = _collect_ch6_cooperative_detection_metrics(patrol_task, timeline_rows)
    control_distance_metrics = _collect_control_distance_metrics(timeline_rows)
    intent_metrics = _collect_intent_target_metrics(timeline_rows, durations)
    low_metrics = _zone_metrics(timeline_rows, durations, "truth_low_count")
    medium_metrics = _zone_metrics(timeline_rows, durations, "truth_medium_count")
    high_metrics = _zone_metrics(timeline_rows, durations, "truth_high_count")
    picture_low_metrics = _zone_metrics(timeline_rows, durations, "picture_low_count")
    picture_medium_metrics = _zone_metrics(timeline_rows, durations, "picture_medium_count")
    picture_high_metrics = _zone_metrics(timeline_rows, durations, "picture_high_count")

    zone_key_map = {
        "LOW": "truth_low_count",
        "MEDIUM": "truth_medium_count",
        "HIGH": "truth_high_count",
    }
    picture_zone_key_map = {
        "LOW": "picture_low_count",
        "MEDIUM": "picture_medium_count",
        "HIGH": "picture_high_count",
    }
    expected_zone_key = zone_key_map.get(str(spec.expected_zone).upper(), "")
    picture_expected_zone_key = picture_zone_key_map.get(str(spec.expected_zone).upper(), "")
    expected_zone_metrics = (
        _zone_metrics(timeline_rows, durations, expected_zone_key)
        if expected_zone_key
        else {"peak_count": 0, "occupied_time_s": 0.0, "breach_events": 0, "first_time_s": None}
    )
    picture_expected_zone_metrics = (
        _zone_metrics(timeline_rows, durations, picture_expected_zone_key)
        if picture_expected_zone_key
        else {"peak_count": 0, "occupied_time_s": 0.0, "breach_events": 0, "first_time_s": None}
    )

    stable_peak = int(_safe_max_float(timeline_rows, "stable_ready_count") or 0)
    stable_tracking_target_peak = int(_safe_max_float(timeline_rows, "stable_tracking_target_count") or 0)
    guided_peak = int(_safe_max_float(timeline_rows, "active_guided_missile_peak") or 0)
    detect_mode_time_s = _duration_breakdown(timeline_rows, durations, "detect_mode")
    cap_state_time_s = _duration_breakdown(timeline_rows, durations, "cap_state")
    mission_threat_time_s = _duration_breakdown(timeline_rows, durations, "mission_threat_level", default="NONE")
    intent_threat_time_s = _duration_breakdown(timeline_rows, durations, "intent_threat", default="NONE")
    enemy_script_phase_time_s = _duration_breakdown(timeline_rows, durations, "enemy_script_phase_name", default="NONE")
    left_tactic_time_s = _duration_breakdown(timeline_rows, durations, "left_tactic")
    right_tactic_time_s = _duration_breakdown(timeline_rows, durations, "right_tactic")
    left_phase_time_s = _duration_breakdown(timeline_rows, durations, "left_phase")
    right_phase_time_s = _duration_breakdown(timeline_rows, durations, "right_phase")
    left_target_time_s = _duration_breakdown(timeline_rows, durations, "left_target", default="-")
    right_target_time_s = _duration_breakdown(timeline_rows, durations, "right_target", default="-")
    enemy_awacs_available_time_s = sum(
        float(dt) for row, dt in zip(timeline_rows, durations) if int(row.get("enemy_script_awacs_available", 0) or 0) > 0
    )
    enemy_awacs_denied_time_s = sum(
        float(dt) for row, dt in zip(timeline_rows, durations) if int(row.get("enemy_script_awacs_available", 0) or 0) <= 0
    )

    left_expected_rows = sum(
        1 for row in timeline_rows if _matches_expected_tactic(str(row.get("left_tactic", "") or ""), spec.expected_tactic_keywords)
    )
    right_expected_rows = sum(
        1 for row in timeline_rows if _matches_expected_tactic(str(row.get("right_tactic", "") or ""), spec.expected_tactic_keywords)
    )
    expected_tactic_rows = max(left_expected_rows, right_expected_rows)
    expected_tactic_time_s = 0.0
    for row, dt in zip(timeline_rows, durations):
        if (
            _matches_expected_tactic(str(row.get("left_tactic", "") or ""), spec.expected_tactic_keywords)
            or _matches_expected_tactic(str(row.get("right_tactic", "") or ""), spec.expected_tactic_keywords)
        ):
            expected_tactic_time_s += float(dt)
    first_expected_tactic_time_s = None
    for row in timeline_rows:
        if (
            _matches_expected_tactic(str(row.get("left_tactic", "") or ""), spec.expected_tactic_keywords)
            or _matches_expected_tactic(str(row.get("right_tactic", "") or ""), spec.expected_tactic_keywords)
        ):
            first_expected_tactic_time_s = float(row.get("time_s", 0.0) or 0.0)
            break

    mission_result = str(_last_row_value(timeline_rows, "mission_result", "") or "")
    mission_threat_level = str(_last_row_value(timeline_rows, "mission_threat_level", "") or "")
    friendly_alive_end = int(_last_row_value(timeline_rows, "friendly_alive", 0) or 0)
    enemy_alive_end = int(_last_row_value(timeline_rows, "enemy_alive", 0) or 0)
    friendly_loss_end = int(_last_row_value(timeline_rows, "friendly_loss_count", max(0, 4 - friendly_alive_end)) or 0)
    enemy_kill_end = int(_last_row_value(timeline_rows, "enemy_kill_count", max(0, 4 - enemy_alive_end)) or 0)
    friendly_min_altitude_m = _safe_min_float(timeline_rows, "friendly_min_altitude_m")
    enemy_min_altitude_m = _safe_min_float(timeline_rows, "enemy_min_altitude_m")
    friendly_min_speed_mps = _safe_min_float(timeline_rows, "friendly_min_speed_mps")
    enemy_min_speed_mps = _safe_min_float(timeline_rows, "enemy_min_speed_mps")
    friendly_max_descent_rate_mps = _safe_max_float(timeline_rows, "friendly_max_descent_rate_mps")
    enemy_max_descent_rate_mps = _safe_max_float(timeline_rows, "enemy_max_descent_rate_mps")
    friendly_sim_recreate_total = int(_last_row_value(timeline_rows, "friendly_sim_recreate_total", 0) or 0)
    enemy_sim_recreate_total = int(_last_row_value(timeline_rows, "enemy_sim_recreate_total", 0) or 0)
    try:
        friendly_sim_recreate_by_agent = json.loads(
            str(_last_row_value(timeline_rows, "friendly_sim_recreate_by_agent_json", "{}") or "{}")
        )
    except Exception:
        friendly_sim_recreate_by_agent = {}
    try:
        enemy_sim_recreate_by_agent = json.loads(
            str(_last_row_value(timeline_rows, "enemy_sim_recreate_by_agent_json", "{}") or "{}")
        )
    except Exception:
        enemy_sim_recreate_by_agent = {}

    event_type_counts: Dict[str, int] = defaultdict(int)
    event_field_counts: Dict[str, int] = defaultdict(int)
    for event in list(event_rows or []):
        event_type = str(event.get("event_type", "") or "")
        field = str(event.get("field", "") or "")
        if event_type:
            event_type_counts[event_type] += 1
        if event_type or field:
            event_field_counts[f"{event_type}::{field}"] += 1

    threat_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

    summary = {
        "scenario_id": spec.scenario_id,
        "title": spec.title,
        "thesis_focus": spec.thesis_focus,
        "design_intent": spec.design_intent,
        "control_contract": list(spec.control_contract),
        "enemy_control_mode": spec.enemy_control_mode,
        "expected_zone": spec.expected_zone,
        "expected_tactic_keywords": list(spec.expected_tactic_keywords),
        "time_s": float(_last_row_value(timeline_rows, "time_s", 0.0) or 0.0),
        "step_count": len(timeline_rows),
        "friendly_alive_end": friendly_alive_end,
        "enemy_alive_end": enemy_alive_end,
        "friendly_loss_count": friendly_loss_end,
        "enemy_kill_count": enemy_kill_end,
        "friendly_missiles_left_end": int(_last_row_value(timeline_rows, "friendly_missiles_left", 0) or 0),
        "friendly_min_altitude_m": friendly_min_altitude_m,
        "enemy_min_altitude_m": enemy_min_altitude_m,
        "friendly_min_speed_mps": friendly_min_speed_mps,
        "enemy_min_speed_mps": enemy_min_speed_mps,
        "friendly_max_descent_rate_mps": friendly_max_descent_rate_mps,
        "enemy_max_descent_rate_mps": enemy_max_descent_rate_mps,
        "friendly_sim_recreate_total": friendly_sim_recreate_total,
        "enemy_sim_recreate_total": enemy_sim_recreate_total,
        "friendly_sim_recreate_by_agent": friendly_sim_recreate_by_agent,
        "enemy_sim_recreate_by_agent": enemy_sim_recreate_by_agent,
        "mission_result": mission_result,
        "mission_threat_level": mission_threat_level,
        "mission_threat_peak": _max_label_by_order(timeline_rows, "mission_threat_level", threat_order, default="NONE"),
        "mission_runtime_kills": int(_last_row_value(timeline_rows, "mission_runtime_kills", 0) or 0),
        "mission_total_threats_end": int(_last_row_value(timeline_rows, "mission_total_threats", 0) or 0),
        "mission_total_threats_peak": int(_safe_max_float(timeline_rows, "mission_total_threats") or 0),
        "mission_high_zone_threats_end": int(_last_row_value(timeline_rows, "mission_high_zone_threats", 0) or 0),
        "mission_high_zone_threats_peak": int(_safe_max_float(timeline_rows, "mission_high_zone_threats") or 0),
        "mission_medium_zone_threats_end": int(_last_row_value(timeline_rows, "mission_medium_zone_threats", 0) or 0),
        "mission_medium_zone_threats_peak": int(_safe_max_float(timeline_rows, "mission_medium_zone_threats") or 0),
        "mission_friendly_lost_end": int(_last_row_value(timeline_rows, "mission_friendly_lost", 0) or 0),
        "mission_missiles_fired_end": int(_last_row_value(timeline_rows, "mission_missiles_fired", 0) or 0),
        "mission_high_zone_breach_events_end": int(_last_row_value(timeline_rows, "mission_high_zone_breach_events", 0) or 0),
        "awacs_track_count_end": int(_last_row_value(timeline_rows, "awacs_track_count", 0) or 0),
        "radar_track_count_end": int(_last_row_value(timeline_rows, "radar_track_count", 0) or 0),
        "awacs_track_peak": int(_safe_max_float(timeline_rows, "awacs_track_count") or 0),
        "radar_track_peak": int(_safe_max_float(timeline_rows, "radar_track_count") or 0),
        "stable_ready_peak": int(stable_peak),
        "stable_tracking_target_peak": int(stable_tracking_target_peak),
        "first_stable_ready_time_s": _first_time_when(timeline_rows, "stable_ready_count", lambda value: int(value or 0) > 0),
        "first_awacs_track_time_s": _first_time_when(timeline_rows, "awacs_track_count", lambda value: int(value or 0) > 0),
        "first_radar_track_time_s": _first_time_when(timeline_rows, "radar_track_count", lambda value: int(value or 0) > 0),
        "first_scan_assignment_time_s": _first_time_when(
            timeline_rows, "scan_assignment_count", lambda value: int(value or 0) > 0
        ),
        "first_scan_multi_target_time_s": _first_time_when(
            timeline_rows, "scan_target_count", lambda value: int(value or 0) > 1
        ),
        "detect_mode_switch_count": _count_switches(timeline_rows, "detect_mode"),
        "cap_state_switch_count": _count_switches(timeline_rows, "cap_state"),
        "enemy_script_phase_switch_count": _count_switches(timeline_rows, "enemy_script_phase_name"),
        "left_tactic_switch_count": _count_switches(timeline_rows, "left_tactic"),
        "right_tactic_switch_count": _count_switches(timeline_rows, "right_tactic"),
        "left_phase_switch_count": _count_switches(timeline_rows, "left_phase"),
        "right_phase_switch_count": _count_switches(timeline_rows, "right_phase"),
        "left_target_switch_count": _count_switches(timeline_rows, "left_target"),
        "right_target_switch_count": _count_switches(timeline_rows, "right_target"),
        "A0100_scan_target_switch_count": _count_switches(timeline_rows, "A0100_scan_target"),
        "A0200_scan_target_switch_count": _count_switches(timeline_rows, "A0200_scan_target"),
        "A0300_scan_target_switch_count": _count_switches(timeline_rows, "A0300_scan_target"),
        "A0400_scan_target_switch_count": _count_switches(timeline_rows, "A0400_scan_target"),
        "cap_state_time_s": cap_state_time_s,
        "detect_mode_time_s": detect_mode_time_s,
        "mission_threat_time_s": mission_threat_time_s,
        "intent_threat_time_s": intent_threat_time_s,
        "enemy_script_phase_time_s": enemy_script_phase_time_s,
        "left_tactic_time_s": left_tactic_time_s,
        "right_tactic_time_s": right_tactic_time_s,
        "left_phase_time_s": left_phase_time_s,
        "right_phase_time_s": right_phase_time_s,
        "left_target_time_s": left_target_time_s,
        "right_target_time_s": right_target_time_s,
        "enemy_script_family": str(_last_row_value(timeline_rows, "enemy_script_family", "") or ""),
        "enemy_script_phase_end": str(_last_row_value(timeline_rows, "enemy_script_phase_name", "") or ""),
        "enemy_script_phase_end_id": _safe_int(_last_row_value(timeline_rows, "enemy_script_phase_id", -1), -1),
        "enemy_script_phase_count": len(list(getattr(spec.scenario, "get_phase_catalog", lambda: [])() or [])),
        "enemy_script_phase_catalog": list(getattr(spec.scenario, "get_phase_catalog", lambda: [])() or []),
        "enemy_awacs_available_time_s": float(enemy_awacs_available_time_s),
        "enemy_awacs_denied_time_s": float(enemy_awacs_denied_time_s),
        "enemy_awacs_denied_ratio": (
            float(enemy_awacs_denied_time_s) / float(sum(durations)) if durations else 0.0
        ),
        "first_enemy_phase_change_time_s": _first_time_when(
            timeline_rows, "enemy_script_phase_id", lambda value: int(value or 0) > 0
        ),
        "left_expected_tactic_rows": int(left_expected_rows),
        "right_expected_tactic_rows": int(right_expected_rows),
        "expected_tactic_rows": int(expected_tactic_rows),
        "expected_tactic_time_s": float(expected_tactic_time_s),
        "expected_tactic_observed": bool(expected_tactic_rows > 0),
        "first_expected_tactic_time_s": first_expected_tactic_time_s,
        "low_risk_peak_count": int(low_metrics["peak_count"]),
        "low_risk_breach_events": int(low_metrics["breach_events"]),
        "low_risk_enemy_time_s": float(low_metrics["occupied_time_s"]),
        "first_low_time_s": low_metrics["first_time_s"],
        "medium_risk_peak_count": int(medium_metrics["peak_count"]),
        "medium_risk_breach_events": int(medium_metrics["breach_events"]),
        "medium_risk_enemy_time_s": float(medium_metrics["occupied_time_s"]),
        "first_medium_time_s": medium_metrics["first_time_s"],
        "high_risk_peak_count": int(high_metrics["peak_count"]),
        "high_risk_breach_events": int(high_metrics["breach_events"]),
        "high_risk_breach_time_s": float(high_metrics["occupied_time_s"]),
        "first_high_time_s": high_metrics["first_time_s"],
        "picture_low_risk_peak_count": int(picture_low_metrics["peak_count"]),
        "picture_low_risk_enemy_time_s": float(picture_low_metrics["occupied_time_s"]),
        "picture_low_first_time_s": picture_low_metrics["first_time_s"],
        "picture_medium_risk_peak_count": int(picture_medium_metrics["peak_count"]),
        "picture_medium_risk_enemy_time_s": float(picture_medium_metrics["occupied_time_s"]),
        "picture_medium_first_time_s": picture_medium_metrics["first_time_s"],
        "picture_high_risk_peak_count": int(picture_high_metrics["peak_count"]),
        "picture_high_risk_enemy_time_s": float(picture_high_metrics["occupied_time_s"]),
        "picture_high_breach_events": int(picture_high_metrics["breach_events"]),
        "picture_high_first_time_s": picture_high_metrics["first_time_s"],
        "expected_zone_peak_count": int(expected_zone_metrics["peak_count"]),
        "expected_zone_enemy_time_s": float(expected_zone_metrics["occupied_time_s"]),
        "expected_zone_breach_events": int(expected_zone_metrics["breach_events"]),
        "expected_zone_triggered": bool(expected_zone_metrics["peak_count"]),
        "expected_zone_first_time_s": expected_zone_metrics["first_time_s"],
        "picture_expected_zone_peak_count": int(picture_expected_zone_metrics["peak_count"]),
        "picture_expected_zone_enemy_time_s": float(picture_expected_zone_metrics["occupied_time_s"]),
        "picture_expected_zone_breach_events": int(picture_expected_zone_metrics["breach_events"]),
        "picture_expected_zone_triggered": bool(picture_expected_zone_metrics["peak_count"]),
        "picture_expected_zone_first_time_s": picture_expected_zone_metrics["first_time_s"],
        "low_risk_picture_match_ratio": _exact_match_ratio(timeline_rows, "truth_low_count", "picture_low_count"),
        "medium_risk_picture_match_ratio": _exact_match_ratio(timeline_rows, "truth_medium_count", "picture_medium_count"),
        "high_risk_picture_match_ratio": _exact_match_ratio(timeline_rows, "truth_high_count", "picture_high_count"),
        "risk_picture_exact_match_ratio": _all_exact_match_ratio(
            timeline_rows,
            (
                ("truth_low_count", "picture_low_count"),
                ("truth_medium_count", "picture_medium_count"),
                ("truth_high_count", "picture_high_count"),
            ),
        ),
        "expected_zone_picture_match_ratio": (
            _exact_match_ratio(timeline_rows, expected_zone_key, picture_expected_zone_key)
            if expected_zone_key and picture_expected_zone_key
            else 0.0
        ),
        "min_enemy_bullseye_km": _safe_min_float(timeline_rows, "nearest_enemy_bullseye_km"),
        "min_enemy_to_friendly_km": _safe_min_float(timeline_rows, "nearest_enemy_to_friendly_km"),
        "picture_total_peak": int(_safe_max_float(timeline_rows, "picture_total_count") or 0),
        "search_picture_total_peak": int(_safe_max_float(timeline_rows, "search_picture_total_count") or 0),
        "scan_assignment_peak": int(_safe_max_float(timeline_rows, "scan_assignment_count") or 0),
        "scan_target_peak": int(_safe_max_float(timeline_rows, "scan_target_count") or 0),
        "assignment_target_peak": int(_safe_max_float(timeline_rows, "assignment_target_count") or 0),
        "scan_coverage_peak_deg": _safe_max_float(timeline_rows, "scan_coverage_total_deg"),
        "scan_coverage_mean_deg": _safe_mean_float(timeline_rows, "scan_coverage_total_deg"),
        "scan_overlap_peak_deg": _safe_max_float(timeline_rows, "scan_overlap_deg"),
        "scan_overlap_mean_deg": _safe_mean_float(timeline_rows, "scan_overlap_deg"),
        "confidence_radius_peak_km": _safe_max_float(timeline_rows, "confidence_radius_km"),
        "confidence_radius_mean_km": _safe_mean_float(timeline_rows, "confidence_radius_km"),
        "gate_pass_count": int(_last_row_value(timeline_rows, "gate_pass_count", 0) or 0),
        "gate_block_count": int(_last_row_value(timeline_rows, "gate_block_count", 0) or 0),
        "gate_total_request_count": int(_last_row_value(timeline_rows, "gate_total_request_count", 0) or 0),
        "relay_attempt_count": int(_last_row_value(timeline_rows, "relay_attempt_count", 0) or 0),
        "relay_success_count": int(_last_row_value(timeline_rows, "relay_success_count", 0) or 0),
        "relay_fail_count": int(_last_row_value(timeline_rows, "relay_fail_count", 0) or 0),
        "relay_active_time_s": sum(float(dt) for row, dt in zip(timeline_rows, durations) if int(row.get("active_relay_flag", 0) or 0) > 0),
        "active_guided_missile_peak": int(guided_peak),
        "missile_launch_count": int(_last_row_value(timeline_rows, "missile_launch_count", 0) or 0),
        "missile_outcome_count": int(_last_row_value(timeline_rows, "missile_outcome_count", 0) or 0),
        "missile_state_ready_peak": int(_safe_max_float(timeline_rows, "missile_state_ready") or 0),
        "missile_state_launched_peak": int(_safe_max_float(timeline_rows, "missile_state_launched") or 0),
        "missile_state_guiding_peak": int(_safe_max_float(timeline_rows, "missile_state_guiding") or 0),
        "missile_state_terminal_peak": int(_safe_max_float(timeline_rows, "missile_state_terminal") or 0),
        "missile_state_hit_peak": int(_safe_max_float(timeline_rows, "missile_state_hit") or 0),
        "missile_state_miss_peak": int(_safe_max_float(timeline_rows, "missile_state_miss") or 0),
        "first_gate_pass_time_s": _first_time_when(timeline_rows, "gate_pass_count", lambda value: int(value or 0) > 0),
        "first_relay_success_time_s": _first_time_when(
            timeline_rows, "relay_success_count", lambda value: int(value or 0) > 0
        ),
        "first_missile_launch_time_s": _first_time_when(
            timeline_rows, "missile_launch_count", lambda value: int(value or 0) > 0
        ),
        "first_enemy_kill_time_s": _first_time_when(timeline_rows, "enemy_kill_count", lambda value: int(value or 0) > 0),
        "first_friendly_loss_time_s": _first_time_when(
            timeline_rows, "friendly_loss_count", lambda value: int(value or 0) > 0
        ),
        "intent_target_end": str(_last_row_value(timeline_rows, "intent_target", "") or ""),
        "intent_name_end": str(_last_row_value(timeline_rows, "intent_name", "") or ""),
        "intent_threat_end": str(_last_row_value(timeline_rows, "intent_threat", "") or ""),
        "intent_confidence_end": float(_last_row_value(timeline_rows, "intent_confidence", 0.0) or 0.0),
        "intent_threat_peak": _max_label_by_order(timeline_rows, "intent_threat", threat_order, default="NONE"),
        "intent_contact_peak": int(_safe_max_float(timeline_rows, "intent_contact_count") or 0),
        "intent_classified_peak": int(_safe_max_float(timeline_rows, "intent_classified_count") or 0),
        "intent_unclassified_peak": int(_safe_max_float(timeline_rows, "intent_unclassified_count") or 0),
        "intent_ready_peak": int(_safe_max_float(timeline_rows, "intent_ready_count") or 0),
        "intent_confidence_peak": _safe_max_float(timeline_rows, "intent_confidence"),
        "intent_confidence_mean": _safe_mean_float(timeline_rows, "intent_confidence"),
        "intent_attack_peak": int(_safe_max_float(timeline_rows, "intent_attack_count") or 0),
        "intent_defense_peak": int(_safe_max_float(timeline_rows, "intent_defense_count") or 0),
        "intent_retreat_peak": int(_safe_max_float(timeline_rows, "intent_retreat_count") or 0),
        "intent_recon_peak": int(_safe_max_float(timeline_rows, "intent_recon_count") or 0),
        "intent_high_target_peak": int(_safe_max_float(timeline_rows, "intent_high_count") or 0),
        "intent_critical_target_peak": int(_safe_max_float(timeline_rows, "intent_critical_count") or 0),
        "first_intent_contact_time_s": _first_time_when(
            timeline_rows, "intent_contact_count", lambda value: int(value or 0) > 0
        ),
        "first_intent_time_s": _first_time_when(
            timeline_rows, "intent_classified_count", lambda value: int(value or 0) > 0
        ),
        "first_intent_ready_time_s": _first_time_when(
            timeline_rows, "intent_ready_count", lambda value: int(value or 0) > 0
        ),
        "first_intent_full_ready_time_s": _first_time_when(
            timeline_rows, "intent_ready_count", lambda value: int(value or 0) >= len(ENEMY_AGENT_IDS)
        ),
        "first_intent_high_time_s": _first_time_when(
            timeline_rows,
            "intent_threat",
            lambda value: str(value or "").strip().lower() in {"high", "critical"},
        ),
        "first_mission_high_time_s": _first_time_when(
            timeline_rows,
            "mission_threat_level",
            lambda value: str(value or "").strip().lower() in {"high", "critical"},
        ),
        "event_count_total": int(len(event_rows or [])),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "event_field_counts": dict(sorted(event_field_counts.items())),
    }
    summary.update(cooperative_detection_metrics)
    summary.update(control_distance_metrics)
    summary.update(intent_metrics)
    summary["enemy_zone_profiles"] = _enemy_zone_profile_summary(timeline_rows, getattr(patrol_task, "faor", None))
    summary["intent_evade_peak"] = summary.get("intent_defense_peak", 0)

    radar_detect_map = dict(summary.get("radar_first_detect_time_by_target_s", {}) or {})
    if patrol_task is not None:
        summary["target_first_awacs_time_s"] = dict(sorted((getattr(patrol_task, "_target_first_awacs_time", {}) or {}).items()))
        summary["target_first_fcr_time_s"] = dict(sorted((getattr(patrol_task, "_target_first_fcr_time", {}) or {}).items()))
        summary["target_first_ready_time_s"] = dict(sorted((getattr(patrol_task, "_target_first_ready_time", {}) or {}).items()))
        summary["target_zone_transition_history"] = {
            str(target_id): list(history or [])
            for target_id, history in dict(getattr(patrol_task, "_zone_transition_history", {}) or {}).items()
        }
        guidance_snapshot = {}
        if hasattr(patrol_task, "get_guidance_verification_snapshot"):
            guidance_snapshot = patrol_task.get_guidance_verification_snapshot() or {}
        if guidance_snapshot:
            summary["guidance_snapshot"] = guidance_snapshot
            summary["gate_pass_raw_count"] = int(guidance_snapshot.get("prelaunch_gate_pass_count", 0))
            summary["gate_block_raw_count"] = int(guidance_snapshot.get("prelaunch_gate_block_count", 0))
            summary["gate_pass_unique_count"] = int(guidance_snapshot.get("prelaunch_gate_unique_pass_count", summary["gate_pass_raw_count"]))
            summary["gate_block_unique_count"] = int(guidance_snapshot.get("prelaunch_gate_unique_block_count", summary["gate_block_raw_count"]))
            summary["gate_total_request_raw_count"] = int(guidance_snapshot.get("prelaunch_gate_raw_total_requests", summary["gate_pass_raw_count"] + summary["gate_block_raw_count"]))
            summary["gate_total_request_unique_count"] = int(guidance_snapshot.get("prelaunch_gate_unique_total_requests", summary["gate_pass_unique_count"] + summary["gate_block_unique_count"]))
            summary["gate_pass_rate_raw"] = float(guidance_snapshot.get("prelaunch_gate_raw_pass_rate", 0.0))
            summary["gate_pass_rate_unique"] = float(guidance_snapshot.get("prelaunch_gate_unique_pass_rate", 0.0))
            summary["gate_pass_rate"] = float(guidance_snapshot.get("prelaunch_gate_unique_pass_rate", guidance_snapshot.get("prelaunch_gate_raw_pass_rate", 0.0)))
            summary["gate_ready_pass_count"] = int(guidance_snapshot.get("prelaunch_gate_ready_unique_pass_count", guidance_snapshot.get("prelaunch_gate_ready_pass_count", 0)))
            summary["gate_ready_block_count"] = int(guidance_snapshot.get("prelaunch_gate_ready_unique_block_count", guidance_snapshot.get("prelaunch_gate_ready_block_count", 0)))
            summary["gate_ready_total_request_count"] = int(guidance_snapshot.get("prelaunch_gate_ready_unique_total_requests", guidance_snapshot.get("prelaunch_gate_ready_total_requests", 0)))
            summary["gate_ready_pass_rate"] = float(guidance_snapshot.get("prelaunch_gate_ready_pass_rate", 0.0))
            summary["gate_ready_pass_rate_raw"] = float(guidance_snapshot.get("prelaunch_gate_ready_pass_rate_raw", 0.0))
            summary["gate_pass_per_launch_rate"] = float(guidance_snapshot.get("gate_pass_per_launch_rate", 0.0))
            summary["gate_ready_pass_per_launch_rate"] = float(guidance_snapshot.get("gate_ready_pass_per_launch_rate", 0.0))
            summary["gate_block_stable_not_ready_count"] = int(guidance_snapshot.get("prelaunch_gate_block_stable_not_ready_count", 0))
            summary["gate_block_friendly_safe_count"] = int(guidance_snapshot.get("prelaunch_gate_block_friendly_safe_count", 0))
            summary["gate_block_long_shot_count"] = int(guidance_snapshot.get("prelaunch_gate_block_long_shot_count", 0))
            summary["gate_block_launch_exec_fail_count"] = int(guidance_snapshot.get("prelaunch_gate_block_launch_exec_fail_count", 0))
            summary["gate_block_other_count"] = int(guidance_snapshot.get("prelaunch_gate_block_other_count", 0))
            summary["relay_attempt_raw_count"] = int(guidance_snapshot.get("relay_attempt_count", 0))
            summary["relay_success_raw_count"] = int(guidance_snapshot.get("relay_success_count", 0))
            summary["relay_attempt_unique_count"] = int(guidance_snapshot.get("relay_unique_attempt_count", summary["relay_attempt_raw_count"]))
            summary["relay_success_unique_count"] = int(guidance_snapshot.get("relay_unique_success_count", summary["relay_success_raw_count"]))
            summary["relay_success_rate_raw"] = float(guidance_snapshot.get("relay_raw_success_rate", 0.0))
            summary["relay_success_rate_unique"] = float(guidance_snapshot.get("relay_unique_success_rate", 0.0))
            summary["relay_success_rate"] = float(guidance_snapshot.get("relay_success_rate", 0.0))
            summary["relay_success_per_launch_rate"] = float(guidance_snapshot.get("relay_success_per_launch_rate", 0.0))
            summary["gate_pass_count"] = int(summary["gate_pass_unique_count"])
            summary["gate_block_count"] = int(summary["gate_block_unique_count"])
            summary["gate_total_request_count"] = int(summary["gate_total_request_unique_count"])
            summary["relay_attempt_count"] = int(summary["relay_attempt_unique_count"])
            summary["relay_success_count"] = int(summary["relay_success_unique_count"])
        else:
            relay_attempts = int(summary["relay_attempt_count"])
            gate_total = int(summary["gate_pass_count"]) + int(summary["gate_block_count"])
            summary["relay_success_rate"] = (
                float(summary["relay_success_count"]) / float(relay_attempts) if relay_attempts > 0 else 0.0
            )
            summary["gate_pass_rate"] = (
                float(summary["gate_pass_count"]) / float(gate_total) if gate_total > 0 else 0.0
            )
            summary["gate_pass_raw_count"] = int(summary["gate_pass_count"])
            summary["gate_block_raw_count"] = int(summary["gate_block_count"])
            summary["gate_pass_unique_count"] = int(summary["gate_pass_count"])
            summary["gate_block_unique_count"] = int(summary["gate_block_count"])
            summary["gate_total_request_raw_count"] = int(gate_total)
            summary["gate_total_request_unique_count"] = int(gate_total)
            summary["gate_pass_rate_raw"] = float(summary["gate_pass_rate"])
            summary["gate_pass_rate_unique"] = float(summary["gate_pass_rate"])
            summary["gate_ready_pass_count"] = int(summary["gate_pass_count"])
            summary["gate_ready_block_count"] = 0
            summary["gate_ready_total_request_count"] = int(summary["gate_ready_pass_count"])
            summary["gate_ready_pass_rate"] = 1.0 if int(summary["gate_ready_total_request_count"]) > 0 else 0.0
            summary["gate_ready_pass_rate_raw"] = float(summary["gate_ready_pass_rate"])
            summary["gate_ready_pass_per_launch_rate"] = 0.0
            summary["gate_block_stable_not_ready_count"] = 0
            summary["gate_block_friendly_safe_count"] = 0
            summary["gate_block_long_shot_count"] = 0
            summary["gate_block_launch_exec_fail_count"] = 0
            summary["gate_block_other_count"] = int(summary["gate_block_count"])
            summary["relay_attempt_raw_count"] = int(summary["relay_attempt_count"])
            summary["relay_success_raw_count"] = int(summary["relay_success_count"])
            summary["relay_attempt_unique_count"] = int(summary["relay_attempt_count"])
            summary["relay_success_unique_count"] = int(summary["relay_success_count"])
            summary["relay_success_rate_raw"] = float(summary["relay_success_rate"])
            summary["relay_success_rate_unique"] = float(summary["relay_success_rate"])

        normalized_awacs = dict(summary.get("target_first_awacs_time_s", {}) or {})
        normalized_fcr = dict(summary.get("target_first_fcr_time_s", {}) or {})
        normalized_ready = dict(summary.get("target_first_ready_time_s", {}) or {})
        all_target_ids = sorted(set(normalized_awacs.keys()) | set(normalized_fcr.keys()) | set(normalized_ready.keys()) | set(radar_detect_map.keys()))
        for target_id in all_target_ids:
            radar_time = radar_detect_map.get(target_id)
            ready_time = normalized_ready.get(target_id)
            awacs_time = normalized_awacs.get(target_id)
            fcr_time = normalized_fcr.get(target_id)
            try:
                if radar_time not in (None, ""):
                    radar_time = float(radar_time)
                    if fcr_time in (None, "") or float(fcr_time) < 0.0 or radar_time < float(fcr_time):
                        normalized_fcr[target_id] = radar_time
                        fcr_time = radar_time
            except Exception:
                pass
            try:
                if ready_time not in (None, ""):
                    ready_time = float(ready_time)
                    if fcr_time not in (None, "") and float(fcr_time) > ready_time:
                        normalized_fcr[target_id] = ready_time
                    if awacs_time not in (None, "") and float(awacs_time) > ready_time:
                        normalized_awacs[target_id] = ready_time
            except Exception:
                pass
        summary["target_first_awacs_time_s"] = dict(sorted(normalized_awacs.items()))
        summary["target_first_fcr_time_s"] = dict(sorted(normalized_fcr.items()))
        summary["target_first_ready_time_s"] = dict(sorted(normalized_ready.items()))
    else:
        relay_attempts = int(summary["relay_attempt_count"])
        gate_total = int(summary["gate_pass_count"]) + int(summary["gate_block_count"])
        summary["relay_success_rate"] = (
            float(summary["relay_success_count"]) / float(relay_attempts) if relay_attempts > 0 else 0.0
        )
        summary["gate_pass_rate"] = (
            float(summary["gate_pass_count"]) / float(gate_total) if gate_total > 0 else 0.0
        )
        summary["gate_pass_raw_count"] = int(summary["gate_pass_count"])
        summary["gate_block_raw_count"] = int(summary["gate_block_count"])
        summary["gate_pass_unique_count"] = int(summary["gate_pass_count"])
        summary["gate_block_unique_count"] = int(summary["gate_block_count"])
        summary["gate_total_request_raw_count"] = int(gate_total)
        summary["gate_total_request_unique_count"] = int(gate_total)
        summary["gate_pass_rate_raw"] = float(summary["gate_pass_rate"])
        summary["gate_pass_rate_unique"] = float(summary["gate_pass_rate"])
        summary["gate_ready_pass_count"] = int(summary["gate_pass_count"])
        summary["gate_ready_block_count"] = 0
        summary["gate_ready_total_request_count"] = int(summary["gate_ready_pass_count"])
        summary["gate_ready_pass_rate"] = 1.0 if int(summary["gate_ready_total_request_count"]) > 0 else 0.0
        summary["gate_ready_pass_rate_raw"] = float(summary["gate_ready_pass_rate"])
        summary["gate_ready_pass_per_launch_rate"] = 0.0
        summary["gate_block_stable_not_ready_count"] = 0
        summary["gate_block_friendly_safe_count"] = 0
        summary["gate_block_long_shot_count"] = 0
        summary["gate_block_launch_exec_fail_count"] = 0
        summary["gate_block_other_count"] = int(summary["gate_block_count"])
        summary["relay_attempt_raw_count"] = int(summary["relay_attempt_count"])
        summary["relay_success_raw_count"] = int(summary["relay_success_count"])
        summary["relay_attempt_unique_count"] = int(summary["relay_attempt_count"])
        summary["relay_success_unique_count"] = int(summary["relay_success_count"])
        summary["relay_success_rate_raw"] = float(summary["relay_success_rate"])
        summary["relay_success_rate_unique"] = float(summary["relay_success_rate"])

    return summary


def _metric_cell(value: object, *, digits: int = 3) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return round(float(value), digits)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _summary_metric_rows(summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key in sorted(summary.keys()):
        rows.append({"metric": key, "value": _metric_cell(summary.get(key), digits=6)})
    return rows


def _duration_rows_from_summary(summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    mappings = [
        ("cap_state", "cap_state_time_s"),
        ("detect_mode", "detect_mode_time_s"),
        ("mission_threat", "mission_threat_time_s"),
        ("intent_threat", "intent_threat_time_s"),
        ("enemy_phase", "enemy_script_phase_time_s"),
        ("left_tactic", "left_tactic_time_s"),
        ("right_tactic", "right_tactic_time_s"),
        ("left_phase", "left_phase_time_s"),
        ("right_phase", "right_phase_time_s"),
        ("left_target", "left_target_time_s"),
        ("right_target", "right_target_time_s"),
    ]
    for category, key in mappings:
        duration_map = dict(summary.get(key, {}) or {})
        for label, time_s in sorted(duration_map.items(), key=lambda item: (-float(item[1]), str(item[0]))):
            rows.append({"category": category, "label": label, "time_s": _metric_cell(time_s, digits=3)})
    return rows


def _intent_target_rows_from_summary(summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    target_metrics = dict(summary.get("intent_target_metrics", {}) or {})
    for target_id in ENEMY_AGENT_IDS:
        metrics = dict(target_metrics.get(target_id, {}) or {})
        rows.append(
            {
                "target_id": target_id,
                "first_present_time_s": _metric_cell(metrics.get("first_present_time_s"), digits=3),
                "first_truth_time_s": _metric_cell(metrics.get("first_truth_time_s"), digits=3),
                "first_ready_time_s": _metric_cell(metrics.get("first_ready_time_s"), digits=3),
                "first_attack_time_s": _metric_cell(metrics.get("first_attack_time_s"), digits=3),
                "first_defense_time_s": _metric_cell(metrics.get("first_defense_time_s"), digits=3),
                "first_retreat_time_s": _metric_cell(metrics.get("first_retreat_time_s"), digits=3),
                "first_recon_time_s": _metric_cell(metrics.get("first_recon_time_s"), digits=3),
                "first_high_time_s": _metric_cell(metrics.get("first_high_time_s"), digits=3),
                "peak_threat": metrics.get("peak_threat", ""),
                "label_switch_count": _metric_cell(metrics.get("label_switch_count")),
                "threat_switch_count": _metric_cell(metrics.get("threat_switch_count")),
                "present_time_s": _metric_cell(metrics.get("present_time_s"), digits=3),
                "ready_time_s": _metric_cell(metrics.get("ready_time_s"), digits=3),
                "ready_ratio_in_present": _metric_cell(metrics.get("ready_ratio_in_present"), digits=6),
                "truth_time_s": _metric_cell(metrics.get("truth_time_s"), digits=3),
                "truth_match_time_s": _metric_cell(metrics.get("truth_match_time_s"), digits=3),
                "truth_accuracy": _metric_cell(metrics.get("truth_accuracy"), digits=6),
                "ready_truth_time_s": _metric_cell(metrics.get("ready_truth_time_s"), digits=3),
                "ready_truth_match_time_s": _metric_cell(metrics.get("ready_truth_match_time_s"), digits=3),
                "ready_truth_accuracy": _metric_cell(metrics.get("ready_truth_accuracy"), digits=6),
                "attack_time_s": _metric_cell(metrics.get("attack_time_s"), digits=3),
                "defense_time_s": _metric_cell(metrics.get("defense_time_s"), digits=3),
                "retreat_time_s": _metric_cell(metrics.get("retreat_time_s"), digits=3),
                "recon_time_s": _metric_cell(metrics.get("recon_time_s"), digits=3),
                "high_time_s": _metric_cell(metrics.get("high_time_s"), digits=3),
                "confidence_peak": _metric_cell(metrics.get("confidence_peak"), digits=6),
                "confidence_mean_present": _metric_cell(metrics.get("confidence_mean_present"), digits=6),
                "end_intent_name": metrics.get("end_intent_name", ""),
                "end_truth_name": metrics.get("end_truth_name", ""),
                "end_intent_threat": metrics.get("end_intent_threat", ""),
            }
        )

    group_metrics = dict(summary.get("intent_group_metrics", {}) or {})
    for group_prefix in ("enemy_left", "enemy_right"):
        metrics = dict(group_metrics.get(group_prefix, {}) or {})
        rows.append(
            {
                "target_id": group_prefix,
                "first_present_time_s": _metric_cell(metrics.get("first_present_time_s"), digits=3),
                "first_ready_time_s": _metric_cell(metrics.get("first_ready_time_s"), digits=3),
                "first_attack_time_s": _metric_cell(metrics.get("first_attack_time_s"), digits=3),
                "first_defense_time_s": _metric_cell(metrics.get("first_defense_time_s"), digits=3),
                "first_retreat_time_s": _metric_cell(metrics.get("first_retreat_time_s"), digits=3),
                "first_recon_time_s": _metric_cell(metrics.get("first_recon_time_s"), digits=3),
                "first_high_time_s": _metric_cell(metrics.get("first_high_time_s"), digits=3),
                "peak_threat": metrics.get("peak_threat", ""),
                "label_switch_count": _metric_cell(metrics.get("dominant_switch_count")),
                "threat_switch_count": "",
                "present_time_s": _metric_cell(metrics.get("classified_time_s"), digits=3),
                "ready_time_s": _metric_cell(metrics.get("ready_time_s"), digits=3),
                "ready_ratio_in_present": (
                    _metric_cell(
                        float(metrics.get("ready_time_s", 0.0) or 0.0)
                        / float(metrics.get("classified_time_s", 0.0) or 1.0),
                        digits=6,
                    )
                    if float(metrics.get("classified_time_s", 0.0) or 0.0) > 0
                    else 0.0
                ),
                "attack_time_s": _metric_cell(metrics.get("attack_time_s"), digits=3),
                "defense_time_s": _metric_cell(metrics.get("defense_time_s"), digits=3),
                "retreat_time_s": _metric_cell(metrics.get("retreat_time_s"), digits=3),
                "recon_time_s": _metric_cell(metrics.get("recon_time_s"), digits=3),
                "high_time_s": _metric_cell(metrics.get("high_time_s"), digits=3),
                "confidence_peak": "",
                "confidence_mean_present": "",
                "end_intent_name": metrics.get("end_intent_name", ""),
                "end_intent_threat": metrics.get("end_intent_threat", ""),
            }
        )
    return rows


def _control_distance_node_rows_from_summary(summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    route_order = {"left": 0, "right": 1}
    phase_order = {phase: index for index, phase in enumerate(CONTROL_PHASE_ORDER)}
    node_rows = list(summary.get("control_distance_nodes", []) or [])
    node_rows = sorted(
        node_rows,
        key=lambda item: (
            route_order.get(str(item.get("route", "") or ""), 99),
            phase_order.get(str(item.get("phase", "") or ""), 99),
            float(item.get("first_time_s", 0.0) or 0.0),
        ),
    )
    for row in node_rows:
        rows.append(
            {
                "route": row.get("route", ""),
                "phase": row.get("phase", ""),
                "first_time_s": _metric_cell(row.get("first_time_s"), digits=3),
                "distance_km": _metric_cell(row.get("distance_km"), digits=3),
                "description": row.get("description", ""),
            }
        )
    return rows


def _control_distance_segment_rows_from_summary(summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    route_order = {"left": 0, "right": 1}
    phase_order = {phase: index for index, phase in enumerate(CONTROL_PHASE_ORDER)}
    segment_rows = list(summary.get("control_distance_segments", []) or [])
    segment_rows = sorted(
        segment_rows,
        key=lambda item: (
            route_order.get(str(item.get("route", "") or ""), 99),
            phase_order.get(str(item.get("from_phase", "") or ""), 99),
            float(item.get("start_time_s", 0.0) or 0.0),
        ),
    )
    for row in segment_rows:
        rows.append(
            {
                "route": row.get("route", ""),
                "from_phase": row.get("from_phase", ""),
                "to_phase": row.get("to_phase", ""),
                "start_time_s": _metric_cell(row.get("start_time_s"), digits=3),
                "end_time_s": _metric_cell(row.get("end_time_s"), digits=3),
                "start_distance_km": _metric_cell(row.get("start_distance_km"), digits=3),
                "end_distance_km": _metric_cell(row.get("end_distance_km"), digits=3),
                "compression_km": _metric_cell(row.get("compression_km"), digits=3),
                "duration_s": _metric_cell(row.get("duration_s"), digits=3),
                "compression_rate_kmps": _metric_cell(row.get("compression_rate_kmps"), digits=6),
            }
        )
    return rows


def _cooperative_detection_agent_rows_from_summary(summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    activation_map = dict(summary.get("radar_activation_times_s", {}) or {})
    detected_map = dict(summary.get("radar_detected_target_count_by_agent", {}) or {})
    for agent_id in FRIENDLY_AGENT_IDS:
        rows.append(
            {
                "agent_id": agent_id,
                "activation_time_s": _metric_cell(activation_map.get(agent_id), digits=3),
                "detected_target_count": _metric_cell(detected_map.get(agent_id)),
            }
        )
    return rows


def _cooperative_detection_target_rows_from_summary(summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    first_detect_map = dict(summary.get("radar_first_detect_time_by_target_s", {}) or {})
    best_rows = {
        str(item.get("target", "") or ""): dict(item)
        for item in list(summary.get("per_target_best_first_detect", []) or [])
    }
    for target_id in ENEMY_AGENT_IDS:
        best = dict(best_rows.get(target_id, {}) or {})
        rows.append(
            {
                "target_id": target_id,
                "first_detect_time_s": _metric_cell(first_detect_map.get(target_id), digits=3),
                "best_detector": best.get("agent", ""),
                "best_delay_s": _metric_cell(best.get("delay_s"), digits=3),
                "best_detect_time_s": _metric_cell(best.get("detect_time_s"), digits=3),
            }
        )
    return rows


def _target_timeline_rows_from_summary(summary: Dict[str, object]) -> List[Dict[str, object]]:
    awacs_map = dict(summary.get("target_first_awacs_time_s", {}) or {})
    fcr_map = dict(summary.get("target_first_fcr_time_s", {}) or {})
    ready_map = dict(summary.get("target_first_ready_time_s", {}) or {})
    radar_detect_map = dict(summary.get("radar_first_detect_time_by_target_s", {}) or {})
    zone_map = dict(summary.get("target_zone_transition_history", {}) or {})
    rows: List[Dict[str, object]] = []
    for target_id in ENEMY_AGENT_IDS:
        zones = list(zone_map.get(target_id, []) or [])
        zone_text = " -> ".join(
            f"{str(zone_name)}@{int(round(float(time_s)))}s"
            for time_s, zone_name in zones
            if time_s not in (None, "")
        )
        awacs_value = awacs_map.get(target_id)
        fcr_value = fcr_map.get(target_id)
        ready_value = ready_map.get(target_id)
        radar_value = radar_detect_map.get(target_id)
        try:
            if radar_value not in (None, ""):
                radar_time = float(radar_value)
                if fcr_value in (None, "") or float(fcr_value) < 0.0 or radar_time < float(fcr_value):
                    fcr_value = radar_time
        except Exception:
            pass
        try:
            if ready_value not in (None, ""):
                ready_time = float(ready_value)
                if fcr_value not in (None, "") and float(fcr_value) > ready_time:
                    fcr_value = ready_time
                if awacs_value not in (None, "") and float(awacs_value) > ready_time:
                    awacs_value = ready_time
        except Exception:
            pass
        rows.append(
            {
                "target": target_id,
                "awacs_s": _safe_int(round(float(awacs_value))) if awacs_value not in (None, "") else -1,
                "fcr_s": _safe_int(round(float(fcr_value))) if fcr_value not in (None, "") else -1,
                "ready_s": _safe_int(round(float(ready_value))) if ready_value not in (None, "") else -1,
                "zones": zone_text,
            }
        )
    return rows


def _cooperative_chain_metric_rows(summary: Dict[str, object]) -> List[Dict[str, object]]:
    metrics = [
        ("radar_first_activation_time_s", "首次雷达激活时间(s)"),
        ("full_detect_time_s", "联队全探完成时间(s)"),
        ("full_detect_delay_s", "相对首激活全探延迟(s)"),
        ("radar_detection_coverage_ratio_alive_targets_post_activation", "激活后平均探测覆盖率(按存活目标)"),
        ("radar_full_coverage_continuity_alive_targets_post_activation", "激活后全覆盖连续性(按存活目标)"),
        ("radar_detection_coverage_ratio_post_activation", "激活后平均探测覆盖率"),
        ("radar_full_coverage_continuity_post_activation", "激活后全覆盖连续性"),
        ("radar_any_track_continuity_post_activation", "激活后任意目标连续性"),
        ("first_stable_ready_time_s", "首次稳定就绪时间(s)"),
        ("stable_ready_peak", "稳定就绪峰值"),
        ("stable_tracking_target_peak", "稳定跟踪目标峰值"),
        ("first_gate_pass_time_s", "首次发射门通过时间(s)"),
        ("gate_pass_count", "发射门通过次数(unique)"),
        ("gate_block_count", "发射门阻断次数(unique)"),
        ("gate_total_request_count", "发射门总请求次数(unique)"),
        ("gate_pass_rate", "发射门通过率(unique)"),
        ("gate_pass_raw_count", "发射门通过次数(raw)"),
        ("gate_block_raw_count", "发射门阻断次数(raw)"),
        ("gate_total_request_raw_count", "发射门总请求次数(raw)"),
        ("gate_pass_rate_raw", "发射门通过率(raw)"),
        ("first_relay_success_time_s", "首次接力制导成功时间(s)"),
        ("relay_success_count", "接力制导成功次数(unique)"),
        ("relay_attempt_count", "接力制导尝试次数(unique)"),
        ("relay_success_rate", "接力制导成功率(unique)"),
        ("relay_success_raw_count", "接力制导成功次数(raw)"),
        ("relay_attempt_raw_count", "接力制导尝试次数(raw)"),
        ("relay_success_rate_raw", "接力制导成功率(raw)"),
        ("relay_active_time_s", "接力链激活时长(s)"),
        ("active_guided_missile_peak", "在制导导弹峰值"),
    ]
    rows: List[Dict[str, object]] = []
    for key, label in metrics:
        value = summary.get(key)
        if key.endswith("_rate") or key.endswith("_rate_raw"):
            value = float(value or 0.0) * 100.0 if value not in (None, "") else value
            rows.append({"metric_key": key, "metric_name": label, "value": _metric_cell(value, digits=3), "unit": "%"})
        else:
            rows.append({"metric_key": key, "metric_name": label, "value": _metric_cell(value, digits=3), "unit": ""})
    return rows


def _milestone_rows_from_summary(summary: Dict[str, object]) -> List[Dict[str, object]]:
    milestones = [
        ("first_awacs_track_time_s", "First AWACS track"),
        ("radar_first_activation_time_s", "First radar activation"),
        ("first_radar_track_time_s", "First radar track"),
        ("first_scan_assignment_time_s", "First scan assignment"),
        ("full_detect_time_s", "Full radar detection complete"),
        ("first_intent_contact_time_s", "First intent contact"),
        ("first_intent_time_s", "First intent output"),
        ("first_intent_ready_time_s", "First intent ready"),
        ("first_intent_full_ready_time_s", "All intent models ready"),
        ("first_intent_high_time_s", "First high intent"),
        ("first_enemy_phase_change_time_s", "First enemy phase change"),
        ("first_stable_ready_time_s", "First stable ready"),
        ("first_gate_pass_time_s", "First gate pass"),
        ("first_relay_success_time_s", "First relay success"),
        ("first_missile_launch_time_s", "First missile launch"),
        ("first_enemy_kill_time_s", "First enemy kill"),
        ("first_friendly_loss_time_s", "First friendly loss"),
        ("expected_zone_first_time_s", "Truth enters expected zone"),
        ("picture_expected_zone_first_time_s", "Picture enters expected zone"),
        ("first_mission_high_time_s", "Mission high threat"),
    ]
    rows: List[Dict[str, object]] = []
    for key, label in milestones:
        rows.append({"milestone": label, "key": key, "time_s": _metric_cell(summary.get(key), digits=3)})
    return rows


def _event_count_rows(event_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    type_counts: Dict[str, int] = defaultdict(int)
    field_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    field_value_max: Dict[Tuple[str, str], int] = defaultdict(int)
    for event in event_rows:
        event_type = str(event.get("event_type", "") or "")
        field = str(event.get("field", "") or "")
        if event_type:
            type_counts[event_type] += 1
        field_counts[(event_type, field)] += 1
        if event_type == "counter_increase" and field:
            field_value_max[(event_type, field)] = max(
                field_value_max[(event_type, field)],
                _safe_int(event.get("value"), 0),
            )

    rows: List[Dict[str, object]] = []
    for event_type, count in sorted(type_counts.items()):
        rows.append({"scope": "event_type", "event_type": event_type, "field": "", "count": count, "final_value": ""})
    for (event_type, field), count in sorted(field_counts.items()):
        final_value: object = ""
        if event_type == "counter_increase":
            final_value = field_value_max.get((event_type, field), 0)
        rows.append(
            {
                "scope": "event_field",
                "event_type": event_type,
                "field": field,
                "count": count,
                "final_value": final_value,
            }
        )
    return rows


def _enemy_phase_catalog_rows(summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in list(summary.get("enemy_script_phase_catalog", []) or []):
        rows.append(
            {
                "phase_id": item.get("phase_id"),
                "phase_name": item.get("phase_name"),
                "trigger": item.get("trigger"),
                "objective": item.get("objective"),
                "target_offsets_deg": _metric_cell(item.get("target_offsets_deg"), digits=6),
                "target_altitudes_km": _metric_cell(item.get("target_altitudes_km"), digits=6),
            }
        )
    return rows


def _decision_trace_rows(timeline_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    columns = [
        "time_s",
        "cap_state",
        "cap_state_reason",
        "detect_mode",
        "enemy_script_phase_name",
        "enemy_script_phase_trigger",
        "enemy_script_phase_objective",
        "enemy_script_phase_elapsed_s",
        "enemy_script_reference_distance_km",
        "enemy_script_awacs_state",
        "enemy_left_group_phase",
        "enemy_right_group_phase",
        "enemy_left_wave_index",
        "enemy_right_wave_index",
        "enemy_left_pressure_tag",
        "enemy_right_pressure_tag",
        "mission_threat_level",
        "left_tactic",
        "right_tactic",
        "left_phase",
        "right_phase",
        "left_target",
        "right_target",
        "left_stage_reason",
        "left_tactic_reason",
        "left_maneuver_reason",
        "left_parameter_reason",
        "left_decision_snapshot_json",
        "right_stage_reason",
        "right_tactic_reason",
        "right_maneuver_reason",
        "right_parameter_reason",
        "right_decision_snapshot_json",
        "stable_ready_count",
        "stable_tracking_target_count",
        "intent_target",
        "intent_name",
        "intent_class",
        "intent_threat",
        "intent_confidence",
        "intent_contact_count",
        "intent_classified_count",
        "intent_unclassified_count",
        "intent_ready_count",
        "intent_attack_count",
        "intent_defense_count",
        "intent_retreat_count",
        "intent_recon_count",
        "intent_high_count",
        "enemy_left_intent_name",
        "enemy_left_intent_class",
        "enemy_left_intent_threat",
        "enemy_right_intent_name",
        "enemy_right_intent_class",
        "enemy_right_intent_threat",
        "B0100_intent_name",
        "B0100_intent_class",
        "B0100_intent_threat",
        "B0200_intent_name",
        "B0200_intent_class",
        "B0200_intent_threat",
        "B0300_intent_name",
        "B0300_intent_class",
        "B0300_intent_threat",
        "B0400_intent_name",
        "B0400_intent_class",
        "B0400_intent_threat",
        "truth_low_count",
        "truth_medium_count",
        "truth_high_count",
        "picture_low_count",
        "picture_medium_count",
        "picture_high_count",
        "scan_assignment_count",
        "scan_target_count",
        "scan_coverage_total_deg",
        "scan_overlap_deg",
        "gate_pass_count",
        "relay_success_count",
        "missile_launch_count",
        "enemy_kill_count",
        "friendly_loss_count",
    ]
    rows: List[Dict[str, object]] = []
    for row in timeline_rows:
        rows.append({column: _metric_cell(row.get(column), digits=6) for column in columns})
    return rows


def _artifact_index_rows(output_dir: Path, summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    fixed_files = [
        ("summary", output_dir / "summary.json", "场景指标 JSON 汇总"),
        ("report", output_dir / "summary.md", "场景 Markdown 报告"),
        ("timeline", output_dir / "timeline.csv", "全量时间序列"),
        ("events", output_dir / "events.csv", "事件时间序列"),
        ("plot", output_dir / "overview.png", "总览曲线"),
        ("plot", output_dir / "guidance.png", "制导与门禁曲线"),
    ]
    for artifact_type, path, description in fixed_files:
        if path.exists():
            rows.append({"type": artifact_type, "path": str(path), "description": description})

    for pattern, artifact_type, description in (
        ("figures/*.png", "figure", "runner 细分图"),
        ("tables/*.csv", "table", "runner 明细表"),
        ("thesis_assets/figures/*.png", "figure", "论文素材图"),
        ("thesis_assets/csv/*.csv", "table", "论文素材原始表"),
    ):
        for path in sorted(output_dir.glob(pattern)):
            rows.append({"type": artifact_type, "path": str(path), "description": description})

    thesis_summary = output_dir / "thesis_assets" / "README_ch6_assets.md"
    if thesis_summary.exists():
        rows.append({"type": "report", "path": str(thesis_summary), "description": "论文素材摘要"})
    if summary.get("thesis_assets_error"):
        rows.append({"type": "warning", "path": "", "description": str(summary.get("thesis_assets_error"))})
    return rows


def _write_detail_tables(
    output_dir: Path,
    summary: Dict[str, object],
    timeline_rows: List[Dict[str, object]],
    event_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(tables_dir / "summary_metrics.csv", _summary_metric_rows(summary))
    _write_csv(tables_dir / "duration_breakdown.csv", _duration_rows_from_summary(summary))
    _write_csv(tables_dir / "milestone_times.csv", _milestone_rows_from_summary(summary))
    _write_csv(tables_dir / "event_counts.csv", _event_count_rows(event_rows))
    _write_csv(tables_dir / "intent_target_summary.csv", _intent_target_rows_from_summary(summary))
    _write_csv(tables_dir / "control_distance_nodes.csv", _control_distance_node_rows_from_summary(summary))
    _write_csv(tables_dir / "control_distance_segments.csv", _control_distance_segment_rows_from_summary(summary))
    _write_csv(tables_dir / "enemy_zone_profile.csv", _enemy_zone_profile_rows_from_summary(summary))
    _write_csv(tables_dir / "cooperative_detection_agents.csv", _cooperative_detection_agent_rows_from_summary(summary))
    _write_csv(tables_dir / "cooperative_detection_targets.csv", _cooperative_detection_target_rows_from_summary(summary))
    _write_csv(tables_dir / "cooperative_chain_metrics.csv", _cooperative_chain_metric_rows(summary))
    _write_csv(tables_dir / "decision_trace.csv", _decision_trace_rows(timeline_rows))
    _write_csv(tables_dir / "enemy_phase_catalog.csv", _enemy_phase_catalog_rows(summary))

    return {
        "tables_dir": str(tables_dir),
        "table_file_count": len(list(tables_dir.glob("*.csv"))),
    }


def _generate_thesis_asset_bundle(log_file: Path, output_dir: Path, summary: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    result = {
        "thesis_assets_dir": "",
        "thesis_assets_figure_dir": "",
        "thesis_assets_csv_dir": "",
        "thesis_assets_figure_count": 0,
        "thesis_assets_csv_count": 0,
        "thesis_assets_generated": False,
        "thesis_assets_error": "",
    }
    if ch6_asset_generator is None:
        result["thesis_assets_error"] = "generate_ch6_validation_assets.py could not be imported."
        return result

    thesis_dir = output_dir / "thesis_assets"
    figure_dir = thesis_dir / "figures"
    csv_dir = thesis_dir / "csv"
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    try:
        timeline_rows: List[Dict[str, object]] = []
        if (output_dir / "timeline.csv").exists():
            with (output_dir / "timeline.csv").open("r", encoding="utf-8", newline="") as handle:
                timeline_rows = list(csv.DictReader(handle))
        lines = ch6_asset_generator.read_lines(Path(log_file))
        time_lookup = ch6_asset_generator.build_time_lookup(lines)
        awacs_rows = ch6_asset_generator.parse_awacs_tracks(lines, time_lookup)
        target_rows = ch6_asset_generator.parse_target_timeline(lines)
        if (not target_rows) and summary:
            target_rows = _target_timeline_rows_from_summary(summary)
        snapshot_rows = ch6_asset_generator.parse_snapshots(lines)
        state_rows = ch6_asset_generator.parse_states(lines)
        ready_rows = ch6_asset_generator.build_ready_curve(target_rows)
        launches = ch6_asset_generator.parse_launch_records(lines)
        hit_rows, miss_reasons_friendly, miss_reasons_enemy = ch6_asset_generator.parse_missile_events(lines, launches)
        thesis_summary = ch6_asset_generator.parse_final_summary(lines)
        if summary:
            runner_thesis_summary = ch6_asset_generator.summary_from_runner_summary(summary)
            if thesis_summary:
                merged_summary = dict(runner_thesis_summary)
                merged_summary.update({key: value for key, value in thesis_summary.items() if value not in (None, "")})
                thesis_summary = merged_summary
            else:
                thesis_summary = runner_thesis_summary

        ch6_asset_generator.export_raw_tables(
            csv_dir=csv_dir,
            awacs_rows=awacs_rows,
            target_rows=target_rows,
            snapshot_rows=snapshot_rows,
            state_rows=state_rows,
            ready_rows=ready_rows,
            hit_rows=hit_rows,
            miss_reasons_friendly=miss_reasons_friendly,
            miss_reasons_enemy=miss_reasons_enemy,
        )

        end_time = max([float(row["time_s"]) for row in snapshot_rows], default=1200.0)
        ch6_asset_generator.plot_awacs_tracks(awacs_rows, figure_dir / "fig01_awacs_tracks")
        ch6_asset_generator.plot_ready_curve(ready_rows, figure_dir / "fig02_ready_targets")
        if timeline_rows:
            state_rows_from_timeline = [
                {
                    "time_s": float(row.get("time_s", 0.0) or 0.0),
                    "state": str(row.get("cap_state", "") or ""),
                    "tactic_text": f"L={str(row.get('left_tactic', '') or '-')}; R={str(row.get('right_tactic', '') or '-')}",
                    "distance_km": float(row.get("nearest_enemy_to_friendly_km", 0.0) or 0.0),
                }
                for row in timeline_rows
            ]
            snapshot_rows_from_timeline = [
                {
                    "time_s": float(row.get("time_s", 0.0) or 0.0),
                    "a_alive": int(float(row.get("friendly_alive", 0) or 0)),
                    "b_alive": int(float(row.get("enemy_alive", 0) or 0)),
                    "a_ms": int(float(row.get("friendly_missiles_left", 0) or 0)),
                    "b_ms": int(float(row.get("enemy_missiles_left", 0) or 0)),
                    "zone_high": int(float(row.get("truth_high_count", 0) or 0)),
                    "zone_medium": int(float(row.get("truth_medium_count", 0) or 0)),
                    "zone_low": int(float(row.get("truth_low_count", 0) or 0)),
                    "zone_outer": max(0, 4 - int(float(row.get("truth_high_count", 0) or 0)) - int(float(row.get("truth_medium_count", 0) or 0)) - int(float(row.get("truth_low_count", 0) or 0))),
                    "left_tactic": str(row.get("left_tactic", "") or "-"),
                    "right_tactic": str(row.get("right_tactic", "") or "-"),
                    "tracking": "ON" if int(float(row.get("stable_tracking_target_count", 0) or 0)) > 0 else "OFF",
                    "relay": "ON" if int(float(row.get("active_relay_flag", 0) or 0)) > 0 else "OFF",
                    "gate_pass": int(float(row.get("gate_pass_count", 0) or 0)),
                    "gate_block": int(float(row.get("gate_block_count", 0) or 0)),
                    "relay_success": int(float(row.get("relay_success_count", 0) or 0)),
                    "relay_attempt": int(float(row.get("relay_attempt_count", 0) or 0)),
                }
                for row in timeline_rows
            ]
            end_time = max([float(row["time_s"]) for row in snapshot_rows_from_timeline], default=end_time)
            ch6_asset_generator.plot_state_tactic_timeline(
                state_rows_from_timeline,
                snapshot_rows_from_timeline,
                figure_dir / "fig03_state_tactic_timeline",
                end_time,
            )
            ch6_asset_generator.plot_alive_and_missiles_from_timeline(timeline_rows, figure_dir / "fig04_alive_and_missiles")
            ch6_asset_generator.plot_gate_and_relay_from_timeline(timeline_rows, figure_dir / "fig05_gate_and_relay")
            ch6_asset_generator.plot_risk_zone_counts_from_timeline(timeline_rows, figure_dir / "fig06_risk_zone_counts")
            gate_relay_curve_basis = "unique_timeline"
        else:
            ch6_asset_generator.plot_state_tactic_timeline(
                state_rows,
                snapshot_rows,
                figure_dir / "fig03_state_tactic_timeline",
                end_time,
            )
            ch6_asset_generator.plot_alive_and_missiles(snapshot_rows, figure_dir / "fig04_alive_and_missiles")
            ch6_asset_generator.plot_gate_and_relay(snapshot_rows, figure_dir / "fig05_gate_and_relay")
            ch6_asset_generator.plot_risk_zone_counts(snapshot_rows, figure_dir / "fig06_risk_zone_counts")
            gate_relay_curve_basis = "battle_snapshot_fallback"
        ch6_asset_generator.plot_missile_end_reasons(
            miss_reasons_friendly,
            miss_reasons_enemy,
            figure_dir / "fig07_missile_end_reasons",
        )
        ch6_asset_generator.plot_kill_timeline(hit_rows, figure_dir / "fig08_kill_timeline")
        ch6_asset_generator.write_summary_md(
            output_path=thesis_dir / "README_ch6_assets.md",
            summary=thesis_summary,
            target_rows=target_rows,
            snapshot_rows=snapshot_rows,
            hit_rows=hit_rows,
            miss_reasons_friendly=miss_reasons_friendly,
            miss_reasons_enemy=miss_reasons_enemy,
            figure_dir=figure_dir,
            gate_relay_curve_basis=gate_relay_curve_basis,
        )
        (thesis_dir / "summary_metrics.json").write_text(
            json.dumps(thesis_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result.update(
            {
                "thesis_assets_dir": str(thesis_dir),
                "thesis_assets_figure_dir": str(figure_dir),
                "thesis_assets_csv_dir": str(csv_dir),
                "thesis_assets_figure_count": len(list(figure_dir.glob("*.png"))),
                "thesis_assets_csv_count": len(list(csv_dir.glob("*.csv"))),
                "thesis_assets_generated": True,
                "thesis_assets_error": "",
            }
        )
    except Exception as exc:
        logging.exception("Failed to generate thesis asset bundle from %s", log_file)
        result["thesis_assets_error"] = str(exc)
    return result


def _append_markdown_table(lines: List[str], title: str, rows: List[Tuple[str, object]]) -> None:
    if not rows:
        return
    lines.extend(["", f"## {title}", "| Metric | Value |", "| --- | --- |"])
    for label, value in rows:
        lines.append(f"| {label} | {_metric_cell(value, digits=4)} |")


def _write_summary_markdown(
    path: Path,
    summary: Dict[str, object],
    output_dir: Path,
    *,
    include_plots: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {summary.get('scenario_id', '')} {summary.get('title', '')}",
        "",
        f"- Thesis focus: {summary.get('thesis_focus', '')}",
        f"- Design intent: {summary.get('design_intent', '')}",
        f"- Enemy control mode: {summary.get('enemy_control_mode', '')}",
        f"- Simulated time: {float(summary.get('time_s', 0.0) or 0.0):.1f}s",
        f"- Step count: {int(summary.get('step_count', 0) or 0)}",
        "",
        "## Control Contract",
    ]
    for item in list(summary.get("control_contract", []) or []):
        lines.append(f"- {item}")

    phase_catalog = list(summary.get("enemy_script_phase_catalog", []) or [])
    if phase_catalog:
        lines.extend(["", "## Enemy Scene Design", "| Phase | Trigger | Objective | Offsets (deg) | Altitudes (km) |", "| --- | --- | --- | --- | --- |"])
        for phase in phase_catalog:
            lines.append(
                "| {name} | {trigger} | {objective} | {offsets} | {alts} |".format(
                    name=phase.get("phase_name", ""),
                    trigger=phase.get("trigger", ""),
                    objective=phase.get("objective", ""),
                    offsets=phase.get("target_offsets_deg", []),
                    alts=phase.get("target_altitudes_km", []),
                )
            )

    _append_markdown_table(
        lines,
        "Outcome Summary",
        [
            ("Expected zone", summary.get("expected_zone")),
            ("Expected zone triggered", summary.get("expected_zone_triggered")),
            ("Expected zone occupied time (truth, s)", summary.get("expected_zone_enemy_time_s")),
            ("Expected zone occupied time (picture, s)", summary.get("picture_expected_zone_enemy_time_s")),
            ("Expected zone picture match ratio", summary.get("expected_zone_picture_match_ratio")),
            ("Enemy kills", summary.get("enemy_kill_count")),
            ("Friendly losses", summary.get("friendly_loss_count")),
            ("Mission result", summary.get("mission_result")),
            ("Mission threat peak", summary.get("mission_threat_peak")),
            ("Mission high-zone breaches", summary.get("mission_high_zone_breach_events_end")),
            ("Enemy phase switches", summary.get("enemy_script_phase_switch_count")),
            ("Enemy AWACS denied time (s)", summary.get("enemy_awacs_denied_time_s")),
        ],
    )
    _append_markdown_table(
        lines,
        "Flight Safety",
        [
            ("Friendly min altitude (m)", summary.get("friendly_min_altitude_m")),
            ("Enemy min altitude (m)", summary.get("enemy_min_altitude_m")),
            ("Friendly min speed (m/s)", summary.get("friendly_min_speed_mps")),
            ("Enemy min speed (m/s)", summary.get("enemy_min_speed_mps")),
            ("Friendly max descent rate (m/s)", summary.get("friendly_max_descent_rate_mps")),
            ("Enemy max descent rate (m/s)", summary.get("enemy_max_descent_rate_mps")),
            ("Friendly recreate total", summary.get("friendly_sim_recreate_total")),
            ("Enemy recreate total", summary.get("enemy_sim_recreate_total")),
            ("Friendly recreate by agent", summary.get("friendly_sim_recreate_by_agent")),
            ("Enemy recreate by agent", summary.get("enemy_sim_recreate_by_agent")),
        ],
    )
    _append_markdown_table(
        lines,
        "Detection And Intent",
        [
            ("AWACS track peak", summary.get("awacs_track_peak")),
            ("Radar track peak", summary.get("radar_track_peak")),
            ("Stable-ready peak", summary.get("stable_ready_peak")),
            ("Stable tracking target peak", summary.get("stable_tracking_target_peak")),
            ("First AWACS track (s)", summary.get("first_awacs_track_time_s")),
            ("First radar track (s)", summary.get("first_radar_track_time_s")),
            ("First intent contact (s)", summary.get("first_intent_contact_time_s")),
            ("First classified intent (s)", summary.get("first_intent_time_s")),
            ("First ready intent (s)", summary.get("first_intent_ready_time_s")),
            ("Intent threat peak", summary.get("intent_threat_peak")),
            ("Intent confidence peak", summary.get("intent_confidence_peak")),
            ("Intent confidence mean", summary.get("intent_confidence_mean")),
        ],
    )
    _append_markdown_table(
        lines,
        "Decision And Engagement",
        [
            ("Expected tactic observed", summary.get("expected_tactic_observed")),
            ("Expected tactic time (s)", summary.get("expected_tactic_time_s")),
            ("Enemy phase end", summary.get("enemy_script_phase_end")),
            ("First enemy phase change (s)", summary.get("first_enemy_phase_change_time_s")),
            ("Left tactic switches", summary.get("left_tactic_switch_count")),
            ("Right tactic switches", summary.get("right_tactic_switch_count")),
            ("Left target switches", summary.get("left_target_switch_count")),
            ("Right target switches", summary.get("right_target_switch_count")),
            ("Gate pass count (unique)", summary.get("gate_pass_count")),
            ("Gate block count (unique)", summary.get("gate_block_count")),
            ("Gate total requests (unique)", summary.get("gate_total_request_count")),
            ("Gate pass rate (unique)", f"{100.0 * float(summary.get('gate_pass_rate', 0.0)):.1f}%"),
            ("Gate pass count (raw)", summary.get("gate_pass_raw_count")),
            ("Gate block count (raw)", summary.get("gate_block_raw_count")),
            ("Gate total requests (raw)", summary.get("gate_total_request_raw_count")),
            ("Gate pass rate (raw)", f"{100.0 * float(summary.get('gate_pass_rate_raw', 0.0)):.1f}%"),
            ("Relay success count (unique)", summary.get("relay_success_count")),
            ("Relay attempt count (unique)", summary.get("relay_attempt_count")),
            ("Relay success rate (unique)", f"{100.0 * float(summary.get('relay_success_rate', 0.0)):.1f}%"),
            ("Relay success count (raw)", summary.get("relay_success_raw_count")),
            ("Relay attempt count (raw)", summary.get("relay_attempt_raw_count")),
            ("Relay success rate (raw)", f"{100.0 * float(summary.get('relay_success_rate_raw', 0.0)):.1f}%"),
            ("Missile launch count", summary.get("missile_launch_count")),
            ("Active guided missile peak", summary.get("active_guided_missile_peak")),
            ("First missile launch (s)", summary.get("first_missile_launch_time_s")),
            ("First enemy kill (s)", summary.get("first_enemy_kill_time_s")),
        ],
    )
    _append_markdown_table(
        lines,
        "Risk Picture And Scan",
        [
            ("Risk exact match ratio", summary.get("risk_picture_exact_match_ratio")),
            ("Low risk match ratio", summary.get("low_risk_picture_match_ratio")),
            ("Medium risk match ratio", summary.get("medium_risk_picture_match_ratio")),
            ("High risk match ratio", summary.get("high_risk_picture_match_ratio")),
            ("Picture threat peak", summary.get("picture_total_peak")),
            ("Search picture peak", summary.get("search_picture_total_peak")),
            ("Scan assignment peak", summary.get("scan_assignment_peak")),
            ("Scan target peak", summary.get("scan_target_peak")),
            ("Scan coverage peak (deg)", summary.get("scan_coverage_peak_deg")),
            ("Scan overlap peak (deg)", summary.get("scan_overlap_peak_deg")),
            ("Confidence radius peak (km)", summary.get("confidence_radius_peak_km")),
            ("Min enemy-to-friendly distance (km)", summary.get("min_enemy_to_friendly_km")),
        ],
    )

    duration_rows = _duration_rows_from_summary(summary)
    if duration_rows:
        lines.extend(["", "## Duration Breakdown", "| Category | Label | Time (s) |", "| --- | --- | --- |"])
        for row in duration_rows:
            lines.append(f"| {row['category']} | {row['label']} | {row['time_s']} |")

    milestone_rows = _milestone_rows_from_summary(summary)
    if milestone_rows:
        lines.extend(["", "## Milestones", "| Milestone | Time (s) |", "| --- | --- |"])
        for row in milestone_rows:
            lines.append(f"| {row['milestone']} | {row['time_s']} |")

    control_nodes = _control_distance_node_rows_from_summary(summary)
    if control_nodes:
        lines.extend(
            [
                "",
                "## Control-Distance Nodes",
                "| Route | Phase | First Time (s) | Distance (km) | Description |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in control_nodes:
            lines.append(
                f"| {row['route']} | {row['phase']} | {row['first_time_s']} | {row['distance_km']} | {row['description']} |"
            )

    coop_agents = _cooperative_detection_agent_rows_from_summary(summary)
    if coop_agents:
        lines.extend(
            [
                "",
                "## Cooperative Detection Agents",
                "| Agent | Activation Time (s) | Detected Targets |",
                "| --- | --- | --- |",
            ]
        )
        for row in coop_agents:
            lines.append(
                f"| {row['agent_id']} | {row['activation_time_s']} | {row['detected_target_count']} |"
            )

    coop_targets = _cooperative_detection_target_rows_from_summary(summary)
    if coop_targets:
        lines.extend(
            [
                "",
                "## Cooperative Detection Targets",
                "| Target | First Detect (s) | Best Detector | Best Delay (s) | Best Detect Time (s) |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in coop_targets:
            lines.append(
                f"| {row['target_id']} | {row['first_detect_time_s']} | {row['best_detector']} | {row['best_delay_s']} | {row['best_detect_time_s']} |"
            )

    coop_chain = _cooperative_chain_metric_rows(summary)
    if coop_chain:
        lines.extend(["", "## Cooperative Tracking And Relay", "| Metric | Value | Unit |", "| --- | --- | --- |"])
        for row in coop_chain:
            lines.append(f"| {row['metric_name']} | {row['value']} | {row['unit']} |")

    lines.extend(["", "## Artifacts", f"- Summary JSON: {output_dir / 'summary.json'}", f"- Timeline CSV: {output_dir / 'timeline.csv'}", f"- Events CSV: {output_dir / 'events.csv'}", f"- Tables directory: {output_dir / 'tables'}"])
    if include_plots:
        lines.append(f"- Overview plot: {output_dir / 'overview.png'}")
        lines.append(f"- Guidance plot: {output_dir / 'guidance.png'}")
        lines.append(f"- Detailed figure directory: {output_dir / 'figures'}")
    if summary.get("thesis_assets_dir"):
        lines.append(f"- Thesis asset directory: {summary.get('thesis_assets_dir')}")
    if summary.get("thesis_assets_error"):
        lines.append(f"- Thesis asset generation warning: {summary.get('thesis_assets_error')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_scenario_summary(summary: Dict[str, object]) -> None:
    logging.info("[验证指标汇总]")
    logging.info(
        "  总仿真时间: %.1fs | 步数: %d",
        float(summary.get("time_s", 0.0) or 0.0),
        int(summary.get("step_count", 0) or 0),
    )
    logging.info(
        "  联队全探完成: %s | 首次稳定双机就绪: %s | 稳定跟踪峰值: %s",
        _metric_cell(summary.get("full_detect_time_s"), digits=3),
        _metric_cell(summary.get("first_stable_ready_time_s"), digits=3),
        _metric_cell(summary.get("stable_tracking_target_peak")),
    )
    logging.info(
        "  发射门通过/阻断(unique): %s/%s | 接力成功/尝试(unique): %s/%s",
        _metric_cell(summary.get("gate_pass_count")),
        _metric_cell(summary.get("gate_block_count")),
        _metric_cell(summary.get("relay_success_count")),
        _metric_cell(summary.get("relay_attempt_count")),
    )
    logging.info(
        "  发射门通过/阻断(raw): %s/%s | 接力成功/尝试(raw): %s/%s",
        _metric_cell(summary.get("gate_pass_raw_count")),
        _metric_cell(summary.get("gate_block_raw_count")),
        _metric_cell(summary.get("relay_success_raw_count")),
        _metric_cell(summary.get("relay_attempt_raw_count")),
    )
    activation_map = dict(summary.get("radar_activation_times_s", {}) or {})
    if activation_map:
        activation_text = " | ".join(
            f"{agent}:{_metric_cell(time_s, digits=3)}s"
            for agent, time_s in sorted(activation_map.items(), key=lambda item: item[0])
        )
        logging.info("  雷达激活时间: %s", activation_text)
    detect_map = dict(summary.get("radar_first_detect_time_by_target_s", {}) or {})
    if detect_map:
        detect_text = " | ".join(
            f"{target}:{_metric_cell(time_s, digits=3)}s"
            for target, time_s in sorted(detect_map.items(), key=lambda item: item[0])
        )
        logging.info("  各目标首探时间: %s", detect_text)
    for route_label, key in (("左路", "left_phase_first_entry_distance_km"), ("右路", "right_phase_first_entry_distance_km")):
        distance_map = dict(summary.get(key, {}) or {})
        time_key = "left_phase_first_entry_time_s" if route_label == "左路" else "right_phase_first_entry_time_s"
        time_map = dict(summary.get(time_key, {}) or {})
        if not distance_map:
            continue
        node_text = " | ".join(
            f"{phase}@{_metric_cell(time_map.get(phase), digits=3)}s/{_metric_cell(distance_map.get(phase), digits=3)}km"
            for phase in CONTROL_PHASE_ORDER
            if phase in distance_map
        )
        logging.info("  %s控制距离节点: %s", route_label, node_text)
    logging.info(
        "  意图识别: 首次接触=%ss | 首次分类=%ss | 攻击峰值=%s | 防御峰值=%s | 侦察峰值=%s",
        _metric_cell(summary.get("first_intent_contact_time_s"), digits=3),
        _metric_cell(summary.get("first_intent_time_s"), digits=3),
        _metric_cell(summary.get("intent_attack_peak")),
        _metric_cell(summary.get("intent_defense_peak")),
        _metric_cell(summary.get("intent_recon_peak")),
    )


def _summary_number(summary: Dict[str, object], key: str, *, default: float = float("nan")) -> float:
    value = summary.get(key)
    if value in (None, ""):
        return default
    try:
        numeric = float(value)
    except Exception:
        return default
    return numeric if np.isfinite(numeric) else default


def _batch_summary_rows(summaries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in summaries:
        rows.append(
            {
                "scenario_id": summary.get("scenario_id"),
                "title": summary.get("title"),
                "expected_zone": summary.get("expected_zone"),
                "enemy_control_mode": summary.get("enemy_control_mode"),
                "time_s": _metric_cell(summary.get("time_s"), digits=3),
                "radar_first_activation_time_s": _metric_cell(summary.get("radar_first_activation_time_s"), digits=3),
                "full_detect_time_s": _metric_cell(summary.get("full_detect_time_s"), digits=3),
                "radar_detection_coverage_ratio_post_activation": _metric_cell(
                    summary.get("radar_detection_coverage_ratio_post_activation"), digits=6
                ),
                "radar_detection_coverage_ratio_alive_targets_post_activation": _metric_cell(
                    summary.get("radar_detection_coverage_ratio_alive_targets_post_activation"), digits=6
                ),
                "radar_full_coverage_continuity_post_activation": _metric_cell(
                    summary.get("radar_full_coverage_continuity_post_activation"), digits=6
                ),
                "radar_full_coverage_continuity_alive_targets_post_activation": _metric_cell(
                    summary.get("radar_full_coverage_continuity_alive_targets_post_activation"), digits=6
                ),
                "enemy_kill_count": summary.get("enemy_kill_count"),
                "friendly_loss_count": summary.get("friendly_loss_count"),
                "exchange_ratio": _metric_cell(
                    (
                        float(summary.get("enemy_kill_count", 0) or 0) / float(summary.get("friendly_loss_count", 0) or 1)
                        if int(summary.get("friendly_loss_count", 0) or 0) > 0
                        else float(summary.get("enemy_kill_count", 0) or 0)
                    ),
                    digits=6,
                ),
                "stable_ready_peak": summary.get("stable_ready_peak"),
                "stable_tracking_target_peak": summary.get("stable_tracking_target_peak"),
                "enemy_script_phase_switch_count": summary.get("enemy_script_phase_switch_count"),
                "enemy_awacs_denied_time_s": _metric_cell(summary.get("enemy_awacs_denied_time_s"), digits=3),
                "gate_pass_rate": _metric_cell(summary.get("gate_pass_rate"), digits=6),
                "gate_pass_rate_raw": _metric_cell(summary.get("gate_pass_rate_raw"), digits=6),
                "gate_pass_count": summary.get("gate_pass_count"),
                "gate_block_count": summary.get("gate_block_count"),
                "gate_total_request_count": summary.get("gate_total_request_count"),
                "gate_pass_raw_count": summary.get("gate_pass_raw_count"),
                "gate_block_raw_count": summary.get("gate_block_raw_count"),
                "gate_total_request_raw_count": summary.get("gate_total_request_raw_count"),
                "relay_success_rate": _metric_cell(summary.get("relay_success_rate"), digits=6),
                "relay_success_rate_raw": _metric_cell(summary.get("relay_success_rate_raw"), digits=6),
                "relay_success_count": summary.get("relay_success_count"),
                "relay_attempt_count": summary.get("relay_attempt_count"),
                "relay_success_raw_count": summary.get("relay_success_raw_count"),
                "relay_attempt_raw_count": summary.get("relay_attempt_raw_count"),
                "missile_launch_count": summary.get("missile_launch_count"),
                "active_guided_missile_peak": summary.get("active_guided_missile_peak"),
                "friendly_min_altitude_m": _metric_cell(summary.get("friendly_min_altitude_m"), digits=3),
                "enemy_min_altitude_m": _metric_cell(summary.get("enemy_min_altitude_m"), digits=3),
                "friendly_min_speed_mps": _metric_cell(summary.get("friendly_min_speed_mps"), digits=3),
                "enemy_min_speed_mps": _metric_cell(summary.get("enemy_min_speed_mps"), digits=3),
                "friendly_sim_recreate_total": summary.get("friendly_sim_recreate_total"),
                "enemy_sim_recreate_total": summary.get("enemy_sim_recreate_total"),
                "expected_zone_enemy_time_s": _metric_cell(summary.get("expected_zone_enemy_time_s"), digits=3),
                "picture_expected_zone_enemy_time_s": _metric_cell(
                    summary.get("picture_expected_zone_enemy_time_s"), digits=3
                ),
                "risk_picture_exact_match_ratio": _metric_cell(
                    summary.get("risk_picture_exact_match_ratio"), digits=6
                ),
                "expected_zone_picture_match_ratio": _metric_cell(
                    summary.get("expected_zone_picture_match_ratio"), digits=6
                ),
                "first_awacs_track_time_s": _metric_cell(summary.get("first_awacs_track_time_s"), digits=3),
                "first_intent_ready_time_s": _metric_cell(summary.get("first_intent_ready_time_s"), digits=3),
                "intent_confidence_mean": _metric_cell(summary.get("intent_confidence_mean"), digits=6),
                "intent_truth_accuracy": _metric_cell(summary.get("intent_truth_accuracy"), digits=6),
                "intent_truth_ready_accuracy": _metric_cell(summary.get("intent_truth_ready_accuracy"), digits=6),
                "first_stable_ready_time_s": _metric_cell(summary.get("first_stable_ready_time_s"), digits=3),
                "first_missile_launch_time_s": _metric_cell(summary.get("first_missile_launch_time_s"), digits=3),
                "first_enemy_kill_time_s": _metric_cell(summary.get("first_enemy_kill_time_s"), digits=3),
                "min_enemy_to_friendly_km": _metric_cell(summary.get("min_enemy_to_friendly_km"), digits=3),
                "low_risk_enemy_time_s": _metric_cell(summary.get("low_risk_enemy_time_s"), digits=3),
                "medium_risk_enemy_time_s": _metric_cell(summary.get("medium_risk_enemy_time_s"), digits=3),
                "high_risk_breach_time_s": _metric_cell(summary.get("high_risk_breach_time_s"), digits=3),
                "output_dir": summary.get("output_dir"),
            }
        )
    return rows


def _batch_zone_rows(summaries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in summaries:
        rows.append(
            {
                "scenario_id": summary.get("scenario_id"),
                "truth_low_time_s": _metric_cell(summary.get("low_risk_enemy_time_s"), digits=3),
                "truth_medium_time_s": _metric_cell(summary.get("medium_risk_enemy_time_s"), digits=3),
                "truth_high_time_s": _metric_cell(summary.get("high_risk_breach_time_s"), digits=3),
                "picture_low_time_s": _metric_cell(summary.get("picture_low_risk_enemy_time_s"), digits=3),
                "picture_medium_time_s": _metric_cell(summary.get("picture_medium_risk_enemy_time_s"), digits=3),
                "picture_high_time_s": _metric_cell(summary.get("picture_high_risk_enemy_time_s"), digits=3),
            }
        )
    return rows


def _batch_milestone_rows(summaries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in summaries:
        for milestone_row in _milestone_rows_from_summary(summary):
            rows.append({"scenario_id": summary.get("scenario_id"), **milestone_row})
    return rows


def _plot_batch_bars(ax, labels: List[str], values: List[float], title: str, *, suffix: str = "", color: str = "#2F6B9A") -> None:
    plot_values = [0.0 if not np.isfinite(value) else float(value) for value in values]
    bars = ax.bar(labels, plot_values, color=color, edgecolor="black", linewidth=0.4)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        label = "NA" if not np.isfinite(value) else f"{value:.2f}{suffix}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(plot_values + [1.0]) * 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def _generate_batch_plots(figures_dir: Path, summaries: List[Dict[str, object]]) -> None:
    if not HAS_MPL or not summaries:
        return

    labels = [str(summary.get("scenario_id", "")) for summary in summaries]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    effect_metrics = [
        ("Enemy kills", "enemy_kill_count", 1.0, "#1B9E77", ""),
        ("Friendly losses", "friendly_loss_count", 1.0, "#D95F02", ""),
        ("Stable ready peak", "stable_ready_peak", 1.0, "#7570B3", ""),
        ("Gate pass rate (unique)", "gate_pass_rate", 100.0, "#2F6B9A", "%"),
        ("Relay success rate (unique)", "relay_success_rate", 100.0, "#66A61E", "%"),
        ("Guided missile peak", "active_guided_missile_peak", 1.0, "#E7298A", ""),
    ]
    for ax, (title, key, multiplier, color, suffix) in zip(axes.flat, effect_metrics):
        values = [_summary_number(summary, key) * multiplier for summary in summaries]
        _plot_batch_bars(ax, labels, values, title, suffix=suffix, color=color)
    _save_figure(fig, figures_dir / "comparison_effectiveness.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    timing_metrics = [
        ("First radar track", "first_radar_track_time_s"),
        ("First stable ready", "first_stable_ready_time_s"),
        ("First gate pass", "first_gate_pass_time_s"),
        ("First missile launch", "first_missile_launch_time_s"),
        ("First enemy kill", "first_enemy_kill_time_s"),
        ("First high intent", "first_intent_high_time_s"),
    ]
    for ax, (title, key) in zip(axes.flat, timing_metrics):
        values = [_summary_number(summary, key) for summary in summaries]
        _plot_batch_bars(ax, labels, values, title, suffix="s", color="#4C78A8")
    _save_figure(fig, figures_dir / "comparison_timing.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    low_truth = [_summary_number(summary, "low_risk_enemy_time_s", default=0.0) for summary in summaries]
    medium_truth = [_summary_number(summary, "medium_risk_enemy_time_s", default=0.0) for summary in summaries]
    high_truth = [_summary_number(summary, "high_risk_breach_time_s", default=0.0) for summary in summaries]
    axes[0].bar(labels, low_truth, label="Low", color="#A6CEE3")
    axes[0].bar(labels, medium_truth, bottom=low_truth, label="Medium", color="#FDBF6F")
    axes[0].bar(
        labels,
        high_truth,
        bottom=[low + medium for low, medium in zip(low_truth, medium_truth)],
        label="High",
        color="#FB9A99",
    )
    axes[0].set_title("Truth Zone Occupancy (s)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    low_picture = [_summary_number(summary, "picture_low_risk_enemy_time_s", default=0.0) for summary in summaries]
    medium_picture = [_summary_number(summary, "picture_medium_risk_enemy_time_s", default=0.0) for summary in summaries]
    high_picture = [_summary_number(summary, "picture_high_risk_enemy_time_s", default=0.0) for summary in summaries]
    axes[1].bar(labels, low_picture, label="Low", color="#A6CEE3")
    axes[1].bar(labels, medium_picture, bottom=low_picture, label="Medium", color="#FDBF6F")
    axes[1].bar(
        labels,
        high_picture,
        bottom=[low + medium for low, medium in zip(low_picture, medium_picture)],
        label="High",
        color="#FB9A99",
    )
    axes[1].set_title("Picture Zone Occupancy (s)")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    x = np.arange(len(labels))
    risk_match = np.asarray([_summary_number(summary, "risk_picture_exact_match_ratio", default=0.0) * 100.0 for summary in summaries], dtype=float)
    expected_match = np.asarray(
        [_summary_number(summary, "expected_zone_picture_match_ratio", default=0.0) * 100.0 for summary in summaries],
        dtype=float,
    )
    width = 0.35
    axes[2].bar(x - width / 2.0, risk_match, width, label="All-zone exact", color="#2F6B9A")
    axes[2].bar(x + width / 2.0, expected_match, width, label="Expected-zone exact", color="#66A61E")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Picture Match Ratio (%)")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend()
    _save_figure(fig, figures_dir / "comparison_zone_picture.png")


def _write_batch_reports(batch_dir: Path, summaries: List[Dict[str, object]], *, include_plots: bool) -> Dict[str, object]:
    comparison_tables_dir = batch_dir / "comparison_tables"
    comparison_figures_dir = batch_dir / "comparison_figures"
    comparison_tables_dir.mkdir(parents=True, exist_ok=True)
    if include_plots:
        comparison_figures_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(comparison_tables_dir / "scenario_comparison.csv", _batch_summary_rows(summaries))
    _write_csv(comparison_tables_dir / "zone_comparison.csv", _batch_zone_rows(summaries))
    _write_csv(comparison_tables_dir / "milestone_comparison.csv", _batch_milestone_rows(summaries))

    if include_plots:
        _generate_batch_plots(comparison_figures_dir, summaries)

    lines = [
        "# Chapter 6 Batch Comparison",
        "",
        f"- Scenario count: {len(summaries)}",
        f"- Comparison tables: {comparison_tables_dir}",
    ]
    if include_plots:
        lines.append(f"- Comparison figures: {comparison_figures_dir}")
    lines.extend(
        [
            "",
            "## Scenario Summary",
            "| Scenario | Kills | Losses | Stable Peak | Gate Pass(unique) | Relay Success(unique) | Picture Match |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for summary in summaries:
        lines.append(
            "| {scenario} | {kills} | {losses} | {stable} | {gate:.1f}% | {relay:.1f}% | {match:.1f}% |".format(
                scenario=summary.get("scenario_id", ""),
                kills=int(summary.get("enemy_kill_count", 0) or 0),
                losses=int(summary.get("friendly_loss_count", 0) or 0),
                stable=int(summary.get("stable_ready_peak", 0) or 0),
                gate=100.0 * float(summary.get("gate_pass_rate", 0.0) or 0.0),
                relay=100.0 * float(summary.get("relay_success_rate", 0.0) or 0.0),
                match=100.0 * float(summary.get("risk_picture_exact_match_ratio", 0.0) or 0.0),
            )
        )
    (batch_dir / "README_batch_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "comparison_tables_dir": str(comparison_tables_dir),
        "comparison_figures_dir": str(comparison_figures_dir) if include_plots else "",
        "comparison_table_count": len(list(comparison_tables_dir.glob("*.csv"))),
        "comparison_figure_count": len(list(comparison_figures_dir.glob("*.png"))) if include_plots else 0,
    }


def _resolve_scenario_ids(scenario_arg: str) -> List[str]:
    requested = str(scenario_arg or "").strip()
    available = {spec.scenario_id.upper(): spec.scenario_id for spec in iter_scenario_specs()}
    if not requested or requested.upper() == "ALL":
        return [available[key] for key in sorted(available.keys())]

    scenario_ids: List[str] = []
    for token in requested.replace(";", ",").split(","):
        item = token.strip().upper()
        if not item:
            continue
        if item not in available:
            raise ValueError(f"Unknown scenario id: {token.strip()}")
        scenario_ids.append(available[item])

    deduped: List[str] = []
    seen: set[str] = set()
    for scenario_id in scenario_ids:
        if scenario_id not in seen:
            deduped.append(scenario_id)
            seen.add(scenario_id)
    return deduped


def run_ch6_validation(
    scenario_id: str,
    *,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    mode: str = "proposed",
    output_root: Optional[Path] = None,
    make_plots: bool = True,
) -> Dict[str, object]:
    spec = get_scenario_spec(scenario_id)
    scenario = copy.deepcopy(spec.scenario)
    if hasattr(scenario, "reset_runtime_state"):
        scenario.reset_runtime_state()
    output_root = Path(output_root) if output_root else OUTPUT_BASE_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{spec.scenario_id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_overrides = {
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "FRIEND_BASELINE_MODEL": "SU27",
        "ENEMY_BASELINE_MODEL": "F16",
        "CAP_VERIFICATION": "0",
        "CAP_AWACS_DETERMINISTIC": "0",
        "CAP_AWACS_FIXED_CENTER_KM": "",
        "CAP_VERIFY_TABLE": "0",
        "CAP_DEBUG_PRINT": "0",
        "CAP_ROOTCAUSE_TRACE": "1",
        "CAP_B0100_DEEP_TRACE": "0",
        "CAP_ALL_B_DEEP_TRACE": "0",
        "CAP_ENEMY_F16_NATIVE_ENABLED": "0",
        "CAP_NEW_ENEMY_MANEUVER_AI_ENABLED": "0",
        "CAP_ENEMY_USE_PAIRAWARE_UNIFIED_ENABLED": "0",
        "ENEMY_DISABLE_WAVE_MODE": "0",
        "CAP_ENEMY_FRIENDLY_BRIDGE_ENABLED": "0",
        "CAP_ENEMY_RULE_LOWLEVEL_ENABLED": "0",
        "CAP_ENEMY_DIRECT_RTB_ENABLED": "0",
        "CAP_ENEMY_DIRECT_RTB_DISTANCE_KM": "40",
        "ENEMY_TURNBACK_DISTANCE_KM": "46",
        "ENEMY_HARD_STANDOFF_DISTANCE_KM": "40",
        "ENEMY_REATTACK_DISTANCE_KM": "72",
        "ENEMY_REGROUP_RELEASE_DISTANCE_KM": "72",
        "ENEMY_PULLBACK_DISTANCE_KM": "28",
        "ENEMY_REGROUP_TIMEOUT_S": "10",
        "ENEMY_REGROUP_MIN_HOLD_S": "6",
        "ENEMY_REGROUP_DEPART_MARGIN_KM": "12",
        "ENEMY_TURN_PHASE_S": "5",
        "ENEMY_REATTACK_HOLD_S": "2",
        "ENEMY_TURN_SOUTH_MIN_HOLD_S": "5",
        "ENEMY_TURN_SOUTH_DEPART_MARGIN_KM": "4",
        "ENEMY_RECOVERY_HOLD_S": "5",
        "ENEMY_RTB_ENABLED": "0",
        "ENEMY_SECOND_ATTACK_PROB": "0",
        "CAP_ENEMY_SIM_RECREATE_ENABLED": "1",
        "CAP_ENEMY_SIM_RECREATE_MAX_COUNT": "8",
        "CAP_ENEMY_SIM_RECREATE_WINDOW": "12",
        "CAP_ENEMY_SIM_RECREATE_COOLDOWN_STEPS": "220",
        "CAP_FRIENDLY_SIM_RECREATE_ENABLED": "1",
        "CAP_FRIENDLY_SIM_RECREATE_MAX_COUNT": "4",
        "CAP_FRIENDLY_SIM_RECREATE_WINDOW": "12",
        "CAP_FRIENDLY_SIM_RECREATE_COOLDOWN_STEPS": "260",
        "CAP_FORCE_TACTIC": "",
        "CAP_ENEMY_FORCE_TACTIC": "",
        "CAP_ENEMY_GROUP1_FORCE_TACTIC": "",
        "CAP_ENEMY_GROUP2_FORCE_TACTIC": "",
        "CAP_ENEMY_SIMPLE": "1" if spec.enemy_control_mode in ("scripted", "hybrid_opening") else "0",
    }
    if seed is not None:
        env_overrides["CAP_AWACS_SEED"] = str(int(seed))

    max_steps = int(max_steps or spec.max_steps)
    temp_config_path: Optional[Path] = None
    env = None
    patrol_task = None
    timeline_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    prev_state: Dict[str, object] = {}

    with _temporary_env(env_overrides):
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
        _apply_outer_compat_patches()
        _refresh_cap_debug_flags_from_env()
        log_file = setup_logging(str(output_dir), f"{spec.scenario_id}_{mode}")
        logging.info("=" * 72)
        logging.info("Chapter 6 validation start: %s | %s", spec.scenario_id, spec.title)
        logging.info("Focus: %s", spec.thesis_focus)
        seed_label = "native-random" if seed is None else str(int(seed))
        logging.info("Enemy mode: %s | seed=%s | max_steps=%s", spec.enemy_control_mode, seed_label, max_steps)
        logging.info("=" * 72)
        try:
            base_config = CAP_DIR / "config" / "patrol_config.yaml"
            use_native_base_config = (
                spec.scenario_id == "S1"
                and spec.enemy_control_mode == "freeplay"
                and str(mode).strip().lower() == "proposed"
                and seed is None
            )
            if use_native_base_config:
                config_name = prepare_patrol_config(str(CAP_DIR), str(PROJECT_ROOT))
                with base_config.open("r", encoding="utf-8") as handle:
                    config_dict = yaml.safe_load(handle)
                logging.info(
                    "S1 uses unmodified run_cap_simulation patrol_config with native random flow."
                )
            else:
                config_dict = generate_scenario_config(scenario, str(base_config))
                config_dict["cap_experiment_mode"] = mode
                if seed is not None:
                    config_dict["cap_rng_seed"] = int(seed)
                    config_dict["cap_awacs_seed"] = int(seed)

                config_name = f"ch6_validation_{spec.scenario_id.lower()}_{timestamp}"
                temp_config_path = PROJECT_ROOT / "envs" / "JSBSim" / "configs" / f"{config_name}.yaml"
                with temp_config_path.open("w", encoding="utf-8") as handle:
                    yaml.dump(config_dict, handle, allow_unicode=True, sort_keys=False)

            env = CAPEnv(config_name, my_aircraft_type="su27sk", enemy_aircraft_type="f16")
            env.max_steps = max_steps
            patrol_task = create_patrol_task(env, allow_patrol_fallback=False)
            env.task = patrol_task
            env.reset()
            env.max_steps = max_steps
            set_sim_log_clock(int(getattr(env, "current_step", 0)), 0.0)
            patrol_display = detect_patrol_display(env)
            last_heading_states: Dict[str, Dict[str, float]] = {}
            last_structured_states: Dict[str, object] = {}

            if str(scenario.awacs_status.value).lower() != "normal":
                patrol_task.awacs = ScenarioAwareAwacsProxy(patrol_task.awacs, scenario)

            enemy_target_headings: Dict[str, object] = {}
            enemy_target_altitudes_ft: Dict[str, float] = {}
            opening_profile: Dict[str, Any] = {}
            opening_release_time_s = 0.0
            opening_released = False
            if spec.enemy_control_mode == "scripted":
                for enemy_id in ("B0100", "B0200", "B0300", "B0400"):
                    init_state = config_dict["aircraft_configs"][enemy_id]["init_state"]
                    enemy_target_altitudes_ft[enemy_id] = float(init_state.get("ic_h_sl_ft", 30000.0))
                enemy_target_headings["_target_altitudes_ft"] = dict(enemy_target_altitudes_ft)
                patrol_task.enemy_scenario = scenario
                patrol_task.enemy_target_headings = enemy_target_headings
                patrol_task.enemy_target_altitudes_ft = enemy_target_altitudes_ft
            elif spec.enemy_control_mode == "hybrid_opening":
                patrol_task.enemy_scenario = scenario
                if hasattr(scenario, "get_opening_script_profile"):
                    opening_profile = dict(scenario.get_opening_script_profile() or {})
                for enemy_id in ("B0100", "B0200", "B0300", "B0400"):
                    init_state = config_dict["aircraft_configs"][enemy_id]["init_state"]
                    enemy_target_altitudes_ft[enemy_id] = float(init_state.get("ic_h_sl_ft", 30000.0))
                for enemy_id, heading_deg in dict(opening_profile.get("target_headings_deg", {}) or {}).items():
                    enemy_target_headings[str(enemy_id)] = float(heading_deg)
                for enemy_id, altitude_km in dict(opening_profile.get("target_altitudes_km", {}) or {}).items():
                    enemy_target_altitudes_ft[str(enemy_id)] = float(altitude_km) * 3280.84
                if enemy_target_altitudes_ft:
                    enemy_target_headings["_target_altitudes_ft"] = dict(enemy_target_altitudes_ft)
                patrol_task.enemy_target_headings = enemy_target_headings
                patrol_task.enemy_target_altitudes_ft = enemy_target_altitudes_ft
                opening_release_time_s = float(opening_profile.get("duration_s", 0.0) or 0.0)
                if opening_release_time_s > 0.0:
                    logging.info(
                        "Hybrid opening control enabled for %.1fs before native enemy AI takeover.",
                        opening_release_time_s,
                    )
            elif spec.enemy_control_mode == "native_profile":
                patrol_task.enemy_scenario = scenario
                if hasattr(scenario, "get_validation_profile"):
                    patrol_task.enemy_validation_profile = scenario.get_validation_profile()

            acmi_path = output_dir / f"{spec.scenario_id}_{timestamp}.txt.acmi"
            acmi_recorder = AcmiRecorder()
            acmi_recorder.reset()
            acmi_recorder.write_header(str(acmi_path))

            dt = float(getattr(env, "time_interval", 0.2) or 0.2)
            for step in range(1, max_steps + 1):
                time_s = step * dt
                set_sim_log_clock(step, time_s)
                if spec.enemy_control_mode == "scripted":
                    friendlies = _truth_positions(patrol_task, env, "A")
                    enemies = _truth_positions(patrol_task, env, "B")
                    current_distance = 999.0
                    if friendlies and enemies:
                        friendly_center = np.mean(np.asarray(list(friendlies.values()), dtype=float), axis=0)
                        current_distance = min(
                            float(np.linalg.norm(np.asarray(pos, dtype=float) - friendly_center))
                            for pos in enemies.values()
                        )
                    enemy_target_headings = apply_enemy_maneuver(
                        env,
                        scenario,
                        float(current_distance),
                        float(time_s),
                        enemy_target_headings,
                    )
                    patrol_task.enemy_target_headings = enemy_target_headings
                    if isinstance(enemy_target_headings.get("_target_altitudes_ft"), dict):
                        enemy_target_altitudes_ft.update(enemy_target_headings["_target_altitudes_ft"])
                        patrol_task.enemy_target_altitudes_ft = enemy_target_altitudes_ft
                elif spec.enemy_control_mode == "hybrid_opening":
                    if opening_release_time_s > 0.0 and time_s <= opening_release_time_s:
                        os.environ["CAP_ENEMY_SIMPLE"] = "1"
                        patrol_task.enemy_target_headings = enemy_target_headings
                        patrol_task.enemy_target_altitudes_ft = enemy_target_altitudes_ft
                    else:
                        os.environ["CAP_ENEMY_SIMPLE"] = "0"
                        if not opening_released:
                            enemy_target_headings = {}
                            enemy_target_altitudes_ft = {}
                            patrol_task.enemy_target_headings = enemy_target_headings
                            patrol_task.enemy_target_altitudes_ft = enemy_target_altitudes_ft
                            patrol_task.enemy_scenario = None
                            opening_released = True
                            logging.info(
                                "[%5.1fs] Released CH6 opening controls; enemy AI returned to native adapter.",
                                float(time_s),
                            )

                actions = build_step_actions(env, patrol_task)
                env.step(actions)
                set_sim_log_clock(int(getattr(env, "current_step", step)), time_s)
                maybe_log_guidance_snapshot(patrol_task, step, dt)
                acmi_recorder.write_frame(str(acmi_path), env, time_s)
                log_heading_changes(env, patrol_task, patrol_display, last_heading_states, time_s)
                log_structured_battle_events(env, patrol_task, last_structured_states, time_s)

                row = _collect_step_row(spec, patrol_task, env, time_s)
                timeline_rows.append(row)
                _update_events(event_rows, prev_state, row)

                if step % int(max(1, round(30.0 / dt))) == 0:
                    logging.info(
                        "[%5.1fs] state=%s mode=%s enemy(L/M/H)=%d/%d/%d left=%s right=%s",
                        time_s,
                        row["cap_state"],
                        row["detect_mode"],
                        row["truth_low_count"],
                        row["truth_medium_count"],
                        row["truth_high_count"],
                        row["left_tactic"],
                        row["right_tactic"],
                    )

                if int(row["enemy_alive"]) == 0 or int(row["friendly_alive"]) == 0:
                    logging.info("Simulation terminated early at %.1fs because one side has no surviving aircraft.", time_s)
                    break
                if time_s >= 1200.0:
                    break

            log_final_guidance_summary(patrol_task)
            log_final_document_summary(patrol_task)
            summary = _scenario_summary(spec, patrol_task, timeline_rows, event_rows=event_rows)
            summary["log_file"] = str(log_file)
            summary["acmi_file"] = str(acmi_path)
            summary["output_dir"] = str(output_dir)
            summary["detailed_figures_dir"] = str(output_dir / "figures")
            summary["tables_dir"] = str(output_dir / "tables")

            _write_csv(output_dir / "timeline.csv", timeline_rows)
            _write_csv(output_dir / "events.csv", event_rows)
            if make_plots:
                _generate_plots(output_dir, timeline_rows, summary=summary)
                summary.update(_generate_thesis_asset_bundle(Path(log_file), output_dir, summary=summary))
            else:
                summary.update(
                    {
                        "thesis_assets_dir": "",
                        "thesis_assets_figure_dir": "",
                        "thesis_assets_csv_dir": "",
                        "thesis_assets_figure_count": 0,
                        "thesis_assets_csv_count": 0,
                        "thesis_assets_generated": False,
                        "thesis_assets_error": "",
                    }
                )
            summary.update(_write_detail_tables(output_dir, summary, timeline_rows, event_rows))
            (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_summary_markdown(
                output_dir / "summary.md",
                summary,
                output_dir,
                include_plots=bool(make_plots),
            )
            artifact_rows = _artifact_index_rows(output_dir, summary)
            _write_csv(output_dir / "tables" / "artifact_index.csv", artifact_rows)
            summary["artifact_index_csv"] = str(output_dir / "tables" / "artifact_index.csv")
            summary["artifact_file_count"] = len(artifact_rows)
            _write_csv(output_dir / "tables" / "summary_metrics.csv", _summary_metric_rows(summary))
            (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            _log_scenario_summary(summary)
            return summary
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if temp_config_path is not None and temp_config_path.exists():
                try:
                    temp_config_path.unlink()
                except Exception:
                    logging.warning("Failed to remove temporary config: %s", temp_config_path)


def run_ch6_validation_batch(
    scenario_ids: Iterable[str],
    *,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    mode: str = "proposed",
    output_root: Optional[Path] = None,
    make_plots: bool = True,
) -> Dict[str, object]:
    scenario_list = list(scenario_ids)
    output_root = Path(output_root) if output_root else OUTPUT_BASE_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = "ALL" if len(scenario_list) > 1 else scenario_list[0]
    batch_dir = output_root / f"{batch_name}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    for scenario_id in scenario_list:
        summaries.append(
            run_ch6_validation(
                scenario_id,
                seed=seed,
                max_steps=max_steps,
                mode=mode,
                output_root=batch_dir,
                make_plots=make_plots,
            )
        )

    batch_summary = {
        "batch_output_dir": str(batch_dir),
        "scenario_ids": scenario_list,
        "scenario_count": len(scenario_list),
        "seed": seed,
        "mode": mode,
        "summaries": summaries,
    }
    batch_summary.update(_write_batch_reports(batch_dir, summaries, include_plots=bool(make_plots)))
    (batch_dir / "batch_summary.json").write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return batch_summary


def _scenario_seed(base_seed: int, scenario_id: str) -> int:
    text = str(scenario_id or "").strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    try:
        offset = int(digits) if digits else 0
    except Exception:
        offset = 0
    return int(base_seed) + offset


def _compare_row(proposed: Dict[str, object], baseline: Dict[str, object]) -> Dict[str, object]:
    def _num(item: Dict[str, object], key: str) -> Optional[float]:
        value = item.get(key)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _delta_b_minus_p(key: str) -> Optional[float]:
        p = _num(proposed, key)
        b = _num(baseline, key)
        if p is None or b is None:
            return None
        return b - p

    def _delta_p_minus_b(key: str) -> Optional[float]:
        p = _num(proposed, key)
        b = _num(baseline, key)
        if p is None or b is None:
            return None
        return p - b

    return {
        "scenario_id": proposed.get("scenario_id") or baseline.get("scenario_id"),
        "proposed_output_dir": proposed.get("output_dir", ""),
        "baseline_output_dir": baseline.get("output_dir", ""),
        "proposed_radar_detection_coverage_ratio_post_activation": proposed.get("radar_detection_coverage_ratio_post_activation"),
        "baseline_radar_detection_coverage_ratio_post_activation": baseline.get("radar_detection_coverage_ratio_post_activation"),
        "delta_radar_detection_coverage_ratio_post_activation_p_minus_b": _delta_p_minus_b("radar_detection_coverage_ratio_post_activation"),
        "proposed_radar_detection_coverage_ratio_alive_targets_post_activation": proposed.get("radar_detection_coverage_ratio_alive_targets_post_activation"),
        "baseline_radar_detection_coverage_ratio_alive_targets_post_activation": baseline.get("radar_detection_coverage_ratio_alive_targets_post_activation"),
        "delta_radar_detection_coverage_ratio_alive_targets_post_activation_p_minus_b": _delta_p_minus_b("radar_detection_coverage_ratio_alive_targets_post_activation"),
        "proposed_radar_full_coverage_continuity_post_activation": proposed.get("radar_full_coverage_continuity_post_activation"),
        "baseline_radar_full_coverage_continuity_post_activation": baseline.get("radar_full_coverage_continuity_post_activation"),
        "delta_radar_full_coverage_continuity_post_activation_p_minus_b": _delta_p_minus_b("radar_full_coverage_continuity_post_activation"),
        "proposed_radar_full_coverage_continuity_alive_targets_post_activation": proposed.get("radar_full_coverage_continuity_alive_targets_post_activation"),
        "baseline_radar_full_coverage_continuity_alive_targets_post_activation": baseline.get("radar_full_coverage_continuity_alive_targets_post_activation"),
        "delta_radar_full_coverage_continuity_alive_targets_post_activation_p_minus_b": _delta_p_minus_b("radar_full_coverage_continuity_alive_targets_post_activation"),
        "proposed_first_stable_ready_time_s": proposed.get("first_stable_ready_time_s"),
        "baseline_first_stable_ready_time_s": baseline.get("first_stable_ready_time_s"),
        "delta_first_stable_ready_time_s_b_minus_p": _delta_b_minus_p("first_stable_ready_time_s"),
        "proposed_gate_pass_rate": proposed.get("gate_pass_rate"),
        "baseline_gate_pass_rate": baseline.get("gate_pass_rate"),
        "delta_gate_pass_rate_p_minus_b": _delta_p_minus_b("gate_pass_rate"),
        "proposed_relay_success_rate": proposed.get("relay_success_rate"),
        "baseline_relay_success_rate": baseline.get("relay_success_rate"),
        "delta_relay_success_rate_p_minus_b": _delta_p_minus_b("relay_success_rate"),
        "proposed_intent_truth_accuracy": proposed.get("intent_truth_accuracy"),
        "baseline_intent_truth_accuracy": baseline.get("intent_truth_accuracy"),
        "delta_intent_truth_accuracy_p_minus_b": _delta_p_minus_b("intent_truth_accuracy"),
        "proposed_high_risk_breach_time_s": proposed.get("high_risk_breach_time_s"),
        "baseline_high_risk_breach_time_s": baseline.get("high_risk_breach_time_s"),
        "delta_high_risk_breach_time_s_b_minus_p": _delta_b_minus_p("high_risk_breach_time_s"),
        "proposed_enemy_kill_count": proposed.get("enemy_kill_count"),
        "baseline_enemy_kill_count": baseline.get("enemy_kill_count"),
        "proposed_friendly_loss_count": proposed.get("friendly_loss_count"),
        "baseline_friendly_loss_count": baseline.get("friendly_loss_count"),
        "proposed_exchange_ratio": (
            float(proposed.get("enemy_kill_count", 0) or 0) / float(proposed.get("friendly_loss_count", 0) or 1)
            if int(proposed.get("friendly_loss_count", 0) or 0) > 0
            else float(proposed.get("enemy_kill_count", 0) or 0)
        ),
        "baseline_exchange_ratio": (
            float(baseline.get("enemy_kill_count", 0) or 0) / float(baseline.get("friendly_loss_count", 0) or 1)
            if int(baseline.get("friendly_loss_count", 0) or 0) > 0
            else float(baseline.get("enemy_kill_count", 0) or 0)
        ),
    }


def run_ch6_validation_compare(
    scenario_ids: Iterable[str],
    *,
    base_seed: int = 1,
    max_steps: Optional[int] = None,
    output_root: Optional[Path] = None,
    make_plots: bool = True,
) -> Dict[str, object]:
    scenario_list = list(scenario_ids)
    output_root = Path(output_root) if output_root else OUTPUT_BASE_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = "COMPARE_ALL" if len(scenario_list) > 1 else f"COMPARE_{scenario_list[0]}"
    batch_dir = output_root / f"{batch_name}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    proposed_summaries: List[Dict[str, object]] = []
    baseline_summaries: List[Dict[str, object]] = []
    compare_rows: List[Dict[str, object]] = []
    all_summaries: List[Dict[str, object]] = []

    for scenario_id in scenario_list:
        scenario_seed = _scenario_seed(int(base_seed), scenario_id)
        proposed = run_ch6_validation(
            scenario_id,
            seed=scenario_seed,
            max_steps=max_steps,
            mode="proposed",
            output_root=batch_dir,
            make_plots=make_plots,
        )
        baseline = run_ch6_validation(
            scenario_id,
            seed=scenario_seed,
            max_steps=max_steps,
            mode="baseline",
            output_root=batch_dir,
            make_plots=make_plots,
        )
        proposed_summaries.append(proposed)
        baseline_summaries.append(baseline)
        all_summaries.extend((proposed, baseline))
        compare_rows.append(_compare_row(proposed, baseline))

    batch_summary = {
        "batch_output_dir": str(batch_dir),
        "scenario_ids": scenario_list,
        "scenario_count": len(scenario_list),
        "compare_mode": True,
        "base_seed": int(base_seed),
        "summaries": all_summaries,
        "proposed_summaries": proposed_summaries,
        "baseline_summaries": baseline_summaries,
        "compare_rows": compare_rows,
    }
    batch_summary.update(_write_batch_reports(batch_dir, all_summaries, include_plots=bool(make_plots)))
    compare_dir = Path(batch_summary["comparison_tables_dir"])
    _write_csv(compare_dir / "proposed_vs_baseline.csv", compare_rows)
    (batch_dir / "batch_summary.json").write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return batch_summary


def _build_parser(default_scenario: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Chapter 6 CAP validation scenarios.")
    parser.add_argument("--scenario", default=default_scenario or "S1", help="Scenario id: S1 / S2 / S3 / ALL / comma list")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed. Omit it to match run_cap_simulation native randomness.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override max step count.")
    parser.add_argument("--mode", default="proposed", help="Experiment mode written into CAP config.")
    parser.add_argument("--compare", action="store_true", help="Run proposed and baseline pairwise under the same per-scenario seed.")
    parser.add_argument("--output-root", default=str(OUTPUT_BASE_DIR), help="Output root directory.")
    parser.add_argument("--no-plots", action="store_true", help="Disable png figure generation.")
    parser.add_argument("--list", action="store_true", help="List available Chapter 6 scenarios.")
    return parser


def main(default_scenario: Optional[str] = None) -> int:
    parser = _build_parser(default_scenario=default_scenario)
    args = parser.parse_args()
    if args.list:
        for spec in iter_scenario_specs():
            print(f"{spec.scenario_id}: {spec.title} | mode={spec.enemy_control_mode} | zone={spec.expected_zone}")
        return 0

    try:
        scenario_ids = _resolve_scenario_ids(args.scenario)
    except ValueError as exc:
        parser.error(str(exc))

    if args.compare:
        result = run_ch6_validation_compare(
            scenario_ids,
            base_seed=int(args.seed if args.seed is not None else 1),
            max_steps=args.steps,
            output_root=Path(args.output_root),
            make_plots=not bool(args.no_plots),
        )
    elif len(scenario_ids) == 1:
        result = run_ch6_validation(
            scenario_ids[0],
            seed=args.seed,
            max_steps=args.steps,
            mode=args.mode,
            output_root=Path(args.output_root),
            make_plots=not bool(args.no_plots),
        )
    else:
        result = run_ch6_validation_batch(
            scenario_ids,
            seed=args.seed,
            max_steps=args.steps,
            mode=args.mode,
            output_root=Path(args.output_root),
            make_plots=not bool(args.no_plots),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cooperative_chain_metric_rows(summary: Dict[str, object]) -> List[Dict[str, object]]:
    metrics = [
        ("radar_first_activation_time_s", "首次雷达激活时间(s)"),
        ("full_detect_time_s", "全目标探测完成时间(s)"),
        ("full_detect_delay_s", "相对雷达激活的全探测时延(s)"),
        ("radar_detection_coverage_ratio_alive_targets_post_activation", "雷达激活后平均探测覆盖率(存活目标)"),
        ("radar_full_coverage_continuity_alive_targets_post_activation", "雷达激活后全覆盖连续率(存活目标)"),
        ("radar_detection_coverage_ratio_post_activation", "雷达激活后平均探测覆盖率"),
        ("radar_full_coverage_continuity_post_activation", "雷达激活后全覆盖连续率"),
        ("radar_any_track_continuity_post_activation", "雷达激活后任意目标连续率"),
        ("first_stable_ready_time_s", "首次稳定就绪时间(s)"),
        ("stable_ready_peak", "稳定就绪峰值"),
        ("stable_tracking_target_peak", "稳定跟踪目标峰值"),
        ("first_gate_pass_time_s", "首次发射门通过时间(s)"),
        ("gate_pass_count", "发射门通过次数(unique)"),
        ("gate_block_count", "发射门阻断次数(unique)"),
        ("gate_total_request_count", "发射门总请求次数(unique)"),
        ("gate_pass_rate", "发射门通过率(unique)"),
        ("gate_ready_pass_count", "稳定就绪后发射门通过次数(unique)"),
        ("gate_ready_block_count", "稳定就绪后发射门阻断次数(unique)"),
        ("gate_ready_total_request_count", "稳定就绪后发射门总请求次数(unique)"),
        ("gate_ready_pass_rate", "稳定就绪后发射门通过率(unique)"),
        ("gate_pass_raw_count", "发射门通过次数(raw)"),
        ("gate_block_raw_count", "发射门阻断次数(raw)"),
        ("gate_total_request_raw_count", "发射门总请求次数(raw)"),
        ("gate_pass_rate_raw", "发射门通过率(raw)"),
        ("gate_block_stable_not_ready_count", "稳定跟踪未满足阻断次数"),
        ("gate_block_friendly_safe_count", "友机安全阻断次数"),
        ("gate_block_long_shot_count", "远距长射阻断次数"),
        ("gate_block_launch_exec_fail_count", "发射执行失败次数"),
        ("gate_block_other_count", "其他阻断次数"),
        ("first_relay_success_time_s", "首次接力制导成功时间(s)"),
        ("relay_success_count", "接力制导成功次数(unique)"),
        ("relay_attempt_count", "接力制导尝试次数(unique)"),
        ("relay_success_rate", "接力制导成功率(unique)"),
        ("relay_success_raw_count", "接力制导成功次数(raw)"),
        ("relay_attempt_raw_count", "接力制导尝试次数(raw)"),
        ("relay_success_rate_raw", "接力制导成功率(raw)"),
        ("relay_active_time_s", "接力链激活时长(s)"),
        ("active_guided_missile_peak", "在制导导弹峰值"),
    ]
    percent_keys = {
        "gate_pass_rate",
        "gate_ready_pass_rate",
        "gate_pass_rate_raw",
        "relay_success_rate",
        "relay_success_rate_raw",
    }
    rows: List[Dict[str, object]] = []
    for key, label in metrics:
        value = summary.get(key)
        if key in percent_keys:
            value = float(value or 0.0) * 100.0 if value not in (None, "") else value
            rows.append({"metric_key": key, "metric_name": label, "value": _metric_cell(value, digits=3), "unit": "%"})
        else:
            rows.append({"metric_key": key, "metric_name": label, "value": _metric_cell(value, digits=3), "unit": ""})
    return rows


def _enemy_zone_profile_rows_from_summary(summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    zone_profiles = list(summary.get("enemy_zone_profiles", []) or [])
    zone_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "OUTSIDE": 0}
    for item in sorted(
        zone_profiles,
        key=lambda row: (
            -zone_rank.get(str(row.get("deepest_zone", "") or "").upper(), -1),
            str(row.get("enemy_id", "") or ""),
        ),
    ):
        rows.append(
            {
                "enemy_id": item.get("enemy_id", ""),
                "deepest_zone": item.get("deepest_zone", ""),
                "position_source": item.get("position_source", ""),
                "first_medium_time_s": _metric_cell(item.get("first_medium_time_s"), digits=3),
                "first_high_time_s": _metric_cell(item.get("first_high_time_s"), digits=3),
                "medium_time_s": _metric_cell(item.get("medium_time_s"), digits=3),
                "high_time_s": _metric_cell(item.get("high_time_s"), digits=3),
                "closest_to_high_line_km": _metric_cell(item.get("closest_to_high_line_km"), digits=3),
                "closest_to_medium_line_km": _metric_cell(item.get("closest_to_medium_line_km"), digits=3),
                "high_penetration_km": _metric_cell(item.get("high_penetration_km"), digits=3),
                "medium_penetration_km": _metric_cell(item.get("medium_penetration_km"), digits=3),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
