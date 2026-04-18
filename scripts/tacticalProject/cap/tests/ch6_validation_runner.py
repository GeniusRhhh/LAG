from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


from cap.cap_task import CAPTask
from cap.faor_manager import FAORManager, RiskZone
from run_detection_verification_real import (  # noqa: E402
    A0100_LAT,
    A0100_LON,
    CAPEnvForVerification,
    DEG_TO_KM,
    apply_enemy_maneuver,
    generate_scenario_config,
    setup_logging,
    write_acmi_frame,
    write_acmi_header,
)

from ch6_validation_scenarios import (  # noqa: E402
    Ch6ScenarioSpec,
    get_scenario_spec,
    iter_scenario_specs,
)


OUTPUT_BASE_DIR = TACTICAL_DIR / "cap_results" / "Chapter6_validation"


def _apply_outer_compat_patches() -> None:
    try:
        import cap.cap_task_impl as cap_task_impl_module
        from cap import cap_task_action_helpers

        if not hasattr(cap_task_impl_module, "_capact"):
            cap_task_impl_module._capact = cap_task_action_helpers
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


def _plot_categorical(ax, times: List[float], values: List[str], title: str) -> None:
    labels = ["UNKNOWN"]
    for item in values:
        norm = str(item or "").strip() or "UNKNOWN"
        if norm not in labels:
            labels.append(norm)
    mapping = {label: idx for idx, label in enumerate(labels)}
    y_values = [mapping[str(item or "").strip() or "UNKNOWN"] for item in values]
    ax.step(times, y_values, where="post")
    ax.set_title(title)
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels(labels)
    ax.grid(True, alpha=0.3)


def _generate_plots(output_dir: Path, timeline_rows: List[Dict[str, object]]) -> None:
    if not HAS_MPL or not timeline_rows:
        return
    times = [float(row["time_s"]) for row in timeline_rows]
    bullseye_series = [
        float(row["nearest_enemy_bullseye_km"]) if row.get("nearest_enemy_bullseye_km") not in (None, "") else np.nan
        for row in timeline_rows
    ]
    friendly_distance_series = [
        float(row["nearest_enemy_to_friendly_km"]) if row.get("nearest_enemy_to_friendly_km") not in (None, "") else np.nan
        for row in timeline_rows
    ]
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    axes[0].plot(times, [int(row["truth_low_count"]) for row in timeline_rows], label="Low")
    axes[0].plot(times, [int(row["truth_medium_count"]) for row in timeline_rows], label="Medium")
    axes[0].plot(times, [int(row["truth_high_count"]) for row in timeline_rows], label="High")
    axes[0].set_title("Enemy Count in Risk Zones")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    _plot_categorical(axes[1], times, [str(row["left_tactic"]) for row in timeline_rows], "Left Pair Tactic")
    _plot_categorical(axes[2], times, [str(row["right_tactic"]) for row in timeline_rows], "Right Pair Tactic")
    axes[3].plot(times, bullseye_series, label="Nearest enemy to bullseye")
    axes[3].plot(times, friendly_distance_series, label="Nearest enemy to friendly")
    axes[3].plot(times, [float(row["friendly_alive"]) for row in timeline_rows], label="Friendly alive")
    axes[3].plot(times, [float(row["enemy_alive"]) for row in timeline_rows], label="Enemy alive")
    axes[3].set_title("Distance / Alive Curves")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "overview.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    axes[0].plot(times, [int(row["awacs_track_count"]) for row in timeline_rows], label="AWACS")
    axes[0].plot(times, [int(row["radar_track_count"]) for row in timeline_rows], label="Radar unique")
    axes[0].plot(times, [int(row["stable_ready_count"]) for row in timeline_rows], label="Stable dual-track")
    axes[0].set_title("Detection / Tracking")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, [int(row["gate_pass_count"]) for row in timeline_rows], label="Gate pass")
    axes[1].plot(times, [int(row["gate_block_count"]) for row in timeline_rows], label="Gate block")
    axes[1].set_title("Prelaunch Gate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, [int(row["relay_attempt_count"]) for row in timeline_rows], label="Relay attempt")
    axes[2].plot(times, [int(row["relay_success_count"]) for row in timeline_rows], label="Relay success")
    axes[2].set_title("Relay Guidance")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times, [int(row["missile_launch_count"]) for row in timeline_rows], label="Launch total")
    axes[3].plot(times, [int(row["missile_outcome_count"]) for row in timeline_rows], label="Outcome total")
    axes[3].set_title("Missile Events")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "guidance.png", dpi=160)
    plt.close(fig)


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

    intent_target = ""
    intent_name = ""
    intent_threat = ""
    intent_conf = 0.0
    intent_adapter = getattr(patrol_task, "intent_adapter", None)
    if intent_adapter is not None and hasattr(intent_adapter, "get_highest_threat"):
        highest = intent_adapter.get_highest_threat()
        if highest is not None:
            intent_target = str(getattr(highest, "track_id", "") or "")
            intent_name = str(getattr(getattr(highest, "intent", None), "value", "") or "")
            intent_threat = str(getattr(getattr(highest, "threat_level", None), "value", "") or "")
            intent_conf = float(getattr(highest, "confidence", 0.0) or 0.0)

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

    mission_summary = {}
    mission_evaluator = getattr(patrol_task, "mission_evaluator", None)
    if mission_evaluator is not None and hasattr(mission_evaluator, "get_summary"):
        mission_summary = mission_evaluator.get_summary() or {}

    cap_state = getattr(getattr(patrol_task, "cap_state_machine", None), "state", "")
    detect_mode = getattr(getattr(patrol_task, "coop_detection", None), "_mode", "")
    row = {
        "time_s": float(time_s),
        "scenario_id": spec.scenario_id,
        "cap_state": getattr(cap_state, "value", str(cap_state)),
        "detect_mode": getattr(detect_mode, "value", str(detect_mode)),
        "friendly_alive": len(friendly_truth),
        "enemy_alive": len(enemy_truth),
        "friendly_missiles_left": _friendly_missiles_left(env),
        "truth_low_count": low_count,
        "truth_medium_count": medium_count,
        "truth_high_count": high_count,
        "nearest_enemy_bullseye_km": None if not np.isfinite(nearest_bullseye) else float(nearest_bullseye),
        "nearest_enemy_to_friendly_km": None if not np.isfinite(nearest_friendly) else float(nearest_friendly),
        "awacs_track_count": len(awacs_tracks or {}),
        "radar_track_count": len(radar_track_ids),
        "stable_ready_count": len(guidance_snapshot.get("stable_tracking_ready_targets", []) or []),
        "left_tactic": _normalize_tactic_name((left_info or {}).get("tactic")),
        "right_tactic": _normalize_tactic_name((right_info or {}).get("tactic")),
        "left_phase": str((left_info or {}).get("phase", "") or ""),
        "right_phase": str((right_info or {}).get("phase", "") or ""),
        "intent_target": intent_target,
        "intent_name": intent_name,
        "intent_threat": intent_threat,
        "intent_confidence": intent_conf,
        "gate_pass_count": int(guidance_snapshot.get("prelaunch_gate_pass_count", 0)),
        "gate_block_count": int(guidance_snapshot.get("prelaunch_gate_block_count", 0)),
        "relay_attempt_count": int(guidance_snapshot.get("relay_attempt_count", 0)),
        "relay_success_count": int(guidance_snapshot.get("relay_success_count", 0)),
        "active_guided_missile_peak": int(guidance_snapshot.get("active_guided_missile_peak", 0)),
        "missile_launch_count": int(missile_summary.get("launch_count", 0)),
        "missile_outcome_count": int(missile_summary.get("outcome_count", 0)),
        "mission_result": str(mission_summary.get("result", "") or ""),
        "mission_threat_level": str(mission_summary.get("threat_level", "") or ""),
        "mission_runtime_kills": int(mission_summary.get("kills", 0) or 0),
        "zone_by_enemy_json": json.dumps(zone_by_enemy, ensure_ascii=False, sort_keys=True),
    }
    return row


def _update_events(
    events: List[Dict[str, object]],
    prev_state: Dict[str, object],
    row: Dict[str, object],
) -> None:
    time_s = float(row["time_s"])
    for label in ("left_tactic", "right_tactic"):
        prev_value = prev_state.get(label)
        current_value = row[label]
        if prev_value is not None and prev_value != current_value:
            events.append({"time_s": time_s, "event_type": "tactic_change", "field": label, "value": current_value})
        prev_state[label] = current_value

    for label in ("gate_pass_count", "gate_block_count", "relay_attempt_count", "relay_success_count", "missile_launch_count", "missile_outcome_count"):
        prev_value = int(prev_state.get(label, 0))
        current_value = int(row[label])
        if current_value > prev_value:
            events.append({"time_s": time_s, "event_type": "counter_increase", "field": label, "value": current_value})
        prev_state[label] = current_value

    intent_key = f"{row['intent_target']}|{row['intent_name']}"
    prev_intent = prev_state.get("intent_key")
    if row["intent_target"] and intent_key != prev_intent:
        events.append({"time_s": time_s, "event_type": "intent_change", "field": "highest_threat", "value": intent_key})
    prev_state["intent_key"] = intent_key

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


def _scenario_summary(
    spec: Ch6ScenarioSpec,
    patrol_task: CAPTask,
    timeline_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    durations = _row_durations(timeline_rows)
    low_metrics = _zone_metrics(timeline_rows, durations, "truth_low_count")
    medium_metrics = _zone_metrics(timeline_rows, durations, "truth_medium_count")
    high_metrics = _zone_metrics(timeline_rows, durations, "truth_high_count")

    zone_key_map = {
        "LOW": "truth_low_count",
        "MEDIUM": "truth_medium_count",
        "HIGH": "truth_high_count",
    }
    expected_zone_key = zone_key_map.get(str(spec.expected_zone).upper(), "")
    expected_zone_metrics = (
        _zone_metrics(timeline_rows, durations, expected_zone_key)
        if expected_zone_key
        else {"peak_count": 0, "occupied_time_s": 0.0, "breach_events": 0, "first_time_s": None}
    )

    stable_peak = max((int(row.get("stable_ready_count", 0) or 0) for row in timeline_rows), default=0)
    guided_peak = max((int(row.get("active_guided_missile_peak", 0) or 0) for row in timeline_rows), default=0)
    detect_modes = [str(row.get("detect_mode", "") or "") for row in timeline_rows]
    detect_mode_time_s: Dict[str, float] = {}
    for row, dt in zip(timeline_rows, durations):
        mode = str(row.get("detect_mode", "") or "UNKNOWN")
        detect_mode_time_s[mode] = float(detect_mode_time_s.get(mode, 0.0) + float(dt))

    left_expected_rows = sum(
        1 for row in timeline_rows if _matches_expected_tactic(str(row.get("left_tactic", "") or ""), spec.expected_tactic_keywords)
    )
    right_expected_rows = sum(
        1 for row in timeline_rows if _matches_expected_tactic(str(row.get("right_tactic", "") or ""), spec.expected_tactic_keywords)
    )
    expected_tactic_rows = max(left_expected_rows, right_expected_rows)

    mission_result = str(_last_row_value(timeline_rows, "mission_result", "") or "")
    mission_threat_level = str(_last_row_value(timeline_rows, "mission_threat_level", "") or "")
    friendly_alive_end = int(_last_row_value(timeline_rows, "friendly_alive", 0) or 0)
    enemy_alive_end = int(_last_row_value(timeline_rows, "enemy_alive", 0) or 0)

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
        "friendly_loss_count": max(0, 4 - friendly_alive_end),
        "enemy_kill_count": max(0, 4 - enemy_alive_end),
        "friendly_missiles_left_end": int(_last_row_value(timeline_rows, "friendly_missiles_left", 0) or 0),
        "mission_result": mission_result,
        "mission_threat_level": mission_threat_level,
        "mission_runtime_kills": int(_last_row_value(timeline_rows, "mission_runtime_kills", 0) or 0),
        "awacs_track_count_end": int(_last_row_value(timeline_rows, "awacs_track_count", 0) or 0),
        "radar_track_count_end": int(_last_row_value(timeline_rows, "radar_track_count", 0) or 0),
        "stable_ready_peak": int(stable_peak),
        "first_stable_ready_time_s": _first_time_when(timeline_rows, "stable_ready_count", lambda value: int(value or 0) > 0),
        "first_awacs_track_time_s": _first_time_when(timeline_rows, "awacs_track_count", lambda value: int(value or 0) > 0),
        "first_radar_track_time_s": _first_time_when(timeline_rows, "radar_track_count", lambda value: int(value or 0) > 0),
        "detect_mode_switch_count": _count_switches(timeline_rows, "detect_mode"),
        "left_tactic_switch_count": _count_switches(timeline_rows, "left_tactic"),
        "right_tactic_switch_count": _count_switches(timeline_rows, "right_tactic"),
        "detect_mode_time_s": detect_mode_time_s,
        "left_expected_tactic_rows": int(left_expected_rows),
        "right_expected_tactic_rows": int(right_expected_rows),
        "expected_tactic_rows": int(expected_tactic_rows),
        "expected_tactic_observed": bool(expected_tactic_rows > 0),
        "low_risk_peak_count": int(low_metrics["peak_count"]),
        "low_risk_enemy_time_s": float(low_metrics["occupied_time_s"]),
        "first_low_time_s": low_metrics["first_time_s"],
        "medium_risk_peak_count": int(medium_metrics["peak_count"]),
        "medium_risk_enemy_time_s": float(medium_metrics["occupied_time_s"]),
        "first_medium_time_s": medium_metrics["first_time_s"],
        "high_risk_peak_count": int(high_metrics["peak_count"]),
        "high_risk_breach_events": int(high_metrics["breach_events"]),
        "high_risk_breach_time_s": float(high_metrics["occupied_time_s"]),
        "first_high_time_s": high_metrics["first_time_s"],
        "expected_zone_peak_count": int(expected_zone_metrics["peak_count"]),
        "expected_zone_enemy_time_s": float(expected_zone_metrics["occupied_time_s"]),
        "expected_zone_triggered": bool(expected_zone_metrics["peak_count"]),
        "expected_zone_first_time_s": expected_zone_metrics["first_time_s"],
        "min_enemy_bullseye_km": _safe_min_float(timeline_rows, "nearest_enemy_bullseye_km"),
        "min_enemy_to_friendly_km": _safe_min_float(timeline_rows, "nearest_enemy_to_friendly_km"),
        "gate_pass_count": int(_last_row_value(timeline_rows, "gate_pass_count", 0) or 0),
        "gate_block_count": int(_last_row_value(timeline_rows, "gate_block_count", 0) or 0),
        "relay_attempt_count": int(_last_row_value(timeline_rows, "relay_attempt_count", 0) or 0),
        "relay_success_count": int(_last_row_value(timeline_rows, "relay_success_count", 0) or 0),
        "active_guided_missile_peak": int(guided_peak),
        "missile_launch_count": int(_last_row_value(timeline_rows, "missile_launch_count", 0) or 0),
        "missile_outcome_count": int(_last_row_value(timeline_rows, "missile_outcome_count", 0) or 0),
        "intent_target_end": str(_last_row_value(timeline_rows, "intent_target", "") or ""),
        "intent_name_end": str(_last_row_value(timeline_rows, "intent_name", "") or ""),
        "intent_threat_end": str(_last_row_value(timeline_rows, "intent_threat", "") or ""),
        "intent_confidence_end": float(_last_row_value(timeline_rows, "intent_confidence", 0.0) or 0.0),
    }

    relay_attempts = int(summary["relay_attempt_count"])
    gate_total = int(summary["gate_pass_count"]) + int(summary["gate_block_count"])
    summary["relay_success_rate"] = (
        float(summary["relay_success_count"]) / float(relay_attempts) if relay_attempts > 0 else 0.0
    )
    summary["gate_pass_rate"] = (
        float(summary["gate_pass_count"]) / float(gate_total) if gate_total > 0 else 0.0
    )

    if patrol_task is not None:
        guidance_snapshot = {}
        if hasattr(patrol_task, "get_guidance_verification_snapshot"):
            guidance_snapshot = patrol_task.get_guidance_verification_snapshot() or {}
        if guidance_snapshot:
            summary["guidance_snapshot"] = guidance_snapshot

    return summary


def _write_summary_markdown(
    path: Path,
    summary: Dict[str, object],
    output_dir: Path,
    *,
    include_plots: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("Expected zone", summary.get("expected_zone")),
        ("Expected zone triggered", summary.get("expected_zone_triggered")),
        ("Expected zone peak count", summary.get("expected_zone_peak_count")),
        ("Expected zone occupied time (s)", f"{float(summary.get('expected_zone_enemy_time_s', 0.0)):.1f}"),
        ("Low-risk peak count", summary.get("low_risk_peak_count")),
        ("Low-risk occupied time (s)", f"{float(summary.get('low_risk_enemy_time_s', 0.0)):.1f}"),
        ("Medium-risk peak count", summary.get("medium_risk_peak_count")),
        ("Medium-risk occupied time (s)", f"{float(summary.get('medium_risk_enemy_time_s', 0.0)):.1f}"),
        ("High-risk peak count", summary.get("high_risk_peak_count")),
        ("High-risk breach events", summary.get("high_risk_breach_events")),
        ("High-risk occupied time (s)", f"{float(summary.get('high_risk_breach_time_s', 0.0)):.1f}"),
        ("First low time (s)", summary.get("first_low_time_s")),
        ("First medium time (s)", summary.get("first_medium_time_s")),
        ("First high time (s)", summary.get("first_high_time_s")),
        ("Stable-ready peak", summary.get("stable_ready_peak")),
        ("First stable-ready time (s)", summary.get("first_stable_ready_time_s")),
        ("Detection mode switches", summary.get("detect_mode_switch_count")),
        ("Expected tactic observed", summary.get("expected_tactic_observed")),
        ("Gate pass count", summary.get("gate_pass_count")),
        ("Gate pass rate", f"{100.0 * float(summary.get('gate_pass_rate', 0.0)):.1f}%"),
        ("Relay success count", summary.get("relay_success_count")),
        ("Relay success rate", f"{100.0 * float(summary.get('relay_success_rate', 0.0)):.1f}%"),
        ("Missile launch count", summary.get("missile_launch_count")),
        ("Enemy kills", summary.get("enemy_kill_count")),
        ("Friendly losses", summary.get("friendly_loss_count")),
        ("Min enemy-to-friendly distance (km)", summary.get("min_enemy_to_friendly_km")),
        ("Min enemy-to-bullseye distance (km)", summary.get("min_enemy_bullseye_km")),
        ("Mission result", summary.get("mission_result")),
        ("Mission threat level", summary.get("mission_threat_level")),
    ]

    lines = [
        f"# {summary.get('scenario_id', '')} {summary.get('title', '')}",
        "",
        f"- Thesis focus: {summary.get('thesis_focus', '')}",
        f"- Design intent: {summary.get('design_intent', '')}",
        f"- Enemy control mode: {summary.get('enemy_control_mode', '')}",
        f"- Simulated time: {float(summary.get('time_s', 0.0) or 0.0):.1f}s",
        "",
        "## Control Contract",
    ]
    for item in list(summary.get("control_contract", []) or []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Key Metrics",
            "| Metric | Value |",
            "| --- | --- |",
        ]
    )
    for label, value in metrics:
        lines.append(f"| {label} | {value} |")

    detect_mode_time_s = dict(summary.get("detect_mode_time_s", {}) or {})
    if detect_mode_time_s:
        lines.extend(
            [
                "",
                "## Detection Mode Time",
                "| Mode | Time (s) |",
                "| --- | --- |",
            ]
        )
        for mode in sorted(detect_mode_time_s):
            lines.append(f"| {mode} | {float(detect_mode_time_s[mode]):.1f} |")

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Summary JSON: {output_dir / 'summary.json'}",
            f"- Timeline CSV: {output_dir / 'timeline.csv'}",
            f"- Events CSV: {output_dir / 'events.csv'}",
        ]
    )
    if include_plots:
        lines.append(f"- Overview plot: {output_dir / 'overview.png'}")
        lines.append(f"- Guidance plot: {output_dir / 'guidance.png'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ch6_validation(
    scenario_id: str,
    *,
    seed: int = 1,
    max_steps: Optional[int] = None,
    mode: str = "proposed",
    output_root: Optional[Path] = None,
    make_plots: bool = True,
) -> Dict[str, object]:
    spec = get_scenario_spec(scenario_id)
    scenario = spec.scenario
    output_root = Path(output_root) if output_root else OUTPUT_BASE_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{spec.scenario_id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_overrides = {
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "CAP_VERIFICATION": "1",
        "CAP_AWACS_DETERMINISTIC": "1",
        "CAP_AWACS_FIXED_CENTER_KM": "100,0,9",
        "CAP_AWACS_SEED": str(seed),
        "CAP_VERIFY_TABLE": "0",
        "CAP_DEBUG_PRINT": "0",
        "CAP_ROOTCAUSE_TRACE": "0",
        "CAP_FORCE_TACTIC": "",
        "CAP_ENEMY_FORCE_TACTIC": "",
        "CAP_ENEMY_GROUP1_FORCE_TACTIC": "",
        "CAP_ENEMY_GROUP2_FORCE_TACTIC": "",
        "CAP_ENEMY_SIMPLE": "1" if spec.enemy_control_mode == "scripted" else "0",
    }

    max_steps = int(max_steps or spec.max_steps)
    temp_config_path: Optional[Path] = None
    env = None
    patrol_task = None
    timeline_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    prev_state: Dict[str, object] = {}

    with _temporary_env(env_overrides):
        random.seed(seed)
        np.random.seed(seed)
        _apply_outer_compat_patches()
        log_file = setup_logging(str(output_dir), f"{spec.scenario_id}_{mode}")
        logging.info("=" * 72)
        logging.info("Chapter 6 validation start: %s | %s", spec.scenario_id, spec.title)
        logging.info("Focus: %s", spec.thesis_focus)
        logging.info("Enemy mode: %s | seed=%s | max_steps=%s", spec.enemy_control_mode, seed, max_steps)
        logging.info("=" * 72)
        try:
            base_config = CAP_DIR / "config" / "patrol_config.yaml"
            config_dict = generate_scenario_config(scenario, str(base_config))
            config_dict["cap_experiment_mode"] = mode
            config_dict["cap_rng_seed"] = int(seed)
            config_dict["cap_awacs_seed"] = int(seed)

            config_name = f"ch6_validation_{spec.scenario_id.lower()}_{timestamp}"
            temp_config_path = PROJECT_ROOT / "envs" / "JSBSim" / "configs" / f"{config_name}.yaml"
            with temp_config_path.open("w", encoding="utf-8") as handle:
                yaml.dump(config_dict, handle, allow_unicode=True, sort_keys=False)

            env = CAPEnvForVerification(config_name)
            env.max_steps = max_steps
            patrol_task = CAPTask(env.config)
            env.task = patrol_task
            env.reset()
            env.max_steps = max_steps

            if str(scenario.awacs_status.value).lower() != "normal":
                patrol_task.awacs = ScenarioAwareAwacsProxy(patrol_task.awacs, scenario)

            enemy_target_headings: Dict[str, object] = {}
            enemy_target_altitudes_ft: Dict[str, float] = {}
            if spec.enemy_control_mode == "scripted":
                for enemy_id in ("B0100", "B0200", "B0300", "B0400"):
                    init_state = config_dict["aircraft_configs"][enemy_id]["init_state"]
                    enemy_target_altitudes_ft[enemy_id] = float(init_state.get("ic_h_sl_ft", 30000.0))
                enemy_target_headings["_target_altitudes_ft"] = dict(enemy_target_altitudes_ft)
                patrol_task.enemy_scenario = scenario
                patrol_task.enemy_target_headings = enemy_target_headings
                patrol_task.enemy_target_altitudes_ft = enemy_target_altitudes_ft

            acmi_path = output_dir / f"{spec.scenario_id}_{timestamp}.txt.acmi"
            write_acmi_header(str(acmi_path), scenario)

            dt = float(getattr(env, "time_interval", 0.2) or 0.2)
            for step in range(1, max_steps + 1):
                time_s = step * dt
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

                actions = []
                agent_ids = list(env.agents.keys())
                for agent_id in agent_ids:
                    if getattr(env.agents[agent_id], "is_alive", False):
                        alt_cmd, hdg_cmd, spd_cmd = patrol_task.get_action(env, agent_id)
                        actions.append([alt_cmd, hdg_cmd, spd_cmd, 0])
                    else:
                        actions.append([7, 8, 3, 0])
                env.step(np.asarray(actions, dtype=np.float32).reshape(1, len(agent_ids), 4))
                write_acmi_frame(str(acmi_path), env, time_s, patrol_task)

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

            summary = _scenario_summary(spec, patrol_task, timeline_rows)
            summary["log_file"] = str(log_file)
            summary["acmi_file"] = str(acmi_path)
            summary["output_dir"] = str(output_dir)

            _write_csv(output_dir / "timeline.csv", timeline_rows)
            _write_csv(output_dir / "events.csv", event_rows)
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if make_plots:
                _generate_plots(output_dir, timeline_rows)
            _write_summary_markdown(
                output_dir / "summary.md",
                summary,
                output_dir,
                include_plots=bool(make_plots),
            )
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


def _build_parser(default_scenario: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Chapter 6 CAP validation scenarios.")
    parser.add_argument("--scenario", default=default_scenario or "S1", help="Scenario id: S1 / S2 / S3")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic seed for the validation run.")
    parser.add_argument("--steps", type=int, default=None, help="Override max step count.")
    parser.add_argument("--mode", default="proposed", help="Experiment mode written into CAP config.")
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

    summary = run_ch6_validation(
        args.scenario,
        seed=args.seed,
        max_steps=args.steps,
        mode=args.mode,
        output_root=Path(args.output_root),
        make_plots=not bool(args.no_plots),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
