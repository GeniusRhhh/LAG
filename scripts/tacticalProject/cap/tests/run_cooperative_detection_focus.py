from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CAP_DIR = os.path.dirname(CURRENT_DIR)
TACTICAL_DIR = os.path.dirname(CAP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(TACTICAL_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, TACTICAL_DIR)
sys.path.insert(0, CAP_DIR)

from cap.cap_task import CAPTask
from cap.tests.cooperative_detection_focus_scenarios import (
    DetectionFocusScenarioSpec,
    get_scenario_spec,
    iter_scenario_specs,
)
from cap.tests.run_detection_verification_real import (
    A0100_LAT,
    A0100_LON,
    CAPEnvForVerification,
    DEG_TO_KM,
    OUTPUT_BASE_DIR as BASE_OUTPUT_DIR,
    apply_enemy_maneuver,
    generate_scenario_config,
    setup_logging,
    write_acmi_frame,
    write_acmi_header,
)


OUTPUT_BASE_DIR = Path(BASE_OUTPUT_DIR).parent / "Cooperative_detection_focus"


def _resolve_scenario_ids(arg: str) -> List[str]:
    value = str(arg or "").strip().upper()
    available = {spec.scenario_id.upper(): spec.scenario_id for spec in iter_scenario_specs()}
    if value in ("", "ALL"):
        return [available[key] for key in sorted(available)]
    result: List[str] = []
    for token in value.replace(",", " ").split():
        if token not in available:
            raise ValueError(f"Unknown scenario id: {token}")
        result.append(available[token])
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _front_reference_distance(spec: DetectionFocusScenarioSpec) -> float:
    base_y = float(spec.scenario.initial_distance)
    own_center_y = 50.0
    return max(0.0, base_y - own_center_y)


def _script_reference_distance(spec: DetectionFocusScenarioSpec, time_s: float) -> float:
    scenario = spec.scenario
    velocities = scenario.generate_initial_velocities()
    closing_speed = 0.0
    for _, vy, _ in velocities:
        closing_speed += abs(float(vy))
    closing_speed = closing_speed / max(1, len(velocities))
    return max(0.0, _front_reference_distance(spec) - closing_speed * float(time_s))


def _awacs_available_for_focus(spec: DetectionFocusScenarioSpec, current_time_s: float, reference_distance_km: float) -> bool:
    loss_windows = tuple(getattr(spec, "awacs_loss_windows_s", ()) or ())
    t = float(current_time_s)
    if loss_windows:
        for start_s, end_s in loss_windows:
            if float(start_s) <= t < float(end_s):
                return False
        return True
    scenario = spec.scenario
    if hasattr(scenario, "is_awacs_available"):
        try:
            return bool(scenario.is_awacs_available(float(current_time_s), float(reference_distance_km)))
        except Exception:
            return True
    return True


def _configure_focus_awacs_runtime(spec: DetectionFocusScenarioSpec, patrol_task, mode: str) -> None:
    patrol_task._focus_awacs_loss_windows_s = tuple(getattr(spec, "awacs_loss_windows_s", ()) or ())
    patrol_task._focus_awacs_force_control = True
    patrol_task._focus_awacs_disable_source_random_loss = True
    patrol_task._focus_awacs_disable_internal_scan_cycle = True
    patrol_task._focus_awacs_mode = str(mode or "proposed").strip().lower()
    patrol_task._focus_force_all_hot = True
    patrol_task._picture_awacs_only_max_age_s = 0.8
    patrol_task._picture_awacs_partner_max_age_s = 0.4
    patrol_task._picture_decision_track_max_age_s = 1.0
    patrol_task._picture_decision_include_recent_lost = False
    patrol_task._picture_decision_lost_grace_s = 0.0
    patrol_task._picture_search_track_max_age_s = 1.2
    patrol_task._picture_search_lost_grace_s = 0.0
    patrol_task._picture_radar_track_max_age_s = 0.8
    awacs = getattr(patrol_task, "awacs", None)
    if awacs is not None:
        try:
            awacs._disable_random_loss = True
            awacs._disable_internal_scan_cycle = True
        except Exception:
            pass
    fusion = getattr(patrol_task, "track_fusion", None)
    if fusion is not None:
        try:
            fusion.awacs_only_max_age_s = 0.8
            fusion.awacs_partner_max_age_s = 0.4
            fusion.fresh_track_window_s = 0.8
        except Exception:
            pass


def _configure_focus_friendly_geometry(config_dict: dict) -> None:
    aircraft_configs = dict(config_dict.get("aircraft_configs", {}) or {})
    # Preserve the original two-hot two-cold CAP opening:
    # A0100/A0300 hot, northbound; A0200/A0400 cold, southbound.
    heading_map = {
        "A0100": 0.0,
        "A0200": 180.0,
        "A0300": 0.0,
        "A0400": 180.0,
    }
    for aid, heading_deg in heading_map.items():
        init_state = dict((aircraft_configs.get(aid) or {}).get("init_state", {}) or {})
        init_state["ic_psi_true_deg"] = float(heading_deg)
        aircraft_configs[aid]["init_state"] = init_state
    config_dict["aircraft_configs"] = aircraft_configs


def _apply_focus_maneuver_script(
    spec: DetectionFocusScenarioSpec,
    time_s: float,
    enemy_target_headings: Dict[str, object],
) -> Dict[str, object]:
    script = tuple(getattr(spec, "maneuver_script", ()) or ())
    enemy_ids = ("B0100", "B0200", "B0300", "B0400")
    if not script:
        return enemy_target_headings

    active = None
    for item in script:
        if float(item.get("time_s", 0.0)) <= float(time_s):
            active = item
        else:
            break
    if active is None:
        return enemy_target_headings

    base_key = "_base_headings"
    if base_key not in enemy_target_headings or not isinstance(enemy_target_headings.get(base_key), dict):
        enemy_target_headings[base_key] = {eid: float(enemy_target_headings.get(eid, 180.0)) for eid in enemy_ids}

    offsets = tuple(active.get("heading_offsets_deg", ()) or ())
    for idx, eid in enumerate(enemy_ids):
        if idx < len(offsets):
            heading = (180.0 + float(offsets[idx])) % 360.0
            enemy_target_headings[eid] = heading
            enemy_target_headings[base_key][eid] = heading

    target_altitudes = dict(active.get("target_altitudes_km", {}) or {})
    if target_altitudes:
        alt_key = "_target_altitudes_ft"
        alt_map = enemy_target_headings.get(alt_key)
        if not isinstance(alt_map, dict):
            alt_map = {}
            enemy_target_headings[alt_key] = alt_map
        rate_key = "_altitude_rate_km_s"
        rate_map = enemy_target_headings.get(rate_key)
        if not isinstance(rate_map, dict):
            rate_map = {}
            enemy_target_headings[rate_key] = rate_map
        for eid, alt_km in target_altitudes.items():
            alt_map[str(eid)] = float(alt_km) * 3280.84
            rate_map[str(eid)] = 0.0

    return enemy_target_headings


def _battlefield_distance_to_friendlies(env) -> float:
    my_positions = []
    enemy_positions = []
    for aid, agent in env.agents.items():
        if not agent.is_alive:
            continue
        lon, lat, _ = agent.get_geodetic()
        x = (lon - A0100_LON) * DEG_TO_KM + 75
        y = (lat - A0100_LAT) * DEG_TO_KM
        if aid.startswith("A"):
            my_positions.append((x, y))
        elif aid.startswith("B"):
            enemy_positions.append((x, y))
    if not my_positions or not enemy_positions:
        return 999.0
    my_center = np.mean(my_positions, axis=0)
    return float(
        min(np.hypot(ex - my_center[0], ey - my_center[1]) for ex, ey in enemy_positions)
    )


def _get_detection_mode(patrol_task) -> str:
    if hasattr(patrol_task, "coop_detection"):
        det = patrol_task.coop_detection
        if hasattr(det, "_mode"):
            return str(det._mode)
        if hasattr(det, "mode"):
            return str(det.mode)
    return ""


def _collect_tracks(patrol_task) -> Dict[str, List[str]]:
    detections: Dict[str, List[str]] = {}
    radar = getattr(patrol_task, "cap_radar", None)
    if radar is None or not hasattr(radar, "get_tracks"):
        return detections
    for aid in ("A0100", "A0200", "A0300", "A0400"):
        try:
            tracks = radar.get_tracks(aid)
        except Exception:
            tracks = None
        if tracks:
            detections[aid] = list(tracks.keys())
    return detections


def _first_activation_time(patrol_task) -> Optional[float]:
    activation_times = getattr(patrol_task, "_radar_range_entry_times", {}) or {}
    if not activation_times:
        return None
    try:
        return float(min(float(v) for v in activation_times.values()))
    except Exception:
        return None


def _build_actions(env, patrol_task) -> np.ndarray:
    actions = []
    for aid in list(env.agents.keys()):
        if env.agents[aid].is_alive:
            alt_cmd, hdg_cmd, spd_cmd = patrol_task.get_action(env, aid)
            actions.append([alt_cmd, hdg_cmd, spd_cmd, 0])
        else:
            actions.append([7, 8, 3, 0])
    return np.array(actions, dtype=np.float32).reshape(1, len(actions), 4)


def _summarize_detection_metrics(samples: List[Dict[str, object]]) -> Dict[str, object]:
    expected_targets = ["B0100", "B0200", "B0300", "B0400"]
    if not samples:
        return {}

    coverage = [int(sample.get("coverage_count", 0)) for sample in samples]
    first_full_time = next(
        (float(sample["time_s"]) for sample in samples if int(sample.get("coverage_count", 0)) >= len(expected_targets)),
        None,
    )
    mode_switches = 0
    prev_mode = None
    for sample in samples:
        mode = str(sample.get("mode", "") or "")
        if prev_mode is not None and mode != prev_mode:
            mode_switches += 1
        prev_mode = mode

    agent_target_counts: Dict[str, Dict[str, int]] = {
        aid: {tid: 0 for tid in expected_targets} for aid in ("A0100", "A0200", "A0300", "A0400")
    }
    for sample in samples:
        detections = dict(sample.get("detections", {}) or {})
        for aid, targets in detections.items():
            for tid in list(targets or []):
                if aid in agent_target_counts and tid in agent_target_counts[aid]:
                    agent_target_counts[aid][tid] += 1

    target_allocation_primary = {}
    for tid in expected_targets:
        counts = {aid: agent_target_counts[aid][tid] for aid in agent_target_counts}
        if sum(counts.values()) <= 0:
            continue
        target_allocation_primary[tid] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    return {
        "sample_count": len(samples),
        "first_sample_time_s": float(samples[0]["time_s"]),
        "last_sample_time_s": float(samples[-1]["time_s"]),
        "avg_coverage_ratio": float(np.mean(coverage) / len(expected_targets)),
        "full_coverage_continuity": float(np.mean([1.0 if c >= len(expected_targets) else 0.0 for c in coverage])),
        "first_full_coverage_time_s": first_full_time,
        "mode_switch_count": int(mode_switches),
        "target_primary_responsibility": target_allocation_primary,
    }


def run_detection_focus_scenario(
    spec: DetectionFocusScenarioSpec,
    *,
    mode: str = "proposed",
    seed: int = 1,
    make_acmi: bool = True,
) -> Dict[str, object]:
    mode = str(mode or "proposed").strip().lower()
    scenario = copy.deepcopy(spec.scenario)
    if hasattr(scenario, "reset_runtime_state"):
        scenario.reset_runtime_state()

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    scenario_tag = f"{spec.scenario_id}_{mode.upper()}"
    log_file = setup_logging(str(OUTPUT_BASE_DIR), f"{scenario_tag}_{timestamp}")

    result: Dict[str, object] = {
        "scenario_id": spec.scenario_id,
        "title": spec.title,
        "mode": mode,
        "seed": int(seed),
        "log_file": log_file,
        "acmi_path": None,
        "focus_stop_reason": "",
        "activation_time_s": None,
        "focus_start_time_s": None,
        "focus_end_time_s": None,
        "samples": [],
    }

    base_config = os.path.join(os.path.dirname(__file__), "..", "config", "patrol_config.yaml")
    base_config = os.path.abspath(base_config)
    config_dict = generate_scenario_config(scenario, base_config)
    _configure_focus_friendly_geometry(config_dict)
    config_dict["cap_experiment_mode"] = mode
    config_dict["cap_rng_seed"] = int(seed)
    config_dict["cap_awacs_seed"] = int(seed)

    temp_config_name = f"detection_focus_{spec.scenario_id.lower()}_{timestamp}"
    jsbsim_config_dir = _repo_root() / "envs" / "JSBSim" / "configs"
    jsbsim_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = jsbsim_config_dir / f"{temp_config_name}.yaml"
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, allow_unicode=True)

    env = None
    try:
        env = CAPEnvForVerification(temp_config_name)
        env.max_steps = int(spec.max_steps)
        patrol_task = CAPTask(env.config)
        env.task = patrol_task
        os.environ["CAP_ENEMY_SIMPLE"] = "1"
        env.reset()
        env.max_steps = int(spec.max_steps)

        acmi_path = OUTPUT_BASE_DIR / f"{timestamp}_{scenario_tag}.txt.acmi"
        if make_acmi:
            write_acmi_header(str(acmi_path), scenario)
            result["acmi_path"] = str(acmi_path)

        enemy_target_headings: Dict[str, object] = {}
        enemy_target_altitudes_ft: Dict[str, float] = {}
        enemy_ids = ["B0100", "B0200", "B0300", "B0400"]
        enemy_target_headings["_target_altitudes_ft"] = {}
        for eid in enemy_ids:
            if eid in config_dict["aircraft_configs"]:
                init_alt_ft = config_dict["aircraft_configs"][eid]["init_state"].get("ic_h_sl_ft", 30000.0)
                enemy_target_altitudes_ft[eid] = init_alt_ft
                enemy_target_headings["_target_altitudes_ft"][eid] = init_alt_ft
        patrol_task.enemy_scenario = scenario
        patrol_task.enemy_target_headings = enemy_target_headings
        patrol_task.enemy_target_altitudes_ft = enemy_target_altitudes_ft
        _configure_focus_awacs_runtime(spec, patrol_task, mode)

        dt = float(env.time_interval)
        focus_started = False
        focus_start_time = None
        min_post_activation_observe_s = max(float(spec.stop_window_s), 120.0)
        full_coverage_streak_s = 0.0
        patrol_task._awacs_grace_seconds = 0.0
        patrol_task._detection_focus_mode = True
        for step in range(1, int(spec.max_steps) + 1):
            time_s = step * dt
            current_distance = _battlefield_distance_to_friendlies(env)
            reference_distance = _script_reference_distance(spec, time_s)
            enemy_target_headings = apply_enemy_maneuver(
                env, scenario, reference_distance, time_s, enemy_target_headings
            )
            enemy_target_headings = _apply_focus_maneuver_script(spec, time_s, enemy_target_headings)
            patrol_task.enemy_target_headings = enemy_target_headings
            if isinstance(enemy_target_headings.get("_target_altitudes_ft"), dict):
                enemy_target_altitudes_ft.update(enemy_target_headings["_target_altitudes_ft"])
                patrol_task.enemy_target_altitudes_ft = enemy_target_altitudes_ft

            patrol_task._awacs_grace_seconds = 0.0
            patrol_task._detection_focus_reference_distance_km = float(reference_distance)
            patrol_task._detection_focus_time_s = float(time_s)
            if _awacs_available_for_focus(spec, time_s, reference_distance):
                pass
            else:
                try:
                    patrol_task._clear_awacs_tracks(current_time=float(time_s))
                except Exception:
                    try:
                        patrol_task._clear_awacs_tracks(None, float(time_s))
                    except Exception:
                        pass
                try:
                    patrol_task._last_awacs_info_time = None
                except Exception:
                    pass

            action_array = _build_actions(env, patrol_task)
            env.step(action_array)
            if make_acmi:
                write_acmi_frame(str(acmi_path), env, time_s, patrol_task)

            activation_time = _first_activation_time(patrol_task)
            if activation_time is not None and result["activation_time_s"] is None:
                result["activation_time_s"] = float(activation_time)

            if activation_time is not None and not focus_started:
                if time_s >= activation_time:
                    focus_started = True
                    focus_start_time = float(time_s)
                    result["focus_start_time_s"] = float(time_s)

            if focus_started:
                detections = _collect_tracks(patrol_task)
                detected_targets = set()
                for targets in detections.values():
                    detected_targets.update(targets)
                result["samples"].append(
                    {
                        "time_s": float(time_s),
                        "distance_km": float(current_distance),
                        "mode": _get_detection_mode(patrol_task),
                        "detections": detections,
                        "coverage_count": int(len([t for t in detected_targets if t.startswith("B")])),
                    }
                )
                full_cover = len([t for t in detected_targets if t.startswith("B")]) >= 4
                if full_cover:
                    full_coverage_streak_s += dt
                    if (
                        full_coverage_streak_s >= float(spec.stop_window_s)
                        and focus_start_time is not None
                        and (time_s - focus_start_time) >= min_post_activation_observe_s
                    ):
                        result["focus_stop_reason"] = "full_coverage_stable"
                        result["focus_end_time_s"] = float(time_s)
                        break
                else:
                    full_coverage_streak_s = 0.0
                if focus_start_time is not None and (time_s - focus_start_time) >= float(spec.max_post_activation_s):
                    result["focus_stop_reason"] = "max_focus_window_reached"
                    result["focus_end_time_s"] = float(time_s)
                    break

            all_dead = all(not env.agents[aid].is_alive for aid in env.agents)
            if all_dead:
                result["focus_stop_reason"] = "all_aircraft_dead"
                result["focus_end_time_s"] = float(time_s)
                break
        else:
            result["focus_stop_reason"] = "max_steps_reached"
            if focus_start_time is not None:
                result["focus_end_time_s"] = float(spec.max_steps * dt)

        result.update(_summarize_detection_metrics(list(result.get("samples", []) or [])))
        return result
    finally:
        try:
            os.environ.pop("CAP_ENEMY_SIMPLE", None)
        except Exception:
            pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            if temp_config_path.exists():
                temp_config_path.unlink()
        except Exception:
            pass


def run_detection_focus_compare(
    scenario_ids: Iterable[str],
    *,
    base_seed: int = 1,
) -> Dict[str, object]:
    scenario_list = list(scenario_ids)
    summaries: List[Dict[str, object]] = []
    for index, scenario_id in enumerate(scenario_list):
        spec = get_scenario_spec(scenario_id)
        seed = int(base_seed) + index + 1
        proposed = run_detection_focus_scenario(spec, mode="proposed", seed=seed)
        baseline = run_detection_focus_scenario(spec, mode="baseline", seed=seed)
        summaries.extend((proposed, baseline))
    return {
        "scenario_ids": scenario_list,
        "summaries": summaries,
    }


def _write_compare_summary(result: Dict[str, object]) -> Path:
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_BASE_DIR / f"focus_compare_summary_{timestamp}.csv"
    rows = list(result.get("summaries", []) or [])
    fieldnames = [
        "scenario_id",
        "title",
        "mode",
        "seed",
        "activation_time_s",
        "focus_start_time_s",
        "focus_end_time_s",
        "focus_stop_reason",
        "sample_count",
        "avg_coverage_ratio",
        "full_coverage_continuity",
        "first_full_coverage_time_s",
        "mode_switch_count",
        "target_primary_responsibility",
        "log_file",
        "acmi_path",
    ]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key) for key in fieldnames}
            payload["target_primary_responsibility"] = json.dumps(
                payload.get("target_primary_responsibility", {}),
                ensure_ascii=False,
            )
            writer.writerow(payload)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cooperative detection focus-only validation.")
    parser.add_argument("--scenario", default="ALL", help="Scenario id: D1 / D2 / D3 / ALL / comma list")
    parser.add_argument("--mode", default="proposed", choices=["proposed", "baseline"])
    parser.add_argument("--compare", action="store_true", help="Run proposed and baseline pairwise.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-acmi", action="store_true", help="Disable ACMI recording.")
    parser.add_argument("--list", action="store_true", help="List available scenarios.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.list:
        for spec in iter_scenario_specs():
            print(f"{spec.scenario_id}: {spec.title}")
        return 0
    scenario_ids = _resolve_scenario_ids(args.scenario)
    if args.compare:
        result = run_detection_focus_compare(scenario_ids, base_seed=int(args.seed))
        result["summary_csv"] = str(_write_compare_summary(result))
    else:
        summaries = []
        for scenario_id in scenario_ids:
            spec = get_scenario_spec(scenario_id)
            summaries.append(
                run_detection_focus_scenario(
                    spec,
                    mode=str(args.mode),
                    seed=int(args.seed),
                    make_acmi=not bool(args.no_acmi),
                )
            )
        result = {"scenario_ids": scenario_ids, "summaries": summaries}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
