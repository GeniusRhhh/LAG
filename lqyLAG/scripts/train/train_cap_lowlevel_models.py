#!/usr/bin/env python
import argparse
import csv
import glob
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import gymnasium as gym
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
LQYLAG_ROOT = REPO_ROOT / "lqyLAG"

sys.path.append(str(REPO_ROOT))

from lqyLAG.algorithms.ppo.ppo_actor import PPOActor
from lqyLAG.algorithms.ppo.ppo_critic import PPOCritic
from lqyLAG.config import get_config
from lqyLAG.envs.JSBSim.model.baseline_actor import BaselineActor


FT_PER_M = 3.280839895013123
FPS_PER_MPS = 3.280839895013123
FPM_PER_MPS = 196.8503937007874


def parse_args():
    parser = argparse.ArgumentParser(description="Train CAP-native low-level models for tacticalProject.")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--aircraft", type=str, default="f16", choices=["f16", "su27", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-mode", type=str, default="warmstart", choices=["warmstart", "scratch"])

    parser.add_argument(
        "--f16-init-path",
        type=str,
        default=str(REPO_ROOT / "envs" / "JSBSim" / "model" / "baseline_model.pt"),
    )
    parser.add_argument(
        "--su27-init-path",
        type=str,
        default=r"D:\Pycharm\LAG\lqyLAG\scripts\results\SingleControl\1\heading_su27\ppo\su27_baseline_v1\run40\actor_990.pt",
    )

    parser.add_argument("--foundation-steps", type=int, default=2500000)
    parser.add_argument("--hardcase-steps", type=int, default=1500000)
    parser.add_argument("--n-rollout-threads", type=int, default=8)
    parser.add_argument("--buffer-size", type=int, default=240)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ppo-epoch", type=int, default=10)
    parser.add_argument("--num-mini-batch", type=int, default=2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--n-eval-rollout-threads", type=int, default=2)
    parser.add_argument("--use-eval", action="store_true")
    parser.add_argument("--select-best-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-selection-window", type=int, default=8)
    parser.add_argument("--checkpoint-selection-episodes", type=int, default=48)

    parser.add_argument("--foundation-hardcase-prob", type=float, default=0.15)
    parser.add_argument("--hardcase-prob", type=float, default=0.50)
    parser.add_argument("--samples-per-trace", type=int, default=14)
    parser.add_argument("--min-hardcase-alt-m", type=float, default=250.0)
    parser.add_argument("--max-hardcase-alt-m", type=float, default=12000.0)
    parser.add_argument("--max-hardcases", type=int, default=0)
    parser.add_argument("--trace-glob", action="append", default=[])

    parser.add_argument(
        "--export-dir",
        type=str,
        default=str(REPO_ROOT / "scripts" / "tacticalProject" / "models"),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(REPO_ROOT / "lqyLAG" / "scripts" / "results" / "cap_native_train_runs"),
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def default_trace_globs(aircraft: str) -> List[str]:
    prefix = "B" if aircraft == "f16" else "A"
    return [
        str(REPO_ROOT / "envs" / "JSBSim" / "scripts" / "tacticalProject" / "logs" / f"crash_trace_{prefix}*.csv"),
        str(REPO_ROOT / "scripts" / "tacticalProject" / "logs" / f"crash_trace_{prefix}*.csv"),
    ]


def make_run_dir(base_dir: str | Path) -> Path:
    run_dir = Path(base_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_cmd(cmd: List[str], cwd: Path, log_path: Path, env: Dict[str, str] | None = None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("[run] " + " ".join(cmd))
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def actor_uses_mlp_actlayer(state_dict) -> bool:
    return any(key.startswith("act.mlp.") or key.startswith("mlp.") for key in state_dict.keys())


def build_ppo_args(use_mlp_actlayer: bool):
    parser = get_config()
    args = parser.parse_args([])
    args.hidden_size = "128 128"
    args.act_hidden_size = "128 128" if use_mlp_actlayer else ""
    args.activation_id = 1
    args.use_feature_normalization = False
    args.use_recurrent_policy = True
    args.recurrent_hidden_size = 128
    args.recurrent_hidden_layers = 1
    args.gain = 0.01
    args.use_prior = False
    return args


def copy_matching_state(src_state, dst_module, extra_prefix_map=None) -> int:
    dst_state = dst_module.state_dict()
    matched = 0
    for key, value in dst_state.items():
        src_key = key
        if extra_prefix_map:
            for dst_prefix, src_prefix in extra_prefix_map.items():
                if key.startswith(dst_prefix):
                    src_key = src_prefix + key[len(dst_prefix):]
                    break
        if src_key in src_state and src_state[src_key].shape == value.shape:
            dst_state[key] = src_state[src_key]
            matched += 1
    dst_module.load_state_dict(dst_state)
    return matched


def prepare_warmstart_dir(source_model_path: str, use_mlp_actlayer: bool, run_dir: Path) -> Path:
    source_state = torch.load(source_model_path, map_location="cpu", weights_only=True)
    ppo_args = build_ppo_args(use_mlp_actlayer)
    obs_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(12,), dtype=float)
    act_space = gym.spaces.MultiDiscrete([41, 41, 41, 30])

    actor = PPOActor(ppo_args, obs_space, act_space, device=torch.device("cpu"))
    critic = PPOCritic(ppo_args, obs_space, device=torch.device("cpu"))

    actor_match_count = copy_matching_state(source_state, actor)
    critic_match_count = copy_matching_state(source_state, critic, extra_prefix_map={"mlp.": "act.mlp."})

    warmstart_dir = run_dir / "warmstart"
    warmstart_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), warmstart_dir / "actor_latest.pt")
    torch.save(critic.state_dict(), warmstart_dir / "critic_latest.pt")
    write_json(
        warmstart_dir / "warmstart_manifest.json",
        {
            "source_model_path": source_model_path,
            "use_mlp_actlayer": use_mlp_actlayer,
            "actor_match_count": actor_match_count,
            "critic_match_count": critic_match_count,
        },
    )
    return warmstart_dir


def locate_latest_run_dir(base_dir: Path) -> Path:
    run_dirs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run")],
        key=lambda p: p.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return run_dirs[-1]


def predict_next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_nums = [
        int(p.name[3:])
        for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit()
    ]
    next_run_num = max(run_nums) + 1 if run_nums else 1
    return base_dir / f"run{next_run_num}"


def write_watch_paths(path: Path, payload: dict):
    lines = [
        f"launcher_run_dir={payload.get('launcher_run_dir', '')}",
        f"aircraft={payload.get('aircraft', '')}",
        f"current_stage={payload.get('current_stage', '')}",
        f"stage_log={payload.get('stage_log', '')}",
        f"expected_ppo_run_dir={payload.get('expected_ppo_run_dir', '')}",
        f"training_log_txt={payload.get('training_log_txt', '')}",
        f"training_metrics_json={payload.get('training_metrics_json', '')}",
        f"training_progress_json={payload.get('training_progress_json', '')}",
        f"hardcase_json={payload.get('hardcase_json', '')}",
        f"status_json={payload.get('status_json', '')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_ppo_actor_to_baseline(ppo_actor_path: Path, export_path: Path, use_mlp_actlayer: bool):
    ppo_state = torch.load(ppo_actor_path, map_location="cpu", weights_only=True)
    actor = BaselineActor(input_dim=12, use_mlp_actlayer=use_mlp_actlayer)
    copy_matching_state(ppo_state, actor)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), export_path)


def write_model_metadata(
    export_path: Path,
    aircraft: str,
    scenario_name: str,
    final_stage: str,
    latest_run: Path,
    selected_actor: str = "",
    command_semantics: str = "",
):
    semantics = str(command_semantics or "").strip()
    if not semantics:
        semantics = "cap_native_residual_15x17x7" if "residual" in str(scenario_name).lower() else "cap_native_15x17x7"

    if "residual" in semantics.lower():
        observation_layout = [
            "delta_altitude_m_div_1000",
            "delta_heading_rad",
            "delta_velocity_u_mps_div_340",
            "altitude_m_div_5000",
            "roll_sin",
            "roll_cos",
            "pitch_sin",
            "pitch_cos",
            "u_mps_div_340",
            "v_mps_div_340",
            "w_mps_div_340",
            "vc_mps_div_340",
        ]
    else:
        observation_layout = [
            "command_altitude_m_div_1000",
            "command_heading_rad",
            "command_velocity_mps_div_100",
            "altitude_m_div_5000",
            "roll_sin",
            "roll_cos",
            "pitch_sin",
            "pitch_cos",
            "u_mps_div_340",
            "v_mps_div_340",
            "w_mps_div_340",
            "vc_mps_div_340",
        ]

    metadata = {
        "schema_version": 1,
        "aircraft": aircraft,
        "scenario_name": scenario_name,
        "final_stage": final_stage,
        "training_run_dir": str(latest_run),
        "selected_actor_checkpoint": selected_actor,
        "command_semantics": semantics,
        "observation_layout": observation_layout,
        "action_layout": [
            "aileron_index_0_40",
            "elevator_index_0_40",
            "rudder_index_0_40",
            "throttle_index_0_29",
        ],
    }
    write_json(export_path.with_suffix(".json"), metadata)


def _get_float(row, key, default=0.0):
    value = row.get(key, default)
    return float(value if value not in ("", None) else default)


def load_trace_rows(trace_file: str) -> List[dict]:
    rows = []
    with open(trace_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "step": int(float(row["step"])),
                    "alt_m": _get_float(row, "alt_m"),
                    "roll_deg": _get_float(row, "roll_deg"),
                    "pitch_deg": _get_float(row, "pitch_deg"),
                    "tas_mps": _get_float(row, "tas_mps"),
                    "v_up_mps": _get_float(row, "v_up_mps"),
                    "alpha_deg": _get_float(row, "alpha_deg"),
                    "cmd_aileron": _get_float(row, "cmd_aileron"),
                    "cmd_elevator": _get_float(row, "cmd_elevator"),
                    "cmd_rudder": _get_float(row, "cmd_rudder"),
                    "cmd_throttle": _get_float(row, "cmd_throttle", 0.9),
                }
            )
    return rows


def is_risk_row(row: dict) -> bool:
    return bool(
        row["v_up_mps"] < -2.0
        or row["tas_mps"] < 170.0
        or row["alpha_deg"] > 9.0
        or row["pitch_deg"] < 4.0
    )


def row_score(row: dict) -> float:
    low_alt = max(0.0, 2500.0 - row["alt_m"]) / 2500.0
    sink = max(0.0, -row["v_up_mps"] - 2.0) / 25.0
    low_speed = max(0.0, 150.0 - row["tas_mps"]) / 80.0
    high_alpha = max(0.0, row["alpha_deg"] - 8.0) / 12.0
    flat_pitch = max(0.0, 6.0 - row["pitch_deg"]) / 12.0
    return low_alt + sink + low_speed + high_alpha + flat_pitch


def classify_window(index: int, first_risk_idx: int, last_risk_idx: int) -> str:
    span = max(last_risk_idx - first_risk_idx, 1)
    progress = float(index - first_risk_idx) / float(span)
    if progress <= 0.33:
        return "early"
    if progress <= 0.66:
        return "mid"
    return "late"


def pick_rows(rows: List[dict], samples_per_trace: int, min_alt_m: float, max_alt_m: float) -> List[dict]:
    valid_indices = [
        idx
        for idx, row in enumerate(rows)
        if min_alt_m <= row["alt_m"] <= max_alt_m and is_risk_row(row)
    ]
    if not valid_indices:
        return []

    first_risk_idx = valid_indices[0]
    last_risk_idx = valid_indices[-1]
    anchors = {
        first_risk_idx,
        last_risk_idx,
        min(valid_indices, key=lambda idx: rows[idx]["alt_m"]),
        min(valid_indices, key=lambda idx: rows[idx]["tas_mps"]),
        min(valid_indices, key=lambda idx: rows[idx]["v_up_mps"]),
        max(valid_indices, key=lambda idx: rows[idx]["alpha_deg"]),
    }

    if len(valid_indices) > 1:
        for fraction in (0.15, 0.30, 0.50, 0.70, 0.85):
            anchors.add(valid_indices[int(round((len(valid_indices) - 1) * fraction))])

    selected_indices = set()
    for anchor in anchors:
        for lead in (0, 5, 10, 20, 35, 50, 80):
            idx = max(0, anchor - lead)
            row = rows[idx]
            if min_alt_m <= row["alt_m"] <= max_alt_m and row_score(row) > 0.0:
                selected_indices.add(idx)

    selected_indices = sorted(selected_indices)
    if len(selected_indices) > samples_per_trace:
        positions = np.linspace(0, len(selected_indices) - 1, num=samples_per_trace, dtype=int)
        selected_indices = [selected_indices[int(pos)] for pos in positions]

    picked = []
    for idx in selected_indices:
        row = dict(rows[idx])
        row["window_tag"] = classify_window(idx, first_risk_idx, last_risk_idx)
        picked.append(row)
    return picked


def build_first_segment(row: dict) -> dict:
    if row["alt_m"] < 800.0:
        alt_cmd_idx = 12
    elif row["alt_m"] < 1500.0:
        alt_cmd_idx = 11
    elif row["tas_mps"] < 105.0 or row["v_up_mps"] < -12.0:
        alt_cmd_idx = 9
    else:
        alt_cmd_idx = 8

    vel_cmd_idx = 6 if row["tas_mps"] < 125.0 else 5
    return {
        "alt_cmd_idx": int(alt_cmd_idx),
        "hdg_cmd_idx": 8,
        "vel_cmd_idx": int(vel_cmd_idx),
    }


def build_hardcase_case(row: dict, trace_file: str) -> dict:
    alt_m = max(row["alt_m"], 150.0)
    tas_mps = max(row["tas_mps"], 80.0)
    window_tag = row.get("window_tag", "mid")
    if alt_m < 1200.0:
        profile_name = "cap_hardcase_low"
    elif alt_m < 3500.0:
        profile_name = "cap_hardcase_mid"
    else:
        profile_name = "cap_hardcase_high"

    return {
        "profile_name": profile_name,
        "window_tag": window_tag,
        "source_trace": str(trace_file),
        "source_step": int(row["step"]),
        "severity_score": float(row_score(row)),
        "ic_h_sl_ft": float(alt_m * FT_PER_M),
        "ic_u_fps": float(tas_mps * FPS_PER_MPS),
        "ic_theta_deg": float(row["pitch_deg"]),
        "ic_phi_deg": float(row["roll_deg"]),
        "ic_alpha_deg": float(max(-3.0, min(18.0, row["alpha_deg"]))),
        "ic_q_rad_sec": 0.0,
        "ic_p_rad_sec": 0.0,
        "ic_r_rad_sec": 0.0,
        "ic_roc_fpm": float(row["v_up_mps"] * FPM_PER_MPS),
        "first_segment": build_first_segment(row),
    }


def dedupe_cases(cases: Iterable[dict]) -> List[dict]:
    deduped = []
    seen = set()
    for case in sorted(cases, key=lambda item: item["severity_score"], reverse=True):
        key = (
            case["profile_name"],
            int(round(case["ic_h_sl_ft"] / 500.0)),
            int(round(case["ic_u_fps"] / 25.0)),
            int(round(case["ic_theta_deg"] / 2.0)),
            int(round(case["ic_phi_deg"] / 3.0)),
            int(round(case["ic_alpha_deg"] / 2.0)),
            int(round(case["ic_roc_fpm"] / 500.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def build_hardcase_payload(args, aircraft: str, output_path: Path) -> Path | None:
    patterns = args.trace_glob or default_trace_globs(aircraft)
    trace_files = sorted({file for pattern in patterns for file in glob.glob(pattern)})
    if not trace_files:
        logging.warning("[hardcase] no crash traces found for %s", aircraft)
        return None

    cases = []
    for trace_file in trace_files:
        rows = load_trace_rows(trace_file)
        for row in pick_rows(rows, args.samples_per_trace, args.min_hardcase_alt_m, args.max_hardcase_alt_m):
            cases.append(build_hardcase_case(row, trace_file))

    cases = dedupe_cases(cases)
    if args.max_hardcases > 0:
        cases = cases[: args.max_hardcases]

    payload = {
        "aircraft": aircraft,
        "trace_files": trace_files,
        "case_count": len(cases),
        "cases": cases,
    }
    write_json(output_path, payload)
    logging.info("[hardcase] %s traces=%d cases=%d -> %s", aircraft, len(trace_files), len(cases), output_path)
    return output_path


def scenario_name_for(aircraft: str) -> str:
    return "1/cap_lowlevel_f16" if aircraft == "f16" else "1/cap_lowlevel_su27"


def default_export_path(export_dir: Path, aircraft: str) -> Path:
    return export_dir / f"{aircraft}_cap_lowlevel_native.pt"


def run_training_stage(
    args,
    aircraft: str,
    scenario_name: str,
    experiment_name: str,
    model_dir: Path | None,
    use_mlp_actlayer: bool,
    num_env_steps: int,
    stage_log_path: Path,
    env_overrides: dict,
    expected_run_dir: Path,
) -> Path:
    env = os.environ.copy()
    env.update(env_overrides)

    cmd = [
        args.python_exe,
        str(SCRIPT_DIR / "train_jsbsim.py"),
        "--env-name",
        "SingleControl",
        "--algorithm-name",
        "ppo",
        "--scenario-name",
        scenario_name,
        "--experiment-name",
        experiment_name,
        "--seed",
        str(args.seed),
        "--num-env-steps",
        str(num_env_steps),
        "--n-rollout-threads",
        str(args.n_rollout_threads),
        "--buffer-size",
        str(args.buffer_size),
        "--lr",
        str(args.lr),
        "--ppo-epoch",
        str(args.ppo_epoch),
        "--num-mini-batch",
        str(args.num_mini_batch),
        "--entropy-coef",
        str(args.entropy_coef),
        "--log-interval",
        str(args.log_interval),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-episodes",
        str(args.eval_episodes),
        "--n-eval-rollout-threads",
        str(args.n_eval_rollout_threads),
        "--hidden-size",
        "128 128",
        "--act-hidden-size",
        "128 128" if use_mlp_actlayer else "",
        "--activation-id",
        "1",
        "--user-name",
        f"cap_native_{aircraft}",
        "--render-mode",
        "txt",
    ]
    if args.use_eval:
        cmd.append("--use-eval")
    if args.device.lower().startswith("cuda"):
        cmd.append("--cuda")
    if model_dir is not None:
        cmd.extend(["--model-dir", str(model_dir)])

    run_cmd(cmd, LQYLAG_ROOT, stage_log_path, env=env)

    if expected_run_dir.exists():
        return expected_run_dir
    result_root = LQYLAG_ROOT / "scripts" / "results" / "SingleControl" / scenario_name / "ppo" / experiment_name
    return locate_latest_run_dir(result_root)


def run_validation(
    args,
    scenario_name: str,
    export_path: Path,
    output_path: Path,
    hardcase_json: Path | None,
    hardcase_prob: float,
    episodes: int | None = None,
):
    cmd = [
        args.python_exe,
        str(SCRIPT_DIR / "eval_cap_lowlevel_model.py"),
        "--scenario-name",
        scenario_name,
        "--model-path",
        str(export_path),
        "--episodes",
        str(int(episodes if episodes is not None else max(args.eval_episodes, 120))),
        "--seed",
        str(args.seed),
        "--output",
        str(output_path),
    ]
    if hardcase_json is not None:
        cmd.extend(["--hardcase-json", str(hardcase_json), "--hardcase-prob", str(hardcase_prob)])
    run_cmd(cmd, REPO_ROOT, output_path.with_suffix(".runner.log"))


def list_checkpoint_candidates(run_dir: Path, window: int) -> List[Path]:
    numbered = sorted(
        [p for p in run_dir.glob("actor_*.pt") if p.stem.split("_")[-1].isdigit()],
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if window > 0:
        numbered = numbered[-window:]
    latest = run_dir / "actor_latest.pt"
    candidates = numbered[:]
    if latest.exists():
        candidates.append(latest)
    deduped = []
    seen = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def load_eval_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("summary", payload)


def checkpoint_sort_key(curriculum_summary: dict, hardcase_summary: dict) -> tuple:
    reference = hardcase_summary or curriculum_summary
    return (
        float(reference.get("survive_rate", 0.0)),
        float(curriculum_summary.get("survive_rate", 0.0)),
        float(reference.get("avg_min_vc_mps", 0.0)),
        float(reference.get("avg_min_alt_m", 0.0)),
        -float(reference.get("avg_max_aoa_deg", 999.0)),
        float(curriculum_summary.get("avg_reward", 0.0)),
        float(reference.get("avg_steps", 0.0)),
    )


def select_best_checkpoint(
    args,
    scenario_name: str,
    run_dir: Path,
    selector_dir: Path,
    hardcase_json: Path | None,
) -> Path:
    selector_dir.mkdir(parents=True, exist_ok=True)
    candidates = list_checkpoint_candidates(run_dir, args.checkpoint_selection_window)
    if not candidates:
        raise FileNotFoundError(f"No actor checkpoints found under {run_dir}")

    ranking = []
    for idx, candidate in enumerate(candidates):
        curriculum_eval = selector_dir / f"{candidate.stem}_curriculum.json"
        run_validation(
            args=args,
            scenario_name=scenario_name,
            export_path=candidate,
            output_path=curriculum_eval,
            hardcase_json=None,
            hardcase_prob=0.0,
            episodes=args.checkpoint_selection_episodes,
        )
        curriculum_summary = load_eval_summary(curriculum_eval)

        hardcase_summary = {}
        if hardcase_json is not None:
            hardcase_eval = selector_dir / f"{candidate.stem}_hardcase.json"
            run_validation(
                args=args,
                scenario_name=scenario_name,
                export_path=candidate,
                output_path=hardcase_eval,
                hardcase_json=hardcase_json,
                hardcase_prob=1.0,
                episodes=args.checkpoint_selection_episodes,
            )
            hardcase_summary = load_eval_summary(hardcase_eval)

        sort_key = checkpoint_sort_key(curriculum_summary, hardcase_summary)
        ranking.append(
            {
                "candidate": str(candidate),
                "sort_key": list(sort_key),
                "curriculum_summary": curriculum_summary,
                "hardcase_summary": hardcase_summary,
            }
        )

    ranking.sort(key=lambda item: tuple(item["sort_key"]), reverse=True)
    write_json(selector_dir / "checkpoint_ranking.json", {"ranking": ranking})
    return Path(ranking[0]["candidate"])


def train_one_aircraft(args, aircraft: str, root_run_dir: Path):
    aircraft_run_dir = root_run_dir / aircraft
    aircraft_run_dir.mkdir(parents=True, exist_ok=True)
    watch_paths_file = aircraft_run_dir / "watch_paths.txt"

    init_path = Path(args.f16_init_path if aircraft == "f16" else args.su27_init_path)
    if args.init_mode == "warmstart" and not init_path.exists():
        raise FileNotFoundError(f"Warmstart model not found: {init_path}")

    source_state = torch.load(init_path, map_location="cpu", weights_only=True) if init_path.exists() else {}
    use_mlp_actlayer = actor_uses_mlp_actlayer(source_state) if source_state else (aircraft == "su27")
    scenario_name = scenario_name_for(aircraft)
    export_path = default_export_path(Path(args.export_dir), aircraft)
    hardcase_json = build_hardcase_payload(args, aircraft, aircraft_run_dir / f"{aircraft}_hardcases.json")

    stage_status = {
        "aircraft": aircraft,
        "scenario_name": scenario_name,
        "use_mlp_actlayer": use_mlp_actlayer,
        "warmstart_source": str(init_path) if args.init_mode == "warmstart" else "",
        "hardcase_json": str(hardcase_json) if hardcase_json else "",
        "stages": {},
    }
    write_json(aircraft_run_dir / "status.json", stage_status)

    model_dir = None
    if args.init_mode == "warmstart":
        model_dir = prepare_warmstart_dir(str(init_path), use_mlp_actlayer, aircraft_run_dir)
        stage_status["warmstart_dir"] = str(model_dir)
        write_json(aircraft_run_dir / "status.json", stage_status)

    foundation_env = {
        "CAP_LOWLEVEL_PROFILE_MODE": "foundation",
        "CAP_LOWLEVEL_HARDCASE_JSON": str(hardcase_json) if hardcase_json else "",
        "CAP_LOWLEVEL_HARDCASE_PROB": str(args.foundation_hardcase_prob if hardcase_json else 0.0),
    }
    foundation_experiment = f"{aircraft}_cap_lowlevel_foundation"
    foundation_root = LQYLAG_ROOT / "scripts" / "results" / "SingleControl" / scenario_name / "ppo" / foundation_experiment
    foundation_expected_run = predict_next_run_dir(foundation_root)
    write_watch_paths(
        watch_paths_file,
        {
            "launcher_run_dir": str(root_run_dir),
            "aircraft": aircraft,
            "current_stage": "foundation",
            "stage_log": str(aircraft_run_dir / "foundation_stage.log"),
            "expected_ppo_run_dir": str(foundation_expected_run),
            "training_log_txt": str(foundation_expected_run / "training_log.txt"),
            "training_metrics_json": str(foundation_expected_run / "training_metrics.json"),
            "training_progress_json": str(foundation_expected_run / "training_progress.json"),
            "hardcase_json": str(hardcase_json) if hardcase_json else "",
            "status_json": str(aircraft_run_dir / "status.json"),
        },
    )
    foundation_run = run_training_stage(
        args=args,
        aircraft=aircraft,
        scenario_name=scenario_name,
        experiment_name=foundation_experiment,
        model_dir=model_dir,
        use_mlp_actlayer=use_mlp_actlayer,
        num_env_steps=args.foundation_steps,
        stage_log_path=aircraft_run_dir / "foundation_stage.log",
        env_overrides=foundation_env,
        expected_run_dir=foundation_expected_run,
    )
    stage_status["stages"]["foundation"] = {"run_dir": str(foundation_run)}
    write_json(aircraft_run_dir / "status.json", stage_status)

    latest_run = foundation_run
    final_stage = "foundation"
    if args.hardcase_steps > 0:
        hardcase_env = {
            "CAP_LOWLEVEL_PROFILE_MODE": "hardcase",
            "CAP_LOWLEVEL_HARDCASE_JSON": str(hardcase_json) if hardcase_json else "",
            "CAP_LOWLEVEL_HARDCASE_PROB": str(args.hardcase_prob if hardcase_json else 0.0),
        }
        hardcase_experiment = f"{aircraft}_cap_lowlevel_hardcase"
        hardcase_root = LQYLAG_ROOT / "scripts" / "results" / "SingleControl" / scenario_name / "ppo" / hardcase_experiment
        hardcase_expected_run = predict_next_run_dir(hardcase_root)
        write_watch_paths(
            watch_paths_file,
            {
                "launcher_run_dir": str(root_run_dir),
                "aircraft": aircraft,
                "current_stage": "hardcase",
                "stage_log": str(aircraft_run_dir / "hardcase_stage.log"),
                "expected_ppo_run_dir": str(hardcase_expected_run),
                "training_log_txt": str(hardcase_expected_run / "training_log.txt"),
                "training_metrics_json": str(hardcase_expected_run / "training_metrics.json"),
                "training_progress_json": str(hardcase_expected_run / "training_progress.json"),
                "hardcase_json": str(hardcase_json) if hardcase_json else "",
                "status_json": str(aircraft_run_dir / "status.json"),
            },
        )
        hardcase_run = run_training_stage(
            args=args,
            aircraft=aircraft,
            scenario_name=scenario_name,
            experiment_name=hardcase_experiment,
            model_dir=foundation_run,
            use_mlp_actlayer=use_mlp_actlayer,
            num_env_steps=args.hardcase_steps,
            stage_log_path=aircraft_run_dir / "hardcase_stage.log",
            env_overrides=hardcase_env,
            expected_run_dir=hardcase_expected_run,
        )
        latest_run = hardcase_run
        final_stage = "hardcase"
        stage_status["stages"]["hardcase"] = {"run_dir": str(hardcase_run)}
        write_json(aircraft_run_dir / "status.json", stage_status)

    selected_actor = latest_run / "actor_latest.pt"
    if args.select_best_checkpoint:
        selected_actor = select_best_checkpoint(
            args=args,
            scenario_name=scenario_name,
            run_dir=latest_run,
            selector_dir=aircraft_run_dir / "checkpoint_selection",
            hardcase_json=hardcase_json,
        )
        stage_status["selected_actor"] = str(selected_actor)
        write_json(aircraft_run_dir / "status.json", stage_status)

    export_ppo_actor_to_baseline(selected_actor, export_path, use_mlp_actlayer)
    write_model_metadata(export_path, aircraft, scenario_name, final_stage, latest_run, str(selected_actor))
    stage_status["export_path"] = str(export_path)
    write_json(aircraft_run_dir / "status.json", stage_status)
    write_watch_paths(
        watch_paths_file,
        {
            "launcher_run_dir": str(root_run_dir),
            "aircraft": aircraft,
            "current_stage": f"done:{final_stage}",
            "stage_log": "",
            "expected_ppo_run_dir": str(latest_run),
            "training_log_txt": str(latest_run / "training_log.txt"),
            "training_metrics_json": str(latest_run / "training_metrics.json"),
            "training_progress_json": str(latest_run / "training_progress.json"),
            "hardcase_json": str(hardcase_json) if hardcase_json else "",
            "status_json": str(aircraft_run_dir / "status.json"),
        },
    )

    if not args.skip_validation:
        curriculum_eval_path = aircraft_run_dir / f"{aircraft}_validation_curriculum.json"
        run_validation(args, scenario_name, export_path, curriculum_eval_path, None, 0.0)
        stage_status["validation_curriculum"] = str(curriculum_eval_path)

        if hardcase_json is not None:
            hardcase_eval_path = aircraft_run_dir / f"{aircraft}_validation_hardcase.json"
            run_validation(args, scenario_name, export_path, hardcase_eval_path, hardcase_json, 1.0)
            stage_status["validation_hardcase"] = str(hardcase_eval_path)
        write_json(aircraft_run_dir / "status.json", stage_status)

    return {
        "aircraft": aircraft,
        "scenario_name": scenario_name,
        "export_path": str(export_path),
        "latest_run": str(latest_run),
        "hardcase_json": str(hardcase_json) if hardcase_json else "",
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_dir = make_run_dir(args.log_dir)
    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "args": vars(args),
        "results": [],
    }
    write_json(run_dir / "manifest.json", manifest)

    aircraft_list = ["f16", "su27"] if args.aircraft == "all" else [args.aircraft]
    for aircraft in aircraft_list:
        result = train_one_aircraft(args, aircraft, run_dir)
        manifest["results"].append(result)
        write_json(run_dir / "manifest.json", manifest)

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", manifest)
    logging.info(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
