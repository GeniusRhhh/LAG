#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
LQYLAG_ROOT = SCRIPT_DIR.parent
REPO_ROOT = LQYLAG_ROOT.parent
TRAIN_JSBSIM = LQYLAG_ROOT / "scripts" / "train" / "train_jsbsim.py"
EVAL_SCRIPT = LQYLAG_ROOT / "scripts" / "train" / "eval_cap_lowlevel_model.py"
DEFAULT_PROJECT_MODEL_DIR = REPO_ROOT / "scripts" / "tacticalProject" / "models"
DEFAULT_WARMSTART_MODEL = REPO_ROOT / "envs" / "JSBSim" / "model" / "baseline_model.pt"

sys.path.append(str(REPO_ROOT))

from lqyLAG.algorithms.ppo.ppo_actor import PPOActor
from lqyLAG.algorithms.ppo.ppo_critic import PPOCritic
from lqyLAG.config import get_config
from lqyLAG.envs.JSBSim.model.baseline_actor import BaselineActor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an F16 native 15x17x7 low-level model with plain PPO only, no curriculum builder."
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario-name", type=str, default="1/cap_lowlevel_f16_tactical_energy")
    parser.add_argument("--total-steps", type=int, default=10000000)
    parser.add_argument("--n-rollout-threads", type=int, default=16)
    parser.add_argument("--n-eval-rollout-threads", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=240)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ppo-epoch", type=int, default=10)
    parser.add_argument("--num-mini-batch", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--eval-episodes", type=int, default=96)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--use-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-mode", type=str, default="warmstart", choices=["warmstart", "scratch"])
    parser.add_argument("--warmstart-model", type=str, default=str(DEFAULT_WARMSTART_MODEL))
    parser.add_argument("--launcher-runs-dir", type=str, default=str(SCRIPT_DIR / "runs"))
    parser.add_argument("--exports-dir", type=str, default=str(SCRIPT_DIR / "exports"))
    parser.add_argument("--project-model-dir", type=str, default=str(DEFAULT_PROJECT_MODEL_DIR))
    parser.add_argument("--output-model-name", type=str, default="f16_energy_15x17x7_direct.pt")
    parser.add_argument("--experiment-name", type=str, default="f16_energy_15x17x7_plainppo")
    parser.add_argument("--final-eval-episodes", type=int, default=160)
    parser.add_argument("--select-best-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-selection-window", type=int, default=8)
    parser.add_argument("--checkpoint-selection-episodes", type=int, default=40)
    parser.add_argument("--skip-final-eval", action="store_true")
    return parser.parse_args()


def run_and_stream(cmd, cwd: Path):
    print("[run]", " ".join(cmd))
    log_file = None
    if hasattr(run_and_stream, "_log_path") and run_and_stream._log_path:
        log_path = Path(run_and_stream._log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_file is not None:
            log_file.write(line)
            log_file.flush()
    rc = proc.wait()
    if log_file is not None:
        log_file.close()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def prepare_warmstart_dir(source_model_path: Path, run_dir: Path) -> tuple[Path, bool, dict]:
    import gymnasium as gym

    source_state = torch.load(source_model_path, map_location="cpu", weights_only=True)
    use_mlp_actlayer = actor_uses_mlp_actlayer(source_state)
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
    info = {
        "source_model_path": str(source_model_path),
        "use_mlp_actlayer": use_mlp_actlayer,
        "actor_match_count": actor_match_count,
        "critic_match_count": critic_match_count,
    }
    with open(warmstart_dir / "warmstart_manifest.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return warmstart_dir, use_mlp_actlayer, info


def export_ppo_actor_to_baseline(ppo_actor_path: Path, export_path: Path, use_mlp_actlayer: bool):
    ppo_state = torch.load(ppo_actor_path, map_location="cpu", weights_only=True)
    actor = BaselineActor(input_dim=12, use_mlp_actlayer=use_mlp_actlayer)
    copy_matching_state(ppo_state, actor)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), export_path)


def write_model_metadata(export_path: Path, scenario_name: str, selected_actor: Path, training_run_dir: Path):
    metadata = {
        "schema_version": 1,
        "aircraft": "f16",
        "scenario_name": scenario_name,
        "selected_actor_checkpoint": str(selected_actor),
        "training_run_dir": str(training_run_dir),
        "command_semantics": "cap_native_15x17x7",
        "observation_layout": [
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
        ],
        "action_layout": [
            "aileron_index_0_40",
            "elevator_index_0_40",
            "rudder_index_0_40",
            "throttle_index_0_29",
        ],
        "training_mode": "plain_ppo_no_curriculum",
    }
    with open(export_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def copy_with_metadata(src_model: Path, dst_model: Path):
    dst_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_model, dst_model)
    src_meta = src_model.with_suffix(".json")
    if src_meta.exists():
        shutil.copy2(src_meta, dst_model.with_suffix(".json"))


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def collect_training_artifacts(training_run_dir: Path, root_run_dir: Path) -> dict:
    artifacts_dir = root_run_dir / "artifacts"
    checkpoints_dir = artifacts_dir / "checkpoints"
    summaries_dir = artifacts_dir / "summaries"
    copied = {
        "training_run_dir": str(training_run_dir),
        "summaries": {},
        "checkpoints": [],
    }

    summary_files = [
        "training_log.txt",
        "training_metrics.json",
        "training_progress.json",
    ]
    for name in summary_files:
        src = training_run_dir / name
        dst = summaries_dir / name
        if copy_if_exists(src, dst):
            copied["summaries"][name] = str(dst)

    checkpoint_candidates = sorted(training_run_dir.glob("actor_*.pt"), key=lambda p: p.name)
    checkpoint_candidates += sorted(training_run_dir.glob("critic_*.pt"), key=lambda p: p.name)
    for src in checkpoint_candidates:
        dst = checkpoints_dir / src.name
        if copy_if_exists(src, dst):
            copied["checkpoints"].append(str(dst))

    for name in ("actor_latest.pt", "critic_latest.pt"):
        src = training_run_dir / name
        dst = checkpoints_dir / name
        if copy_if_exists(src, dst):
            copied["latest_" + name.split(".")[0]] = str(dst)

    return copied


def locate_latest_run_dir(base_dir: Path) -> Path:
    run_dirs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run")],
        key=lambda p: p.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return run_dirs[-1]


def list_checkpoint_candidates(run_dir: Path, window: int) -> list[Path]:
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
    for item in candidates:
        key = str(item.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def load_eval_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("summary", payload)


def checkpoint_sort_key(summary: dict) -> tuple:
    return (
        float(summary.get("survive_rate", 0.0)),
        float(summary.get("avg_min_alt_m", 0.0)),
        float(summary.get("avg_min_vc_mps", 0.0)),
        -float(summary.get("crash_rate", 1.0)),
        float(summary.get("avg_steps", 0.0)),
        float(summary.get("avg_reward", 0.0)),
        -float(summary.get("avg_max_aoa_deg", 999.0)),
    )


def evaluate_checkpoint(args, candidate_actor: Path, selector_dir: Path, use_mlp_actlayer: bool) -> tuple[Path, dict]:
    selector_dir.mkdir(parents=True, exist_ok=True)
    baseline_export = selector_dir / f"{candidate_actor.stem}.baseline.pt"
    eval_output = selector_dir / f"{candidate_actor.stem}.eval.json"
    export_ppo_actor_to_baseline(candidate_actor, baseline_export, use_mlp_actlayer)
    eval_cmd = [
        args.python_exe,
        str(EVAL_SCRIPT),
        "--scenario-name",
        args.scenario_name,
        "--model-path",
        str(baseline_export),
        "--episodes",
        str(args.checkpoint_selection_episodes),
        "--seed",
        str(args.seed),
        "--output",
        str(eval_output),
    ]
    run_and_stream._log_path = str(selector_dir / f"{candidate_actor.stem}.runner.log")
    run_and_stream(eval_cmd, LQYLAG_ROOT)
    return eval_output, load_eval_summary(eval_output)


def select_best_checkpoint(args, run_dir: Path, selector_dir: Path, use_mlp_actlayer: bool) -> tuple[Path, dict]:
    ranking = []
    for candidate in list_checkpoint_candidates(run_dir, args.checkpoint_selection_window):
        eval_output, summary = evaluate_checkpoint(args, candidate, selector_dir, use_mlp_actlayer)
        ranking.append(
            {
                "candidate": str(candidate),
                "eval_output": str(eval_output),
                "summary": summary,
                "sort_key": list(checkpoint_sort_key(summary)),
            }
        )
    ranking.sort(key=lambda item: tuple(item["sort_key"]), reverse=True)
    ranking_path = selector_dir / "checkpoint_ranking.json"
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump({"ranking": ranking}, f, ensure_ascii=False, indent=2)
    return Path(ranking[0]["candidate"]), {"ranking_path": str(ranking_path), "ranking": ranking}


def main():
    args = parse_args()

    launcher_runs_dir = Path(args.launcher_runs_dir)
    launcher_runs_dir.mkdir(parents=True, exist_ok=True)
    root_run_dir = launcher_runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    root_run_dir.mkdir(parents=True, exist_ok=True)

    model_dir = None
    use_mlp_actlayer = False
    warmstart_info = {}
    if args.init_mode == "warmstart":
        warmstart_model = Path(args.warmstart_model)
        if not warmstart_model.exists():
            raise FileNotFoundError(f"Warmstart model not found: {warmstart_model}")
        model_dir, use_mlp_actlayer, warmstart_info = prepare_warmstart_dir(warmstart_model, root_run_dir)

    train_cmd = [
        args.python_exe,
        str(TRAIN_JSBSIM),
        "--env-name",
        "SingleControl",
        "--algorithm-name",
        "ppo",
        "--scenario-name",
        args.scenario_name,
        "--experiment-name",
        args.experiment_name,
        "--seed",
        str(args.seed),
        "--num-env-steps",
        str(args.total_steps),
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
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-interval",
        str(args.eval_interval),
        "--log-interval",
        str(args.log_interval),
        "--save-interval",
        str(args.save_interval),
        "--hidden-size",
        "128 128",
        "--act-hidden-size",
        "128 128" if use_mlp_actlayer else "",
        "--activation-id",
        "1",
        "--user-name",
        "f16_energy_plainppo",
        "--render-mode",
        "txt",
    ]
    if args.use_eval:
        train_cmd.append("--use-eval")
    if args.device.lower().startswith("cuda"):
        train_cmd.append("--cuda")
    if model_dir is not None:
        train_cmd.extend(["--model-dir", str(model_dir)])

    started_at = datetime.now().isoformat(timespec="seconds")
    run_and_stream._log_path = str(root_run_dir / "train_launcher_output.log")
    run_and_stream(train_cmd, LQYLAG_ROOT)

    ppo_root = LQYLAG_ROOT / "scripts" / "results" / "SingleControl" / args.scenario_name / "ppo" / args.experiment_name
    latest_run = locate_latest_run_dir(ppo_root)
    artifact_manifest = collect_training_artifacts(latest_run, root_run_dir)
    selected_actor = latest_run / "actor_latest.pt"
    if not selected_actor.exists():
        raise FileNotFoundError(f"Latest actor checkpoint not found: {selected_actor}")
    checkpoint_selection = {}
    if args.select_best_checkpoint:
        selected_actor, checkpoint_selection = select_best_checkpoint(
            args,
            latest_run,
            root_run_dir / "checkpoint_selection",
            use_mlp_actlayer,
        )

    project_export_path = Path(args.project_model_dir) / args.output_model_name
    package_export_path = Path(args.exports_dir) / args.output_model_name

    export_ppo_actor_to_baseline(selected_actor, project_export_path, use_mlp_actlayer)
    write_model_metadata(project_export_path, args.scenario_name, selected_actor, latest_run)
    copy_with_metadata(project_export_path, package_export_path)

    final_eval_path = None
    if not args.skip_final_eval:
        final_eval_path = root_run_dir / "final_eval_plainppo.json"
        eval_cmd = [
            args.python_exe,
            str(EVAL_SCRIPT),
            "--scenario-name",
            args.scenario_name,
            "--model-path",
            str(project_export_path),
            "--episodes",
            str(args.final_eval_episodes),
            "--seed",
            str(args.seed),
            "--output",
            str(final_eval_path),
        ]
        run_and_stream._log_path = str(root_run_dir / "final_eval_output.log")
        run_and_stream(eval_cmd, LQYLAG_ROOT)

    status = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "training_mode": "plain_ppo_no_curriculum",
        "scenario_name": args.scenario_name,
        "seed": args.seed,
        "experiment_name": args.experiment_name,
        "training_run_dir": str(latest_run),
        "training_artifacts": artifact_manifest,
        "project_export_path": str(project_export_path),
        "package_export_path": str(package_export_path),
        "selected_actor": str(selected_actor),
        "checkpoint_selection": checkpoint_selection,
        "warmstart": warmstart_info,
        "final_eval_path": str(final_eval_path) if final_eval_path else "",
        "note": (
            "No tactical curriculum, no hardcase curriculum, no log builder. "
            "This is direct plain PPO on cap_lowlevel_f16_tactical_energy."
        ),
    }
    with open(root_run_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    export_manifest = {
        "run_root": str(root_run_dir),
        "launcher_log": str(root_run_dir / "train_launcher_output.log"),
        "training_run_dir": str(latest_run),
        "training_artifacts": artifact_manifest,
        "exported_models": {
            "project_model": str(project_export_path),
            "project_model_meta": str(project_export_path.with_suffix(".json")),
            "package_model": str(package_export_path),
            "package_model_meta": str(package_export_path.with_suffix(".json")),
        },
        "final_eval": {
            "json": str(final_eval_path) if final_eval_path else "",
            "runner_log": str(root_run_dir / "final_eval_output.log") if final_eval_path else "",
        },
    }
    with open(root_run_dir / "export_manifest.json", "w", encoding="utf-8") as f:
        json.dump(export_manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
