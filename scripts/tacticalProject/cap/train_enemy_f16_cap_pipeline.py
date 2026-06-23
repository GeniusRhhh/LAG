#!/usr/bin/env python
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def latest_child_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    children = [item for item in path.iterdir() if item.is_dir()]
    if not children:
        return None
    return max(children, key=lambda item: item.stat().st_mtime)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full enemy F16 CAP pipeline: trace collection -> BC pretrain -> BC eval -> online PPO -> final eval."
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--output-root", type=str, default=str(REPO_ROOT / "scripts" / "tacticalProject" / "models" / "enemy_f16_cap_pipeline"))
    parser.add_argument("--log-glob", action="append", default=[])
    parser.add_argument("--model-key", action="append", default=[])
    parser.add_argument("--collect-episodes", type=int, default=0)
    parser.add_argument("--collect-steps", type=int, default=4200)
    parser.add_argument("--teacher-mode", type=str, choices=["legacy", "native"], default="legacy")
    parser.add_argument("--enable-safe-teacher", action="store_true")
    parser.add_argument("--native-model-path", type=str, default="")
    parser.add_argument("--native-meta-path", type=str, default="")
    parser.add_argument("--dataset-min-alt-m", type=float, default=6000.0)
    parser.add_argument("--dataset-min-vc-mps", type=float, default=170.0)
    parser.add_argument("--dataset-max-abs-g", type=float, default=6.0)
    parser.add_argument("--allow-mixed-semantics", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-chunks", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4200)
    parser.add_argument("--warmstart-baseline", type=str, default=str(REPO_ROOT / "envs" / "JSBSim" / "model" / "baseline_model.pt"))
    parser.add_argument("--warmstart-ppo-actor", type=str, default="")
    parser.add_argument("--online-updates", type=int, default=0)
    parser.add_argument("--online-buffer-size", type=int, default=256)
    parser.add_argument("--online-train-agent-id", type=str, default="B0100")
    parser.add_argument("--online-max-steps", type=int, default=4200)
    parser.add_argument("--online-lr", type=float, default=1e-4)
    parser.add_argument("--online-gamma", type=float, default=0.995)
    parser.add_argument("--online-gae-lambda", type=float, default=0.95)
    parser.add_argument("--online-ppo-epoch", type=int, default=8)
    parser.add_argument("--online-num-mini-batch", type=int, default=4)
    parser.add_argument("--online-data-chunk-length", type=int, default=64)
    parser.add_argument("--online-entropy-coef", type=float, default=0.002)
    parser.add_argument("--online-value-loss-coef", type=float, default=1.0)
    parser.add_argument("--online-clip-param", type=float, default=0.2)
    parser.add_argument("--online-enable-safe-teacher-for-other-enemies", action="store_true")
    parser.add_argument("--online-eval-steps", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_root) / timestamp
    root.mkdir(parents=True, exist_ok=True)

    dataset_path = root / "enemy_f16_cap_dataset.npz"
    bc_dir = root / "bc"
    eval_dir = root / "eval_bc"
    online_dir = root / "online"
    collect_root = root / "collected_traces"

    collected_globs = []
    if args.collect_episodes > 0:
        collect_cmd = [
            args.python_exe,
            str(Path(__file__).with_name("collect_enemy_f16_cap_traces.py")),
            "--output-root",
            str(collect_root),
            "--episodes",
            str(args.collect_episodes),
            "--steps",
            str(args.collect_steps),
            "--teacher-mode",
            args.teacher_mode,
        ]
        if args.enable_safe_teacher:
            collect_cmd.append("--enable-safe-teacher")
        if args.native_model_path:
            collect_cmd.extend(["--native-model-path", args.native_model_path])
        if args.native_meta_path:
            collect_cmd.extend(["--native-meta-path", args.native_meta_path])
        subprocess.run(collect_cmd, cwd=str(REPO_ROOT), check=True)
        collected_globs.extend(
            [
                str(collect_root / "**" / "cap_*.log"),
            ]
        )

    build_cmd = [
        args.python_exe,
        str(Path(__file__).with_name("build_enemy_f16_cap_dataset.py")),
        "--output",
        str(dataset_path),
        "--min-alt-m",
        str(args.dataset_min_alt_m),
        "--min-vc-mps",
        str(args.dataset_min_vc_mps),
        "--max-abs-g",
        str(args.dataset_max_abs_g),
    ]
    if args.allow_mixed_semantics:
        build_cmd.append("--allow-mixed-semantics")
    for pattern in collected_globs + list(args.log_glob):
        build_cmd.extend(["--log-glob", pattern])
    for model_key in args.model_key:
        build_cmd.extend(["--model-key", model_key])
    subprocess.run(build_cmd, cwd=str(REPO_ROOT), check=True)

    bc_cmd = [
        args.python_exe,
        str(Path(__file__).with_name("train_enemy_f16_cap_bc.py")),
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(bc_dir),
        "--epochs",
        str(args.epochs),
        "--seq-len",
        str(args.seq_len),
        "--stride",
        str(args.stride),
        "--batch-chunks",
        str(args.batch_chunks),
    ]
    if args.warmstart_ppo_actor:
        bc_cmd.extend(["--warmstart-ppo-actor", args.warmstart_ppo_actor])
    elif args.warmstart_baseline:
        bc_cmd.extend(["--warmstart-baseline", args.warmstart_baseline])
    subprocess.run(bc_cmd, cwd=str(REPO_ROOT), check=True)

    eval_cmd = [
        args.python_exe,
        str(Path(__file__).with_name("eval_enemy_f16_cap_longrun.py")),
        "--ppo-actor-path",
        str(bc_dir / "actor_bc_best.pt"),
        "--output-root",
        str(eval_dir),
        "--steps",
        str(args.steps),
    ]
    subprocess.run(eval_cmd, cwd=str(REPO_ROOT), check=True)
    bc_eval_run = latest_child_dir(eval_dir)

    online_run = None
    if args.online_updates > 0:
        online_cmd = [
            args.python_exe,
            str(Path(__file__).with_name("train_enemy_f16_cap_online.py")),
            "--output-root",
            str(online_dir),
            "--train-agent-id",
            args.online_train_agent_id,
            "--max-steps",
            str(args.online_max_steps),
            "--buffer-size",
            str(args.online_buffer_size),
            "--updates",
            str(args.online_updates),
            "--lr",
            str(args.online_lr),
            "--gamma",
            str(args.online_gamma),
            "--gae-lambda",
            str(args.online_gae_lambda),
            "--ppo-epoch",
            str(args.online_ppo_epoch),
            "--num-mini-batch",
            str(args.online_num_mini_batch),
            "--data-chunk-length",
            str(args.online_data_chunk_length),
            "--entropy-coef",
            str(args.online_entropy_coef),
            "--value-loss-coef",
            str(args.online_value_loss_coef),
            "--clip-param",
            str(args.online_clip_param),
            "--warmstart-ppo-actor",
            str(bc_dir / "actor_bc_best.pt"),
        ]
        if args.online_enable_safe_teacher_for_other_enemies:
            online_cmd.append("--enable-safe-teacher-for-other-enemies")
        if args.online_eval_steps > 0:
            online_cmd.extend(["--eval-steps", str(args.online_eval_steps)])
        subprocess.run(online_cmd, cwd=str(REPO_ROOT), check=True)
        online_run = latest_child_dir(online_dir)

    summary = {
        "pipeline_root": str(root),
        "dataset_path": str(dataset_path),
        "bc_dir": str(bc_dir),
        "bc_eval_run": str(bc_eval_run) if bc_eval_run is not None else "",
        "online_root": str(online_dir),
        "online_run": str(online_run) if online_run is not None else "",
        "collect_root": str(collect_root),
        "safe_teacher_enabled_for_collection": bool(args.enable_safe_teacher),
        "online_enabled": bool(args.online_updates > 0),
        "online_updates": int(args.online_updates),
    }
    (root / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[pipeline] complete: {root}")


if __name__ == "__main__":
    main()
