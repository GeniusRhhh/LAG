#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect real CAP logs with all enemy F16 aircraft fully traced.")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4200)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--teacher-mode", type=str, choices=["legacy", "native"], default="legacy")
    parser.add_argument("--native-model-path", type=str, default="")
    parser.add_argument("--native-meta-path", type=str, default="")
    parser.add_argument("--command-semantics", type=str, default="cap_native_15x17x7")
    parser.add_argument("--enable-safe-teacher", action="store_true")
    return parser.parse_args()


def count_safe_teacher_events(log_path: Path) -> int:
    count = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[ENEMY_SAFE_TEACHER]" in line:
                count += 1
    return count


def main():
    args = parse_args()
    root = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"runs": []}

    for episode_idx in range(args.episodes):
        run_dir = root / f"episode_{episode_idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CAP_ALL_B_DEEP_TRACE"] = "1"
        env["CAP_B0100_DEEP_TRACE"] = "0"
        env["CAP_ROOTCAUSE_TRACE"] = "0"
        env["CAP_ENEMY_SAFE_TEACHER_ENABLED"] = "1" if args.enable_safe_teacher else "0"
        if args.teacher_mode == "native":
            env["CAP_ENEMY_F16_NATIVE_ENABLED"] = "1"
            if args.native_model_path:
                env["CAP_ENEMY_F16_NATIVE_PATH"] = args.native_model_path
            if args.native_meta_path:
                env["CAP_ENEMY_F16_NATIVE_META_PATH"] = args.native_meta_path
            env["CAP_ENEMY_F16_NATIVE_COMMAND_SEMANTICS"] = args.command_semantics
        else:
            env["CAP_ENEMY_F16_NATIVE_ENABLED"] = "0"

        cmd = [
            args.python_exe,
            str(Path(__file__).with_name("run_cap_simulation.py")),
            "--steps",
            str(args.steps),
            "--output",
            str(run_dir),
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
        log_files = sorted(run_dir.glob("cap_*.log"), key=lambda p: p.stat().st_mtime)
        if not log_files:
            raise FileNotFoundError(f"No CAP log produced under {run_dir}")
        safe_teacher_events = count_safe_teacher_events(log_files[-1])
        manifest["runs"].append(
            {
                "episode": int(episode_idx),
                "run_dir": str(run_dir),
                "log_path": str(log_files[-1]),
                "teacher_mode": args.teacher_mode,
                "safe_teacher_enabled": bool(args.enable_safe_teacher),
                "safe_teacher_events": int(safe_teacher_events),
            }
        )

    manifest_path = root / "trace_collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "root": str(root),
                "manifest": str(manifest_path),
                "episodes": int(args.episodes),
                "safe_teacher_enabled": bool(args.enable_safe_teacher),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
