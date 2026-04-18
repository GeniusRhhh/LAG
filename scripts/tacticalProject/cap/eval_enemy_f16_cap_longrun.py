#!/usr/bin/env python
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.JSBSim.model.baseline_actor import BaselineActor


SURVIVE_RE = re.compile(r"我方存活:\s*(\d+/\d+),\s*敌方存活:\s*(\d+/\d+)")
ENERGY_RE = re.compile(r"\[(?:HARD_RECOVERY|ENVELOPE)\]\[(B\d{4})\]\[step=(\d+)\].*stage=ENERGY_L3")
FINAL_RE = re.compile(
    r"\[(B\d{4})\]\[T\+(\d+\.\d+)s\]\[normalize_action\]\[final_execute\]\[act\].*?"
    r"alt=([-\d.]+)m vc=([-\d.]+)mps .*? v_up=([-\d.]+)mps"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an enemy F16 CAP model in the real long-horizon CAP simulation.")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--steps", type=int, default=4200)
    parser.add_argument("--output-root", type=str, default=str(REPO_ROOT / "scripts" / "tacticalProject" / "cap_results" / "enemy_f16_longrun_eval"))
    parser.add_argument("--baseline-model-path", type=str, default="")
    parser.add_argument("--ppo-actor-path", type=str, default="")
    parser.add_argument("--command-semantics", type=str, default="")
    parser.add_argument("--enable-safe-teacher", action="store_true")
    return parser.parse_args()


def actor_uses_mlp_actlayer(state_dict) -> bool:
    return any(key.startswith("act.mlp.") or key.startswith("mlp.") for key in state_dict.keys())


def copy_matching_state(src_state, dst_module) -> int:
    dst_state = dst_module.state_dict()
    matched = 0
    for key, value in dst_state.items():
        if key in src_state and src_state[key].shape == value.shape:
            dst_state[key] = src_state[key]
            matched += 1
    dst_module.load_state_dict(dst_state, strict=False)
    return matched


def export_ppo_actor_to_baseline(ppo_actor_path: Path, export_path: Path):
    ppo_state = torch.load(ppo_actor_path, map_location="cpu", weights_only=True)
    actor = BaselineActor(input_dim=12, use_mlp_actlayer=actor_uses_mlp_actlayer(ppo_state))
    copy_matching_state(ppo_state, actor)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), export_path)


def load_command_semantics(model_path: Path, explicit_value: str) -> str:
    explicit_value = str(explicit_value or "").strip()
    if explicit_value:
        return explicit_value

    candidate_paths = [
        model_path.with_suffix(".json"),
        model_path.parent / "bc_metrics.json",
        model_path.parent / "online_training_metrics.json",
    ]
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        semantics = str(payload.get("command_semantics", "") or "").strip()
        if semantics:
            return semantics
    return "legacy_3x5x3"


def parse_summary(log_path: Path):
    first_energy = {}
    last_final = {}
    survive = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = ENERGY_RE.search(line)
            if match and match.group(1) not in first_energy:
                first_energy[match.group(1)] = int(match.group(2))
            match = FINAL_RE.search(line)
            if match:
                last_final[match.group(1)] = {
                    "time_s": float(match.group(2)),
                    "alt_m": float(match.group(3)),
                    "vc_mps": float(match.group(4)),
                    "v_up_mps": float(match.group(5)),
                }
            if survive is None:
                match = SURVIVE_RE.search(line)
                if match:
                    survive = {
                        "friendly": match.group(1),
                        "enemy": match.group(2),
                    }

    return {
        "survival": survive or {"friendly": "unknown", "enemy": "unknown"},
        "first_energy_l3_step": first_energy,
        "last_final_execute": last_final,
    }


def main():
    args = parse_args()
    if not args.baseline_model_path and not args.ppo_actor_path:
        raise ValueError("Provide either --baseline-model-path or --ppo-actor-path.")

    run_root = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    if args.baseline_model_path:
        model_path = Path(args.baseline_model_path)
    else:
        model_path = run_root / "temp_eval_baseline.pt"
        export_ppo_actor_to_baseline(Path(args.ppo_actor_path), model_path)

    command_semantics = load_command_semantics(
        Path(args.baseline_model_path) if args.baseline_model_path else Path(args.ppo_actor_path),
        args.command_semantics,
    )

    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "command_semantics": command_semantics,
                    "source": "eval_enemy_f16_cap_longrun",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    env = os.environ.copy()
    env["CAP_ENEMY_F16_NATIVE_ENABLED"] = "1"
    env["CAP_ENEMY_F16_NATIVE_PATH"] = str(model_path)
    env["CAP_ENEMY_F16_NATIVE_META_PATH"] = str(meta_path)
    env["CAP_ENEMY_F16_NATIVE_COMMAND_SEMANTICS"] = command_semantics
    env["CAP_ENEMY_SAFE_TEACHER_ENABLED"] = "1" if args.enable_safe_teacher else "0"
    env["CAP_ENEMY_RULE_LOWLEVEL_ENABLED"] = "0"
    env["CAP_RUNTIME_RESPECT_EXTERNAL_OVERRIDES"] = "1"

    cmd = [
        args.python_exe,
        str(Path(__file__).with_name("run_cap_simulation.py")),
        "--steps",
        str(args.steps),
        "--output",
        str(run_root),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

    log_files = sorted(run_root.glob("cap_*.log"), key=lambda p: p.stat().st_mtime)
    if not log_files:
        raise FileNotFoundError(f"No cap log produced under {run_root}")
    log_path = log_files[-1]

    summary = parse_summary(log_path)
    payload = {
        "run_root": str(run_root),
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "command_semantics": command_semantics,
        "steps": int(args.steps),
        "log_path": str(log_path),
        "summary": summary,
    }
    out_path = run_root / "longrun_eval_summary.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
