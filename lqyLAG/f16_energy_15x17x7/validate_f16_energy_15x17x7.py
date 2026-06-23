#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LQYLAG_ROOT = SCRIPT_DIR.parent
REPO_ROOT = LQYLAG_ROOT.parent
EVAL_SCRIPT = LQYLAG_ROOT / "scripts" / "train" / "eval_cap_lowlevel_model.py"
DEFAULT_MODEL = REPO_ROOT / "scripts" / "tacticalProject" / "models" / "f16_energy_15x17x7_direct.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a trained F16 15x17x7 low-level model.")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument(
        "--scenario-name",
        type=str,
        default="1/cap_lowlevel_f16_tactical_energy",
        help="Use 1/cap_lowlevel_f16_tactical_energy for the optimized direct-command training setup.",
    )
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default=str(SCRIPT_DIR / "validation" / "f16_energy_15x17x7_eval.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_exe,
        str(EVAL_SCRIPT),
        "--scenario-name",
        args.scenario_name,
        "--model-path",
        args.model_path,
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--output",
        str(output_path),
    ]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(LQYLAG_ROOT), check=True)
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
