#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import types
import importlib.util
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TACTICAL_PROJECT_ROOT = REPO_ROOT / "scripts" / "tacticalProject"
CAP_ROOT = TACTICAL_PROJECT_ROOT / "cap"
RUNTIME_ROOT = CAP_ROOT / "runtime"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TACTICAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(TACTICAL_PROJECT_ROOT))

MY_AIRCRAFT_TYPE = "su27sk"
ENEMY_AIRCRAFT_TYPE = "f16"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "scripts" / "tacticalProject" / "acmi_output" / "cap_template_demos"
TEMPLATES = [
    ("drag_shoot", "drag_shoot"),
    ("pincer_attack", "pincer_attack"),
    ("high_low", "high_low"),
    ("side_by_side", "side_by_side"),
    ("front_back", "front_back"),
    ("tactical_evasion", "tactical_evasion"),
    ("tactical_turn", "tactical_turn"),
]


def _ensure_namespace_packages():
    if "cap" not in sys.modules:
        mod = types.ModuleType("cap")
        mod.__path__ = [str(CAP_ROOT)]
        sys.modules["cap"] = mod
    if "cap.runtime" not in sys.modules:
        mod = types.ModuleType("cap.runtime")
        mod.__path__ = [str(RUNTIME_ROOT)]
        sys.modules["cap.runtime"] = mod


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime_functions():
    _ensure_namespace_packages()
    bootstrap_mod = _load_module("cap.runtime.bootstrap", RUNTIME_ROOT / "bootstrap.py")
    _load_module("cap.run_logging", CAP_ROOT / "run_logging.py")
    _load_module("cap.acmi_writer", CAP_ROOT / "acmi_writer.py")
    _load_module("cap.run_helpers", CAP_ROOT / "run_helpers.py")
    engine_mod = _load_module("cap.runtime.engine", RUNTIME_ROOT / "engine.py")
    return bootstrap_mod.bootstrap_runtime, engine_mod.run_cap_simulation_core


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 7 CAP template ACMI demos.")
    parser.add_argument("--steps", type=int, default=3600, help="3600 steps = 12 minutes at 0.2s")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    bootstrap_runtime, run_cap_simulation_core = _load_runtime_functions()
    context = bootstrap_runtime(
        entry_file=__file__,
        my_aircraft_type=MY_AIRCRAFT_TYPE,
        enemy_aircraft_type=ENEMY_AIRCRAFT_TYPE,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for label, force_name in TEMPLATES:
        out_dir = args.output_root / f"{timestamp}_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CAP_FORCE_TACTIC"] = force_name
        run_cap_simulation_core(
            context=context,
            max_steps=args.steps,
            output_dir=str(out_dir),
            allow_patrol_fallback=False,
            my_aircraft_type=MY_AIRCRAFT_TYPE,
            enemy_aircraft_type=ENEMY_AIRCRAFT_TYPE,
        )
        summary.append(
            {
                "template": label,
                "forced_env": force_name,
                "output_dir": str(out_dir),
                "steps": int(args.steps),
            }
        )

    summary_path = args.output_root / f"template_demo_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.environ["CAP_FORCE_TACTIC"] = ""


if __name__ == "__main__":
    main()
