#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
LQY_ROOT = REPO_ROOT / "lqyLAG"
TACTICAL_PROJECT_ROOT = REPO_ROOT / "scripts" / "tacticalProject"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LQY_ROOT) not in sys.path:
    sys.path.insert(0, str(LQY_ROOT))
if str(TACTICAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(TACTICAL_PROJECT_ROOT))

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
from envs.JSBSim.model.baseline_actor import BaselineActor
from acmi_writer import AcmiRecorder
from compare_f16_lowlevel_tacview import (
    actor_uses_mlp_actlayer,
    build_cap_direct_input,
    circular_delta_deg,
    configure_logging,
    get_metrics,
    set_seed,
)


DEFAULT_BASELINE_MODEL = REPO_ROOT / "envs" / "JSBSim" / "model" / "baseline_model.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "tacticalProject" / "acmi_output" / "f16_ppt_maneuver_demos"
DEFAULT_SCENARIO = "1/cap_lowlevel_f16_tactical_residual"


@dataclass(frozen=True)
class DemoCase:
    name: str
    seconds: float
    phases: list[tuple[str, float, tuple[int, int, int]]]
    scenario: str = DEFAULT_SCENARIO
    inference_style: str = "env_obs"


DEMO_CASES = [
    DemoCase(
        name="01_level_hold",
        seconds=18.0,
        phases=[
            ("trim_hold", 18.0, (7, 8, 3)),
        ],
    ),
    DemoCase(
        name="02_turn_left_60",
        seconds=28.0,
        phases=[
            ("trim_hold", 4.0, (7, 8, 3)),
            ("turn_left_60", 24.0, (7, 4, 3)),
        ],
    ),
    DemoCase(
        name="03_turn_right_60",
        seconds=28.0,
        phases=[
            ("trim_hold", 4.0, (7, 8, 3)),
            ("turn_right_60", 24.0, (7, 12, 3)),
        ],
    ),
    DemoCase(
        name="04_climb_1000m",
        seconds=32.0,
        phases=[
            ("trim_hold", 5.0, (7, 8, 3)),
            ("climb_1000m", 27.0, (13, 8, 3)),
        ],
    ),
    DemoCase(
        name="05_descend_1000m",
        seconds=32.0,
        phases=[
            ("trim_hold", 5.0, (7, 8, 3)),
            ("descend_1000m", 27.0, (1, 8, 3)),
        ],
    ),
    DemoCase(
        name="06_tactical_crank_left",
        seconds=36.0,
        phases=[
            ("trim_hold", 4.0, (7, 8, 3)),
            ("crank_left", 16.0, (7, 4, 3)),
            ("level_hold", 16.0, (7, 8, 3)),
        ],
    ),
    DemoCase(
        name="07_short_skate_left",
        seconds=42.0,
        phases=[
            ("trim_hold", 4.0, (7, 8, 3)),
            ("skate_crank", 12.0, (7, 4, 3)),
            ("turn_cold", 10.0, (7, 12, 5)),
            ("escape_accel", 16.0, (7, 8, 6)),
        ],
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 7 Tacview ACMI demo files for PPT maneuver slides.")
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frame-every", type=int, default=1)
    return parser.parse_args()


class BaselineController:
    def __init__(self, model_path: Path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.actor = BaselineActor(input_dim=12, use_mlp_actlayer=actor_uses_mlp_actlayer(state_dict))
        self.actor.load_state_dict(state_dict)
        self.actor.eval()
        self.rnn_state = np.zeros((1, 1, 128), dtype=np.float32)

    def reset(self):
        self.rnn_state = np.zeros((1, 1, 128), dtype=np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        action, next_state = self.actor(obs[np.newaxis, :], self.rnn_state)
        self.rnn_state = next_state.detach().cpu().numpy()
        return action.detach().cpu().numpy().reshape(-1).astype(np.int64)


def apply_phase(env: SingleControlEnv, agent_id: str, phase_name: str, cmd: tuple[int, int, int]):
    task = env.task
    alt_idx, hdg_idx, vel_idx = (int(cmd[0]), int(cmd[1]), int(cmd[2]))
    task._apply_segment_targets(env, agent_id, alt_idx, hdg_idx, vel_idx)
    task._segment_states[agent_id] = {
        "alt_idx": alt_idx,
        "hdg_idx": hdg_idx,
        "vel_idx": vel_idx,
        "hold_steps": 10**9,
        "elapsed_steps": 0,
        "profile_name": phase_name,
    }
    if hasattr(task, "_active_profiles"):
        task._active_profiles[agent_id] = {"name": phase_name, "source": "ppt_demo"}


def build_input(env: SingleControlEnv, cmd: tuple[int, int, int], obs: np.ndarray) -> np.ndarray:
    task = env.task
    alt_idx, hdg_idx, vel_idx = (int(cmd[0]), int(cmd[1]), int(cmd[2]))
    inp = np.zeros(12, dtype=np.float32)
    inp[0] = float(getattr(task, "norm_alt", np.linspace(-1.5, 1.5, 15))[min(alt_idx, 14)])
    inp[1] = float(getattr(task, "norm_hdg", np.linspace(-np.pi, np.pi, 17))[min(hdg_idx, 16)])
    inp[2] = float(getattr(task, "norm_vel", np.linspace(-1.5, 1.5, 7))[min(vel_idx, 6)])
    if len(obs) >= 9:
        inp[3:12] = obs[:9]
    return inp


def build_timeline(case: DemoCase, dt: float):
    timeline = []
    for phase_name, seconds, cmd in case.phases:
        steps = max(1, int(round(seconds / dt)))
        timeline.extend([(phase_name, cmd)] * steps)
    return timeline


def run_case(case: DemoCase, model_path: Path, output_dir: Path, seed: int, frame_every: int):
    env = SingleControlEnv(case.scenario)
    env.seed(seed)
    env.reset()
    agent_id = env.ego_ids[0]
    controller = BaselineController(model_path)
    controller.reset()
    recorder = AcmiRecorder()
    dt = float(env.time_interval)
    timeline = build_timeline(case, dt)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acmi_path = output_dir / f"{case.name}_{timestamp}.acmi"
    meta_path = output_dir / f"{case.name}_{timestamp}.json"
    recorder.write_header(str(acmi_path))

    samples = []
    phase_index = 0
    phase_name, phase_seconds_left, phase_cmd = case.phases[phase_index]

    for step_index, (timeline_phase_name, timeline_cmd) in enumerate(timeline):
        if phase_seconds_left <= 0 and phase_index < len(case.phases) - 1:
            phase_index += 1
            phase_name, phase_seconds_left, phase_cmd = case.phases[phase_index]
        apply_phase(env, agent_id, phase_name, phase_cmd)
        obs = env.get_obs()[agent_id].astype(np.float32)
        if case.inference_style == "cap_direct":
            actor_input = build_input(env, phase_cmd, obs)
        else:
            actor_input = obs
        action = controller.act(actor_input)
        env.current_step += 1
        norm_action = env.task.normalize_action(env, agent_id, np.asarray(action, dtype=np.int64))
        env.agents[agent_id].set_property_values(env.task.action_var, norm_action)
        for _ in range(env.agent_interaction_steps):
            for sim in env._jsbsims.values():
                sim.run()
            for sim in env._tempsims.values():
                sim.run()
        env.task.step(env)
        if step_index % max(1, frame_every) == 0:
            recorder.write_frame(str(acmi_path), env, env.current_step * dt)
        metrics = get_metrics(env, agent_id)
        metrics["step"] = int(step_index)
        metrics["time_s"] = float(env.current_step * dt)
        metrics["phase"] = phase_name
        metrics["command"] = list(phase_cmd)
        samples.append(metrics)
        phase_seconds_left -= dt

    summary = {
        "case_name": case.name,
        "model_path": str(model_path),
        "scenario_name": case.scenario,
        "acmi_path": str(acmi_path),
        "sample_count": len(samples),
        "final_metrics": samples[-1] if samples else {},
        "phases": [{"name": p[0], "seconds": p[1], "cmd": list(p[2])} for p in case.phases],
    }
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()
    return acmi_path, meta_path, summary


def main():
    args = parse_args()
    configure_logging(args.output_dir)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.baseline_model.exists():
        raise FileNotFoundError(f"baseline model not found: {args.baseline_model}")

    results = []
    for case in DEMO_CASES:
        logging.info("running %s", case.name)
        result = run_case(case, args.baseline_model, args.output_dir, args.seed, args.frame_every)
        results.append(result)
        logging.info("done %s -> %s", case.name, result[0])

    final_summary = args.output_dir / f"ppt_maneuver_demo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "model_path": str(args.baseline_model),
        "output_dir": str(args.output_dir),
        "cases": [r[2] for r in results],
    }
    final_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[demo] complete: {args.output_dir}")


if __name__ == "__main__":
    main()
