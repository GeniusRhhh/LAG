#!/usr/bin/env python
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


REPO_ROOT = Path(__file__).resolve().parents[2]
LQY_ROOT = REPO_ROOT / "lqyLAG"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LQY_ROOT) not in sys.path:
    sys.path.insert(0, str(LQY_ROOT))

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
from envs.JSBSim.model.baseline_actor import BaselineActor


DEFAULT_BASELINE_MODEL = REPO_ROOT / "envs" / "JSBSim" / "model" / "baseline_model.pt"
DEFAULT_NEW_MODEL = REPO_ROOT / "scripts" / "tacticalProject" / "models" / "f16_energy_15x17x7_direct.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "tacticalProject" / "acmi_output" / "f16_lowlevel_compare"
DEFAULT_BASELINE_SCENARIO = "1/cap_lowlevel_f16_tactical_residual"
DEFAULT_NEW_SCENARIO = "1/cap_lowlevel_f16_tactical_energy"

CAP_NORM_ALT = np.array([-1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500], dtype=np.float32) / 1000.0
CAP_NORM_HDG = np.array(
    [
        -np.pi,
        -2 * np.pi / 3,
        -np.pi / 2,
        -5 * np.pi / 12,
        -np.pi / 3,
        -np.pi / 4,
        -np.pi / 6,
        -np.pi / 12,
        0.0,
        np.pi / 12,
        np.pi / 6,
        np.pi / 4,
        np.pi / 3,
        5 * np.pi / 12,
        np.pi / 2,
        2 * np.pi / 3,
        np.pi,
    ],
    dtype=np.float32,
)
CAP_NORM_VEL = np.array([-150, -100, -50, 0, 50, 100, 150], dtype=np.float32) / 100.0


@dataclass(frozen=True)
class CommandPhase:
    name: str
    seconds: float
    cmd: tuple[int, int, int]


DEFAULT_PLAN = [
    CommandPhase("trim_hold", 8.0, (7, 8, 3)),
    CommandPhase("turn_left_30", 12.0, (7, 6, 3)),
    CommandPhase("turn_right_30", 12.0, (7, 10, 3)),
    CommandPhase("climb_500m", 16.0, (11, 8, 3)),
    CommandPhase("descend_500m", 16.0, (3, 8, 3)),
    CommandPhase("accelerate_100", 12.0, (7, 8, 5)),
    CommandPhase("decelerate_100", 12.0, (7, 8, 1)),
]

STRONG_PLAN = [
    CommandPhase("trim_hold", 8.0, (7, 8, 3)),
    CommandPhase("turn_left_60", 14.0, (7, 4, 3)),
    CommandPhase("turn_right_60", 14.0, (7, 12, 3)),
    CommandPhase("climb_1000m", 18.0, (13, 8, 3)),
    CommandPhase("descend_1000m", 18.0, (1, 8, 3)),
    CommandPhase("accelerate_150", 14.0, (7, 8, 6)),
    CommandPhase("decelerate_150", 14.0, (7, 8, 0)),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare old/new F16 low-level models in single-aircraft Tacview ACMI.")
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--new-model", type=Path, default=DEFAULT_NEW_MODEL)
    parser.add_argument("--baseline-scenario", type=str, default=DEFAULT_BASELINE_SCENARIO)
    parser.add_argument("--new-scenario", type=str, default=DEFAULT_NEW_SCENARIO)
    parser.add_argument("--baseline-inference-style", type=str, choices=["env_obs", "cap_direct"], default="env_obs")
    parser.add_argument("--new-inference-style", type=str, choices=["env_obs", "cap_direct"], default="env_obs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frame-every", type=int, default=1)
    parser.add_argument("--plan-mode", type=str, choices=["standard", "strong"], default="standard")
    return parser.parse_args()


def configure_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"compare_f16_lowlevel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    logging.info("log: %s", log_path)
    return log_path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def actor_uses_mlp_actlayer(state_dict) -> bool:
    return any(key.startswith("act.mlp.") for key in state_dict.keys())


class BaselineController:
    def __init__(self, model_path: Path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        actor = BaselineActor(input_dim=12, use_mlp_actlayer=actor_uses_mlp_actlayer(state_dict))
        actor.load_state_dict(state_dict)
        actor.eval()
        self.actor = actor
        self.reset()

    def reset(self):
        self.rnn_state = np.zeros((1, 1, 128), dtype=np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        action, next_state = self.actor(obs[np.newaxis, :], self.rnn_state)
        self.rnn_state = next_state.detach().cpu().numpy()
        return action.detach().cpu().numpy().reshape(-1).astype(np.int64)


def force_segment(env: SingleControlEnv, agent_id: str, phase: CommandPhase):
    task = env.task
    alt_idx, hdg_idx, vel_idx = (int(phase.cmd[0]), int(phase.cmd[1]), int(phase.cmd[2]))
    task._apply_segment_targets(env, agent_id, alt_idx, hdg_idx, vel_idx)
    task._segment_states[agent_id] = {
        "alt_idx": alt_idx,
        "hdg_idx": hdg_idx,
        "vel_idx": vel_idx,
        "hold_steps": 10 ** 9,
        "elapsed_steps": 0,
        "profile_name": phase.name,
    }
    if hasattr(task, "_active_profiles"):
        task._active_profiles[agent_id] = {"name": phase.name, "source": "manual_compare"}


def get_metrics(env: SingleControlEnv, agent_id: str) -> dict:
    aircraft = env.agents[agent_id]
    return {
        "heading_deg": float(np.degrees(aircraft.get_property_value(c.attitude_heading_true_rad))) % 360.0,
        "roll_deg": float(np.degrees(aircraft.get_property_value(c.attitude_roll_rad))),
        "pitch_deg": float(np.degrees(aircraft.get_property_value(c.attitude_pitch_rad))),
        "altitude_m": float(aircraft.get_property_value(c.position_h_sl_m)),
        "vc_mps": float(aircraft.get_property_value(c.velocities_vc_mps)),
        "tas_mps": float(np.linalg.norm(aircraft.get_velocity())),
        "v_up_mps": -float(aircraft.get_property_value(c.velocities_v_down_mps)),
        "aoa_deg": float(aircraft.get_property_value(c.aero_alpha_deg)),
    }


def build_cap_direct_input(env: SingleControlEnv, phase: CommandPhase, obs: np.ndarray) -> np.ndarray:
    task = env.task
    alt_idx, hdg_idx, vel_idx = (int(phase.cmd[0]), int(phase.cmd[1]), int(phase.cmd[2]))

    norm_alt = getattr(task, "norm_alt", CAP_NORM_ALT)
    norm_hdg = getattr(task, "norm_hdg", CAP_NORM_HDG)
    norm_vel = getattr(task, "norm_vel", CAP_NORM_VEL)

    inp = np.zeros(12, dtype=np.float32)
    inp[0] = float(norm_alt[min(alt_idx, len(norm_alt) - 1)])
    inp[1] = float(norm_hdg[min(hdg_idx, len(norm_hdg) - 1)])
    inp[2] = float(norm_vel[min(vel_idx, len(norm_vel) - 1)])
    if len(obs) >= 9:
        inp[3:12] = obs[:9]
    return inp


def circular_delta_deg(start_deg: float, end_deg: float) -> float:
    return ((end_deg - start_deg + 180.0) % 360.0) - 180.0


def build_phase_report(phase: CommandPhase, start_metrics: dict, end_metrics: dict) -> dict:
    return {
        "name": phase.name,
        "cmd": list(phase.cmd),
        "heading_delta_deg": circular_delta_deg(start_metrics["heading_deg"], end_metrics["heading_deg"]),
        "altitude_delta_m": end_metrics["altitude_m"] - start_metrics["altitude_m"],
        "vc_delta_mps": end_metrics["vc_mps"] - start_metrics["vc_mps"],
        "tas_delta_mps": end_metrics["tas_mps"] - start_metrics["tas_mps"],
        "start": start_metrics,
        "end": end_metrics,
    }


def build_timeline(plan: list[CommandPhase], dt: float):
    timeline = []
    phase_ranges = []
    step_cursor = 0
    for phase in plan:
        phase_steps = max(1, int(round(phase.seconds / dt)))
        start_step = step_cursor
        end_step = step_cursor + phase_steps
        timeline.extend([phase] * phase_steps)
        phase_ranges.append(
            {
                "name": phase.name,
                "cmd": list(phase.cmd),
                "start_step": start_step,
                "end_step": end_step,
                "seconds": phase_steps * dt,
            }
        )
        step_cursor = end_step
    return timeline, phase_ranges


def run_model_case(
    case_name: str,
    model_path: Path,
    scenario_name: str,
    inference_style: str,
    seed: int,
    output_dir: Path,
    plan: list[CommandPhase],
    frame_every: int,
):
    env = SingleControlEnv(scenario_name)
    env.seed(seed)
    env.reset()

    agent_id = env.ego_ids[0]
    controller = BaselineController(model_path)
    dt = float(env.time_interval)
    timeline, phase_ranges = build_timeline(plan, dt)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acmi_path = output_dir / f"{case_name}_{timestamp}.acmi"
    json_path = output_dir / f"{case_name}_{timestamp}.json"

    active_phase_name = None
    active_phase_start_metrics = None
    prev_phase = None
    samples = []
    phase_reports = []
    min_altitude = float("inf")
    min_vc = float("inf")
    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    max_sink_rate = 0.0

    for step_index, phase in enumerate(timeline):
        if phase.name != active_phase_name:
            if active_phase_name is not None and active_phase_start_metrics is not None and prev_phase is not None:
                phase_reports.append(build_phase_report(prev_phase, active_phase_start_metrics, get_metrics(env, agent_id)))
            force_segment(env, agent_id, phase)
            active_phase_name = phase.name
            prev_phase = phase
            active_phase_start_metrics = get_metrics(env, agent_id)
            logging.info("[%s] phase=%s cmd=%s step=%d", case_name, phase.name, phase.cmd, step_index)

        obs = env.get_obs()[agent_id].astype(np.float32)
        actor_input = build_cap_direct_input(env, phase, obs) if inference_style == "cap_direct" else obs
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
            env.render(mode="txt", filepath=str(acmi_path))

        metrics = get_metrics(env, agent_id)
        metrics["step"] = int(step_index)
        metrics["time_s"] = float(env.current_step * dt)
        metrics["phase"] = phase.name
        metrics["command"] = list(phase.cmd)
        metrics["model_path"] = str(model_path)
        metrics["inference_style"] = inference_style
        metrics["actor_input"] = [float(x) for x in actor_input.tolist()]
        samples.append(metrics)

        min_altitude = min(min_altitude, metrics["altitude_m"])
        min_vc = min(min_vc, metrics["vc_mps"])
        max_abs_roll = max(max_abs_roll, abs(metrics["roll_deg"]))
        max_abs_pitch = max(max_abs_pitch, abs(metrics["pitch_deg"]))
        max_sink_rate = max(max_sink_rate, -metrics["v_up_mps"])

    if active_phase_name is not None and active_phase_start_metrics is not None and prev_phase is not None:
        phase_reports.append(build_phase_report(prev_phase, active_phase_start_metrics, get_metrics(env, agent_id)))

    summary = {
        "case_name": case_name,
        "model_path": str(model_path),
        "scenario_name": scenario_name,
        "inference_style": inference_style,
        "seed": seed,
        "acmi_path": str(acmi_path),
        "phase_ranges": phase_ranges,
        "phase_reports": phase_reports,
        "final_metrics": samples[-1] if samples else {},
        "min_altitude_m": min_altitude,
        "min_vc_mps": min_vc,
        "max_abs_roll_deg": max_abs_roll,
        "max_abs_pitch_deg": max_abs_pitch,
        "max_sink_rate_mps": max_sink_rate,
        "sample_count": len(samples),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": samples}, f, indent=2)

    logging.info("[%s] acmi=%s", case_name, acmi_path)
    logging.info("[%s] summary=%s", case_name, summary)
    env.close()
    return acmi_path, json_path, summary

def main():
    args = parse_args()
    configure_logging(args.output_dir)
    set_seed(args.seed)
    plan = DEFAULT_PLAN if args.plan_mode == "standard" else STRONG_PLAN

    if not args.baseline_model.exists():
        raise FileNotFoundError(f"baseline model not found: {args.baseline_model}")
    if not args.new_model.exists():
        raise FileNotFoundError(f"new model not found: {args.new_model}")

    baseline_result = run_model_case(
        case_name="f16_baseline",
        model_path=args.baseline_model,
        scenario_name=args.baseline_scenario,
        inference_style=args.baseline_inference_style,
        seed=args.seed,
        output_dir=args.output_dir,
        plan=plan,
        frame_every=args.frame_every,
    )
    new_result = run_model_case(
        case_name="f16_native",
        model_path=args.new_model,
        scenario_name=args.new_scenario,
        inference_style=args.new_inference_style,
        seed=args.seed,
        output_dir=args.output_dir,
        plan=plan,
        frame_every=args.frame_every,
    )

    compare_path = args.output_dir / f"compare_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "baseline": baseline_result[2],
        "native": new_result[2],
        "baseline_acmi": str(baseline_result[0]),
        "native_acmi": str(new_result[0]),
        "baseline_scenario": args.baseline_scenario,
        "new_scenario": args.new_scenario,
        "baseline_inference_style": args.baseline_inference_style,
        "new_inference_style": args.new_inference_style,
        "plan": [{"name": p.name, "seconds": p.seconds, "cmd": list(p.cmd)} for p in plan],
    }
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("[compare] %s", compare_path)


if __name__ == "__main__":
    main()
