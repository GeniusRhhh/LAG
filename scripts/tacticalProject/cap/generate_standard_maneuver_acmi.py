#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TACTICAL_PROJECT_ROOT = REPO_ROOT / "scripts" / "tacticalProject"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TACTICAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(TACTICAL_PROJECT_ROOT))

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.core.simulatior import AircraftSimulator
from envs.JSBSim.model.baseline_actor import BaselineActor

from acmi_writer import AcmiRecorder
from run_enemy_single_lowlevel_verification import _build_model_input, _raw_bins_to_norm_act


DEFAULT_MODEL = REPO_ROOT / "envs" / "JSBSim" / "model" / "baseline_model.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "tacticalProject" / "acmi_output" / "standard_maneuver_primitives_baseline"
DT = 0.2
SIM_FREQ = 60
DEFAULT_INITIAL_U_MPS = 381.34

ALTITUDE_CMD_VALUES_M = np.array(
    [-1500.0, -1000.0, -750.0, -500.0, -300.0, -150.0, -50.0, 0.0, 50.0, 150.0, 300.0, 500.0, 750.0, 1000.0, 1500.0],
    dtype=np.float32,
)
HEADING_CMD_VALUES_DEG = np.array(
    [-180.0, -120.0, -90.0, -75.0, -60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0, 180.0],
    dtype=np.float32,
)
VELOCITY_CMD_VALUES_MPS = np.array([-150.0, -100.0, -50.0, 0.0, 50.0, 100.0, 150.0], dtype=np.float32)


@dataclass(frozen=True)
class InitialState:
    heading_deg: float
    altitude_m: float
    u_mps: float


@dataclass(frozen=True)
class ManeuverCase:
    name: str
    seconds: float
    category: str
    params: dict
    description: str


@dataclass
class SingleAircraftEnvView:
    sim: AircraftSimulator

    def __post_init__(self):
        self._jsbsims = {self.sim.uid: self.sim}
        self._tempsims = {}
        self.agents = self._jsbsims
        self.time_interval = DT
        self.current_step = 0


CASES = [
    ManeuverCase(
        name="01_level_flight_60s",
        seconds=60.0,
        category="level",
        params={},
        description="Basic primitive: level flight, hold heading, altitude, and speed.",
    ),
    ManeuverCase(
        name="02_climb_2000m_60s",
        seconds=90.0,
        category="climb",
        params={"altitude_delta_m": 2000.0, "maneuver_time_s": 20.0, "hold_time_s": 70.0},
        description="Basic primitive: climb 2000 m and hold.",
    ),
    ManeuverCase(
        name="03_descend_2000m_60s",
        seconds=90.0,
        category="descend",
        params={"altitude_delta_m": -2000.0, "maneuver_time_s": 20.0, "hold_time_s": 70.0},
        description="Basic primitive: descend 2000 m and hold.",
    ),
    ManeuverCase(
        name="04_accelerate_100mps_60s",
        seconds=60.0,
        category="accelerate",
        params={"speed_delta_mps": 100.0, "maneuver_time_s": 20.0, "initial_u_mps": 280.0},
        description="Basic primitive: accelerate by 100 m/s and hold, from a lower initial speed.",
    ),
    ManeuverCase(
        name="05_decelerate_100mps_60s",
        seconds=60.0,
        category="decelerate",
        params={"speed_delta_mps": -100.0, "maneuver_time_s": 20.0, "initial_u_mps": 450.0},
        description="Basic primitive: decelerate by 100 m/s and hold, from a higher initial speed.",
    ),
    ManeuverCase(
        name="06_turn_left_30deg_60s",
        seconds=60.0,
        category="turn",
        params={"heading_delta_deg": -30.0, "maneuver_time_s": 18.0},
        description="Basic primitive: standard left turn, 30 deg total heading change.",
    ),
    ManeuverCase(
        name="07_turn_right_30deg_60s",
        seconds=60.0,
        category="turn",
        params={"heading_delta_deg": 30.0, "maneuver_time_s": 18.0},
        description="Basic primitive: standard right turn, 30 deg total heading change.",
    ),
    ManeuverCase(
        name="08_crank_left60_right60_60s",
        seconds=60.0,
        category="crank",
        params={"first_heading_delta_deg": -60.0, "second_heading_delta_deg": 60.0, "segment_time_s": 18.0},
        description="Composite primitive: continuous turning crank, left 60 deg then right 60 deg, no altitude change.",
    ),
    ManeuverCase(
        name="09_tactical_crank_left_68deg_climb_2000m_60s",
        seconds=90.0,
        category="tactical_crank",
        params={"heading_delta_deg": -68.0, "altitude_delta_m": 2000.0, "maneuver_time_s": 20.0, "hold_time_s": 70.0},
        description="Composite primitive: crank with simultaneous 2000 m altitude adjustment.",
    ),
    ManeuverCase(
        name="10_short_skate_left_60s",
        seconds=60.0,
        category="short_skate",
        params={
            "crank_delta_deg": -68.0,
            "crank_time_s": 10.0,
            "offset_hold_time_s": 6.0,
            "cold_heading_delta_deg": 180.0,
            "turn_cold_time_s": 12.0,
            "escape_speed_delta_mps": 120.0,
            "escape_accel_time_s": 17.0,
        },
        description="Composite primitive: offset, quick turn cold, and accelerate escape.",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate standard single-aircraft ACMI maneuver primitives with baseline_model.pt.")
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case-filter", type=str, default="")
    return parser.parse_args()


def configure_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"standard_maneuver_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def actor_uses_mlp_actlayer(state_dict: dict) -> bool:
    return any(key.startswith("act.mlp.") for key in state_dict.keys())


def load_actor(model_path: Path) -> BaselineActor:
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")
    actor = BaselineActor(input_dim=12, use_mlp_actlayer=actor_uses_mlp_actlayer(state_dict))
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def select_cases(case_filter: str) -> list[ManeuverCase]:
    if not case_filter.strip():
        return list(CASES)
    tokens = [token.strip().lower() for token in case_filter.split(",") if token.strip()]
    selected = []
    for case in CASES:
        haystack = f"{case.name} {case.category}".lower()
        if any(token in haystack for token in tokens):
            selected.append(case)
    if not selected:
        raise ValueError(f"No cases matched case_filter={case_filter!r}")
    return selected


def build_sim(
    agent_id: str = "A0100",
    lon_deg: float = 120.3604,
    lat_deg: float = 62.3993,
    heading_deg: float = 0.0,
    initial_u_mps: float = DEFAULT_INITIAL_U_MPS,
) -> AircraftSimulator:
    return AircraftSimulator(
        uid=agent_id,
        color="Blue",
        model="f16",
        init_state={
            "ic_long_gc_deg": lon_deg,
            "ic_lat_geod_deg": lat_deg,
            "ic_h_sl_ft": 32800.0,
            "ic_psi_true_deg": heading_deg,
            "ic_u_fps": float(initial_u_mps) * 3.28084,
        },
        origin=(120.0, 60.0, 0.0),
        sim_freq=SIM_FREQ,
        num_missiles=0,
    )


def normalize_heading_deg(value: float) -> float:
    return float(value) % 360.0


def signed_heading_diff_deg(target_heading: float, current_heading: float) -> float:
    return ((float(target_heading) - float(current_heading) + 180.0) % 360.0) - 180.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def smoothstep01(value: float) -> float:
    x = clamp01(value)
    return x * x * (3.0 - 2.0 * x)


def lerp(start: float, end: float, progress: float) -> float:
    return float(start) + (float(end) - float(start)) * float(progress)


def lerp_heading_deg(start: float, end: float, progress: float) -> float:
    delta = signed_heading_diff_deg(end, start)
    return normalize_heading_deg(float(start) + delta * float(progress))


def get_current_state(sim: AircraftSimulator) -> dict:
    return {
        "heading_deg": normalize_heading_deg(float(sim.get_property_value(c.attitude_psi_deg))),
        "altitude_m": float(sim.get_property_value(c.position_h_sl_m)),
        "u_mps": float(sim.get_property_value(c.velocities_u_mps)),
        "vc_mps": float(sim.get_property_value(c.velocities_vc_mps)),
        "v_up_mps": -float(sim.get_property_value(c.velocities_v_down_mps)),
        "pitch_deg": float(np.degrees(sim.get_property_value(c.attitude_theta_rad))),
        "roll_deg": float(np.degrees(sim.get_property_value(c.attitude_phi_rad))),
        "tas_mps": float(np.linalg.norm(sim.get_velocity())),
    }


def build_initial_state(sim: AircraftSimulator) -> InitialState:
    current = get_current_state(sim)
    return InitialState(
        heading_deg=current["heading_deg"],
        altitude_m=current["altitude_m"],
        u_mps=current["u_mps"],
    )


def compute_targets(case: ManeuverCase, elapsed_s: float, initial: InitialState) -> tuple[str, float, float, float]:
    if case.category == "level":
        return "LEVEL_HOLD", initial.heading_deg, initial.altitude_m, initial.u_mps

    if case.category == "climb":
        maneuver_time = float(case.params["maneuver_time_s"])
        target_alt_final = initial.altitude_m + float(case.params["altitude_delta_m"])
        if elapsed_s <= maneuver_time:
            progress = smoothstep01(elapsed_s / maneuver_time)
            target_alt = lerp(initial.altitude_m, target_alt_final, progress)
            return "CLIMB", initial.heading_deg, target_alt, initial.u_mps + 20.0 * progress
        return "CLIMB_HOLD_LEVEL", initial.heading_deg, target_alt_final, initial.u_mps

    if case.category == "descend":
        maneuver_time = float(case.params["maneuver_time_s"])
        target_alt_final = initial.altitude_m + float(case.params["altitude_delta_m"])
        if elapsed_s <= maneuver_time:
            progress = smoothstep01(elapsed_s / maneuver_time)
            target_alt = lerp(initial.altitude_m, target_alt_final, progress)
            return "DESCEND", initial.heading_deg, target_alt, initial.u_mps + 10.0 * progress
        return "DESCEND_HOLD_LEVEL", initial.heading_deg, target_alt_final, initial.u_mps

    if case.category == "accelerate":
        progress = smoothstep01(elapsed_s / float(case.params["maneuver_time_s"]))
        target_speed = lerp(initial.u_mps, initial.u_mps + float(case.params["speed_delta_mps"]), progress)
        return "ACCELERATE", initial.heading_deg, initial.altitude_m, target_speed

    if case.category == "decelerate":
        progress = smoothstep01(elapsed_s / float(case.params["maneuver_time_s"]))
        target_speed = lerp(initial.u_mps, initial.u_mps + float(case.params["speed_delta_mps"]), progress)
        return "DECELERATE", initial.heading_deg, initial.altitude_m, max(220.0, target_speed)

    if case.category == "turn":
        progress = smoothstep01(elapsed_s / float(case.params["maneuver_time_s"]))
        final_heading = normalize_heading_deg(initial.heading_deg + float(case.params["heading_delta_deg"]))
        target_heading = lerp_heading_deg(initial.heading_deg, final_heading, progress)
        return "TURN", target_heading, initial.altitude_m, initial.u_mps

    if case.category == "crank":
        first_delta = float(case.params["first_heading_delta_deg"])
        second_delta = float(case.params["second_heading_delta_deg"])
        segment_time = float(case.params["segment_time_s"])
        first_heading = normalize_heading_deg(initial.heading_deg + first_delta)
        final_heading = normalize_heading_deg(first_heading + second_delta)
        if elapsed_s <= segment_time:
            progress = smoothstep01(elapsed_s / segment_time)
            target_heading = lerp_heading_deg(initial.heading_deg, first_heading, progress)
            return "CRANK_LEFT", target_heading, initial.altitude_m, initial.u_mps
        progress = smoothstep01((elapsed_s - segment_time) / segment_time)
        target_heading = lerp_heading_deg(first_heading, final_heading, progress)
        return "CRANK_RIGHT", target_heading, initial.altitude_m, initial.u_mps

    if case.category == "tactical_crank":
        maneuver_time = float(case.params["maneuver_time_s"])
        final_heading = normalize_heading_deg(initial.heading_deg + float(case.params["heading_delta_deg"]))
        target_alt_final = initial.altitude_m + float(case.params["altitude_delta_m"])
        if elapsed_s <= maneuver_time:
            progress = smoothstep01(elapsed_s / maneuver_time)
            target_heading = lerp_heading_deg(initial.heading_deg, final_heading, progress)
            target_alt = lerp(initial.altitude_m, target_alt_final, progress)
            return "TACTICAL_CRANK", target_heading, target_alt, initial.u_mps + 15.0 * progress
        return "TACTICAL_CRANK_HOLD_LEVEL", final_heading, target_alt_final, initial.u_mps

    if case.category == "short_skate":
        crank_heading = normalize_heading_deg(initial.heading_deg + float(case.params["crank_delta_deg"]))
        cold_heading = normalize_heading_deg(initial.heading_deg + float(case.params["cold_heading_delta_deg"]))
        escape_speed = initial.u_mps + float(case.params["escape_speed_delta_mps"])
        t1 = float(case.params["crank_time_s"])
        t2 = t1 + float(case.params["offset_hold_time_s"])
        t3 = t2 + float(case.params["turn_cold_time_s"])
        t4 = t3 + float(case.params["escape_accel_time_s"])

        if elapsed_s <= t1:
            progress = smoothstep01(elapsed_s / t1)
            target_heading = lerp_heading_deg(initial.heading_deg, crank_heading, progress)
            return "SHORT_SKATE_CRANK", target_heading, initial.altitude_m, initial.u_mps
        if elapsed_s <= t2:
            return "SHORT_SKATE_OFFSET_HOLD", crank_heading, initial.altitude_m, initial.u_mps
        if elapsed_s <= t3:
            progress = smoothstep01((elapsed_s - t2) / max(float(case.params["turn_cold_time_s"]), 1e-6))
            target_heading = lerp_heading_deg(crank_heading, cold_heading, progress)
            return "SHORT_SKATE_TURN_COLD", target_heading, initial.altitude_m, initial.u_mps + 20.0 * progress
        if elapsed_s <= t4:
            progress = smoothstep01((elapsed_s - t3) / max(float(case.params["escape_accel_time_s"]), 1e-6))
            target_speed = lerp(initial.u_mps, escape_speed, progress)
            return "SHORT_SKATE_ESCAPE", cold_heading, initial.altitude_m, target_speed
        return "SHORT_SKATE_HOLD", cold_heading, initial.altitude_m, escape_speed

    raise ValueError(f"Unsupported category: {case.category}")


def index_for_altitude_delta(altitude_delta_m: float) -> int:
    if abs(float(altitude_delta_m)) < 25.0:
        return 7
    return int(np.argmin(np.abs(ALTITUDE_CMD_VALUES_M - float(altitude_delta_m))))


def index_for_heading_delta(heading_delta_deg: float) -> int:
    if abs(float(heading_delta_deg)) < 2.0:
        return 8
    return int(np.argmin(np.abs(HEADING_CMD_VALUES_DEG - float(heading_delta_deg))))


def index_for_velocity_delta(speed_delta_mps: float) -> int:
    if abs(float(speed_delta_mps)) < 8.0:
        return 3
    return int(np.argmin(np.abs(VELOCITY_CMD_VALUES_MPS - float(speed_delta_mps))))


def build_command_indices(current_state: dict, target_heading_deg: float, target_altitude_m: float, target_speed_mps: float) -> tuple[int, int, int]:
    altitude_delta_m = float(target_altitude_m) - float(current_state["altitude_m"])
    heading_delta_deg = signed_heading_diff_deg(float(target_heading_deg), float(current_state["heading_deg"]))
    speed_delta_mps = float(target_speed_mps) - float(current_state["u_mps"])
    return (
        index_for_altitude_delta(altitude_delta_m),
        index_for_heading_delta(heading_delta_deg),
        index_for_velocity_delta(speed_delta_mps),
    )


def step_actor(actor: BaselineActor, sim: AircraftSimulator, cmd: tuple[int, int, int], rnn_state: np.ndarray) -> np.ndarray:
    actor_input = _build_model_input(sim, int(cmd[0]), int(cmd[1]), int(cmd[2]), "direct_15x17x7")
    with torch.no_grad():
        action_raw, rnn_out = actor(
            torch.FloatTensor(actor_input).unsqueeze(0),
            torch.FloatTensor(rnn_state),
        )
    next_rnn = rnn_out.detach().cpu().numpy()
    norm_act = _raw_bins_to_norm_act(action_raw.detach().cpu().numpy().squeeze(0))
    sim.set_property_values(
        [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ],
        norm_act,
    )
    return next_rnn


def run_case(case: ManeuverCase, actor: BaselineActor, output_dir: Path, model_path: Path):
    sim = build_sim(initial_u_mps=float(case.params.get("initial_u_mps", DEFAULT_INITIAL_U_MPS)))
    env_view = SingleAircraftEnvView(sim)
    recorder = AcmiRecorder()
    rnn_state = np.zeros((1, 1, 128), dtype=np.float32)
    step_count = int(round(case.seconds / DT))
    initial = build_initial_state(sim)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acmi_path = output_dir / f"{case.name}_{timestamp}.acmi"
    meta_path = output_dir / f"{case.name}_{timestamp}.json"
    recorder.write_header(str(acmi_path))

    samples = []
    for step in range(step_count):
        env_view.current_step = step
        elapsed_s = step * DT
        phase, target_heading_deg, target_altitude_m, target_speed_mps = compute_targets(case, elapsed_s, initial)
        current_state = get_current_state(sim)
        cmd = build_command_indices(current_state, target_heading_deg, target_altitude_m, target_speed_mps)
        rnn_state = step_actor(actor, sim, cmd, rnn_state)

        for _ in range(int(round(SIM_FREQ * DT))):
            sim.run()

        recorder.write_frame(str(acmi_path), env_view, (step + 1) * DT)
        updated_state = get_current_state(sim)
        samples.append(
            {
                "step": step,
                "time_s": round((step + 1) * DT, 2),
                "phase": phase,
                "command": [int(cmd[0]), int(cmd[1]), int(cmd[2])],
                "target_heading_deg": float(target_heading_deg),
                "target_altitude_m": float(target_altitude_m),
                "target_speed_mps": float(target_speed_mps),
                **updated_state,
            }
        )

    final_state = samples[-1] if samples else {}
    heading_delta = signed_heading_diff_deg(final_state.get("heading_deg", initial.heading_deg), initial.heading_deg)
    altitude_delta = float(final_state.get("altitude_m", initial.altitude_m)) - float(initial.altitude_m)
    speed_delta = float(final_state.get("u_mps", initial.u_mps)) - float(initial.u_mps)
    meta = {
        "case_name": case.name,
        "description": case.description,
        "category": case.category,
        "seconds": case.seconds,
        "model_path": str(model_path),
        "acmi_path": str(acmi_path),
        "initial_state": {
            "heading_deg": initial.heading_deg,
            "altitude_m": initial.altitude_m,
            "u_mps": initial.u_mps,
        },
        "final_state": final_state,
        "final_deltas": {
            "heading_delta_deg": heading_delta,
            "altitude_delta_m": altitude_delta,
            "u_delta_mps": speed_delta,
        },
        "sample_count": len(samples),
        "params": case.params,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    sim.close()
    logging.info("done %s -> %s", case.name, acmi_path)
    return meta


def main():
    args = parse_args()
    if not args.baseline_model.exists():
        raise FileNotFoundError(f"baseline model not found: {args.baseline_model}")
    configure_logging(args.output_dir)
    set_seed(args.seed)
    actor = load_actor(args.baseline_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = select_cases(args.case_filter)
    results = []
    for case in selected_cases:
        logging.info("running %s", case.name)
        results.append(run_case(case, actor, args.output_dir, args.baseline_model))

    summary_path = args.output_dir / f"standard_maneuver_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_path": str(args.baseline_model),
                "output_dir": str(args.output_dir),
                "case_count": len(results),
                "selected_case_count": len(selected_cases),
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[standard-maneuver] complete: {args.output_dir}")


if __name__ == "__main__":
    main()
