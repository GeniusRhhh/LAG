#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

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
from maneuver_library import ManeuverLibrary


DEFAULT_MODEL = REPO_ROOT / "envs" / "JSBSim" / "model" / "baseline_model.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "tacticalProject" / "acmi_output" / "f16_real_maneuver_demos"
DT = 0.2
SIM_FREQ = 60


class _DemoEnvView:
    def __init__(self, sims: dict[str, AircraftSimulator]):
        self._jsbsims = sims
        self._tempsims = {}
        self.time_interval = DT
        self.current_step = 0
        self.agents = sims


class _FakeTask:
    def __init__(self):
        self.short_skate_states = {}
        self.short_skate_start_time = {}
        self.maneuver_states = {}
        self.formation_state = {}
        self.beam_direction_state = {}
        self.beam_maneuver_state = {}
        self.tactical_evasion_direction_state = {}
        self.returning_agents = set()
        self.selected_tactic = "DRAG_SHOOT"
        self.last_missile_launch_time = {}
        self.env = None
        self._skate_completed = {}
        self.state_manager = self

    def _normalize_angle_diff(self, angle_diff: float) -> float:
        return ((float(angle_diff) + 180.0) % 360.0) - 180.0

    def _convert_heading_to_index(self, angle_rad: float) -> int:
        norm_hdg = np.array([
            -np.pi, -2*np.pi/3, -np.pi/2, -5*np.pi/12, -np.pi/3, -np.pi/4, -np.pi/6,
            -np.pi/12, 0.0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, 5*np.pi/12, np.pi/2,
            2*np.pi/3, np.pi
        ])
        return int(np.argmin(np.abs(norm_hdg - float(angle_rad))))

    def _get_enemy_bearing(self, env, agent_id: str):
        own = env.agents.get(agent_id)
        if own is None:
            return None
        enemy_ids = [aid for aid in env.agents.keys() if str(aid).startswith("B")]
        if not enemy_ids:
            return None
        own_pos = np.asarray(own.get_position(), dtype=np.float64)
        best_id = None
        best_dist = float("inf")
        for enemy_id in enemy_ids:
            enemy = env.agents.get(enemy_id)
            if enemy is None:
                continue
            enemy_pos = np.asarray(enemy.get_position(), dtype=np.float64)
            dist = float(np.linalg.norm(enemy_pos[:2] - own_pos[:2]))
            if dist < best_dist:
                best_dist = dist
                best_id = enemy_id
        if best_id is None:
            return None
        enemy = env.agents[best_id]
        enemy_pos = np.asarray(enemy.get_position(), dtype=np.float64)
        dx = float(enemy_pos[0] - own_pos[0])
        dy = float(enemy_pos[1] - own_pos[1])
        return float(np.degrees(np.arctan2(dx, dy)) % 360.0)

    def _get_teammate_id(self, agent_id: str):
        if agent_id == "A0100":
            return "A0200"
        if agent_id == "A0200":
            return "A0100"
        return "A0100"

    def get_agent_phase(self, agent_id: str):
        return type("Phase", (), {"value": "ENGAGE"})()


@dataclass(frozen=True)
class RealDemoCase:
    name: str
    seconds: float
    source_function: str
    invoke: str
    args: tuple
    action_window_s: float | None = None


REAL_CASES = [
    RealDemoCase("01_maintain_heading", 60.0, "ManeuverLibrary.maintain_heading_precise", "maintain_heading_precise", (180.0, 10.0), 60.0),
    RealDemoCase("02_short_skate_left", 70.0, "ManeuverLibrary.execute_short_skate_precise", "execute_short_skate_precise", ("left",), 30.0),
    RealDemoCase("03_beam", 60.0, "ManeuverLibrary.execute_beam_maneuver", "execute_beam_maneuver", (), 20.0),
    RealDemoCase("04_tactical_crank_left", 60.0, "ManeuverLibrary.execute_tactical_crank", "execute_tactical_crank", ("left", True), 12.0),
    RealDemoCase("05_tactical_climb_left", 60.0, "ManeuverLibrary.execute_tactical_climb", "execute_tactical_climb", ("left",), 22.0),
    RealDemoCase("06_notch_back_left", 60.0, "ManeuverLibrary.execute_notch_back", "execute_notch_back", ("left",), 18.0),
    RealDemoCase("07_rear_formation_hold", 60.0, "ManeuverLibrary.maintain_rear_formation", "maintain_rear_formation", (), 60.0),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate real maneuver ACMI demos via ManeuverLibrary.")
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def configure_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"real_maneuver_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_actor(model_path: Path) -> BaselineActor:
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    actor = BaselineActor(input_dim=12, use_mlp_actlayer=False)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def build_sim(agent_id: str = "A0100", lon_deg: float = 120.3604, lat_deg: float = 62.3993, heading_deg: float = 180.0) -> AircraftSimulator:
    return AircraftSimulator(
        uid=agent_id,
        color="Blue",
        model="f16",
        init_state={
            "ic_long_gc_deg": lon_deg,
            "ic_lat_geod_deg": lat_deg,
            "ic_h_sl_ft": 32800.0,
            "ic_psi_true_deg": heading_deg,
            "ic_u_fps": 1251.1,
        },
        origin=(120.0, 60.0, 0.0),
        sim_freq=SIM_FREQ,
        num_missiles=0,
    )


def build_enemy_sim(agent_id: str = "B0100", lon_deg: float = 120.3604, lat_deg: float = 62.4600, heading_deg: float = 0.0) -> AircraftSimulator:
    return AircraftSimulator(
        uid=agent_id,
        color="Red",
        model="f16",
        init_state={
            "ic_long_gc_deg": lon_deg,
            "ic_lat_geod_deg": lat_deg,
            "ic_h_sl_ft": 32800.0,
            "ic_psi_true_deg": heading_deg,
            "ic_u_fps": 1251.1,
        },
        origin=(120.0, 60.0, 0.0),
        sim_freq=SIM_FREQ,
        num_missiles=0,
    )


def call_maneuver(lib: ManeuverLibrary, case: RealDemoCase, env_view: _DemoEnvView, agent_id: str):
    fn: Callable = getattr(lib, case.invoke)
    if case.invoke == "maintain_heading_precise":
        return fn(env_view, agent_id, *case.args)
    if case.invoke == "execute_short_skate_precise":
        current_time = env_view.current_step * DT
        return fn(env_view, agent_id, current_time, *case.args)
    if case.invoke == "execute_tactical_crank":
        return fn(env_view, agent_id, *case.args)
    if case.invoke == "execute_tactical_climb":
        return fn(env_view, agent_id, *case.args)
    if case.invoke == "execute_notch_back":
        return fn(env_view, agent_id, *case.args)
    if case.invoke == "maintain_rear_formation":
        return fn(env_view, agent_id)
    return fn(env_view, agent_id, *case.args)


def hold_command(lib: ManeuverLibrary, env_view: _DemoEnvView, agent_id: str):
    current_heading = float(env_view.agents[agent_id].get_property_value(c.attitude_psi_deg))
    return lib.maintain_heading_precise(env_view, agent_id, current_heading, 10.0)


def step_actor(actor: BaselineActor, sim: AircraftSimulator, cmd: tuple[int, int, int], rnn_state: np.ndarray) -> np.ndarray:
    actor_input = _build_model_input(sim, int(cmd[0]), int(cmd[1]), int(cmd[2]), "cap_legacy_remap")
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


def run_case(case: RealDemoCase, actor: BaselineActor, output_dir: Path):
    if case.invoke == "maintain_rear_formation":
        leader = build_sim("A0100", lon_deg=120.3604, lat_deg=62.4050, heading_deg=180.0)
        wingman = build_sim("A0200", lon_deg=120.3604, lat_deg=62.3920, heading_deg=180.0)
        enemy = build_enemy_sim("B0100", lon_deg=120.3604, lat_deg=62.4700, heading_deg=0.0)
        sims = {"A0100": leader, "A0200": wingman, "B0100": enemy}
        primary_id = "A0200"
    else:
        sim = build_sim("A0100", lon_deg=120.3604, lat_deg=62.3993, heading_deg=180.0)
        wingman = build_sim("A0200", lon_deg=120.3660, lat_deg=62.3920, heading_deg=180.0)
        enemy = build_enemy_sim("B0100", lon_deg=120.3604, lat_deg=62.4700, heading_deg=0.0)
        sims = {"A0100": sim, "A0200": wingman, "B0100": enemy}
        primary_id = sim.uid

    env_view = _DemoEnvView(sims)
    task = _FakeTask()
    task.env = env_view
    lib = ManeuverLibrary(task)
    recorder = AcmiRecorder()
    friendly_ids = [aid for aid in sims.keys() if aid.startswith("A")]
    rnn_states = {aid: np.zeros((1, 1, 128), dtype=np.float32) for aid in friendly_ids}
    step_count = int(round(case.seconds / DT))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acmi_path = output_dir / f"{case.name}_{timestamp}.acmi"
    meta_path = output_dir / f"{case.name}_{timestamp}.json"
    recorder.write_header(str(acmi_path))

    samples = []
    action_started = False
    action_completed = False
    for step in range(step_count):
        env_view.current_step = step
        current_time_s = step * DT

        if case.invoke == "maintain_rear_formation":
            leader_cmd = hold_command(lib, env_view, "A0100")
            wing_cmd = call_maneuver(lib, case, env_view, "A0200")
            rnn_states["A0100"] = step_actor(actor, sims["A0100"], leader_cmd, rnn_states["A0100"])
            rnn_states["A0200"] = step_actor(actor, sims["A0200"], wing_cmd, rnn_states["A0200"])
            cmd_used = wing_cmd
        else:
            wing_hold = hold_command(lib, env_view, "A0200")
            rnn_states["A0200"] = step_actor(actor, sims["A0200"], wing_hold, rnn_states["A0200"])
            state = task.maneuver_states.get(primary_id, {})
            if case.invoke == "execute_tactical_climb" and state.get("phase") == "level_off":
                if current_time_s - float(state.get("phase_start_time", current_time_s)) >= 5.0:
                    action_completed = True
            elif case.invoke in {"execute_tactical_crank", "execute_notch_back"} and action_started and primary_id not in task.maneuver_states:
                action_completed = True
            elif case.invoke == "execute_short_skate_precise" and task._skate_completed.get(f"{primary_id}_skate_done"):
                action_completed = True
            elif case.invoke == "execute_beam_maneuver" and case.action_window_s is not None and current_time_s >= case.action_window_s:
                action_completed = True

            if not action_completed and (case.action_window_s is None or current_time_s < case.action_window_s):
                cmd_used = call_maneuver(lib, case, env_view, primary_id)
                action_started = True
            else:
                cmd_used = hold_command(lib, env_view, primary_id)

            rnn_states[primary_id] = step_actor(actor, sims[primary_id], cmd_used, rnn_states[primary_id])

        for _ in range(int(round(SIM_FREQ * DT))):
            for sim_obj in sims.values():
                sim_obj.run()

        recorder.write_frame(str(acmi_path), env_view, (step + 1) * DT)
        samples.append(
            {
                "step": step,
                "time_s": round((step + 1) * DT, 2),
                "command": [int(cmd_used[0]), int(cmd_used[1]), int(cmd_used[2])],
                "altitude_m": float(sims[primary_id].get_property_value(c.position_h_sl_m)),
                "heading_deg": float(sims[primary_id].get_property_value(c.attitude_psi_deg)),
                "vc_mps": float(sims[primary_id].get_property_value(c.velocities_vc_mps)),
            }
        )

    meta = {
        "case_name": case.name,
        "seconds": case.seconds,
        "source_function": case.source_function,
        "invoke": case.invoke,
        "args": list(case.args),
        "acmi_path": str(acmi_path),
        "sample_count": len(samples),
        "final": samples[-1] if samples else {},
        "note": "This demo uses the real maneuver function path and then feeds the resulting (alt_cmd, hdg_cmd, vel_cmd) into baseline_model.pt.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for sim_obj in sims.values():
        sim_obj.close()
    logging.info("done %s -> %s", case.name, acmi_path)


def main():
    args = parse_args()
    configure_logging(args.output_dir)
    set_seed(args.seed)
    actor = load_actor(args.baseline_model)
    for case in REAL_CASES:
        run_case(case, actor, args.output_dir)


if __name__ == "__main__":
    main()
