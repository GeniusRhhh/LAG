from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.core.simulatior import AircraftSimulator
from envs.JSBSim.model.baseline_actor import BaselineActor

try:
    from .acmi_writer import AcmiRecorder
except ImportError:
    from acmi_writer import AcmiRecorder


LOG_CMD_RE = re.compile(
    r"\[(?P<agent>[AB]\d{4})\]\[T\+(?P<t>[\d\.]+)s\]\[_get_enemy_action\]\[output\]\[decision\] "
    r"cmd=\((?P<alt>\d+),\s*(?P<hdg>\d+),\s*(?P<vel>\d+)\)"
)


@dataclass(frozen=True)
class CommandPoint:
    time_s: float
    alt_idx: int
    hdg_idx: int
    vel_idx: int


@dataclass
class VerificationSummary:
    steps: int
    semantics: str
    command_source: str
    command_points: int
    final_alt_m: float
    final_vc_mps: float
    final_pitch_deg: float
    final_v_up_mps: float
    min_alt_m: float
    min_vc_mps: float
    max_alt_m: float
    max_vc_mps: float
    time_of_min_alt_s: float
    time_of_min_vc_s: float
    first_below_220_step: int
    first_below_180_step: int
    first_below_150_step: int
    first_below_120_step: int
    first_below_100_step: int
    recreate_count: int
    recreate_times_s: list[float]
    top_commands: list[list[object]]


class _SingleAircraftEnvView:
    def __init__(self, sim: AircraftSimulator):
        self._jsbsims = {sim.uid: sim}
        self._tempsims = {}


def _setup_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "single_lowlevel_verify.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    return log_path


def _cap_norm_alt() -> np.ndarray:
    return np.array(
        [-1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500],
        dtype=np.float32,
    ) / 1000.0


def _cap_norm_hdg() -> np.ndarray:
    return np.array(
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


def _cap_norm_vel() -> np.ndarray:
    return np.array([-150, -100, -50, 0, 50, 100, 150], dtype=np.float32) / 100.0


def _remap_for_legacy_cmd(delta_alt_km: float, delta_hdg_rad: float, delta_vel_norm: float) -> tuple[float, float, float]:
    if delta_alt_km > 0.05:
        alt_legacy = 0.1
    elif delta_alt_km < -0.05:
        alt_legacy = -0.1
    else:
        alt_legacy = 0.0

    hdg_deg = float(np.degrees(delta_hdg_rad))
    if hdg_deg <= -22.5:
        hdg_legacy = -np.pi / 6
    elif hdg_deg <= -7.5:
        hdg_legacy = -np.pi / 12
    elif hdg_deg < 7.5:
        hdg_legacy = 0.0
    elif hdg_deg < 22.5:
        hdg_legacy = np.pi / 12
    else:
        hdg_legacy = np.pi / 6

    if delta_vel_norm > 0.025:
        vel_legacy = 0.05
    elif delta_vel_norm < -0.025:
        vel_legacy = -0.05
    else:
        vel_legacy = 0.0

    return float(alt_legacy), float(hdg_legacy), float(vel_legacy)


def _build_raw_obs(sim: AircraftSimulator) -> np.ndarray:
    h_sl_m = float(sim.get_property_value(c.position_h_sl_m))
    roll_rad = float(sim.get_property_value(c.attitude_roll_rad))
    pitch_rad = float(sim.get_property_value(c.attitude_pitch_rad))
    u_mps = float(sim.get_property_value(c.velocities_u_mps))
    v_mps = float(sim.get_property_value(c.velocities_v_mps))
    w_mps = float(sim.get_property_value(c.velocities_w_mps))
    vc_mps = float(sim.get_property_value(c.velocities_vc_mps))

    raw_obs = np.zeros(9, dtype=np.float32)
    raw_obs[0] = h_sl_m / 5000.0
    raw_obs[1] = np.sin(roll_rad)
    raw_obs[2] = np.cos(roll_rad)
    raw_obs[3] = np.sin(pitch_rad)
    raw_obs[4] = np.cos(pitch_rad)
    raw_obs[5] = u_mps / 340.0
    raw_obs[6] = v_mps / 340.0
    raw_obs[7] = w_mps / 340.0
    raw_obs[8] = vc_mps / 340.0
    return raw_obs


def _build_model_input(
    sim: AircraftSimulator,
    alt_idx: int,
    hdg_idx: int,
    vel_idx: int,
    semantics: str,
) -> np.ndarray:
    norm_alt = _cap_norm_alt()
    norm_hdg = _cap_norm_hdg()
    norm_vel = _cap_norm_vel()

    inp = np.zeros(12, dtype=np.float32)
    delta_alt = float(norm_alt[min(int(alt_idx), len(norm_alt) - 1)])
    delta_hdg = float(norm_hdg[min(int(hdg_idx), len(norm_hdg) - 1)])
    delta_vel = float(norm_vel[min(int(vel_idx), len(norm_vel) - 1)])

    if semantics == "cap_legacy_remap":
        delta_alt, delta_hdg, delta_vel = _remap_for_legacy_cmd(delta_alt, delta_hdg, delta_vel)
    elif semantics != "direct_15x17x7":
        raise ValueError(f"Unsupported semantics: {semantics}")

    inp[0] = delta_alt
    inp[1] = delta_hdg
    inp[2] = delta_vel
    inp[3:12] = _build_raw_obs(sim)
    inp = np.nan_to_num(inp, nan=0.0, posinf=1.0, neginf=-1.0)
    return inp


def _raw_bins_to_norm_act(raw_bins: np.ndarray) -> np.ndarray:
    out = np.asarray(raw_bins, dtype=np.float32).reshape(-1)
    return np.array(
        [
            out[0] / 20.0 - 1.0,
            out[1] / 20.0 - 1.0,
            out[2] / 20.0 - 1.0,
            out[3] / 58.0 + 0.4,
        ],
        dtype=np.float32,
    )


def _load_command_schedule(log_path: Path, agent_id: str) -> List[CommandPoint]:
    result: list[CommandPoint] = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            m = LOG_CMD_RE.search(line)
            if not m:
                continue
            if m.group("agent") != agent_id:
                continue
            result.append(
                CommandPoint(
                    time_s=float(m.group("t")),
                    alt_idx=int(m.group("alt")),
                    hdg_idx=int(m.group("hdg")),
                    vel_idx=int(m.group("vel")),
                )
            )
    if not result:
        raise ValueError(f"No command schedule found for {agent_id} in {log_path}")
    result.sort(key=lambda x: x.time_s)
    return result


def _current_command(schedule: Sequence[CommandPoint], time_s: float) -> CommandPoint:
    current = schedule[0]
    for point in schedule:
        if point.time_s > time_s:
            break
        current = point
    return current


def _default_init_state() -> dict:
    return {
        "ic_long_gc_deg": 120.3604,
        "ic_lat_geod_deg": 62.3993,
        "ic_h_sl_ft": 32800.0,
        "ic_psi_true_deg": 180.0,
        "ic_u_fps": 1251.1,
    }


def _resolve_baseline_model_path() -> Path:
    candidates = [
        REPO_ROOT / "envs" / "JSBSim" / "model" / "baseline_model.pt",
        REPO_ROOT / "lqyLAG" / "envs" / "JSBSim" / "model" / "baseline_model.pt",
        REPO_ROOT / "zykLAG" / "envs" / "JSBSim" / "model" / "baseline_model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("baseline_model.pt not found in known model directories")


def _write_summary(output_dir: Path, summary: VerificationSummary) -> Path:
    path = output_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary.__dict__, handle, ensure_ascii=False, indent=2)
    return path


def _build_recreated_state(sim: AircraftSimulator, rescue_speed_mps: float) -> dict:
    lon_deg = float(sim.get_property_value(c.position_long_gc_deg))
    lat_deg = float(sim.get_property_value(c.position_lat_geod_deg))
    alt_ft = float(sim.get_property_value(c.position_h_sl_ft))
    hdg_deg = float(sim.get_property_value(c.attitude_psi_deg))
    return {
        "ic_long_gc_deg": lon_deg,
        "ic_lat_geod_deg": lat_deg,
        "ic_h_sl_ft": alt_ft,
        "ic_psi_true_deg": hdg_deg,
        "ic_theta_deg": 0.0,
        "ic_phi_deg": 0.0,
        "ic_u_fps": float(rescue_speed_mps / 0.3048),
        "ic_v_fps": 0.0,
        "ic_w_fps": 0.0,
        "ic_p_rad_sec": 0.0,
        "ic_q_rad_sec": 0.0,
        "ic_r_rad_sec": 0.0,
        "ic_roc_fpm": 0.0,
    }


def _should_recreate(window: deque[dict], max_alt_for_no_recreate: float = 2500.0) -> bool:
    if len(window) < window.maxlen:
        return False
    first = window[0]
    last = window[-1]
    if last["alt"] < max_alt_for_no_recreate:
        return False
    if last["vc"] > 170.0:
        return False
    if last["vup"] > -4.0:
        return False
    energy_drop = last["energy"] - first["energy"]
    avg_vup = sum(x["vup"] for x in window) / len(window)
    avg_vc = sum(x["vc"] for x in window) / len(window)
    alt_drop = last["alt"] - first["alt"]
    vc_drop = last["vc"] - first["vc"]

    severe_trend = energy_drop < -2500.0 and vc_drop < -8.0 and alt_drop < -120.0 and avg_vup < -3.5
    severe_low_speed_sink = (
        last["vc"] < 145.0
        and avg_vc < 155.0
        and avg_vup < -8.0
        and alt_drop < -80.0
    )
    return severe_trend or severe_low_speed_sink


def run_single_lowlevel_verification(
    steps: int,
    output_dir: Path,
    semantics: str,
    log_path: Path | None,
    agent_id: str,
    fixed_cmd: tuple[int, int, int],
    enable_recreate: bool,
    max_recreates: int,
) -> tuple[Path, Path]:
    log_path_out = _setup_logging(output_dir)
    logging.info("Single enemy low-level verification started")
    logging.info("steps=%d semantics=%s agent=%s", steps, semantics, agent_id)
    logging.info("command_source=%s", str(log_path) if log_path else f"fixed:{fixed_cmd}")

    actor = BaselineActor(input_dim=12, use_mlp_actlayer=False)
    model_path = _resolve_baseline_model_path()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    actor.load_state_dict(state_dict)
    actor.eval()

    sim = AircraftSimulator(
        uid=agent_id,
        color="Blue",
        model="f16",
        init_state=_default_init_state(),
        origin=(120.0, 60.0, 0.0),
        sim_freq=60,
        num_missiles=4,
    )
    env_view = _SingleAircraftEnvView(sim)
    recorder = AcmiRecorder()
    acmi_path = output_dir / f"{datetime.now().strftime('%m%d_%H%M%S')}_single_enemy_lowlevel.txt.acmi"
    recorder.write_header(str(acmi_path))

    schedule = _load_command_schedule(log_path, agent_id) if log_path else []
    rnn_state = np.zeros((1, 1, 128), dtype=np.float32)

    cmd_counter: Counter[tuple[int, int, int]] = Counter()
    min_alt_m = float("inf")
    min_vc_mps = float("inf")
    max_alt_m = float("-inf")
    max_vc_mps = float("-inf")
    time_of_min_alt_s = 0.0
    time_of_min_vc_s = 0.0
    first_below = {220: -1, 180: -1, 150: -1, 120: -1, 100: -1}
    recreate_times_s: list[float] = []
    trend_window: deque[dict] = deque(maxlen=50)

    for step in range(steps):
        time_s = step * 0.2
        if schedule:
            cmd = _current_command(schedule, time_s)
            alt_idx, hdg_idx, vel_idx = cmd.alt_idx, cmd.hdg_idx, cmd.vel_idx
        else:
            alt_idx, hdg_idx, vel_idx = fixed_cmd

        cmd_counter[(alt_idx, hdg_idx, vel_idx)] += 1
        input_obs = _build_model_input(sim, alt_idx, hdg_idx, vel_idx, semantics)

        obs_t = torch.FloatTensor(input_obs).unsqueeze(0)
        rnn_t = torch.FloatTensor(rnn_state)
        with torch.no_grad():
            action_raw, rnn_out = actor(obs_t, rnn_t)
        rnn_state = rnn_out.detach().cpu().numpy()
        raw_bins = action_raw.detach().cpu().numpy().squeeze(0)
        norm_act = _raw_bins_to_norm_act(raw_bins)
        sim.set_property_values(
            [
                c.fcs_aileron_cmd_norm,
                c.fcs_elevator_cmd_norm,
                c.fcs_rudder_cmd_norm,
                c.fcs_throttle_cmd_norm,
            ],
            norm_act,
        )

        for _ in range(12):
            sim.run()
        recorder.write_frame(str(acmi_path), env_view, (step + 1) * 0.2)

        alt_m = float(sim.get_property_value(c.position_h_sl_m))
        vc_mps = float(sim.get_property_value(c.velocities_vc_mps))
        pitch_deg = float(np.degrees(sim.get_property_value(c.attitude_theta_rad)))
        v_up_mps = -float(sim.get_property_value(c.velocities_v_down_mps))
        energy = 9.81 * alt_m + 0.5 * vc_mps * vc_mps
        trend_window.append({"alt": alt_m, "vc": vc_mps, "vup": v_up_mps, "energy": energy})

        if alt_m < min_alt_m:
            min_alt_m = alt_m
            time_of_min_alt_s = time_s
        if vc_mps < min_vc_mps:
            min_vc_mps = vc_mps
            time_of_min_vc_s = time_s
        max_alt_m = max(max_alt_m, alt_m)
        max_vc_mps = max(max_vc_mps, vc_mps)

        for threshold in sorted(first_below.keys(), reverse=True):
            if first_below[threshold] < 0 and vc_mps < threshold:
                first_below[threshold] = step

        if enable_recreate and len(recreate_times_s) < max_recreates and _should_recreate(trend_window):
            rescue_speed = max(210.0, min(260.0, vc_mps + 55.0))
            new_state = _build_recreated_state(sim, rescue_speed)
            logging.warning(
                "[%s][T+%06.1fs] RECREATE trigger alt=%.1fm vc=%.1fm/s v_up=%.2fm/s -> rescue_speed=%.1fm/s",
                agent_id,
                time_s,
                alt_m,
                vc_mps,
                v_up_mps,
                rescue_speed,
            )
            sim.reload(new_state=new_state)
            rnn_state = np.zeros((1, 1, 128), dtype=np.float32)
            recreate_times_s.append(time_s)
            trend_window.clear()
            alt_m = float(sim.get_property_value(c.position_h_sl_m))
            vc_mps = float(sim.get_property_value(c.velocities_vc_mps))
            pitch_deg = float(np.degrees(sim.get_property_value(c.attitude_theta_rad)))
            v_up_mps = -float(sim.get_property_value(c.velocities_v_down_mps))
            energy = 9.81 * alt_m + 0.5 * vc_mps * vc_mps
            trend_window.append({"alt": alt_m, "vc": vc_mps, "vup": v_up_mps, "energy": energy})
            logging.warning(
                "[%s][T+%06.1fs] RECREATE applied alt=%.1fm vc=%.1fm/s pitch=%.2fdeg v_up=%.2fm/s",
                agent_id,
                time_s,
                alt_m,
                vc_mps,
                pitch_deg,
                v_up_mps,
            )

        if step % 300 == 0 or step == steps - 1:
            logging.info(
                "[%s][T+%06.1fs] cmd=(%d,%d,%d) alt=%.1fm vc=%.1fm/s pitch=%.2fdeg v_up=%.2fm/s norm_act=%s",
                agent_id,
                time_s,
                alt_idx,
                hdg_idx,
                vel_idx,
                alt_m,
                vc_mps,
                pitch_deg,
                v_up_mps,
                np.array2string(norm_act, precision=4, separator=","),
            )

        if not sim.is_alive:
            logging.warning("%s no longer alive at step=%d time=%.1fs", agent_id, step, time_s)
            break

    final_alt_m = float(sim.get_property_value(c.position_h_sl_m))
    final_vc_mps = float(sim.get_property_value(c.velocities_vc_mps))
    final_pitch_deg = float(np.degrees(sim.get_property_value(c.attitude_theta_rad)))
    final_v_up_mps = -float(sim.get_property_value(c.velocities_v_down_mps))

    summary = VerificationSummary(
        steps=steps,
        semantics=semantics,
        command_source=str(log_path) if log_path else f"fixed:{fixed_cmd}",
        command_points=len(schedule),
        final_alt_m=final_alt_m,
        final_vc_mps=final_vc_mps,
        final_pitch_deg=final_pitch_deg,
        final_v_up_mps=final_v_up_mps,
        min_alt_m=min_alt_m,
        min_vc_mps=min_vc_mps,
        max_alt_m=max_alt_m,
        max_vc_mps=max_vc_mps,
        time_of_min_alt_s=time_of_min_alt_s,
        time_of_min_vc_s=time_of_min_vc_s,
        first_below_220_step=first_below[220],
        first_below_180_step=first_below[180],
        first_below_150_step=first_below[150],
        first_below_120_step=first_below[120],
        first_below_100_step=first_below[100],
        recreate_count=len(recreate_times_s),
        recreate_times_s=recreate_times_s,
        top_commands=[[str(k), v] for k, v in cmd_counter.most_common(12)],
    )
    summary_path = _write_summary(output_dir, summary)
    logging.info("Single enemy low-level verification finished")
    logging.info("summary=%s", summary_path)
    logging.info("acmi=%s", acmi_path)
    sim.close()
    return summary_path, acmi_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-aircraft enemy low-level verification runner")
    parser.add_argument("--steps", type=int, default=4200)
    parser.add_argument(
        "--semantics",
        type=str,
        default="cap_legacy_remap",
        choices=["cap_legacy_remap", "direct_15x17x7"],
        help="How to feed high-level commands into baseline_model.pt",
    )
    parser.add_argument(
        "--replay-log",
        type=str,
        default="",
        help="Replay B0100 command schedule from a CAP log",
    )
    parser.add_argument("--agent-id", type=str, default="B0100")
    parser.add_argument("--fixed-alt", type=int, default=7)
    parser.add_argument("--fixed-hdg", type=int, default=8)
    parser.add_argument("--fixed-vel", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--enable-recreate", action="store_true")
    parser.add_argument("--max-recreates", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap_results\single_enemy_lowlevel_verify") / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    log_path = Path(args.replay_log) if args.replay_log else None
    run_single_lowlevel_verification(
        steps=args.steps,
        output_dir=output_dir,
        semantics=args.semantics,
        log_path=log_path,
        agent_id=args.agent_id,
        fixed_cmd=(args.fixed_alt, args.fixed_hdg, args.fixed_vel),
        enable_recreate=args.enable_recreate,
        max_recreates=args.max_recreates,
    )


if __name__ == "__main__":
    main()
