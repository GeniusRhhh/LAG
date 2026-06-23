#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate seven standard 2v2 tactical template ACMI demos.

Constraints for this verification script:
- Do not modify existing tactical task/runtime files.
- Reuse existing 2v2 tactical chain and low-level baseline_model.pt.
- Keep enemy behavior fixed to level flight then turn-around RTB.
- Suppress real missile launches so the template motion can be observed fully.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT
TACTICAL_ROOT = REPO_ROOT / "scripts" / "tacticalProject"
CONFIG_SRC = TACTICAL_ROOT / "configs" / "tactical_bvr.yaml"
CONFIG_NAME = "tactical_bvr_template_verify_2v2"
CONFIG_DST = REPO_ROOT / "envs" / "JSBSim" / "configs" / f"{CONFIG_NAME}.yaml"
DEFAULT_OUTPUT_ROOT = TACTICAL_ROOT / "acmi_output" / "tactical_template_demos_2v2"
DEFAULT_STEPS = 2400  # 480s @ 0.2s, fixed 8-minute tactical demonstration.
FRIEND_AIRCRAFT = "f16"
ENEMY_AIRCRAFT = "f16"
BASELINE_MODEL_PATH = REPO_ROOT / "envs" / "JSBSim" / "model" / "baseline_model.pt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TACTICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TACTICAL_ROOT))

os.environ["FRIEND_BASELINE_MODEL"] = "F16"
os.environ["ENEMY_BASELINE_MODEL"] = "F16"

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from tactical_task_impl import TacticalTask
from tactical_types import TacticalPhase
from cap.acmi_writer import AcmiRecorder


TEMPLATES = [
    ("drag_shoot", "DRAG_SHOOT"),
    ("pincer_attack", "PINCER_ATTACK"),
    ("high_low_attack", "HIGH_LOW_ATTACK"),
    ("side_by_side", "SIDE_BY_SIDE"),
    ("front_back", "FRONT_BACK"),
    ("tactical_evasion", "TACTICAL_EVASION"),
    ("tactical_turn", "TACTICAL_TURN"),
]


class VerificationMultipleCombatEnv(MultipleCombatEnv):
    """Local 2v2 env wrapper that only swaps aircraft model names."""

    def __init__(self, config_name: str, my_aircraft: str, enemy_aircraft: str):
        self.my_aircraft = my_aircraft
        self.enemy_aircraft = enemy_aircraft
        super().__init__(config_name)

    def load_simulator(self):
        for uid, conf in self.config.aircraft_configs.items():
            if uid.startswith("A"):
                conf["model"] = self.my_aircraft
            elif uid.startswith("B"):
                conf["model"] = self.enemy_aircraft
        super().load_simulator()


class TemplateVerificationTask(TacticalTask):
    """Runtime-only wrapper that forces friendly templates and a fixed enemy script."""

    def __init__(self, config, forced_tactic: str):
        super().__init__(config, decision_manager=None, force_tactic=None)
        self.forced_tactic = str(forced_tactic).upper()
        self.selected_tactic = self.forced_tactic
        self.enemy_turn_started = False
        self.enemy_turn_start_time = None

    def reset(self, env):
        super().reset(env)
        self.selected_tactic = self.forced_tactic
        self.enemy_turn_started = False
        self.enemy_turn_start_time = None
        self._reset_template_runtime_state()

    def _reset_template_runtime_state(self):
        self.tactic_roles = {}
        try:
            self.short_skate_states.clear()
        except Exception:
            pass
        if hasattr(self, "_skate_completed") and isinstance(self._skate_completed, dict):
            self._skate_completed.clear()
        else:
            self._skate_completed = {}
        self.executor.tactical_turn_states.clear()
        self.executor.evasion_states.clear()
        self.executor.turn_states.clear()
        self.returning_agents.clear()
        self._template_states = {}

    def _handle_missile_launches(self, env, current_time: float):
        # Verification run: suppress all real launches to keep trajectories readable.
        if hasattr(self, "state_manager") and hasattr(self.state_manager, "missile_launched"):
            for aid in list(self.state_manager.missile_launched.keys()):
                self.state_manager.missile_launched[aid] = False
        return None

    def _get_enemy_action(self, env, agent_id: str):
        current_time = float(env.current_step) * float(env.time_interval)
        aircraft = env.agents[agent_id]
        current_heading = float(aircraft.get_property_value(c.attitude_psi_deg))
        current_altitude = float(aircraft.get_property_value(c.position_h_sl_m))

        turn_trigger = self._enemy_should_rtb(env, agent_id, current_time)
        if turn_trigger and not self.enemy_turn_started:
            self.enemy_turn_started = True
            self.enemy_turn_start_time = current_time

        target_heading = 0.0 if self.enemy_turn_started else 180.0
        altitude_cmd = 7
        if current_altitude < 5900.0:
            altitude_cmd = 8
        elif current_altitude > 6300.0:
            altitude_cmd = 6
        heading_cmd = self._get_heading_cmd(env, agent_id, target_heading)

        # Let enemy hold a very plain profile, then turn home and extend.
        velocity_cmd = 4 if self.enemy_turn_started else 3

        # If already roughly aligned after RTB, keep it simple and stable.
        if self.enemy_turn_started and abs(self._normalize_angle_diff(target_heading - current_heading)) <= 6.0:
            return 7, 8, 4
        return altitude_cmd, heading_cmd, velocity_cmd

    def _enemy_should_rtb(self, env, agent_id: str, current_time: float) -> bool:
        if current_time >= 180.0:
            return True
        try:
            my_target = self._get_nearest_alive_enemy_anywhere(env, agent_id)[1]
        except Exception:
            my_target = None
        if my_target is None or not getattr(my_target, "is_alive", False):
            return False
        try:
            distance = float(self._calculate_distance_between(env.agents[agent_id], my_target))
        except Exception:
            return False
        return distance <= 55000.0

    @staticmethod
    def _clamp_heading(heading: float) -> float:
        return heading % 360.0

    def _bearing_to_nearest_enemy(self, env, agent_id: str, default_heading: float) -> float:
        try:
            _, enemy = self._get_nearest_alive_enemy_anywhere(env, agent_id)
            if enemy is None or not getattr(enemy, "is_alive", False):
                return default_heading
            own_pos = np.asarray(env.agents[agent_id].get_position()[:2], dtype=np.float64)
            enemy_pos = np.asarray(enemy.get_position()[:2], dtype=np.float64)
            delta = enemy_pos - own_pos
            if np.linalg.norm(delta) < 1e-6:
                return default_heading
            return self._clamp_heading(float(np.degrees(np.arctan2(delta[1], delta[0]))))
        except Exception:
            return default_heading

    def _command_heading(self, env, agent_id: str, target_heading: float, altitude_cmd: int = 7, velocity_cmd: int = 4):
        return altitude_cmd, self._get_heading_cmd(env, agent_id, self._clamp_heading(target_heading)), velocity_cmd

    def _altitude_hold_cmd(self, env, agent_id: str, center_m: float = 6100.0, tolerance_m: float = 250.0) -> int:
        current_alt = float(env.agents[agent_id].get_property_value(c.position_h_sl_m))
        if current_alt < center_m - tolerance_m:
            return 8
        if current_alt > center_m + tolerance_m:
            return 6
        return 7

    def _template_active_duration_s(self) -> float:
        if self.forced_tactic == "DRAG_SHOOT":
            return 130.0
        if self.forced_tactic == "PINCER_ATTACK":
            return 140.0
        if self.forced_tactic == "HIGH_LOW_ATTACK":
            return 145.0
        if self.forced_tactic == "SIDE_BY_SIDE":
            return 140.0
        if self.forced_tactic == "FRONT_BACK":
            return 145.0
        if self.forced_tactic == "TACTICAL_EVASION":
            return 175.0
        if self.forced_tactic == "TACTICAL_TURN":
            return 215.0
        return 210.0

    def _friendly_should_rtb(self, current_time: float) -> bool:
        return current_time >= self._template_active_duration_s()

    @staticmethod
    def _south_delta_deg(current_heading: float) -> float:
        return ((180.0 - current_heading + 540.0) % 360.0) - 180.0

    def _south_hold_action(self, env, agent_id: str):
        alt_cmd = self._altitude_hold_cmd(env, agent_id, center_m=6100.0, tolerance_m=240.0)
        current_heading = float(env.agents[agent_id].get_property_value(c.attitude_psi_deg)) % 360.0
        abs_delta = abs(self._south_delta_deg(current_heading))
        velocity_cmd = 3 if abs_delta > 10.0 else 4
        return self._command_heading(env, agent_id, 180.0, altitude_cmd=alt_cmd, velocity_cmd=velocity_cmd)

    def _return_short_skate_direction(self, agent_id: str) -> str:
        is_lead = agent_id == "A0100"
        tactic = self.forced_tactic
        if tactic == "DRAG_SHOOT":
            return "west"
        if tactic == "TACTICAL_EVASION":
            return "west" if is_lead else "east"
        if tactic == "FRONT_BACK":
            return "west" if is_lead else "east"
        return "west" if is_lead else "east"

    def _resolve_short_skate_direction(self, env, agent_id: str, desired_side: str) -> str:
        current_heading = float(env.agents[agent_id].get_property_value(c.attitude_psi_deg)) % 360.0
        left_target = (current_heading - 40.0) % 360.0
        right_target = (current_heading + 40.0) % 360.0

        def side_of_south(heading: float) -> str:
            delta = self._south_delta_deg(heading)
            if delta < 0.0:
                return "west"
            if delta > 0.0:
                return "east"
            return "center"

        left_side = side_of_south(left_target)
        right_side = side_of_south(right_target)

        if desired_side == "west":
            if left_side == "west":
                return "left"
            if right_side == "west":
                return "right"
        if desired_side == "east":
            if right_side == "east":
                return "right"
            if left_side == "east":
                return "left"
        return "left" if desired_side == "west" else "right"

    def _front_back_rtb_profile(self, agent_id: str):
        if self.forced_tactic != "FRONT_BACK":
            return None
        if agent_id == "A0100":
            return {"side": "west"}
        if agent_id == "A0200":
            return {"side": "east"}
        return None

    def _front_back_side_rtb_action(self, env, agent_id: str, current_time: float, side: str):
        state = self._template_states.setdefault(agent_id, {})
        rtb = state.setdefault("rtb", {})
        if not rtb:
            state["rtb"] = {
                "start_time": current_time,
                "mode": "front_back_side_capture",
                "phase": "stage",
                "side": side,
            }
            rtb = state["rtb"]

        alt_cmd = self._altitude_hold_cmd(env, agent_id, center_m=6100.0, tolerance_m=240.0)
        current_heading = float(env.agents[agent_id].get_property_value(c.attitude_psi_deg)) % 360.0
        desired_side = str(rtb.get("side", side))
        elapsed = current_time - float(rtb.get("start_time", current_time))
        stage_heading = 270.0 if desired_side == "west" else 90.0
        stage_delta = self._normalize_angle_diff(stage_heading - current_heading)

        # The front-back bug happens because the aircraft enters RTB already offset.
        # Force a side-specific staging turn first, then capture south from that side.
        if str(rtb.get("phase")) == "stage":
            if abs(stage_delta) <= 12.0 or elapsed >= 16.0:
                rtb["phase"] = "capture_south"
            else:
                return self._command_heading(env, agent_id, stage_heading, altitude_cmd=alt_cmd, velocity_cmd=4)

        south_delta = self._south_delta_deg(current_heading)
        if abs(south_delta) <= 8.0:
            rtb["phase"] = "hold_south"
            return self._south_hold_action(env, agent_id)

        return self._command_heading(env, agent_id, 180.0, altitude_cmd=alt_cmd, velocity_cmd=4)

    def _friendly_rtb_action(self, env, agent_id: str, current_time: float):
        state = self._template_states.setdefault(agent_id, {})
        rtb = state.setdefault("rtb", {})
        if not rtb:
            state.pop("rtb", None)
            self.short_skate_states.pop(agent_id, None)
            self.short_skate_start_time.pop(agent_id, None)
            if hasattr(self, "_skate_completed"):
                self._skate_completed[f"{agent_id}_skate_done"] = False
            front_back_profile = self._front_back_rtb_profile(agent_id)
            if front_back_profile is not None:
                state["rtb"] = {
                    "start_time": current_time,
                    "mode": "front_back_side_capture",
                    "phase": "stage",
                    "side": str(front_back_profile.get("side", "west")),
                }
            else:
                state["rtb"] = {
                    "start_time": current_time,
                    "mode": "short_skate_rtb",
                    "direction": self._resolve_short_skate_direction(
                        env,
                        agent_id,
                        self._return_short_skate_direction(agent_id),
                    ),
                }
            rtb = state["rtb"]

        if getattr(self, "_skate_completed", {}).get(f"{agent_id}_skate_done"):
            return self._south_hold_action(env, agent_id)

        if str(rtb.get("mode")) == "front_back_side_capture":
            return self._front_back_side_rtb_action(
                env,
                agent_id,
                current_time,
                side=str(rtb.get("side", "west")),
            )

        return self._execute_short_skate_precise(
            env,
            agent_id,
            current_time,
            skate_direction=str(rtb.get("direction", "left")),
        )

    def _tactical_evasion_phase_cmd(self, env, agent_id: str, elapsed: float):
        is_lead = agent_id == "A0100"
        alt_cmd = self._altitude_hold_cmd(env, agent_id, center_m=6100.0, tolerance_m=220.0)
        hot_bearing = self._bearing_to_nearest_enemy(env, agent_id, default_heading=0.0)
        offset_sign = -1.0 if is_lead else 1.0
        beam_sign = 1.0 if is_lead else -1.0

        if elapsed < 30.0:
            target_heading = hot_bearing + offset_sign * 8.0
            return self._command_heading(env, agent_id, target_heading, altitude_cmd=alt_cmd, velocity_cmd=4)
        if elapsed < 60.0:
            target_heading = hot_bearing + beam_sign * 90.0
            return self._command_heading(env, agent_id, target_heading, altitude_cmd=alt_cmd, velocity_cmd=5)
        if elapsed < 78.0:
            target_heading = hot_bearing + beam_sign * 105.0
            return self._command_heading(env, agent_id, target_heading, altitude_cmd=alt_cmd, velocity_cmd=4)
        if elapsed < 108.0:
            target_heading = hot_bearing + offset_sign * 22.0
            return self._command_heading(env, agent_id, target_heading, altitude_cmd=alt_cmd, velocity_cmd=5)
        if elapsed < 125.0:
            target_heading = hot_bearing
            return self._command_heading(env, agent_id, target_heading, altitude_cmd=alt_cmd, velocity_cmd=4)
        return self._friendly_rtb_action(env, agent_id, current_time=float(env.current_step) * float(env.time_interval))

    def _tactical_turn_phase_cmd(self, env, agent_id: str, elapsed: float):
        is_lead = agent_id == "A0100"
        alt_cmd = self._altitude_hold_cmd(env, agent_id, center_m=6100.0, tolerance_m=240.0)
        hot_bearing = self._bearing_to_nearest_enemy(env, agent_id, default_heading=0.0)
        side_sign = -1.0 if is_lead else 1.0

        if elapsed < 28.0:
            target_heading = hot_bearing + side_sign * 8.0
            return self._command_heading(env, agent_id, target_heading, altitude_cmd=alt_cmd, velocity_cmd=4)
        if elapsed < 62.0:
            target_heading = hot_bearing + 180.0 + side_sign * 6.0
            return self._command_heading(env, agent_id, target_heading, altitude_cmd=alt_cmd, velocity_cmd=5)
        if elapsed < 98.0:
            target_heading = hot_bearing + 180.0
            return self._command_heading(env, agent_id, target_heading, altitude_cmd=alt_cmd, velocity_cmd=4)
        if elapsed < 135.0:
            target_heading = hot_bearing + side_sign * 4.0
            return self._command_heading(env, agent_id, target_heading, altitude_cmd=alt_cmd, velocity_cmd=2)
        if elapsed < 175.0:
            target_heading = hot_bearing
            return self._command_heading(env, agent_id, target_heading, altitude_cmd=alt_cmd, velocity_cmd=4)
        return self._friendly_rtb_action(env, agent_id, current_time=float(env.current_step) * float(env.time_interval))

    def get_action(self, env, agent_id):
        if agent_id.startswith("B"):
            return self._get_enemy_action(env, agent_id)

        current_time = float(env.current_step) * float(env.time_interval)
        self.selected_tactic = self.forced_tactic
        self._apply_template_roles(env, agent_id)

        if self._friendly_should_rtb(current_time):
            return self._friendly_rtb_action(env, agent_id, current_time)

        if self.forced_tactic == "DRAG_SHOOT":
            return self.executor.execute_drag_shoot(env, agent_id)
        if self.forced_tactic == "PINCER_ATTACK":
            return self.executor.execute_pincer_attack(env, agent_id)
        if self.forced_tactic == "HIGH_LOW_ATTACK":
            return self.executor.execute_high_low_attack(env, agent_id)
        if self.forced_tactic == "SIDE_BY_SIDE":
            return self.executor.execute_side_by_side(env, agent_id)
        if self.forced_tactic == "FRONT_BACK":
            return self.executor.execute_front_back(env, agent_id)
        if self.forced_tactic == "TACTICAL_EVASION":
            return self._execute_forced_tactical_evasion(env, agent_id, current_time)
        if self.forced_tactic == "TACTICAL_TURN":
            return self._execute_forced_tactical_turn(env, agent_id, current_time)
        return 7, 8, 3

    def _apply_template_roles(self, env, agent_id: str):
        friendlies = [aid for aid in ("A0100", "A0200") if aid in env.agents]
        enemies = [eid for eid in ("B0100", "B0200") if eid in env.agents]
        lead_id = "A0100" if "A0100" in friendlies else friendlies[0]
        wingman_id = "A0200" if "A0200" in friendlies else friendlies[-1]
        lead_target = enemies[0] if enemies else None
        wingman_target = enemies[1] if len(enemies) > 1 else lead_target

        for aid in friendlies:
            self._set_agent_tactic(aid, self.forced_tactic)

        if self.forced_tactic == "PINCER_ATTACK":
            self.tactic_roles = {
                "lead": lead_id,
                "wingman": wingman_id,
                "lead_target": lead_target,
                "wingman_target": wingman_target,
                "tactic": "PINCER_ATTACK",
            }
        elif self.forced_tactic == "HIGH_LOW_ATTACK":
            self.tactic_roles = {
                "high": lead_id,
                "low": wingman_id,
                "high_target": lead_target,
                "low_target": wingman_target,
                "tactic": "HIGH_LOW_ATTACK",
            }
        elif self.forced_tactic == "FRONT_BACK":
            self.tactic_roles = {
                "leader": lead_id,
                "wingman": wingman_id,
                "leader_target": lead_target,
                "wingman_target": wingman_target,
                "tactic": "FRONT_BACK",
            }
        elif self.forced_tactic == "SIDE_BY_SIDE":
            self.tactic_roles = {
                "left": lead_id,
                "right": wingman_id,
                "left_target": lead_target,
                "right_target": wingman_target,
                "tactic": "SIDE_BY_SIDE",
            }
        else:
            self.tactic_roles = {
                "lead": lead_id,
                "wingman": wingman_id,
                "lead_target": lead_target,
                "wingman_target": wingman_target,
                "tactic": self.forced_tactic,
            }

        # Keep phase progression available for executor branches.
        for aid in friendlies:
            self._update_phase(env, aid, float(env.current_step) * float(env.time_interval))

    def _execute_forced_tactical_evasion(self, env, agent_id: str, current_time: float):
        state = self._template_states.setdefault(agent_id, {})
        if not state:
            state["start_time"] = current_time

        elapsed = current_time - float(state["start_time"])
        return self._tactical_evasion_phase_cmd(env, agent_id, elapsed)

    def _execute_forced_tactical_turn(self, env, agent_id: str, current_time: float):
        state = self._template_states.setdefault(agent_id, {})
        if not state:
            state["start_time"] = current_time

        elapsed = current_time - float(state["start_time"])
        return self._tactical_turn_phase_cmd(env, agent_id, elapsed)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2v2 tactical template ACMI demos.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Simulation steps at 0.2s per step.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def ensure_config_copy():
    if not CONFIG_SRC.exists():
        raise FileNotFoundError(f"Config source not found: {CONFIG_SRC}")
    CONFIG_DST.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(CONFIG_SRC, CONFIG_DST)
    except PermissionError:
        if not CONFIG_DST.exists():
            raise


def setup_logging(output_root: Path, timestamp: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f"template_demo_2v2_{timestamp}.log"
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    return log_path


def build_env() -> VerificationMultipleCombatEnv:
    env = VerificationMultipleCombatEnv(
        CONFIG_NAME,
        my_aircraft=FRIEND_AIRCRAFT,
        enemy_aircraft=ENEMY_AIRCRAFT,
    )
    return env


def build_task(env, forced_tactic: str) -> TemplateVerificationTask:
    task = TemplateVerificationTask(env.config, forced_tactic=forced_tactic)
    task.task = task
    env.task = task
    return task


def init_acmi(acmi_path: Path) -> AcmiRecorder:
    recorder = AcmiRecorder()
    recorder.reset()
    recorder.write_header(str(acmi_path))
    return recorder


def pack_dummy_actions(env) -> np.ndarray:
    return np.zeros((1, len(env.agents), 4), dtype=np.float32)


def drive_one_step(env, task: TemplateVerificationTask):
    env.current_step += 1
    dummy_action = np.zeros(4, dtype=np.int32)

    for agent_id in list(env._jsbsims.keys()):
        sim = env._jsbsims.get(agent_id)
        if sim is None:
            continue
        norm_action = task.normalize_action(env, agent_id, dummy_action)
        sim.set_property_values(task.action_var, norm_action)

    for _ in range(int(env.agent_interaction_steps)):
        for sim in list(env._jsbsims.values()):
            sim.run()
        for sim in list(env._tempsims.values()):
            sim.run()


def get_alive_counts(env) -> Dict[str, int]:
    friendly_alive = sum(1 for aid in ("A0100", "A0200") if aid in env.agents and env.agents[aid].is_alive)
    enemy_alive = sum(1 for eid in ("B0100", "B0200") if eid in env.agents and env.agents[eid].is_alive)
    return {"friendly_alive": int(friendly_alive), "enemy_alive": int(enemy_alive)}


def get_final_snapshot(env) -> Dict[str, Dict[str, float]]:
    snapshot = {}
    for aid in ("A0100", "A0200", "B0100", "B0200"):
        if aid not in env.agents:
            continue
        sim = env.agents[aid]
        if not sim.is_alive:
            snapshot[aid] = {"alive": False}
            continue
        pos = sim.get_position()
        vel = sim.get_velocity()
        snapshot[aid] = {
            "alive": True,
            "north_m": float(pos[0]),
            "east_m": float(pos[1]),
            "altitude_m": float(pos[2]),
            "speed_mps": float(np.linalg.norm(vel)),
            "heading_deg": float(sim.get_property_value(c.attitude_psi_deg)) % 360.0,
        }
    return snapshot


def run_one_template(template_label: str, forced_tactic: str, steps: int, output_root: Path) -> Dict[str, object]:
    env = build_env()
    task = build_task(env, forced_tactic)
    env.max_steps = steps
    env.reset()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acmi_path = output_root / f"{template_label}_{timestamp}.acmi"
    recorder = init_acmi(acmi_path)

    step = 0
    dt = float(env.time_interval)
    while step < steps:
        step += 1
        drive_one_step(env, task)
        recorder.write_frame(str(acmi_path), env, env.current_step * dt)

    result = {
        "template": template_label,
        "forced_tactic": forced_tactic,
        "steps": int(steps),
        "duration_s": round(float(steps) * dt, 1),
        "acmi_path": str(acmi_path),
        "model_path": str(BASELINE_MODEL_PATH),
        "aircraft": {
            "friendly": FRIEND_AIRCRAFT,
            "enemy": ENEMY_AIRCRAFT,
        },
        "alive_counts": get_alive_counts(env),
        "final_snapshot": get_final_snapshot(env),
    }
    json_path = acmi_path.with_suffix(".json")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(
        "[TEMPLATE_DONE] %s -> %s duration=%.1fs friendly_alive=%d enemy_alive=%d",
        template_label,
        acmi_path.name,
        result["duration_s"],
        result["alive_counts"]["friendly_alive"],
        result["alive_counts"]["enemy_alive"],
    )
    try:
        env.close()
    except Exception:
        pass
    return result


def main():
    args = parse_args()
    ensure_config_copy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root / f"template_demo_2v2_{timestamp}"
    log_path = setup_logging(output_root, timestamp)

    logging.info("[START] Generate 2v2 tactical template demos")
    logging.info("Output root: %s", output_root)
    logging.info("Model path: %s", BASELINE_MODEL_PATH)
    logging.info("Enemy profile: level flight then turn-around RTB, no missile launch")

    summary = []
    for template_label, forced_tactic in TEMPLATES:
        only = os.environ.get("TEMPLATE_DEMO_ONLY", "").strip().lower()
        if only and template_label.lower() != only and forced_tactic.lower() != only:
            continue
        logging.info("[RUN] %s / %s", template_label, forced_tactic)
        summary.append(run_one_template(template_label, forced_tactic, args.steps, output_root))

    summary_path = output_root / f"template_demo_2v2_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("[DONE] Summary written: %s", summary_path)
    logging.info("[DONE] Log written: %s", log_path)


if __name__ == "__main__":
    main()
