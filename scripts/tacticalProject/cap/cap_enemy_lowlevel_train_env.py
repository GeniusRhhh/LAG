from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TACTICAL_ROOT = REPO_ROOT / "scripts" / "tacticalProject"


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    while path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def _clear_import_cache() -> None:
    for prefix in ("envs", "envs.JSBSim", "lqyLAG.envs", "lqyLAG.envs.JSBSim"):
        for module_name in list(sys.modules.keys()):
            if module_name == prefix or module_name.startswith(prefix + "."):
                del sys.modules[module_name]


_prepend_sys_path(REPO_ROOT)
_prepend_sys_path(TACTICAL_ROOT)
_clear_import_cache()

try:
    from .cap_task_refactor_helpers import _build_lowlevel_input_for_agent
    from .enemy_safe_teacher import EnemyEnergySnapshot, EnemySafeTeacher, TeacherMode
    from .run_helpers import create_patrol_task, prepare_patrol_config
    from .runtime.bootstrap import bootstrap_runtime
    from .runtime.engine import CAPEnv
except ImportError:
    from cap.cap_task_refactor_helpers import _build_lowlevel_input_for_agent
    from cap.enemy_safe_teacher import EnemyEnergySnapshot, EnemySafeTeacher, TeacherMode
    from cap.run_helpers import create_patrol_task, prepare_patrol_config
    from cap.runtime.bootstrap import bootstrap_runtime
    from cap.runtime.engine import CAPEnv


MY_AIRCRAFT_TYPE = "su27sk"
ENEMY_AIRCRAFT_TYPE = "f16"


@dataclass(frozen=True)
class RewardTerms:
    alive_bonus: float
    speed_term: float
    sink_term: float
    roll_term: float
    altitude_track_term: float
    heading_track_term: float
    safety_mode_term: float
    terminal_term: float

    @property
    def total(self) -> float:
        return (
            self.alive_bonus
            + self.speed_term
            + self.sink_term
            + self.roll_term
            + self.altitude_track_term
            + self.heading_track_term
            + self.safety_mode_term
            + self.terminal_term
        )


class CapEnemyLowlevelTrainEnv(gym.Env):
    """
    Real 4v4 CAP environment wrapper for training a single enemy low-level actor.

    The high-level tactical logic remains the real CAP unified enemy AI. Only the
    selected enemy aircraft's low-level actuator bins are overridden by the
    online policy on each step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        train_agent_id: str = "B0100",
        max_steps: int = 4200,
        allow_patrol_fallback: bool = False,
        enable_safe_teacher_for_other_enemies: bool = True,
    ) -> None:
        super().__init__()
        self.train_agent_id = str(train_agent_id)
        self.max_steps = int(max_steps)
        self.allow_patrol_fallback = bool(allow_patrol_fallback)
        self.enable_safe_teacher_for_other_enemies = bool(enable_safe_teacher_for_other_enemies)

        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(12,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([41, 41, 41, 30])

        self.context = bootstrap_runtime(
            entry_file=__file__,
            my_aircraft_type=MY_AIRCRAFT_TYPE,
            enemy_aircraft_type=ENEMY_AIRCRAFT_TYPE,
        )
        self.config_name = prepare_patrol_config(self.context.current_dir, self.context.project_root)
        os.environ["CAP_ENEMY_SAFE_TEACHER_ENABLED"] = "1" if self.enable_safe_teacher_for_other_enemies else "0"

        self.env = CAPEnv(
            self.config_name,
            my_aircraft_type=MY_AIRCRAFT_TYPE,
            enemy_aircraft_type=ENEMY_AIRCRAFT_TYPE,
        )
        self.env.max_steps = self.max_steps
        self.task = create_patrol_task(self.env, allow_patrol_fallback=self.allow_patrol_fallback)
        self.env.task = self.task

        self.teacher: EnemySafeTeacher = getattr(self.task, "enemy_safe_teacher", EnemySafeTeacher())
        self._pending_actions: Optional[np.ndarray] = None
        self._pending_cmd: Optional[Tuple[int, int, int]] = None
        self._last_snapshot: Optional[EnemyEnergySnapshot] = None
        self._last_teacher_count: int = 0
        self._episode_step: int = 0

    def close(self):
        if getattr(self, "env", None) is not None:
            try:
                self.env.close()
            except Exception:
                pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.env.max_steps = self.max_steps
        self.env.reset()

        self.task._enemy_lowlevel_action_overrides = {}
        self.task._enemy_safe_teacher_skip_agents = {self.train_agent_id}
        self._episode_step = 0
        self._last_teacher_count = self._teacher_event_count()
        self._pending_actions, self._pending_cmd = self._build_joint_actions_and_target_command()
        self._last_snapshot = self._snapshot()

        obs = self._build_observation(self._pending_cmd)
        info = {
            "train_agent_id": self.train_agent_id,
            "command": tuple(int(x) for x in self._pending_cmd),
            "snapshot": asdict(self._last_snapshot) if self._last_snapshot is not None else None,
            "teacher_events": int(self._last_teacher_count),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(4)
        prev_snapshot = self._last_snapshot
        prev_cmd = tuple(int(x) for x in self._pending_cmd)
        teacher_before = self._teacher_event_count()

        self.task._enemy_lowlevel_action_overrides[self.train_agent_id] = action.copy()
        self.env.step(self._pending_actions)
        self._episode_step += 1

        next_snapshot = self._snapshot()
        terminated = next_snapshot is None or not self.env.agents[self.train_agent_id].is_alive or (
            next_snapshot.altitude_m < 200.0
        )
        truncated = self._episode_step >= self.max_steps or int(getattr(self.env, "current_step", 0)) >= self.max_steps

        reward_terms = self._compute_reward_terms(prev_snapshot, next_snapshot, prev_cmd, action, terminated)
        reward = float(reward_terms.total)

        teacher_after = self._teacher_event_count()
        teacher_delta = int(max(0, teacher_after - teacher_before))

        if terminated or truncated:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            self._pending_actions, self._pending_cmd = self._build_joint_actions_and_target_command()
            next_obs = self._build_observation(self._pending_cmd)

        self._last_snapshot = next_snapshot
        info = {
            "command": prev_cmd,
            "snapshot": asdict(next_snapshot) if next_snapshot is not None else None,
            "reward_terms": asdict(reward_terms),
            "teacher_events_delta": teacher_delta,
            "teacher_events_total": teacher_after,
        }
        return next_obs, reward, terminated, truncated, info

    def _snapshot(self) -> Optional[EnemyEnergySnapshot]:
        return self.teacher.snapshot(self.env, self.train_agent_id)

    def _teacher_event_count(self) -> int:
        stats = getattr(self.task, "_enemy_safe_teacher_stats", {})
        agent_stats = stats.get(self.train_agent_id, {})
        return int(agent_stats.get("safe_hold", 0)) + int(agent_stats.get("energy_recover", 0))

    def _build_joint_actions_and_target_command(self) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        agent_ids = list(getattr(self.env, "ego_ids", [])) + list(getattr(self.env, "enm_ids", []))
        if not agent_ids:
            agent_ids = list(self.env.agents.keys())

        act_dim = 4
        actions = np.zeros((1, len(agent_ids), act_dim), dtype=np.float32)
        target_cmd = (7, 8, 3)

        for index, agent_id in enumerate(agent_ids):
            alt_cmd, hdg_cmd, vel_cmd = self.task.get_action(self.env, agent_id)
            actions[0, index, 0] = float(alt_cmd)
            actions[0, index, 1] = float(hdg_cmd)
            actions[0, index, 2] = float(vel_cmd)
            if agent_id == self.train_agent_id:
                target_cmd = (int(alt_cmd), int(hdg_cmd), int(vel_cmd))

        return actions, target_cmd

    def _build_observation(self, command: Tuple[int, int, int]) -> np.ndarray:
        model_key, _ = self.task._select_lowlevel_model_for_agent(self.train_agent_id)
        _, inp = _build_lowlevel_input_for_agent(
            self.task,
            self.env,
            self.train_agent_id,
            int(command[0]),
            int(command[1]),
            int(command[2]),
            model_key,
        )
        return np.asarray(inp, dtype=np.float32)

    def _heading_change_deg(self, prev_deg: float, next_deg: float) -> float:
        return ((float(next_deg) - float(prev_deg) + 180.0) % 360.0) - 180.0

    def _compute_reward_terms(
        self,
        prev_snapshot: Optional[EnemyEnergySnapshot],
        next_snapshot: Optional[EnemyEnergySnapshot],
        command: Tuple[int, int, int],
        action: np.ndarray,
        terminated: bool,
    ) -> RewardTerms:
        if prev_snapshot is None or next_snapshot is None:
            return RewardTerms(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0 if terminated else 0.0)

        alive_bonus = 0.05
        terminal_term = -10.0 if terminated else 0.0

        vc_change = float(next_snapshot.vc_mps - prev_snapshot.vc_mps)
        heading_change = self._heading_change_deg(prev_snapshot.heading_deg, next_snapshot.heading_deg)

        high_altitude = next_snapshot.altitude_m >= 8000.0
        speed_floor = 210.0 if high_altitude else 190.0
        speed_term = 0.04 * np.clip(vc_change / 2.0, -2.0, 2.0)
        speed_term -= 0.05 * max(0.0, speed_floor - float(next_snapshot.vc_mps)) / 10.0

        sink_term = -0.06 * max(0.0, -float(next_snapshot.v_up_mps) - 5.0) / 5.0
        roll_term = -0.04 * max(0.0, abs(float(next_snapshot.roll_deg)) - 70.0) / 10.0

        alt_idx, hdg_idx, vel_idx = (int(x) for x in command)
        alt_delta = float(self.task.norm_alt[alt_idx])
        hdg_delta = float(self.task.norm_hdg[hdg_idx])
        vel_delta = float(self.task.norm_vel[vel_idx])

        if alt_delta > 0.05:
            altitude_track_term = 0.05 * np.clip(float(next_snapshot.v_up_mps) / 10.0, -1.0, 1.0)
        elif alt_delta < -0.05:
            altitude_track_term = 0.05 * np.clip(-float(next_snapshot.v_up_mps) / 10.0, -1.0, 1.0)
        else:
            altitude_track_term = 0.03 * max(0.0, 1.0 - abs(float(next_snapshot.v_up_mps)) / 8.0)

        if hdg_delta > 0.05:
            heading_track_term = 0.03 * np.clip(heading_change / 6.0, -1.0, 1.0)
        elif hdg_delta < -0.05:
            heading_track_term = 0.03 * np.clip(-heading_change / 6.0, -1.0, 1.0)
        else:
            heading_track_term = 0.02 * max(0.0, 1.0 - abs(heading_change) / 8.0)

        if vel_delta > 0.01:
            speed_term += 0.05 * np.clip(vc_change / 2.0, -1.0, 1.0)
        elif vel_delta < -0.01:
            speed_term += 0.02 * np.clip(-vc_change / 2.0, -1.0, 1.0)

        safety_mode = self.teacher.classify_mode(next_snapshot)
        if safety_mode == TeacherMode.ENERGY_RECOVER:
            safety_mode_term = -0.40
        elif safety_mode == TeacherMode.SAFE_HOLD:
            safety_mode_term = -0.10
        else:
            safety_mode_term = 0.02

        return RewardTerms(
            alive_bonus=alive_bonus,
            speed_term=float(speed_term),
            sink_term=float(sink_term),
            roll_term=float(roll_term),
            altitude_track_term=float(altitude_track_term),
            heading_track_term=float(heading_track_term),
            safety_mode_term=float(safety_mode_term),
            terminal_term=float(terminal_term),
        )
