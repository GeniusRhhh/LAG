"""
Auxiliary speed coordination module.

The original implementation forced pairwise ETA synchronization, which could
slow the faster aircraft and waste interception opportunities. The current
version downgrades speed coordination to a lightweight compatibility layer:

1. Keep the existing interface unchanged.
2. Preserve each aircraft's current speed as much as possible.
3. Do not decelerate a faster aircraft just to wait for its wingman.
4. Only map the current speed to the nearest discrete command level.

This matches the current design intent: speed is no longer a primary
coordination driver in cooperative detection.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SpeedCommand:
    """Discrete speed command returned to the CAP action layer."""

    fighter_id: str
    target_speed: float
    speed_index: int
    eta: float


class MissionPhase:
    PATROL = "PATROL"
    INTERCEPT = "INTERCEPT"
    ENGAGE = "ENGAGE"
    EVADE = "EVADE"
    RTB = "RTB"


class VelocityCoordination:
    """
    Lightweight speed command adapter.

    The class is intentionally conservative now. It no longer tries to solve a
    pairwise synchronization problem. Instead, it simply preserves each
    aircraft's current speed and converts it into the nearest supported command
    level so the rest of the CAP pipeline can stay unchanged.
    """

    SPEED_LEVELS = [150, 180, 200, 220, 250, 280, 300, 330]

    # Keep the phase table for compatibility and future reporting hooks, but
    # do not use it to force synchronization or suppress faster aircraft.
    PHASE_SPEED_CONSTRAINTS = {
        MissionPhase.PATROL: {"v_min": 200.0, "v_max": 250.0, "v_nominal": 220.0},
        MissionPhase.INTERCEPT: {"v_min": 250.0, "v_max": 300.0, "v_nominal": 280.0},
        MissionPhase.ENGAGE: {"v_min": 250.0, "v_max": 330.0, "v_nominal": 280.0},
        MissionPhase.EVADE: {"v_min": 280.0, "v_max": 330.0, "v_nominal": 300.0},
        MissionPhase.RTB: {"v_min": 220.0, "v_max": 280.0, "v_nominal": 250.0},
    }

    V_MIN = 150.0
    V_MAX = 330.0
    V_NOMINAL = 250.0

    def __init__(self, formation_pairs: List[Tuple[str, str]] = None):
        self.formation_pairs = formation_pairs or [
            ("A0100", "A0200"),
            ("A0300", "A0400"),
        ]
        self._last_commands: Dict[str, SpeedCommand] = {}
        self._current_phase = MissionPhase.PATROL

    def set_mission_phase(self, phase: str) -> None:
        self._current_phase = phase

    def _get_phase_constraints(self) -> dict:
        return self.PHASE_SPEED_CONSTRAINTS.get(
            self._current_phase,
            {"v_min": self.V_MIN, "v_max": self.V_MAX, "v_nominal": self.V_NOMINAL},
        )

    def compute_coordinated_speeds(
        self,
        fighter_positions: Dict[str, Tuple[float, float]],
        fighter_speeds: Dict[str, float],
        target_positions: Dict[str, Tuple[float, float]],
        mission_phase: str = None,
    ) -> Dict[str, SpeedCommand]:
        """
        Return per-aircraft speed commands without pairwise synchronization.

        Behavior:
        - Keep the aircraft's current speed.
        - Clip only to the platform-wide absolute bounds.
        - Estimate ETA using the preserved speed.
        """

        if mission_phase is not None:
            self.set_mission_phase(mission_phase)

        # Compatibility hook: read the phase table so callers can still set it,
        # but do not use the narrower bounds to slow down fast aircraft.
        _ = self._get_phase_constraints()

        if not fighter_positions or not target_positions:
            self._last_commands = {}
            return {}

        commands: Dict[str, SpeedCommand] = {}
        for fighter_id, position in fighter_positions.items():
            if fighter_id not in target_positions:
                continue

            target = target_positions[fighter_id]
            distance_m = float(
                np.hypot(float(target[0]) - float(position[0]), float(target[1]) - float(position[1]))
            ) * 1000.0
            current_speed = float(fighter_speeds.get(fighter_id, self.V_NOMINAL))
            bounded_speed = float(np.clip(current_speed, self.V_MIN, self.V_MAX))
            eta = distance_m / bounded_speed if bounded_speed > 1e-6 else 9999.0
            commands[fighter_id] = self._make_command(fighter_id, bounded_speed, eta)

        self._last_commands = commands
        return commands

    def _make_command(self, fighter_id: str, target_speed: float, eta: float) -> SpeedCommand:
        speed_index = int(np.argmin([abs(level - target_speed) for level in self.SPEED_LEVELS]))
        return SpeedCommand(
            fighter_id=fighter_id,
            target_speed=self.SPEED_LEVELS[speed_index],
            speed_index=speed_index,
            eta=eta,
        )

    def get_command(self, fighter_id: str) -> Optional[SpeedCommand]:
        return self._last_commands.get(fighter_id)

    def get_speed_index(self, fighter_id: str, default: int = 4) -> int:
        command = self._last_commands.get(fighter_id)
        return command.speed_index if command else default
