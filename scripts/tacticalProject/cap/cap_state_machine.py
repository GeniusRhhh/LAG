"""
CAP state machine.
Manage the five CAP task states: PATROL -> INTERCEPT -> ENGAGE -> EVADE -> RTB.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .control_ranges import ControlRanges, DEFAULT_RANGES


AWACS_DETECTION_RANGE = 400.0
RADAR_DETECTION_RANGE = 200.0


class CAPState(Enum):
    PATROL = "PATROL"
    INTERCEPT = "INTERCEPT"
    ENGAGE = "ENGAGE"
    EVADE = "EVADE"
    RTB = "RTB"


@dataclass
class StateContext:
    min_threat_distance: float = float("inf")
    formation_center_y: float = 50.0
    has_awacs_info: bool = False
    has_hostile_in_high_zone: bool = False
    has_hostile_in_medium_zone: bool = False
    is_missile_incoming: bool = False
    fuel_critical: bool = False
    mission_time: float = 0.0
    hostile_alive_count: int = 0


class CAPStateMachine:
    HYSTERESIS = 30.0
    MIN_STATE_DURATION = 10.0

    def __init__(self, ranges: ControlRanges = None):
        self.ranges = ranges or DEFAULT_RANGES
        self.state = CAPState.PATROL
        self._prev_state = CAPState.PATROL
        self._state_enter_time = 0.0
        self._state_hold_until = 0.0
        self._ever_engaged = False

    def update(self, ctx: StateContext, current_time: float) -> Tuple[CAPState, bool]:
        self._prev_state = self.state

        if self._should_rtb(ctx):
            self.state = CAPState.RTB
            self._state_hold_until = current_time + self.MIN_STATE_DURATION
        elif self._should_evade(ctx):
            self.state = CAPState.EVADE
            self._ever_engaged = True
            self._state_hold_until = current_time + self.MIN_STATE_DURATION
        elif current_time < self._state_hold_until:
            pass
        elif self._should_engage(ctx):
            if self.state != CAPState.ENGAGE:
                self._state_hold_until = current_time + self.MIN_STATE_DURATION
            self.state = CAPState.ENGAGE
            self._ever_engaged = True
        elif self._should_intercept(ctx):
            if self.state != CAPState.INTERCEPT:
                self._state_hold_until = current_time + self.MIN_STATE_DURATION
            self.state = CAPState.INTERCEPT
        elif self._should_patrol(ctx):
            if self.state != CAPState.PATROL:
                self._state_hold_until = current_time + self.MIN_STATE_DURATION
            self.state = CAPState.PATROL

        changed = self.state != self._prev_state
        if changed:
            self._state_enter_time = current_time
        return self.state, changed

    def _should_rtb(self, ctx: StateContext) -> bool:
        if int(getattr(ctx, "hostile_alive_count", 0)) > 0:
            return False
        return ctx.fuel_critical or ctx.mission_time > 20 * 60

    def _should_evade(self, ctx: StateContext) -> bool:
        if ctx.is_missile_incoming:
            return True
        if ctx.min_threat_distance < self.ranges.MAR:
            return True
        return False

    def _should_engage(self, ctx: StateContext) -> bool:
        if self.state == CAPState.EVADE:
            return False

        if ctx.has_hostile_in_high_zone or ctx.has_hostile_in_medium_zone:
            return True

        if self.state == CAPState.ENGAGE:
            exit_threshold = RADAR_DETECTION_RANGE + self.HYSTERESIS
            return ctx.min_threat_distance < exit_threshold
        enter_threshold = RADAR_DETECTION_RANGE
        return ctx.min_threat_distance < enter_threshold

    def _should_intercept(self, ctx: StateContext) -> bool:
        if self._should_engage(ctx):
            return self.state == CAPState.INTERCEPT

        if not ctx.has_awacs_info:
            return False

        if self.state == CAPState.INTERCEPT:
            exit_threshold = AWACS_DETECTION_RANGE + self.HYSTERESIS
            return ctx.min_threat_distance < exit_threshold
        enter_threshold = AWACS_DETECTION_RANGE
        return ctx.min_threat_distance < enter_threshold

    def _should_patrol(self, ctx: StateContext) -> bool:
        if ctx.is_missile_incoming:
            return False
        if ctx.has_hostile_in_high_zone or ctx.has_hostile_in_medium_zone:
            return False
        if ctx.has_awacs_info and ctx.min_threat_distance < (AWACS_DETECTION_RANGE + self.HYSTERESIS):
            return False
        return True

    def get_current_control_range(
        self, distance: float, ranges_dict: dict[str, float] = None
    ) -> Optional[str]:
        return self.ranges.get_current_node(distance, ranges_dict)

    @property
    def is_patrol(self) -> bool:
        return self.state == CAPState.PATROL

    @property
    def is_intercept(self) -> bool:
        return self.state == CAPState.INTERCEPT

    @property
    def is_engage(self) -> bool:
        return self.state == CAPState.ENGAGE

    @property
    def is_evade(self) -> bool:
        return self.state == CAPState.EVADE

    @property
    def is_rtb(self) -> bool:
        return self.state == CAPState.RTB
