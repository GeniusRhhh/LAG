"""
Common action codec helpers for CAP tactical modules.

Provides:
- Normalized delta arrays for altitude/heading/velocity command spaces
- Discrete command aliases
- Safe clipping helpers
- Heading delta to discrete command mapping helpers
"""

from typing import Union

import numpy as np


# 3-level altitude command set: descend / hold / climb (meters)
NORM_ALTITUDE_3 = np.array([-500.0, 0.0, 500.0], dtype=float)

# 5-level heading command set: [-30, -15, 0, 15, 30] degrees in radians
NORM_HEADING_5 = np.deg2rad(np.array([-30.0, -15.0, 0.0, 15.0, 30.0], dtype=float))

# 3-level speed command set: decel / hold / accel (m/s)
NORM_VELOCITY_3 = np.array([-30.0, 0.0, 30.0], dtype=float)


# Discrete command aliases
ALT_DESCEND = 0
ALT_HOLD = 1
ALT_CLIMB = 2

HDG_HOLD = 2

VEL_DECEL = 0
VEL_HOLD = 1
VEL_ACCEL = 2


def clip_alt_cmd(cmd: Union[int, float]) -> int:
    return int(np.clip(int(cmd), 0, len(NORM_ALTITUDE_3) - 1))


def clip_hdg_cmd(cmd: Union[int, float]) -> int:
    return int(np.clip(int(cmd), 0, len(NORM_HEADING_5) - 1))


def clip_vel_cmd(cmd: Union[int, float]) -> int:
    return int(np.clip(int(cmd), 0, len(NORM_VELOCITY_3) - 1))


def heading_cmd_from_delta_deg(delta_deg: Union[int, float]) -> int:
    """Map heading delta in degrees to nearest 5-level discrete heading index."""
    delta_rad = np.deg2rad(float(delta_deg))
    idx = int(np.argmin(np.abs(NORM_HEADING_5 - delta_rad)))
    return clip_hdg_cmd(idx)


def heading_cmd_from_delta_rad(delta_rad: Union[int, float]) -> int:
    """Map heading delta in radians to nearest 5-level discrete heading index."""
    idx = int(np.argmin(np.abs(NORM_HEADING_5 - float(delta_rad))))
    return clip_hdg_cmd(idx)
