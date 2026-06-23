import logging
import numpy as np
from typing import List, Any, Dict
from jsbsim import FDMExec
from ..core.catalog import Catalog as c

class JSBSimController:
    """JSBSim flight dynamics controller."""

    def __init__(self, fdm: FDMExec, dt: float = 0.1):
        """Initialize controller.

        Args:
            fdm: JSBSim flight dynamics model instance.
            dt: Time step in seconds (default: 0.1).
        """
        self.fdm = fdm
        self.dt = dt
        self.pid_params = {
            'pitch': {'Kp': 3.0, 'Ki': 0.5, 'Kd': 1.0},  # MODIFIED: 提高 Kp 和 Ki 以增强响应
            'roll': {'Kp': 2.5, 'Ki': 0.3, 'Kd': 0.9},  # MODIFIED: 略微提高控制精度
            'yaw': {'Kp': 1.5, 'Ki': 0.1, 'Kd': 0.3}
        }
        self.error_integral = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0}
        self.prev_error = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0}
        self.altitude_min = 3000  # Minimum altitude in meters
        logging.info(f"JSBSimController initialized: dt={dt}s, altitude_min={self.altitude_min}m")

    def control(self, action: np.ndarray, target_state: Dict[str, Any] = None) -> List[float]:
        """Generate control commands.

        Args:
            action: Normalized action vector [aileron, elevator, rudder, throttle].
            target_state: Target state (position, velocity) for missile guidance.

        Returns:
            List: Control commands [aileron, elevator, rudder, throttle].
        """
        aileron_cmd = np.clip(action[0], -1.0, 1.0)
        elevator_cmd = np.clip(action[1], -1.0, 1.0)
        rudder_cmd = np.clip(action[2], -1.0, 1.0)
        throttle_cmd = np.clip(action[3], 0.5, 0.9)

        # Get current state
        pitch_rate = self.fdm[c.attitude_pitch_rad]
        roll_rate = self.fdm[c.attitude_roll_rad]
        altitude = self.fdm[c.position_h_sl_m]

        # Constrain pitch and roll rates
        pitch_rate_cmd = np.clip(elevator_cmd * 0.6, -0.6, 0.6)  # MODIFIED: 放宽俯仰速率限制
        roll_rate_cmd = np.clip(aileron_cmd * 1.2, -1.2, 1.2)  # MODIFIED: 放宽滚转速率限制

        # PID control for pitch
        pitch_error = pitch_rate_cmd - pitch_rate
        self.error_integral['pitch'] += pitch_error * self.dt
        pitch_derivative = (pitch_error - self.prev_error['pitch']) / self.dt
        elevator_cmd = (
            self.pid_params['pitch']['Kp'] * pitch_error +
            self.pid_params['pitch']['Ki'] * self.error_integral['pitch'] +
            self.pid_params['pitch']['Kd'] * pitch_derivative
        )
        self.prev_error['pitch'] = pitch_error

        # PID control for roll
        roll_error = roll_rate_cmd - roll_rate
        self.error_integral['roll'] += roll_error * self.dt
        roll_derivative = (roll_error - self.prev_error['roll']) / self.dt
        aileron_cmd = (
            self.pid_params['roll']['Kp'] * roll_error +
            self.pid_params['roll']['Ki'] * self.error_integral['roll'] +
            self.pid_params['roll']['Kd'] * roll_derivative
        )
        self.prev_error['roll'] = roll_error

        # Altitude protection
        if altitude < self.altitude_min:
            elevator_cmd = np.clip(elevator_cmd + 0.3, -1.0, 1.0)  # MODIFIED: 增强低空保护
            throttle_cmd = np.clip(throttle_cmd + 0.2, 0.5, 0.9)  # MODIFIED: 增加油门
            logging.warning(f"Altitude low: {altitude:.1f}m, applying elevator and throttle correction")

        # Missile guidance input (for MissileSimulator)
        if target_state:
            target_pos = target_state.get("target_position", np.zeros(3))
            target_vel = target_state.get("target_velocity", np.zeros(3))
            self.fdm[c.target_position_x_m] = target_pos[0]
            self.fdm[c.target_position_y_m] = target_pos[1]
            self.fdm[c.target_position_z_m] = target_pos[2]
            self.fdm[c.target_velocity_x_mps] = target_vel[0]
            self.fdm[c.target_velocity_y_mps] = target_vel[1]
            self.fdm[c.target_velocity_z_mps] = target_vel[2]

        control_cmd = [aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd]
        logging.debug(f"Control cmd: aileron={aileron_cmd:.3f}, elevator={elevator_cmd:.3f}, "
                      f"rudder={rudder_cmd:.3f}, throttle={throttle_cmd:.3f}, "
                      f"altitude={altitude:.1f}m")
        return control_cmd

    def reset(self):
        """Reset controller state."""
        self.error_integral = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0}
        self.prev_error = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0}
        logging.info("JSBSimController reset")