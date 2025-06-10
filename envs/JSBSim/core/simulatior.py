import os
import logging
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from typing import Literal, Union, List

import jsbsim
from .catalog import Property, Catalog
from ..utils.utils import get_root_dir, LLA2NEU, NEU2LLA

TeamColors = Literal["Red", "Blue", "Green", "Violet", "Orange"]


class BaseSimulator(ABC):
    def __init__(self, uid: str, color: TeamColors, dt: float):
        self.__uid = uid
        self.__color = color
        self.__dt = dt
        self.model = ""
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)
        logging.debug(f"{self.__class__.__name__}:{self.__uid} is created!")

    @property
    def uid(self) -> str:
        return self.__uid

    @property
    def color(self) -> str:
        return self.__color

    @property
    def dt(self) -> float:
        return self.__dt

    def get_geodetic(self):
        return self._geodetic

    def get_position(self):
        return self._position

    def get_rpy(self):
        return self._posture

    def get_velocity(self):
        return self._velocity

    def reload(self):
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)

    @abstractmethod
    def run(self, **kwargs):
        pass

    def log(self):
        lon, lat, alt = self.get_geodetic()
        roll, pitch, yaw = self.get_rpy() * 180 / np.pi
        log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
        log_msg += f"Name={self.model.upper()},"
        log_msg += f"Color={self.color}"
        return log_msg

    @abstractmethod
    def close(self):
        pass

    def __del__(self):
        logging.debug(f"{self.__class__.__name__}:{self.uid} is deleted!")

class AircraftSimulator(BaseSimulator):
    ALIVE = 0
    CRASH = 1
    SHOTDOWN = 2

    def __init__(self, uid: str = "A0100", color: TeamColors = "Red", model: str = 'f16',
                 init_state: dict = {}, origin: tuple = (120.0, 60.0, 0.0), sim_freq: int = 60, **kwargs):
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model
        self.init_state = init_state
        self.lon0, self.lat0, self.alt0 = origin
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        self._is_leader = uid.endswith("100")
        self.num_missiles = kwargs.get('num_missiles', 4)
        self.num_left_missiles = self.num_missiles
        self.partners: List[AircraftSimulator] = []
        self.enemies: List[AircraftSimulator] = []
        self.launch_missiles: List[MissileSimulator] = []
        self.under_missiles: List[MissileSimulator] = []
        self.reload()
        logging.info(f"AircraftSimulator {uid} initialized: model={model}, missiles={self.num_missiles}")

    @property
    def is_alive(self):
        return self.__status == AircraftSimulator.ALIVE

    @property
    def is_crash(self):
        return self.__status == AircraftSimulator.CRASH

    @property
    def is_shotdown(self):
        return self.__status == AircraftSimulator.SHOTDOWN

    def is_leader(self):
        return self._is_leader

    def set_leader(self, is_leader: bool):
        self._is_leader = is_leader
        logging.info(f"Set Agent {self.uid} as {'Leader' if is_leader else 'Wingman'}")

    def crash(self):
        self.__status = AircraftSimulator.CRASH
        logging.info(f"Agent {self.uid} crashed")

    def shotdown(self):
        self.__status = AircraftSimulator.SHOTDOWN
        logging.info(f"Agent {self.uid} shot down")

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        super().reload()
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        self.launch_missiles.clear()
        self.under_missiles.clear()
        self.num_left_missiles = self.num_missiles

        self.jsbsim_exec = jsbsim.FGFDMExec(os.path.join(get_root_dir(), 'data'))
        self.jsbsim_exec.set_debug_level(0)
        self.jsbsim_exec.load_model(self.model)

        # 预处理 JSBSim 属性，添加默认访问权限
        jsbsim_props = self.jsbsim_exec.query_property_catalog("")
        processed_props = []
        for prop in jsbsim_props:
            prop = prop.strip()
            if " " not in prop:
                prop = f"{prop} R"  # 默认添加只读权限
            processed_props.append(prop)
        try:
            Catalog.add_jsbsim_props(processed_props)
        except Exception as e:
            logging.error(f"Failed to add JSBSim properties: {e}")
            raise

        self.jsbsim_exec.set_dt(self.dt)
        self.clear_default_condition()

        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin
        for key, value in self.init_state.items():
            self.set_property_value(Catalog[key], value)
        success = self.jsbsim_exec.run_ic()
        if not success:
            logging.error("JSBSim initialization failed")
            raise RuntimeError("JSBSim failed to init simulation conditions.")

        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        for j in range(n):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()
        self._update_properties()

    def clear_default_condition(self):
        default_condition = {
            Catalog.ic_long_gc_deg: 120.0,
            Catalog.ic_lat_geod_deg: 60.0,
            Catalog.ic_h_sl_ft: 20000,
            Catalog.ic_psi_true_deg: 0.0,
            Catalog.ic_u_fps: 800.0,
            Catalog.ic_v_fps: 0.0,
            Catalog.ic_w_fps: 0.0,
            Catalog.ic_p_rad_sec: 0.0,
            Catalog.ic_q_rad_sec: 0.0,
            Catalog.ic_r_rad_sec: 0.0,
            Catalog.ic_roc_fpm: 0.0,
            Catalog.ic_terrain_elevation_ft: 0,
        }
        for prop, value in default_condition.items():
            self.set_property_value(prop, value)

    def run(self):
        if self.is_alive:
            if self.bloods <= 0:
                self.shotdown()
            result = self.jsbsim_exec.run()
            if not result:
                logging.error("JSBSim simulation failed")
                raise RuntimeError("JSBSim failed.")
            self._update_properties()
            # Constrain actions to prevent crashes
            pitch_rate = self.get_property_value(Catalog.ic_q_rad_sec)
            roll_rate = self.get_property_value(Catalog.ic_p_rad_sec)
            if abs(pitch_rate) > 0.5:
                self.set_property_value(Catalog.ic_q_rad_sec, np.clip(pitch_rate, -0.5, 0.5))
                logging.debug(f"Agent {self.uid} pitch rate constrained: {pitch_rate:.3f} -> {np.clip(pitch_rate, -0.5, 0.5):.3f}")
            if abs(roll_rate) > 0.7:
                self.set_property_value(Catalog.ic_p_rad_sec, np.clip(roll_rate, -0.7, 0.7))
                logging.debug(f"Agent {self.uid} roll rate constrained: {roll_rate:.3f} -> {np.clip(roll_rate, -0.7, 0.7):.3f}")
            logging.debug(f"Agent {self.uid} state: alt={self._geodetic[2]:.1f}m, "
                          f"vel={np.linalg.norm(self._velocity):.1f}m/s, "
                          f"pitch_rate={pitch_rate:.3f}rad/s, roll_rate={roll_rate:.3f}rad/s")
            return result
        return False

    def close(self):
        if hasattr(self, 'jsbsim_exec') and self.jsbsim_exec:
            self.jsbsim_exec = None
        self.partners = []
        self.enemies = []
        logging.info(f"Agent {self.uid} simulator closed")

    def _update_properties(self):
        self._geodetic[:] = self.get_property_values([
            Catalog.position_long_gc_deg,
            Catalog.position_lat_geod_deg,
            Catalog.position_h_sl_m
        ])
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
        self._posture[:] = self.get_property_values([
            Catalog.attitude_roll_rad,
            Catalog.attitude_pitch_rad,
            Catalog.attitude_heading_true_rad
        ])
        self._velocity[:] = self.get_property_values([
            Catalog.velocities_v_north_mps,
            Catalog.velocities_v_east_mps,
            Catalog.velocities_v_down_mps
        ])
        self._velocity[2] = -self._velocity[2]

    def get_sim_time(self):
        return self.jsbsim_exec.get_sim_time()

    def get_property_values(self, props):
        return [self.get_property_value(prop) for prop in props]

    def set_property_values(self, props, values):
        if len(props) != len(values):
            logging.error(f"Property-value mismatch: props={len(props)}, values={len(values)}")
            raise ValueError("mismatch between properties and values size")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def get_property_value(self, prop):
        if isinstance(prop, Property):
            if prop.access == "R" and prop.update:
                prop.update(self)
            return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        logging.error(f"Invalid prop type: {type(prop)}")
        raise ValueError(f"prop type unhandled: {type(prop)}")

    def set_property_value(self, prop, value):
        if isinstance(prop, Property):
            value = np.clip(value, prop.min, prop.max)
            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)
            if "W" in prop.access and prop.update:
                prop.update(self)
        else:
            logging.error(f"Invalid prop type: {type(prop)}")
            raise ValueError(f"prop type unhandled: {type(prop)}")

    def check_missile_warning(self, multi=False):
        """
        Check for missile warnings.

        Args:
            multi (bool): If True, return a list of all threatening missiles; if False, return the first threatening missile.

        Returns:
            Union[MissileSimulator, List[MissileSimulator], None]: Single missile or list of missiles if threatening, None otherwise.
        """
        threatening_missiles = []
        for missile in self.under_missiles:
            if missile.is_alive:
                distance = np.linalg.norm(missile.get_position() - self.get_position())
                # 假设威胁范围为 5000m
                if distance < 5000:  # 可自定义威胁距离
                    logging.debug(f"Missile warning for {self.uid}: distance={distance:.1f}m from {missile.uid}")
                    if not multi:
                        return missile  # 返回第一个威胁导弹
                    threatening_missiles.append(missile)
        if multi and threatening_missiles:
            return threatening_missiles
        return None if not threatening_missiles else threatening_missiles[0]  # 兼容旧逻辑

class MissileSimulator(BaseSimulator):
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str, missile_model: str = "AIM-120C7"):
        assert parent.dt == target.dt, "Integration timestep must be same!"
        missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)
        missile.target(target)
        return missile

    def __init__(self, uid="A0101", color="Red", model="AIM-120C7", dt=1 / 12):
        super().__init__(uid, color, dt)
        self.__status = MissileSimulator.INACTIVE
        self.model = model
        self.parent_aircraft = None
        self.target_aircraft = None
        self.render_explosion = False
        self._g = 9.81
        self._t_max = 120
        self._t_thrust = 15  # 调整为 15s
        self._Isp = 250  # 提高比冲
        self._Length = 3.66
        self._Diameter = 0.18
        self._cD = 0.35
        self._m0 = 150
        self._dm = 5
        self._K = 3.5  # 调整导航常数
        self._nyz_max = 25  # 调整为 25G
        self._Rc = 20  # 调整为 20m
        self._v_min = 200
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(5 / self.dt))
        self._left_t = int(1 / self.dt)
        self._target_pos_history = deque(maxlen=5)  # For acceleration estimation
        logging.info(f"MissileSimulator {uid} initialized: model={model}, dt={dt}")

    @property
    def is_alive(self):
        return self.__status == MissileSimulator.LAUNCHED

    @property
    def is_success(self):
        return self.__status == MissileSimulator.HIT

    @property
    def is_done(self):
        return self.__status == MissileSimulator.HIT or self.__status == MissileSimulator.MISS

    @property
    def Isp(self):
        return self._Isp if self._t < self._t_thrust else 0

    @property
    def K(self):
        return max(self._K * (self._t_max - self._t) / self._t_max, 0)

    @property
    def S(self):
        S0 = np.pi * (self._Diameter / 2)**2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)

    @property
    def target_distance(self) -> float:
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, parent: AircraftSimulator):
        self.parent_aircraft = parent
        self.parent_aircraft.launch_missiles.append(self)
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()+ np.array([300, 0, 0])  # 增加初始速度
        if np.any(np.isnan(self._velocity)):
            logging.error(f"NaN in velocity for missile {self.uid} from parent {parent.uid}")
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self.__status = MissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(5 / self.dt))
        self._left_t = int(1 / self.dt)
        logging.info(f"Missile {self.uid} launched from {parent.uid}")

    def target(self, target: AircraftSimulator):
        self.target_aircraft = target
        self.target_aircraft.under_missiles.append(self)
        logging.info(f"Missile {self.uid} targeting {target.uid}")

    def run(self):
        self._t += self.dt
        action, distance = self._guidance()
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance
        if distance < self._Rc and self.target_aircraft.is_alive:
            self.__status = MissileSimulator.HIT
            self.target_aircraft.shotdown()
            logging.info(f"Missile {self.uid} hit target {self.target_aircraft.uid}, distance={distance:.1f}m")
        elif (self._t > self._t_max) or (self._t > self._t_thrust and np.linalg.norm(self.get_velocity()) < self._v_min) or \
             np.sum(self._distance_increment) >= self._distance_increment.maxlen or not self.target_aircraft.is_alive:
            self.__status = MissileSimulator.MISS
            reason = "timeout" if self._t > self._t_max else \
                     "low_velocity" if np.linalg.norm(self.get_velocity()) < self._v_min else \
                     "diverging" if np.sum(self._distance_increment) >= self._distance_increment.maxlen else "target_dead"
            logging.info(f"Missile {self.uid} missed: reason={reason}, distance={distance:.1f}m")
        else:
            self._state_trans(action)
            logging.debug(f"Missile {self.uid} state: pos={self._position.tolist()}, "
                          f"vel={np.linalg.norm(self._velocity):.1f}m/s, distance={distance:.1f}m, "
                          f"ny={action[0]:.2f}, nz={action[1]:.2f}")

    def log(self):
        if self.is_alive or self.is_done:
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Name={self.model.upper()},Color={self.color},Type=Weapon + Missile"
            if self.is_alive:
                log_msg += f",Parent={self.parent_aircraft.uid}"
            logging.info(f"Missile {self.uid} rendering: {log_msg}")
            return log_msg
        elif self.is_done and not self.render_explosion:
            self.render_explosion = True
            log_msg = f"-{self.uid}\n"
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
            logging.info(f"Missile {self.uid} explosion: {log_msg}")
            return log_msg
        return None

    def close(self):
        self.target_aircraft = None
        logging.info(f"Missile {self.uid} simulator closed")

    def _guidance(self):
        # Augmented Proportional Navigation (APN)
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])
        if v_m < 1e-6: v_m = 1e-6  # 避免除零
        theta_m = np.arcsin(dz_m / v_m) if v_m > 0 else 0

        x_t, y_t, z_t = self.target_aircraft.get_position()
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()

        # Estimate target acceleration
        self._target_pos_history.append(np.array([x_t, y_t, z_t]))
        a_t = np.zeros(3)
        if len(self._target_pos_history) >= 3:
            pos_prev2, pos_prev1, pos_curr = list(self._target_pos_history)[-3:]
            v_t_prev = (pos_curr - pos_prev1) / self.dt
            v_t_prev2 = (pos_prev1 - pos_prev2) / self.dt
            a_t = (v_t_prev - v_t_prev2) / self.dt

        # Relative position and velocity
        r = np.array([x_t - x_m, y_t - y_m, z_t - z_m])
        v_r = np.array([dx_t - dx_m, dy_t - dy_m, dz_t - dz_m])
        R = np.linalg.norm(r)
        if R < 1e-6: R = 1e-6  # 避免除零
        v_c = -np.dot(r, v_r) / R  # Closing velocity

        # Line-of-sight rates
        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])
        Rxyz = R
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / (Rxy ** 2 + 1e-6)
        deps = ((dz_t - dz_m) * Rxy ** 2 - (z_t - z_m) * (
                (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz ** 2 * Rxy + 1e-6)

        # APN guidance law
        ny = self.K * v_c * dbeta + 0.5 * self.K * a_t[1] / self._g
        nz = self.K * v_c * deps + 0.5 * self.K * a_t[2] / self._g + np.cos(theta_m)

        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), Rxyz

    def _state_trans(self, action):
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self._position, self.lon0, self.lat0, self.alt0)

        v = np.linalg.norm(self.get_velocity())
        v = max(v, 1e-6)  # 避免除零
        theta, phi = self.get_rpy()[1:]
        self._m = max(self._m, 1e-6)  # 确保质量不变为负或零
        T = self._g * self.Isp * self._dm
        D = 0.5 * self._cD * self.S * self.rho * v ** 2
        nx = (T - D) / (self._m * self._g)
        ny, nz = action

        dv = self._g * (nx - np.sin(theta))
        self._dphi = self._g / v * (ny / np.cos(theta) if abs(np.cos(theta)) > 1e-6 else 0)
        self._dtheta = self._g / v * (nz - np.cos(theta))

        v += self.dt * dv
        phi += self.dt * self._dphi
        theta += self.dt * self._dtheta

        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta)
        ])
        self._posture[:] = np.array([0, theta, phi])

        if self._t < self._t_thrust:
            self._m = self._m - self.dt * self._dm
