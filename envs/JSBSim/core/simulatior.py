import os
import logging
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from typing import Literal, Union, List, Dict, Any

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

        # 新增：战术状态跟踪
        self.tactical_state = {
            "current_maneuver": "cruise",
            "energy_state": "balanced",
            "threat_level": "low",
            "engagement_phase": "bvr"
        }

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

    def get_tactical_state(self) -> Dict[str, Any]:
        """获取战术状态"""
        return self.tactical_state.copy()

    def update_tactical_state(self, **kwargs):
        """更新战术状态"""
        self.tactical_state.update(kwargs)

    def reload(self):
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)
        self.tactical_state = {
            "current_maneuver": "cruise",
            "energy_state": "balanced",
            "threat_level": "low",
            "engagement_phase": "bvr"
        }

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

        # 新增：飞行性能参数
        self.flight_envelope = {
            "max_altitude": 18000,  # 最大高度 (m)
            "service_ceiling": 15000,  # 实用升限 (m)
            "max_speed": 600,  # 最大速度 (m/s)
            "min_speed": 150,  # 最小速度 (m/s)
            "max_g": 9.0,  # 最大过载
            "max_climb_rate": 250,  # 最大爬升率 (m/s)
            "max_turn_rate": 25  # 最大转弯率 (deg/s)
        }

        # 新增：传感器和武器系统
        self.sensors = {
            "radar_range": 120000,  # 雷达探测距离 (m)
            "radar_lock_range": 80000,  # 雷达锁定距离 (m)
            "iff_range": 150000,  # 敌我识别距离 (m)
            "rwr_range": 200000  # 雷达告警距离 (m)
        }

        self.weapons = {
            "missiles": {
                "aim120": {"count": self.num_missiles, "range": 100000, "speed": 1200},
            },
            "gun": {"rounds": 500, "range": 2000, "rate": 6000}
        }

        # 新增：战术控制器
        self.tactical_controller = None

        self.reload()
        logging.info(f"Enhanced AircraftSimulator {uid} initialized: model={model}, "
                     f"missiles={self.num_missiles}, sensors={self.sensors}")

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

    def get_energy_state(self) -> Dict[str, float]:
        """获取能量状态"""
        velocity = np.linalg.norm(self.get_velocity())
        altitude = self.get_position()[2]

        # 计算比能量 (Specific Energy)
        kinetic_energy = 0.5 * velocity ** 2
        potential_energy = 9.81 * altitude
        specific_energy = kinetic_energy + potential_energy

        # 归一化到0-1范围
        max_se = 0.5 * self.flight_envelope["max_speed"] ** 2 + 9.81 * self.flight_envelope["max_altitude"]
        normalized_se = specific_energy / max_se

        return {
            "specific_energy": specific_energy,
            "normalized_energy": normalized_se,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "velocity": velocity,
            "altitude": altitude
        }

    def get_maneuver_capability(self) -> Dict[str, float]:
        """获取机动能力"""
        energy_state = self.get_energy_state()
        velocity = energy_state["velocity"]
        altitude = energy_state["altitude"]

        # 基于当前状态计算机动能力
        g_available = self.flight_envelope["max_g"] * min(velocity / 300, 1.0)
        turn_rate_available = self.flight_envelope["max_turn_rate"] * min(velocity / 250, 1.0)
        climb_rate_available = self.flight_envelope["max_climb_rate"] * max(
            1 - altitude / self.flight_envelope["service_ceiling"], 0.1)

        return {
            "max_g_available": g_available,
            "max_turn_rate": turn_rate_available,
            "max_climb_rate": climb_rate_available,
            "maneuver_margin": min(g_available / self.flight_envelope["max_g"], 1.0)
        }

    def get_sensor_contacts(self) -> List[Dict[str, Any]]:
        """获取传感器接触"""
        contacts = []

        # 雷达接触
        for enemy in self.enemies:
            if enemy.is_alive:
                distance = np.linalg.norm(enemy.get_position() - self.get_position())
                if distance <= self.sensors["radar_range"]:
                    relative_vel = enemy.get_velocity() - self.get_velocity()
                    bearing = np.arctan2(
                        enemy.get_position()[1] - self.get_position()[1],
                        enemy.get_position()[0] - self.get_position()[0]
                    )

                    contact = {
                        "uid": enemy.uid,
                        "type": "radar",
                        "distance": distance,
                        "bearing": np.rad2deg(bearing),
                        "velocity": np.linalg.norm(enemy.get_velocity()),
                        "relative_velocity": np.linalg.norm(relative_vel),
                        "altitude": enemy.get_position()[2],
                        "locked": distance <= self.sensors["radar_lock_range"]
                    }
                    contacts.append(contact)

        # 导弹威胁
        for missile in self.under_missiles:
            if missile.is_alive:
                distance = np.linalg.norm(missile.get_position() - self.get_position())
                if distance <= self.sensors["rwr_range"]:
                    contact = {
                        "uid": missile.uid,
                        "type": "missile_threat",
                        "distance": distance,
                        "threat_level": "high" if distance < 30000 else "medium"
                    }
                    contacts.append(contact)

        return contacts

    def get_weapon_status(self) -> Dict[str, Any]:
        """获取武器状态"""
        return {
            "missiles_remaining": self.num_left_missiles,
            "missiles_launched": len(self.launch_missiles),
            "missiles_active": len([m for m in self.launch_missiles if m.is_alive]),
            "missiles_hit": len([m for m in self.launch_missiles if m.is_success]),
            "gun_rounds": self.weapons["gun"]["rounds"],
            "weapon_ready": self.num_left_missiles > 0
        }

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

            # 增强的飞行包线保护
            self._apply_flight_envelope_protection()

            # 更新战术状态
            self._update_tactical_state()

            return result
        return False

    def _apply_flight_envelope_protection(self):
        """应用飞行包线保护"""
        current_alt = self.get_position()[2]
        current_vel = np.linalg.norm(self.get_velocity())

        # 高度保护
        if current_alt > self.flight_envelope["max_altitude"]:
            logging.warning(f"Agent {self.uid} exceeding max altitude: {current_alt:.1f}m")
        elif current_alt < 500:  # 最小安全高度
            logging.warning(f"Agent {self.uid} too low: {current_alt:.1f}m")

        # 速度保护
        if current_vel > self.flight_envelope["max_speed"]:
            logging.warning(f"Agent {self.uid} exceeding max speed: {current_vel:.1f}m/s")
        elif current_vel < self.flight_envelope["min_speed"]:
            logging.warning(f"Agent {self.uid} below min speed: {current_vel:.1f}m/s")

        # 过载保护
        pitch_rate = self.get_property_value(Catalog.ic_q_rad_sec)
        roll_rate = self.get_property_value(Catalog.ic_p_rad_sec)

        if abs(pitch_rate) > 0.8:  # 放宽限制以适应战术机动
            self.set_property_value(Catalog.ic_q_rad_sec, np.clip(pitch_rate, -0.8, 0.8))
            logging.debug(f"Agent {self.uid} pitch rate limited: {pitch_rate:.3f}")

        if abs(roll_rate) > 1.0:  # 放宽限制
            self.set_property_value(Catalog.ic_p_rad_sec, np.clip(roll_rate, -1.0, 1.0))
            logging.debug(f"Agent {self.uid} roll rate limited: {roll_rate:.3f}")

    def _update_tactical_state(self):
        """更新战术状态"""
        energy = self.get_energy_state()
        maneuver = self.get_maneuver_capability()

        # 更新能量状态
        if energy["normalized_energy"] > 0.7:
            self.tactical_state["energy_state"] = "advantage"
        elif energy["normalized_energy"] < 0.3:
            self.tactical_state["energy_state"] = "disadvantage"
        else:
            self.tactical_state["energy_state"] = "balanced"

        # 更新威胁等级
        threats = len([m for m in self.under_missiles if m.is_alive])
        if threats > 0:
            min_threat_distance = min([np.linalg.norm(m.get_position() - self.get_position())
                                       for m in self.under_missiles if m.is_alive], default=np.inf)
            if min_threat_distance < 15000:
                self.tactical_state["threat_level"] = "critical"
            elif min_threat_distance < 30000:
                self.tactical_state["threat_level"] = "high"
            else:
                self.tactical_state["threat_level"] = "medium"
        else:
            self.tactical_state["threat_level"] = "low"

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
        """检查导弹威胁"""
        threatening_missiles = []
        for missile in self.under_missiles:
            if missile.is_alive:
                distance = np.linalg.norm(missile.get_position() - self.get_position())
                # 威胁距离阈值
                if distance < 60000:  # 60km威胁范围
                    logging.debug(f"Missile warning for {self.uid}: distance={distance:.1f}m from {missile.uid}")
                    if not multi:
                        return missile  # 返回第一个威胁导弹
                    threatening_missiles.append(missile)
        if multi and threatening_missiles:
            return threatening_missiles
        return None if not threatening_missiles else threatening_missiles[0]


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

        # 增强的导弹参数
        self._g = 9.81
        self._t_max = 150  # 增加最大飞行时间
        self._t_thrust = 20  # 增加推力时间
        self._Isp = 280  # 提高比冲
        self._Length = 3.66
        self._Diameter = 0.18
        self._cD = 0.3  # 降低阻力系数
        self._m0 = 150
        self._dm = 4  # 降低燃料消耗率
        self._K = 4.0  # 增强导航常数
        self._nyz_max = 30  # 增加最大过载
        self._Rc = 15  # 减小毁伤半径以提高精度要求
        self._v_min = 180  # 降低最小速度

        # 初始化状态
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(8 / self.dt))  # 增加判断窗口
        self._left_t = int(1 / self.dt)
        self._target_pos_history = deque(maxlen=8)  # 增加历史窗口

        # 新增：导弹制导状态
        self.guidance_state = {
            "phase": "inactive",  # inactive, boost, midcourse, terminal
            "lock_time": 0,
            "target_acquired": False,
            "intercept_point": np.zeros(3)
        }

        logging.info(f"Enhanced MissileSimulator {uid} initialized: model={model}, "
                     f"max_range={self._t_max * 300:.0f}m, max_g={self._nyz_max}")

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
        # 动态导航增益
        return max(self._K * (self._t_max - self._t) / self._t_max, 1.0)

    @property
    def S(self):
        S0 = np.pi * (self._Diameter / 2) ** 2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)

    @property
    def target_distance(self) -> float:
        if self.target_aircraft and self.target_aircraft.is_alive:
            return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())
        return np.inf

    def launch(self, parent: AircraftSimulator):
        self.parent_aircraft = parent
        self.parent_aircraft.launch_missiles.append(self)
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()

        # 增强的初始速度
        parent_velocity = parent.get_velocity()
        launch_velocity_boost = 350  # 增加发射速度
        self._velocity[:] = parent_velocity + np.array([launch_velocity_boost, 0, 0])

        if np.any(np.isnan(self._velocity)):
            logging.error(f"NaN in velocity for missile {self.uid} from parent {parent.uid}")
            self._velocity[:] = np.array([400, 0, 0])  # 默认速度

        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self.__status = MissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(8 / self.dt))
        self._left_t = int(1 / self.dt)

        # 初始化制导状态
        self.guidance_state = {
            "phase": "boost",
            "lock_time": 0,
            "target_acquired": False,
            "intercept_point": np.zeros(3)
        }

        logging.info(
            f"Enhanced missile {self.uid} launched from {parent.uid} with velocity {np.linalg.norm(self._velocity):.1f}m/s")

    def target(self, target: AircraftSimulator):
        self.target_aircraft = target
        self.target_aircraft.under_missiles.append(self)
        logging.info(f"Missile {self.uid} targeting {target.uid}")

    def run(self):
        if not self.is_alive:
            return

        self._t += self.dt

        # 更新制导阶段
        self._update_guidance_phase()

        # 制导计算
        action, distance = self._guidance()
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance

        # 命中判断
        if distance < self._Rc and self.target_aircraft and self.target_aircraft.is_alive:
            self.__status = MissileSimulator.HIT
            self.target_aircraft.shotdown()
            self.guidance_state["phase"] = "hit"
            logging.info(f"Missile {self.uid} hit target {self.target_aircraft.uid}, distance={distance:.1f}m")
            return

        # 失效判断
        miss_conditions = [
            self._t > self._t_max,  # 超时
            (self._t > self._t_thrust and np.linalg.norm(self.get_velocity()) < self._v_min),  # 速度过低
            np.sum(self._distance_increment) >= self._distance_increment.maxlen,  # 发散
            not self.target_aircraft or not self.target_aircraft.is_alive,  # 目标失效
            distance > 150000  # 距离过远
        ]

        if any(miss_conditions):
            self.__status = MissileSimulator.MISS
            reasons = ["timeout", "low_velocity", "diverging", "target_dead", "out_of_range"]
            reason = reasons[miss_conditions.index(True)]
            self.guidance_state["phase"] = "miss"
            logging.info(f"Missile {self.uid} missed: reason={reason}, distance={distance:.1f}m, time={self._t:.1f}s")
            return

        # 状态转换
        self._state_trans(action)

        if self._t % (5 / self.dt) == 0:  # 每5秒记录一次
            logging.debug(f"Missile {self.uid} guidance: phase={self.guidance_state['phase']}, "
                          f"distance={distance:.1f}m, velocity={np.linalg.norm(self._velocity):.1f}m/s, "
                          f"time={self._t:.1f}s, ny={action[0]:.2f}, nz={action[1]:.2f}")

    def _update_guidance_phase(self):
        """更新制导阶段"""
        if self._t <= self._t_thrust:
            self.guidance_state["phase"] = "boost"
        elif self._t <= self._t_max * 0.8:
            self.guidance_state["phase"] = "midcourse"
        else:
            self.guidance_state["phase"] = "terminal"

    def log(self):
        if self.is_alive or (self.is_done and not self.render_explosion):
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Name={self.model.upper()},Color={self.color}"

            if self.is_alive:
                log_msg += f",Type=Weapon + Missile,Parent={self.parent_aircraft.uid if self.parent_aircraft else 'None'}"
            elif self.is_done and not self.render_explosion:
                self.render_explosion = True
                explosion_msg = f"-{self.uid}\n"
                explosion_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
                explosion_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
                return explosion_msg

            return log_msg
        return None

    def close(self):
        if self.target_aircraft and self in self.target_aircraft.under_missiles:
            self.target_aircraft.under_missiles.remove(self)
        if self.parent_aircraft and self in self.parent_aircraft.launch_missiles:
            self.parent_aircraft.launch_missiles.remove(self)
        self.target_aircraft = None
        self.parent_aircraft = None
        logging.info(f"Missile {self.uid} simulator closed")

    def _guidance(self):
        """增强比例导航制导法 (Enhanced Proportional Navigation)"""
        if not self.target_aircraft or not self.target_aircraft.is_alive:
            return np.array([0, 0]), np.inf

        # 导弹状态
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])

        if v_m < 1e-6:
            v_m = 1e-6  # 避免除零

        # 目标状态
        x_t, y_t, z_t = self.target_aircraft.get_position()
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()

        # 目标加速度估计（增强预测）
        self._target_pos_history.append(np.array([x_t, y_t, z_t]))
        a_t = np.zeros(3)

        if len(self._target_pos_history) >= 4:
            # 使用4点差分提高精度
            positions = list(self._target_pos_history)[-4:]
            v_t_3 = (positions[3] - positions[2]) / self.dt
            v_t_2 = (positions[2] - positions[1]) / self.dt
            v_t_1 = (positions[1] - positions[0]) / self.dt

            a_t_2 = (v_t_3 - v_t_2) / self.dt
            a_t_1 = (v_t_2 - v_t_1) / self.dt
            a_t = (a_t_2 + a_t_1) / 2  # 平滑加速度

        # 相对位置和速度
        r = np.array([x_t - x_m, y_t - y_m, z_t - z_m])
        v_r = np.array([dx_t - dx_m, dy_t - dy_m, dz_t - dz_m])
        R = np.linalg.norm(r)

        if R < 1e-6:
            return np.array([0, 0]), R

        # 逼近速度
        v_c = -np.dot(r, v_r) / R

        # 视线角速率计算
        r_unit = r / R
        h = np.cross(r, v_r)  # 角动量向量
        omega_los = h / (R * R)  # 视线角速率向量

        # 分解到导弹坐标系
        # 简化处理：假设导弹机体坐标系
        omega_y = omega_los[1]  # 偏航角速率
        omega_z = omega_los[2]  # 俯仰角速率

        # 增强比例导航律
        N = self.K  # 导航比

        # 基础PN项
        ny_pn = N * v_c * omega_y
        nz_pn = N * v_c * omega_z

        # 目标加速度补偿项（APN）
        ny_apn = 0.5 * N * a_t[1]
        nz_apn = 0.5 * N * a_t[2]

        # 重力补偿
        g_comp = self._g * np.cos(np.arcsin(dz_m / v_m)) if v_m > 0 else self._g

        # 总制导指令
        ny = ny_pn + ny_apn
        nz = nz_pn + nz_apn + g_comp / self._g

        # 制导阶段调整
        if self.guidance_state["phase"] == "boost":
            # 助推段：保持稳定
            ny *= 0.5
            nz *= 0.5
        elif self.guidance_state["phase"] == "terminal":
            # 末制导段：增强机动
            ny *= 1.5
            nz *= 1.5

        # 过载限制
        total_n = np.sqrt(ny ** 2 + nz ** 2)
        if total_n > self._nyz_max:
            scale = self._nyz_max / total_n
            ny *= scale
            nz *= scale

        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), R

    def _state_trans(self, action):
        """状态转换"""
        # 位置更新
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self._position, self.lon0, self.lat0, self.alt0)

        # 动力学计算
        v = np.linalg.norm(self.get_velocity())
        v = max(v, 1e-6)  # 避免除零

        theta, phi = self.get_rpy()[1:]
        self._m = max(self._m - self.dt * self._dm if self._t < self._t_thrust else self._m,
                      self._m0 * 0.3)  # 保留30%结构质量

        # 推力计算
        T = self._g * self.Isp * self._dm if self._t < self._t_thrust else 0

        # 阻力计算（改进）
        D = 0.5 * self._cD * self.S * self.rho * v ** 2

        # 轴向加速度
        nx = (T - D) / (self._m * self._g)

        # 法向加速度
        ny, nz = action

        # 速度更新
        dv = self._g * (nx - np.sin(theta))
        v_new = v + self.dt * dv
        v_new = max(v_new, 50)  # 最小速度限制

        # 角速度更新
        self._dphi = self._g / v_new * (ny / np.cos(theta) if abs(np.cos(theta)) > 1e-6 else 0)
        self._dtheta = self._g / v_new * (nz - np.cos(theta))

        # 角度更新
        phi_new = phi + self.dt * self._dphi
        theta_new = theta + self.dt * self._dtheta

        # 限制角度变化率
        max_angle_rate = np.radians(45)  # 最大45度/秒
        self._dphi = np.clip(self._dphi, -max_angle_rate, max_angle_rate)
        self._dtheta = np.clip(self._dtheta, -max_angle_rate, max_angle_rate)

        # 速度向量更新
        self._velocity[:] = np.array([
            v_new * np.cos(theta_new) * np.cos(phi_new),
            v_new * np.cos(theta_new) * np.sin(phi_new),
            v_new * np.sin(theta_new)
        ])

        # 姿态更新
        self._posture[:] = np.array([0, theta_new, phi_new])

    def get_missile_status(self) -> Dict[str, Any]:
        """获取导弹状态信息"""
        return {
            "uid": self.uid,
            "status": self.__status,
            "guidance_phase": self.guidance_state["phase"],
            "time_of_flight": self._t,
            "remaining_time": max(0, self._t_max - self._t),
            "velocity": np.linalg.norm(self.get_velocity()),
            "altitude": self.get_position()[2],
            "target_distance": self.target_distance,
            "fuel_remaining": max(0, 1 - self._t / self._t_thrust) if self._t < self._t_thrust else 0,
            "mass": self._m,
            "is_thrusting": self._t < self._t_thrust,
            "target_uid": self.target_aircraft.uid if self.target_aircraft else None,
            "parent_uid": self.parent_aircraft.uid if self.parent_aircraft else None
        }