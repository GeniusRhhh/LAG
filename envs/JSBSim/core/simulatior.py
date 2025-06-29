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
        self.sensors = {
            "radar_range": 120000,  # 雷达探测距离 (m)
            "radar_lock_range": 80000,  # 雷达锁定距离 (m)
            "iff_range": 150000,  # 敌我识别距离 (m)
            "rwr_range": 200000  # 雷达告警距离 (m)
        }
        self.weapons = {
            "missiles": {
                "aim120": {"count": self.num_missiles, "range": 100000, "speed": 1200},
            }
        }
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

            # 只在运行前检查最基本的状态
            current_alt = self.get_property_value(Catalog.position_h_sl_m)
            if current_alt < 100:  # 低于100米
                logging.error(f"Agent {self.uid} too low: {current_alt:.1f}m")
                self.crash()
                return False

            # 运行仿真
            result = self.jsbsim_exec.run()
            if not result:
                logging.error("JSBSim simulation failed")
                raise RuntimeError("JSBSim failed.")

            self._update_properties()

            # 运行后检查
            if self._geodetic[2] < 0:
                logging.error(f"Agent {self.uid} crashed into ground")
                self.crash()
                return False

            # 飞行包线保护
            self._apply_flight_envelope_protection()

            # 更新战术状态
            self._update_tactical_state()

            return result
        return False

    def _apply_flight_envelope_protection(self):
        """应用飞行包线保护"""
        current_alt = self.get_position()[2]
        current_vel = np.linalg.norm(self.get_velocity())

        # 过载保护
        pitch_rate = self.get_property_value(Catalog.ic_q_rad_sec)
        roll_rate = self.get_property_value(Catalog.ic_p_rad_sec)

        if abs(pitch_rate) > 1.0:  # 放宽限制以适应战术机动
            self.set_property_value(Catalog.ic_q_rad_sec, np.clip(pitch_rate, -1.0, 1.0))
            logging.debug(f"Agent {self.uid} pitch rate limited: {pitch_rate:.3f}")

        if abs(roll_rate) > 1.5:  # 放宽限制
            self.set_property_value(Catalog.ic_p_rad_sec, np.clip(roll_rate, -1.5, 1.5))
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
    """增强版导弹模拟器 - 更准确的AIM-120C7建模"""

    # 类常量定义
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: 'AircraftSimulator', target: 'AircraftSimulator', uid: str, missile_model: str = "AIM-120C7"):
        """创建导弹实例并初始化"""
        assert parent.dt == target.dt, "Integration timestep must be same!"
        missile = cls(uid, parent.color, missile_model, parent.dt)
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

        # 物理参数
        self._g = 9.81
        self._t_max = 120  # 最大飞行时间
        self._t_thrust = 15  # 推力时间
        self._Isp = 250  # 比冲
        self._Length = 3.66
        self._Diameter = 0.18
        self._cD = 0.35  # 阻力系数
        self._m0 = 150  # 初始质量
        self._dm = 3.0  # 燃料消耗率
        self._nyz_max = 30  # 最大过载
        self._Rc = 10  # 毁伤半径
        self._v_min = 180  # 最小速度

        # 制导参数
        self.guidance_params = {
            "boost": {"K": 2.0, "max_accel": 15},
            "midcourse": {"K": 4.0, "max_accel": 25},
            "terminal": {"K": 6.0, "max_accel": 30}
        }

        # 初始化状态
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(12 / self.dt))
        self._left_t = int(1 / self.dt)
        self._target_pos_history = deque(maxlen=12)
        self._distance_traveled = 0  # 新增：累积飞行距离

        # 制导状态
        self.guidance_state = {
            "phase": "inactive",
            "lock_time": 0,
            "target_acquired": False,
            "intercept_point": np.zeros(3),
            "seeker_fov": np.radians(20),
            "seeker_range": 15000,
            "last_guidance_update": 0,
            "target_lost_time": 0,
            "inertial_nav_active": True
        }

        # 预测参数
        self.prediction_params = {
            "prediction_time": 5.0,
            "update_frequency": 0.1,
            "noise_factor": 0.02
        }

        logging.info(f"Enhanced MissileSimulator {uid}: thrust_time={self._t_thrust}s, "
                     f"max_range={self._t_max * 400:.0f}m, max_g={self._nyz_max}, "
                     f"seeker_range={self.guidance_state['seeker_range']}m")

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
    def S(self):
        """迎风面积"""
        S0 = np.pi * (self._Diameter / 2) ** 2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        """空气密度"""
        return 1.225 * np.exp(-self._geodetic[2] / 9300)  # 假设_geodetic[2]是高度

    def launch(self, parent: 'AircraftSimulator'):
        """导弹发射 - 设置初始条件"""
        try:
            self.parent_aircraft = parent
            parent.launch_missiles.append(self)

            self._geodetic[:] = parent.get_geodetic()  # 确保顺序与NEU2LLA一致
            self._position[:] = parent.get_position()

            parent_rpy = parent.get_rpy()
            parent_velocity = parent.get_velocity()
            parent_speed = np.linalg.norm(parent_velocity)

            if parent_speed < 100:
                logging.warning(f"Parent aircraft {parent.uid} speed too low: {parent_speed:.1f}m/s")
                parent_velocity = np.array([250.0, 0.0, 0.0])
                parent_speed = 250.0

            ejection_speed = 20
            launch_direction = parent_velocity / parent_speed
            self._velocity[:] = parent_velocity + ejection_speed * launch_direction

            self._posture[0] = 0
            self._posture[1] = parent_rpy[1]
            self._posture[2] = parent_rpy[2]

            self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
            self._t = 0
            self._m = self._m0
            self._dtheta, self._dphi = 0, 0
            self.__status = MissileSimulator.LAUNCHED
            self._distance_pre = np.inf
            self._distance_increment = deque(maxlen=int(12 / self.dt))
            self._left_t = int(1 / self.dt)
            self._distance_traveled = 0

            self.guidance_state = {
                "phase": "boost",
                "lock_time": 0,
                "target_acquired": False,
                "intercept_point": self.target_aircraft.get_position() if self.target_aircraft else np.zeros(3),
                "seeker_fov": np.radians(20),
                "seeker_range": 15000,
                "last_guidance_update": 0,
                "target_lost_time": 0,
                "inertial_nav_active": True
            }

            if self.target_aircraft and self.target_aircraft.is_alive:
                self._update_intercept_point()

            logging.info(f"🚀 Missile {self.uid} launched: "
                         f"pos={self._position}, vel={np.linalg.norm(self._velocity):.1f}m/s, "
                         f"heading={np.rad2deg(self._posture[2]):.1f}deg, "
                         f"target={self.target_aircraft.uid if self.target_aircraft else 'None'}")

        except Exception as e:
            logging.error(f"Error launching missile {self.uid}: {e}")
            self.__status = MissileSimulator.MISS

    def target(self, target: 'AircraftSimulator'):
        """设置目标"""
        self.target_aircraft = target
        self.target_aircraft.under_missiles.append(self)
        logging.info(f"Missile {self.uid} targeting {target.uid}")

    def run(self):
        """运行导弹模拟"""
        if not self.is_alive:
            return

        self._t += self.dt
        self._update_guidance_phase()

        if self._t % (0.1 / self.dt) == 0:
            self._update_intercept_point()

        action, distance = self._enhanced_guidance()
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance

        if distance < self._Rc and self.target_aircraft and self.target_aircraft.is_alive:
            relative_vel = self.get_velocity() - self.target_aircraft.get_velocity()
            relative_pos = self.target_aircraft.get_position() - self.get_position()
            if np.dot(relative_vel, relative_pos) > 0 or distance < self._Rc * 0.5:
                self.__status = MissileSimulator.HIT
                self.target_aircraft.shotdown()
                self.guidance_state["phase"] = "hit"
                logging.info(f"🎯 Missile {self.uid} HIT target {self.target_aircraft.uid}, "
                             f"distance={distance:.1f}m, time={self._t:.1f}s")
                return

        miss_conditions = [
            self._t > self._t_max,
            (self._t > self._t_thrust and np.linalg.norm(self.get_velocity()) < self._v_min),
            np.sum(self._distance_increment) >= self._distance_increment.maxlen * 0.8,
            not self.target_aircraft or not self.target_aircraft.is_alive,
            distance > 200000,
            self.get_position()[2] < 0
        ]

        if any(miss_conditions):
            self.__status = MissileSimulator.MISS
            reasons = ["timeout", "low_velocity", "diverging", "target_dead", "out_of_range", "ground_impact"]
            reason = reasons[miss_conditions.index(True)]
            self.guidance_state["phase"] = "miss"
            logging.info(f"💥 Missile {self.uid} missed: reason={reason}, "
                         f"distance={distance:.1f}m, time={self._t:.1f}s, "
                         f"velocity={np.linalg.norm(self.get_velocity()):.1f}m/s")
            return

        self._enhanced_state_trans(action)

        if self._t % (5 / self.dt) == 0:
            logging.debug(f"🚀 Missile {self.uid}: phase={self.guidance_state['phase']}, "
                          f"distance={distance:.1f}m, velocity={np.linalg.norm(self._velocity):.1f}m/s, "
                          f"time={self._t:.1f}s, accel={np.linalg.norm(action):.2f}G")

    def _update_guidance_phase(self):
        """更新制导阶段"""
        if self._t <= self._t_thrust:
            self.guidance_state["phase"] = "boost"
        elif self._t <= self._t_max * 0.7:
            self.guidance_state["phase"] = "midcourse"
        else:
            self.guidance_state["phase"] = "terminal"

    def _update_intercept_point(self):
        """计算拦截点"""
        if not self.target_aircraft or not self.target_aircraft.is_alive:
            return

        try:
            target_pos = self.target_aircraft.get_position()
            target_vel = self.target_aircraft.get_velocity()
            missile_pos = self.get_position()

            relative_pos = target_pos - missile_pos
            distance = np.linalg.norm(relative_pos)

            if distance < 1e-6:
                return

            missile_speed = np.linalg.norm(self.get_velocity())
            if missile_speed < 100:
                missile_speed = 400

            intercept_time = distance / (missile_speed + 50)
            predicted_target_pos = target_pos + target_vel * intercept_time
            self.guidance_state["intercept_point"] = predicted_target_pos

        except Exception as e:
            logging.error(f"Error updating intercept point: {e}")

    def _enhanced_guidance(self):
        """增强制导算法"""
        if not self.target_aircraft or not self.target_aircraft.is_alive:
            return np.array([0, 0]), np.inf

        current_params = self.guidance_params[self.guidance_state["phase"]]
        K = current_params["K"]
        max_accel = current_params["max_accel"]

        missile_pos = self.get_position()
        missile_vel = self.get_velocity()
        v_m = np.linalg.norm(missile_vel)

        if v_m < 1e-6:
            v_m = 1e-6

        target_pos = self.target_aircraft.get_position()
        target_vel = self.target_aircraft.get_velocity()

        self._target_pos_history.append(target_pos.copy())
        a_t = np.zeros(3)

        if len(self._target_pos_history) >= 3:
            positions = list(self._target_pos_history)[-3:]
            v_t_2 = (positions[2] - positions[1]) / self.dt
            v_t_1 = (positions[1] - positions[0]) / self.dt
            a_t = (v_t_2 - v_t_1) / self.dt

        if self.guidance_state["phase"] == "terminal":
            aim_point = target_pos
        else:
            aim_point = self.guidance_state.get("intercept_point", target_pos)

        r = aim_point - missile_pos
        v_r = target_vel - missile_vel
        R = np.linalg.norm(r)

        if R < 1e-6:
            return np.array([0, 0]), R

        r_unit = r / R
        h = np.cross(r, v_r)
        omega_los = h / (R * R)

        omega_y = omega_los[1]
        omega_z = omega_los[2]

        v_c = -np.dot(r, v_r) / R

        ny_pn = K * v_c * omega_y
        nz_pn = K * v_c * omega_z

        ny_apn = 0.5 * K * a_t[1]
        nz_apn = 0.5 * K * a_t[2]

        missile_pitch = self.get_rpy()[1]
        g_comp = self._g * np.cos(missile_pitch) if abs(missile_pitch) < np.pi / 2 else self._g

        if self.guidance_state["phase"] == "boost":
            ny = ny_pn * 0.3
            nz = nz_pn * 0.3 + g_comp / self._g
        elif self.guidance_state["phase"] == "midcourse":
            ny = ny_pn + ny_apn * 0.7
            nz = nz_pn + nz_apn * 0.7 + g_comp / self._g
        else:
            ny = (ny_pn + ny_apn) * 1.3
            nz = (nz_pn + nz_apn) * 1.3 + g_comp / self._g

        total_n = np.sqrt(ny ** 2 + nz ** 2)
        if total_n > max_accel:
            scale = max_accel / total_n
            ny *= scale
            nz *= scale

        if self.guidance_state["phase"] == "terminal":
            los_angle = np.arccos(np.clip(np.dot(r_unit, missile_vel / v_m), -1, 1))
            if los_angle > self.guidance_state["seeker_fov"]:
                ny *= 0.3
                nz *= 0.3
                self.guidance_state["target_lost_time"] += self.dt

        return np.clip([ny, nz], -max_accel, max_accel), R

    def _enhanced_state_trans(self, action):
        """状态转换"""
        velocity_step = self.get_velocity()
        self._position[:] += self.dt * velocity_step
        self._geodetic[:] = NEU2LLA(*self._position, self.lon0, self.lat0, self.alt0)
        self._distance_traveled += np.linalg.norm(self.dt * velocity_step)  # 更新飞行距离

        if self._t < self._t_thrust:
            self._m = max(self._m - self.dt * self._dm, self._m0 * 0.2)
        else:
            self._m = max(self._m, self._m0 * 0.2)

        if self._t < self._t_thrust:
            altitude_factor = max(0.3, 1 - self._position[2] / 20000)
            T = self._g * self._Isp * self._dm * altitude_factor
        else:
            T = 0

        v = np.linalg.norm(velocity_step)
        v = max(v, 1e-6)

        mach = v / 340
        if mach < 0.8:
            cd_factor = 1.0
        elif mach < 1.2:
            cd_factor = 1.5
        else:
            cd_factor = 1.2

        D = 0.5 * self._cD * cd_factor * self.S * self.rho * v ** 2

        theta, phi = self.get_rpy()[1:]
        nx = (T - D) / (self._m * self._g)

        ny, nz = action

        dv = self._g * (nx - np.sin(theta))
        v_new = v + self.dt * dv
        v_new = max(v_new, 50)

        dynamic_pressure = 0.5 * self.rho * v_new ** 2
        if dynamic_pressure > 1e-6:
            control_effectiveness = min(1.0, dynamic_pressure / 10000)
        else:
            control_effectiveness = 0.1

        self._dphi = self._g / v_new * (ny * control_effectiveness / max(np.cos(theta), 0.1))
        self._dtheta = self._g / v_new * (nz * control_effectiveness - np.cos(theta))

        phi_new = phi + self.dt * self._dphi
        theta_new = theta + self.dt * self._dtheta

        max_angle_rate = np.radians(60)
        self._dphi = np.clip(self._dphi, -max_angle_rate, max_angle_rate)
        self._dtheta = np.clip(self._dtheta, -max_angle_rate, max_angle_rate)

        self._velocity[:] = np.array([
            v_new * np.cos(theta_new) * np.cos(phi_new),
            v_new * np.cos(theta_new) * np.sin(phi_new),
            v_new * np.sin(theta_new)
        ])

        self._posture[:] = np.array([0, theta_new, phi_new])

        if self._position[2] < 0:
            self._position[2] = 0
            self._velocity[2] = max(0, self._velocity[2])

    def log(self):
        """导弹日志输出"""
        if self.is_alive:
            try:
                lon, lat, alt = self.get_geodetic()
                roll, pitch, yaw = self.get_rpy() * 180 / np.pi

                if np.any(np.isnan([lon, lat, alt, roll, pitch, yaw])):
                    logging.warning(f"NaN in missile {self.uid} state")
                    return None

                log_msg = f"{self.uid},T={lon:.6f}|{lat:.6f}|{alt:.1f}|{roll:.1f}|{pitch:.1f}|{yaw:.1f},"
                log_msg += f"Name={self.model.upper()},Color={self.color},"
                log_msg += "Type=Weapon+Missile"

                if self.parent_aircraft:
                    log_msg += f",Parent={self.parent_aircraft.uid}"

                return log_msg

            except Exception as e:
                logging.error(f"Error in missile log for {self.uid}: {e}")
                return None

        elif self.is_done and not self.render_explosion:
            self.render_explosion = True
            try:
                lon, lat, alt = self.get_geodetic()

                explosion_msg = f"-{self.uid}\n"

                if self.is_success:
                    explosion_msg += f"{self.uid}F,T={lon:.6f}|{lat:.6f}|{alt:.1f}|0|0|0,"
                    explosion_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc * 2}"
                else:
                    explosion_msg += f"{self.uid}F,T={lon:.6f}|{lat:.6f}|{alt:.1f}|0|0|0,"
                    explosion_msg += f"Type=Misc+Explosion,Color=Gray,Radius={self._Rc}"

                return explosion_msg

            except Exception as e:
                logging.error(f"Error in explosion log for {self.uid}: {e}")
                return None

        return None

    def close(self):
        """清理导弹与飞机之间的引用"""
        if self.target_aircraft and self in self.target_aircraft.under_missiles:
            self.target_aircraft.under_missiles.remove(self)
        if self.parent_aircraft and self in self.parent_aircraft.launch_missiles:
            self.parent_aircraft.launch_missiles.remove(self)
        self.target_aircraft = None
        self.parent_aircraft = None
        logging.info(f"Missile {self.uid} simulator closed")

    @property
    def target_distance(self) -> float:
        if self.target_aircraft and self.target_aircraft.is_alive:
            return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())
        return np.inf

    def get_missile_info(self) -> Dict[str, Any]:
        """获取导弹状态和性能统计"""
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
            "parent_uid": self.parent_aircraft.uid if self.parent_aircraft else None,
            "distance_traveled": self._distance_traveled,  # 使用累积距离
            "current_speed": np.linalg.norm(self.get_velocity()),
            "max_g_achieved": self._nyz_max,
            "mass_remaining": self._m / self._m0
        }