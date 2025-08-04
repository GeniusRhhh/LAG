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
        # 飞行性能参数
        self.flight_envelope = {
            "max_altitude": 18000,  # 最大高度 (m)
            "service_ceiling": 15000,  # 实用升限 (m)
            "max_speed": 600,  # 最大速度 (m/s)
            "min_speed": 150,  # 最小速度 (m/s)
            "max_g": 9.0,  # 最大过载
            "max_climb_rate": 250,  # 最大爬升率 (m/s)
            "max_turn_rate": 25  # 最大转弯率 (deg/s)
        }
        # 战术控制器
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
            Catalog.ic_u_fps: 1200.0,
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
        """飞机运行逻辑"""
        if self.is_alive:
            if self.bloods <= 0:
                self.shotdown()
                return False

            # 状态检查
            current_alt = self.get_property_value(Catalog.position_h_sl_m)
            current_vel = np.linalg.norm(self.get_velocity())

            # 只在极端情况下crash
            if current_alt < 0:  # 撞地
                logging.error(f"Agent {self.uid} crashed into ground: {current_alt:.1f}m")
                self.crash()
                return False

            if current_vel < 50:  # 极低速度
                logging.error(f"Agent {self.uid} crashed due to low speed: {current_vel:.1f}m/s")
                self.crash()
                return False

            # 运行仿真
            try:
                result = self.jsbsim_exec.run()
                if not result:
                    logging.error(f"JSBSim simulation failed for {self.uid}")
                    self.crash()
                    return False
            except Exception as e:
                logging.error(f"JSBSim error for {self.uid}: {e}")
                self.crash()
                return False

            self._update_properties()

            # 检查
            if self._geodetic[2] < -100:  # 给100m容差
                logging.error(f"Agent {self.uid} crashed: altitude={self._geodetic[2]:.1f}m")
                self.crash()
                return False

            # 飞行包线保护
            self._apply_flight_envelope_protection()

            # 更新战术状态
            self._update_tactical_state()

            return True
        return False

    def _apply_flight_envelope_protection(self):
        """飞行包线保护"""
        current_alt = self.get_position()[2]
        current_vel = np.linalg.norm(self.get_velocity())

        # 放宽过载限制，避免过度限制战术机动
        pitch_rate = self.get_property_value(Catalog.ic_q_rad_sec)
        roll_rate = self.get_property_value(Catalog.ic_p_rad_sec)

        # 宽松角速度限制
        if abs(pitch_rate) > 2.0:  # 从1.0放宽到2.0
            self.set_property_value(Catalog.ic_q_rad_sec, np.clip(pitch_rate, -2.0, 2.0))
            logging.debug(f"Agent {self.uid} pitch rate limited: {pitch_rate:.3f}")

        if abs(roll_rate) > 2.5:  # 从1.5放宽到2.5
            self.set_property_value(Catalog.ic_p_rad_sec, np.clip(roll_rate, -2.5, 2.5))
            logging.debug(f"Agent {self.uid} roll rate limited: {roll_rate:.3f}")

        # 高度保护
        if current_alt < 200:  # 只在极低高度干预
            # 温和上升
            current_elevator = self.get_property_value(Catalog.fcs_elevator_cmd_norm)
            self.set_property_value(Catalog.fcs_elevator_cmd_norm, max(current_elevator, 0.1))

            # 增加推力
            current_throttle = self.get_property_value(Catalog.fcs_throttle_cmd_norm)
            self.set_property_value(Catalog.fcs_throttle_cmd_norm, max(current_throttle, 0.8))

            # logging.warning(f"Agent {self.uid} low altitude protection: {current_alt:.1f}m")

        # 速度保护
        if current_vel < 120:  # 防失速
            current_throttle = self.get_property_value(Catalog.fcs_throttle_cmd_norm)
            self.set_property_value(Catalog.fcs_throttle_cmd_norm, max(current_throttle, 0.9))
            # logging.warning(f"Agent {self.uid} low speed protection: {current_vel:.1f}m/s")

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
            if min_threat_distance < 10000:
                self.tactical_state["threat_level"] = "critical"
            elif min_threat_distance < 25000:
                self.tactical_state["threat_level"] = "high"
            elif min_threat_distance < 40000:
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
        """导弹威胁检测"""
        threatening_missiles = []

        for missile in self.under_missiles:
            if not missile.is_alive:
                continue

            distance = np.linalg.norm(missile.get_position() - self.get_position())
            velocity = np.linalg.norm(missile.get_velocity())

            # 威胁判定条件
            threat_conditions = [
                distance < 40000,  # 40km威胁距离
                velocity > 200,  # 最小威胁速度
                distance < 60000 and velocity > 350  # 或者较远但高速
            ]

            # 计算接近率
            relative_pos = self.get_position() - missile.get_position()
            relative_vel = self.get_velocity() - missile.get_velocity()

            # 如果导弹在远离，不算威胁
            if np.dot(relative_pos, relative_vel) > 0:
                continue

            is_threat = any(threat_conditions)

            if is_threat:
                logging.debug(f"Agent {self.uid} missile threat: {missile.uid}, "
                              f"distance={distance:.1f}m, velocity={velocity:.1f}m/s")
                if not multi:
                    return missile
                threatening_missiles.append(missile)

        return threatening_missiles if multi and threatening_missiles else (
            threatening_missiles[0] if threatening_missiles else None)


class MissileSimulator(BaseSimulator):
    """AIM-120C7 三段制导导弹模拟器"""

    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    # 飞行阶段定义
    BOOST_PHASE = 0  # 助推段
    MIDCOURSE_PHASE = 1  # 中段制导
    TERMINAL_PHASE = 2  # 末段制导

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str, missile_model: str = "AIM-120C7"):
        assert parent.dt == target.dt, "integration timestep must be same!"
        missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)
        missile.target(target)
        return missile

    def __init__(self,
                 uid="A0101",
                 color="Red",
                 model="AIM-120C7",
                 dt=1 / 60):
        super().__init__(uid, color, dt)
        self.__status = MissileSimulator.INACTIVE
        self.model = model
        self.parent_aircraft = None  # type: AircraftSimulator
        self.target_aircraft = None  # type: AircraftSimulator
        self.render_explosion = False
        self.print_interval = 10  # 每10秒打印一次
        # AIM-120C7 导弹参数
        self._g = 9.81  # 重力加速度
        self._t_max = 120  # 导弹最大飞行时间 (增加到120秒)
        self._t_boost = 12  # 助推时间 (增加助推时间以获得更高速度)
        self._t_terminal = 15  # 末段制导开始时间(距离目标)
        self._Isp = 450  # 比冲 (大幅提升到更真实的AIM-120C值)
        self._Length = 3.66  # 长度
        self._Diameter = 0.178  # 直径
        self._cD = 0.35  # 阻力系数
        self._m0 = 152  # 初始质量 kg
        self._dm = 6  # 质量损失率 kg/s (增加推力)
        self._K = 4  # 比例导引系数
        self._nyz_max = 35  # 最大过载 (AIM-120C7典型值)
        self._Rc = 40  # 爆炸半径 m (减小以提高精度要求)
        self._v_min = 150  # 最小速度 m/s (降低最小速度限制)

        # 制导参数
        self._phase = MissileSimulator.BOOST_PHASE
        self._intercept_point = np.zeros(3)  # 预测拦截点
        self._terminal_distance = 8000  # 末段制导启动距离 (调整到6km)

        self._phase_changed = False

    @property
    def is_alive(self):
        """Missile is still flying"""
        return self.__status == MissileSimulator.LAUNCHED

    @property
    def is_success(self):
        """Missile has hit the target"""
        return self.__status == MissileSimulator.HIT

    @property
    def is_done(self):
        """Missile is already exploded"""
        return self.__status == MissileSimulator.HIT \
               or self.__status == MissileSimulator.MISS

    @property
    def Isp(self):
        return self._Isp if self._t < self._t_boost else 0

    @property
    def K(self):
        """比例导引系数 - 末段制导时动态调整"""
        if self._phase == MissileSimulator.TERMINAL_PHASE:
            # 末段制导时增强机动性
            return self._K * 1.5
        return self._K

    @property
    def S(self):
        """横截面积, unit m^2"""
        S0 = np.pi * (self._Diameter / 2) ** 2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        """空气密度, unit: kg/m^3"""
        h = self._geodetic[-1]
        if h <= 11000:  # 对流层
            T = 288.15 - 0.0065 * h
            return 1.225 * (T / 288.15) ** 4.25588
        elif h <= 20000:  # 平流层下部
            return 0.36392 * np.exp((11000 - h) / 6341.62)
        else:  # 平流层上部
            T = 216.65 + 0.001 * (h - 20000)
            return 0.088035 * (T / 216.65) ** (-35.1632)

    @property
    def target_distance(self) -> float:
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, parent: AircraftSimulator):
        # 继承发射平台的运动参数
        self.parent_aircraft = parent
        self.parent_aircraft.launch_missiles.append(self)
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0  # 导弹滚转角保持为零
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0

        # 初始化状态
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self.__status = MissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(10 / self.dt))  # 10s距离增量检查
        self._left_t = int(1 / self.dt)
        self._phase = MissileSimulator.BOOST_PHASE
        self._phase_changed = False

        # 速度补偿：如果发射速度过低，给予一定的初始速度补偿
        current_velocity = np.linalg.norm(self._velocity)
        if current_velocity < 300:  # 如果发射速度低于300m/s
            # 计算需要的速度补偿，确保导弹有足够的初始速度
            velocity_boost = 400 - current_velocity  # 补偿到400m/s
            # 在发射方向上增加速度
            velocity_direction = self._velocity / current_velocity if current_velocity > 0 else np.array([1, 0, 0])
            self._velocity[:] += velocity_direction * velocity_boost
            print(f" {self.model} {self.uid} velocity compensated: {current_velocity:.1f} -> {np.linalg.norm(self._velocity):.1f}m/s")

        print(
            f" {self.model} {self.uid} launched: v={np.linalg.norm(self._velocity):.1f}m/s, alt={self._geodetic[2]:.0f}m")

    def target(self, target: AircraftSimulator):
        self.target_aircraft = target
        self.target_aircraft.under_missiles.append(self)

    def run(self):
        self._t += self.dt

        # 阶段转换逻辑
        self._update_phase()

        # 根据阶段选择制导律
        action, distance = self._guidance()

        # 距离变化监控
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance

        # 命中判定
        if distance < self._Rc and self.target_aircraft.is_alive:
            self.__status = MissileSimulator.HIT
            self.target_aircraft.shotdown()
            if self._t % self.print_interval < self.dt:  # 只在特定间隔打印
                print(f" {self.model} {self.uid} HIT target at t={self._t:.1f}s, dist={distance:.1f}m")
        elif self._should_miss():
            self.__status = MissileSimulator.MISS
            miss_reason = self._get_miss_reason()
            if self._t % self.print_interval < self.dt:  # 只在特定间隔打印
                print(
                    f" {self.model} {self.uid} missed: {miss_reason}, t={self._t:.1f}s, v={np.linalg.norm(self.get_velocity()):.0f}m/s, dist={distance:.0f}m")
        else:
            self._state_trans(action)

    def _update_phase(self):
        """更新飞行阶段"""
        old_phase = self._phase
        distance = self.target_distance

        if self._t <= self._t_boost:
            self._phase = MissileSimulator.BOOST_PHASE
        elif distance <= self._terminal_distance:
            self._phase = MissileSimulator.TERMINAL_PHASE
        else:
            self._phase = MissileSimulator.MIDCOURSE_PHASE
        #打印
        if old_phase != self._phase and not self._phase_changed:
            phase_names = {0: "boost", 1: "midcourse", 2: "terminal"}
            old_name = phase_names.get(old_phase, "unknown")
            new_name = phase_names.get(self._phase, "unknown")
            print(f"{self.model} {self.uid}: {old_name} -> {new_name} at t={self._t:.1f}s")
            self._phase_changed = True

    def _should_miss(self):
        """判断导弹是否应该失效"""
        distance = self.target_distance
        velocity = np.linalg.norm(self.get_velocity())

        # 时间超限
        if self._t > self._t_max:
            return True

        # 速度过低
        if velocity < self._v_min:
            return True

        # 目标已死亡
        if not self.target_aircraft.is_alive:
            return True

        # 距离持续增大(发散检测)
        if len(self._distance_increment) >= self._distance_increment.maxlen:
            diverging_count = sum(self._distance_increment)
            if diverging_count >= self._distance_increment.maxlen * 0.6:  # 降低到60%的时间在远离
                return True

        return False

    def _get_miss_reason(self):
        """获取失效原因"""
        distance = self.target_distance
        velocity = np.linalg.norm(self.get_velocity())

        if self._t > self._t_max:
            return "timeout"
        elif velocity < self._v_min:
            return "low_velocity"
        elif not self.target_aircraft.is_alive:
            return "target_dead"
        elif len(self._distance_increment) >= self._distance_increment.maxlen:
            diverging_count = sum(self._distance_increment)
            if diverging_count >= self._distance_increment.maxlen * 0.6:
                return "diverging"
        return "unknown"

    def _guidance(self):
        """三段制导律"""
        distance = self.target_distance

        if self._phase == MissileSimulator.BOOST_PHASE:
            return self._boost_guidance(), distance
        elif self._phase == MissileSimulator.MIDCOURSE_PHASE:
            return self._midcourse_guidance(), distance
        else:  # TERMINAL_PHASE
            return self._terminal_guidance(), distance

    def _boost_guidance(self):
        """助推段制导 - 改进的初始指向和能量管理"""
        # 计算目标方向
        target_pos = self.target_aircraft.get_position()
        missile_pos = self.get_position()
        direction = target_pos - missile_pos
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1:
            return np.array([0, 0])

        # 计算期望的俯仰角和偏航角
        direction_unit = direction / direction_norm
        target_pitch = np.arcsin(direction_unit[2])
        target_yaw = np.arctan2(direction_unit[1], direction_unit[0])

        # 当前姿态
        current_pitch = self._posture[1]
        current_yaw = self._posture[2]

        # 角度误差
        pitch_error = target_pitch - current_pitch
        yaw_error = target_yaw - current_yaw

        # 角度归一化
        if yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        elif yaw_error < -np.pi:
            yaw_error += 2 * np.pi

        # 改进的比例控制，考虑能量管理和速度状态
        current_velocity = np.linalg.norm(self.get_velocity())
        
        # 根据速度调整控制参数
        if current_velocity < 400:  # 低速时使用更温和的控制
            k_p = 2.0  # 降低比例增益
            k_d = 0.3  # 降低微分增益
            max_overload = self._nyz_max * 0.5  # 限制过载
        else:
            k_p = 4.0  # 正常比例增益
            k_d = 0.5  # 正常微分增益
            max_overload = self._nyz_max * 0.8  # 正常过载限制
        
        # 计算角速度误差
        pitch_rate_error = 0 - self._dtheta  # 期望角速度为0
        yaw_rate_error = 0 - self._dphi
        
        ny = k_p * yaw_error + k_d * yaw_rate_error
        nz = k_p * pitch_error + k_d * pitch_rate_error + np.cos(current_pitch)  # 保持升力平衡

        # 限制过载，避免过度机动
        return np.clip([ny, nz], -max_overload, max_overload)

    def _midcourse_guidance(self):
        """中段制导 - 预测拦截制导"""
        # 计算预测拦截点
        intercept_point = self._calculate_intercept_point()

        # 计算到拦截点的方向
        missile_pos = self.get_position()
        direction = intercept_point - missile_pos
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1:
            return np.array([0, 0])

        # 计算期望速度方向
        direction_unit = direction / direction_norm
        target_pitch = np.arcsin(direction_unit[2])
        target_yaw = np.arctan2(direction_unit[1], direction_unit[0])

        # 当前姿态
        current_pitch = self._posture[1]
        current_yaw = self._posture[2]

        # 角度误差
        pitch_error = target_pitch - current_pitch
        yaw_error = target_yaw - current_yaw

        # 角度归一化
        if yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        elif yaw_error < -np.pi:
            yaw_error += 2 * np.pi

        # 中段制导的比例控制(较温和)
        k_p = 3.0
        ny = k_p * yaw_error
        nz = k_p * pitch_error + np.cos(current_pitch)  # 保持升力平衡

        return np.clip([ny, nz], -self._nyz_max * 0.7, self._nyz_max * 0.7)  # 中段制导限制过载

    def _terminal_guidance(self):
        """末段制导 -比例导引"""
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])

        if v_m < 1:
            return np.array([0, 0])

        theta_m = np.arcsin(dz_m / v_m)

        x_t, y_t, z_t = self.target_aircraft.get_position()
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()

        # 相对位置和距离
        rel_x, rel_y, rel_z = x_t - x_m, y_t - y_m, z_t - z_m
        Rxy = np.linalg.norm([rel_x, rel_y])
        Rxyz = np.linalg.norm([rel_x, rel_y, rel_z])

        if Rxyz < 1 or Rxy < 1:
            return np.array([0, 0])

        # 相对速度
        rel_dx, rel_dy, rel_dz = dx_t - dx_m, dy_t - dy_m, dz_t - dz_m

        # 视线角速率计算
        dbeta = (rel_dy * rel_x - rel_dx * rel_y) / Rxy ** 2
        deps = (rel_dz * Rxy ** 2 - rel_z * (rel_x * rel_dx + rel_y * rel_dy)) / (Rxyz ** 2 * Rxy)

        # 比例导引律
        ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
        nz = self.K * v_m / self._g * deps + np.cos(theta_m)

        # 添加制导噪声模拟
        guidance_noise = 0.1  # 制导噪声系数
        noise_ny = np.random.normal(0, guidance_noise)
        noise_nz = np.random.normal(0, guidance_noise)
        
        ny += noise_ny
        nz += noise_nz

        return np.clip([ny, nz], -self._nyz_max, self._nyz_max)

    def _calculate_intercept_point(self):
        """计算预测拦截点 - 改进算法考虑目标机动性"""
        missile_pos = self.get_position()
        missile_vel = self.get_velocity()
        target_pos = self.target_aircraft.get_position()
        target_vel = self.target_aircraft.get_velocity()

        # 相对位置和速度
        rel_pos = target_pos - missile_pos
        rel_vel = target_vel - missile_vel

        # 预测时间计算
        missile_speed = np.linalg.norm(missile_vel)
        target_speed = np.linalg.norm(target_vel)
        
        if missile_speed < 1 or target_speed < 1:
            return target_pos

        # 计算接近速度
        closing_velocity = np.dot(rel_pos, rel_vel) / np.linalg.norm(rel_pos)
        
        # 考虑目标机动性的预测时间
        distance = np.linalg.norm(rel_pos)
        
        # 如果目标在远离，使用更保守的预测
        if closing_velocity < 0:
            t_intercept = distance / (missile_speed * 0.8)  # 假设目标会机动
        else:
            # 目标在接近，使用更乐观的预测
            t_intercept = distance / (missile_speed + target_speed * 0.5)

        # 限制预测时间范围
        t_intercept = np.clip(t_intercept, 2.0, 45.0)  # 2-45秒范围

        # 计算预测拦截点，考虑目标可能的机动
        intercept_point = target_pos + target_vel * t_intercept

        return intercept_point

    def _state_trans(self, action):
        """状态转换函数"""
        # 更新位置
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)

        # 当前速度和姿态
        v = np.linalg.norm(self.get_velocity())
        theta, phi = self.get_rpy()[1:]

        # 推力和阻力
        T = self._g * self.Isp * self._dm if self._t < self._t_boost else 0
        D = 0.5 * self._cD * self.S * self.rho * v ** 2

        # 轴向过载
        nx = (T - D) / (self._m * self._g) if self._m > 0 else 0
        ny, nz = action

        # 速度变化
        dv = self._g * (nx - np.sin(theta))

        # 角速度
        if v > 1:
            self._dphi = self._g / v * (ny / np.cos(theta)) if abs(np.cos(theta)) > 0.1 else 0
            self._dtheta = self._g / v * (nz - np.cos(theta))
        else:
            self._dphi = 0
            self._dtheta = 0

        # 更新速度和姿态
        v = max(v + self.dt * dv, 0)
        phi += self.dt * self._dphi
        theta += self.dt * self._dtheta

        # 限制姿态角
        theta = np.clip(theta, -np.pi / 2 + 0.1, np.pi / 2 - 0.1)

        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta)
        ])
        self._posture[:] = np.array([0, theta, phi])

        # 更新质量
        if self._t < self._t_boost:
            self._m = max(self._m - self.dt * self._dm, self._m0 * 0.3)  # 保留30%质量

    def log(self):
        if self.is_alive:
            log_msg = super().log()
        elif self.is_done and (not self.render_explosion):
            self.render_explosion = True
            # 移除导弹模型
            log_msg = f"-{self.uid}\n"
            # 添加爆炸效果
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
        else:
            log_msg = None
        return log_msg

    def close(self):
        self.target_aircraft = None