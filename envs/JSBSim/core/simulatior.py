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
    """修复版AIM-120C7 - 回归比例导引法本质"""

    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: 'AircraftSimulator', target: 'AircraftSimulator', uid: str,
               missile_model: str = "AIM-120C7"):
        """创建导弹实例 - 确保目标正确设置"""
        assert parent.dt == target.dt, "Integration timestep must be same!"
        missile = cls(uid, parent.color, missile_model, parent.dt)
        # 关键修复：先设置目标再发射
        missile.target(target)
        missile.launch(parent)
        return missile

    def __init__(self, uid="A0101", color="Red", model="AIM-120C7", dt=1 / 12):
        super().__init__(uid, color, dt)
        self.__status = MissileSimulator.INACTIVE
        self.model = model
        self.parent_aircraft = None
        self.target_aircraft = None
        self.render_explosion = False

        # AIM-120C7参数 - 基于真实数据
        self._g = 9.81
        self._t_max = 100
        self._t_thrust = 8
        self._Isp = 250
        self._Length = 3.66
        self._Diameter = 0.18
        self._cD = 0.35
        self._m0 = 152
        self._dm = 5.0
        self._nyz_max = 25  # 降低过载，提高稳定性
        self._Rc = 15
        self._v_min = 120

        # **核心修复：统一使用比例导引法**
        self._K_midcourse = 3.0  # 中段制导增益
        self._K_terminal = 4.0  # 终段制导增益
        self._seeker_range = 20000  # 主动雷达搜索距离

        # 初始化状态
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(12 / self.dt))  # 12秒发散检查

        # 制导状态
        self.current_phase = "inactive"
        self.seeker_active = False
        self.guidance_history = []

        # **目标状态跟踪（模拟数据链）**
        self.target_last_pos = np.zeros(3)
        self.target_last_vel = np.zeros(3)
        self.target_update_time = 0

        logging.info(
            f"AIM-120C7 {uid}: unified proportional navigation, K_mid={self._K_midcourse}, K_term={self._K_terminal}")

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
        return np.pi * (self._Diameter / 2) ** 2

    @property
    def rho(self):
        altitude = max(self._geodetic[2], 0)
        return 1.225 * np.exp(-altitude / 9300)

    def launch(self, parent: 'AircraftSimulator'):
        """导弹发射 - 确保目标信息正确"""
        if not self.target_aircraft:
            logging.error(f"❌ Missile {self.uid}: No target set before launch!")
            return

        self.parent_aircraft = parent
        parent.launch_missiles.append(self)

        # 继承父机状态
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0

        # 初始化导弹状态
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self.__status = MissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment.clear()

        # 设置初始制导阶段
        self.current_phase = "midcourse"

        # 初始化目标跟踪
        self.target_last_pos = self.target_aircraft.get_position()
        self.target_last_vel = self.target_aircraft.get_velocity()
        self.target_update_time = 0

        logging.info(f"🚀 AIM-120C7 {self.uid} launched: target={self.target_aircraft.uid}, "
                     f"initial_distance={self.target_distance:.0f}m, phase={self.current_phase}")

    def target(self, target: 'AircraftSimulator'):
        """设置目标"""
        self.target_aircraft = target
        target.under_missiles.append(self)
        logging.debug(f"Missile {self.uid} targeting {target.uid}")

    def run(self):
        """运行导弹模拟"""
        if not self.is_alive:
            return

        self._t += self.dt
        self._update_guidance_phase()
        self._update_target_info()  # 模拟数据链更新

        action, distance = self._unified_proportional_navigation()

        # 记录距离变化
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance

        # 命中判定
        if distance < self._Rc and self.target_aircraft and self.target_aircraft.is_alive:
            # 确保真正的拦截
            relative_vel = self.get_velocity() - self.target_aircraft.get_velocity()
            relative_pos = self.target_aircraft.get_position() - self.get_position()

            if np.dot(relative_vel, relative_pos) > 0 or distance < self._Rc * 0.7:
                self.__status = MissileSimulator.HIT
                self.target_aircraft.shotdown()
                logging.info(f"🎯 AIM-120C7 {self.uid} HIT target {self.target_aircraft.uid}! "
                             f"distance={distance:.1f}m, time={self._t:.1f}s, phase={self.current_phase}")
                return

        # 失效条件
        miss_conditions = [
            self._t > self._t_max,  # 超时
            (self._t > self._t_thrust and np.linalg.norm(self.get_velocity()) < self._v_min),  # 速度过低
            (len(self._distance_increment) >= self._distance_increment.maxlen and
             np.sum(self._distance_increment) >= len(self._distance_increment) * 0.9),  # 持续发散
            not self.target_aircraft or not self.target_aircraft.is_alive,  # 目标死亡
            self.get_position()[2] < -20,  # 撞地
            distance > 200000  # 距离过远
        ]

        if any(miss_conditions):
            self.__status = MissileSimulator.MISS
            reasons = ["timeout", "low_velocity", "diverging", "target_dead", "ground_impact", "out_of_range"]
            reason = reasons[miss_conditions.index(True)]

            logging.info(f"💥 AIM-120C7 {self.uid} missed: {reason}, "
                         f"dist={distance:.0f}m, t={self._t:.1f}s, "
                         f"v={np.linalg.norm(self.get_velocity()):.1f}m/s, "
                         f"alt={self.get_position()[2]:.1f}m, phase={self.current_phase}")
            return

        # 状态转移
        self._state_trans(action)

        # 定期日志
        if int(self._t * 5) % 25 == 0:  # 每5秒
            self._log_status(distance)

    def _update_guidance_phase(self):
        """更新制导阶段"""
        if not self.target_aircraft or not self.target_aircraft.is_alive:
            return

        distance = np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

        # 简化的二阶段切换
        if distance <= self._seeker_range and self._t > 10:  # 确保稳定飞行后再切换
            if self.current_phase != "terminal":
                logging.info(f"AIM-120C7 {self.uid}: midcourse -> terminal at t={self._t:.1f}s, dist={distance:.0f}m")
                self.seeker_active = True
            self.current_phase = "terminal"
        else:
            self.current_phase = "midcourse"

    def _update_target_info(self):
        """模拟数据链更新目标信息"""
        if not self.target_aircraft or not self.target_aircraft.is_alive:
            return

        # 每2秒更新一次（模拟数据链）
        if self._t - self.target_update_time >= 2.0:
            self.target_last_pos = self.target_aircraft.get_position()
            self.target_last_vel = self.target_aircraft.get_velocity()
            self.target_update_time = self._t

    def _unified_proportional_navigation(self):
        """统一的比例导引法 - 核心制导算法"""
        if not self.target_aircraft or not self.target_aircraft.is_alive:
            return np.array([0, 0]), np.inf

        # **关键修复：统一使用比例导引法，就像AIM-9L一样**

        # 获取目标信息
        if self.current_phase == "terminal" and self.seeker_active:
            # 终段：直接追踪目标（类似AIM-9L）
            target_pos = self.target_aircraft.get_position()
            target_vel = self.target_aircraft.get_velocity()
            K = self._K_terminal
        else:
            # 中段：使用数据链信息（避免追踪机动目标的延迟）
            target_pos = self.target_last_pos
            target_vel = self.target_last_vel
            K = self._K_midcourse

        # **使用与AIM-9L相同的比例导引法公式**
        missile_pos = self.get_position()
        missile_vel = self.get_velocity()

        # 相对位置和速度
        r = target_pos - missile_pos
        v_r = target_vel - missile_vel
        distance = np.linalg.norm(r)

        if distance < 1e-6:
            return np.array([0, 0]), distance

        v_m = np.linalg.norm(missile_vel)
        if v_m < 1e-6:
            return np.array([0, 1]), distance

        # **经典比例导引法公式（与AIM-9L完全一致）**
        # 计算视线角速率
        Rxy = np.sqrt(r[0] ** 2 + r[1] ** 2)  # X-Y平面距离
        Rxyz = distance  # 3D距离

        if Rxy < 1e-6:
            dbeta = 0
        else:
            # 视线角速率β（偏航方向）
            dbeta = (v_r[1] * r[0] - v_r[0] * r[1]) / (Rxy * Rxy)

        # 视线角速率ε（俯仰方向）
        if Rxyz < 1e-6:
            deps = 0
        else:
            deps = (v_r[2] * Rxy * Rxy - r[2] * (r[0] * v_r[0] + r[1] * v_r[1])) / (Rxyz * Rxyz * Rxy)

        # 当前导弹姿态
        theta_m = self.get_rpy()[1]  # 俯仰角

        # **比例导引指令（与AIM-9L完全相同的公式）**
        ny = K * v_m / self._g * np.cos(theta_m) * dbeta
        nz = K * v_m / self._g * deps + np.cos(theta_m)  # 重力补偿

        # 限制过载
        total_g = np.sqrt(ny ** 2 + (nz - np.cos(theta_m)) ** 2)
        if total_g > self._nyz_max:
            scale = self._nyz_max / total_g
            ny *= scale
            nz = np.cos(theta_m) + (nz - np.cos(theta_m)) * scale

        # 高度保护
        current_alt = missile_pos[2]
        if current_alt < 1000:
            nz = max(nz, np.cos(theta_m) + 2.0)  # 强制爬升

        # 记录制导历史用于调试
        self.guidance_history.append({
            "time": self._t,
            "phase": self.current_phase,
            "distance": distance,
            "dbeta": dbeta,
            "deps": deps,
            "ny": ny,
            "nz": nz,
            "v_m": v_m
        })

        return np.array([ny, nz]), distance

    def _state_trans(self, action):
        """状态转移 - 基于AIM-9L的成功实现"""
        # 更新位置
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self._position, self.lon0, self.lat0, self.alt0)

        # 当前状态
        v = np.linalg.norm(self.get_velocity())
        v = max(v, 1e-6)
        theta, phi = self.get_rpy()[1:]

        # 推力计算
        if self._t < self._t_thrust:
            T = self._g * self._Isp * self._dm
        else:
            T = 0

        # 阻力计算
        D = 0.5 * self._cD * self.S * self.rho * v ** 2

        # 过载指令
        ny, nz = action

        # **使用与AIM-9L相同的状态转移方程**
        nx = (T - D) / (self._m * self._g)

        # 速度变化
        dv = self._g * (nx - np.sin(theta))

        # 姿态变化率（与AIM-9L完全一致）
        cos_theta = max(np.cos(theta), 0.1)
        self._dphi = self._g / v * (ny / cos_theta)
        self._dtheta = self._g / v * (nz - np.cos(theta))

        # 更新状态
        v += self.dt * dv
        v = max(v, 80)  # 保持最小速度
        phi += self.dt * self._dphi
        theta += self.dt * self._dtheta

        # 更新速度向量
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta)
        ])

        # 更新姿态
        self._posture[:] = np.array([0, theta, phi])

        # 更新质量
        if self._t < self._t_thrust:
            self._m = self._m - self.dt * self._dm

    def _log_status(self, distance):
        """状态日志"""
        v = np.linalg.norm(self.get_velocity())
        alt = self.get_position()[2]

        # 从制导历史中获取最新的制导信息
        if self.guidance_history:
            last_guidance = self.guidance_history[-1]
            dbeta = last_guidance["dbeta"]
            deps = last_guidance["deps"]
            ny = last_guidance["ny"]
            nz = last_guidance["nz"]

            logging.info(f"📊 AIM-120C7 {self.uid}: t={self._t:.1f}s, {self.current_phase}, "
                         f"dist={distance:.0f}m, v={v:.0f}m/s, alt={alt:.0f}m, "
                         f"LOS_rates=[{dbeta:.4f}, {deps:.4f}], commands=[{ny:.2f}, {nz:.2f}]g")
        else:
            logging.info(f"📊 AIM-120C7 {self.uid}: t={self._t:.1f}s, {self.current_phase}, "
                         f"dist={distance:.0f}m, v={v:.0f}m/s, alt={alt:.0f}m")

    @property
    def target_distance(self) -> float:
        if self.target_aircraft and self.target_aircraft.is_alive:
            return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())
        return np.inf

    def log(self):
        """渲染日志"""
        if self.is_alive:
            return super().log()
        elif self.is_done and not self.render_explosion:
            self.render_explosion = True
            lon, lat, alt = self.get_geodetic()
            log_msg = f"-{self.uid}\n"
            log_msg += f"{self.uid}F,T={lon:.6f}|{lat:.6f}|{alt:.1f}|0|0|0,"
            if self.is_success:
                log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc * 2}"
            else:
                log_msg += f"Type=Misc+Explosion,Color=Gray,Radius={self._Rc}"
            return log_msg
        return None

    def close(self):
        """清理资源"""
        if self.target_aircraft and self in self.target_aircraft.under_missiles:
            self.target_aircraft.under_missiles.remove(self)
        if self.parent_aircraft and self in self.parent_aircraft.launch_missiles:
            self.parent_aircraft.launch_missiles.remove(self)
        self.target_aircraft = None
        self.parent_aircraft = None

    def get_guidance_analysis(self):
        """获取制导分析数据"""
        return {
            "total_flight_time": self._t,
            "guidance_history": self.guidance_history[-50:],  # 最近50个数据点
            "final_phase": self.current_phase,
            "seeker_activated": self.seeker_active,
            "target_updates": int(self.target_update_time / 2.0)
        }