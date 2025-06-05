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
        """(lontitude, latitude, altitude), unit: °, m"""
        return self._geodetic

    def get_position(self):
        """(north, east, up), unit: m"""
        return self._position

    def get_rpy(self):
        """(roll, pitch, yaw), unit: rad"""
        return self._posture

    def get_velocity(self):
        """(v_north, v_east, v_up), unit: m/s"""
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
    """封装 JSBSim 实例并管理通信的飞机模拟器类。"""

    ALIVE = 0
    CRASH = 1       # 坠毁：低高度/极端状态/过载
    SHOTDOWN = 2    # 被击落：导弹攻击

    def __init__(self,
                 uid: str = "A0100",
                 color: TeamColors = "Red",
                 model: str = 'f16',
                 init_state: dict = {},
                 origin: tuple = (120.0, 60.0, 0.0),
                 sim_freq: int = 60,
                 **kwargs):
        """构造函数：创建 JSBSim 实例，加载飞机模型并设置初始条件。

        Args:
            uid (str): 5 位十六进制唯一标识符，默认为 "A0100"。
            color (TeamColors): 团队颜色，用于区分红蓝方。
            model (str): 飞机模型名称，默认为 "f16"。
            init_state (dict): 初始状态字典，映射属性到初始值，空字典使用默认值。
            origin (tuple): 全局战斗场的原点（经度，纬度，高度），默认为 (120.0, 60.0, 0.0)。
            sim_freq (int): JSBSim 仿真频率，默认为 60 Hz。
            **kwargs: 其他参数，如 num_missiles。
        """
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model
        self.init_state = init_state
        self.lon0, self.lat0, self.alt0 = origin
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        self._is_leader = uid.endswith("100")  # 若 uid 以 "100" 结尾（如 A0100, B0100），设为领机
        for key, value in kwargs.items():
            if key == 'num_missiles':
                self.num_missiles = value
                self.num_left_missiles = self.num_missiles
        # 固定模拟器链接
        self.partners: List[AircraftSimulator] = []  # 队友列表
        self.enemies: List[AircraftSimulator] = []   # 敌方列表
        # 临时模拟器链接
        self.launch_missiles: List[MissileSimulator] = []   # 已发射的导弹
        self.under_missiles: List[MissileSimulator] = []    # 正在追踪的导弹
        # 初始化模拟器
        self.reload()

    @property
    def is_alive(self):
        """检查飞机是否存活。

        Returns:
            bool: 如果存活，返回 True。
        """
        return self.__status == AircraftSimulator.ALIVE

    @property
    def is_crash(self):
        """检查飞机是否坠毁。

        Returns:
            bool: 如果坠毁，返回 True。
        """
        return self.__status == AircraftSimulator.CRASH

    @property
    def is_shotdown(self):
        """检查飞机是否被击落。

        Returns:
            bool: 如果被击落，返回 True。
        """
        return self.__status == AircraftSimulator.SHOTDOWN

    def is_leader(self):
        """检查是否为领机。

        Returns:
            bool: 如果是领机，返回 True，否则返回 False。
        """
        return self._is_leader

    def set_leader(self, is_leader: bool):
        """设置领机状态。

        Args:
            is_leader (bool): 是否为领机。
        """
        self._is_leader = is_leader
        # logging.info(f"Set Agent {self.uid} as {'Leader' if is_leader else 'Wingman'}")

    def crash(self):
        """标记飞机为坠毁状态。"""
        self.__status = AircraftSimulator.CRASH
        logging.info(f"Agent {self.uid} crashed")

    def shotdown(self):
        """标记飞机为被击落状态。"""
        self.__status = AircraftSimulator.SHOTDOWN
        logging.info(f"Agent {self.uid} shot down")

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        """重新加载飞机模拟器，恢复初始状态。

        Args:
            new_state (dict, optional): 新的状态字典，覆盖初始状态。
            new_origin (tuple, optional): 新的战场原点坐标，覆盖默认原点。
        """
        super().reload()

        # 重置状态
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        self.launch_missiles.clear()
        self.under_missiles.clear()
        self.num_left_missiles = self.num_missiles

        # 加载 JSBSim
        self.jsbsim_exec = jsbsim.FGFDMExec(os.path.join(get_root_dir(), 'data'))
        self.jsbsim_exec.set_debug_level(0)
        self.jsbsim_exec.load_model(self.model)
        Catalog.add_jsbsim_props(self.jsbsim_exec.query_property_catalog(""))
        self.jsbsim_exec.set_dt(self.dt)
        self.clear_default_condition()

        # 分配新属性
        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin
        for key, value in self.init_state.items():
            self.set_property_value(Catalog[key], value)
        success = self.jsbsim_exec.run_ic()
        if not success:
            logging.error("JSBSim 初始化仿真条件失败")
            raise RuntimeError("JSBSim failed to init simulation conditions.")

        # 初始化引擎
        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        for j in range(n):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()
        # 更新内部属性
        self._update_properties()

    def clear_default_condition(self):
        """清除默认仿真条件，设置初始值。"""
        default_condition = {
            Catalog.ic_long_gc_deg: 120.0,  # 地理经度 [°]
            Catalog.ic_lat_geod_deg: 60.0,  # 地理纬度 [°]
            Catalog.ic_h_sl_ft: 20000,      # 海拔高度 [ft]
            Catalog.ic_psi_true_deg: 0.0,   # 初始航向 [°] (0, 360]
            Catalog.ic_u_fps: 800.0,        # x轴速度 [ft/s] (-2200, 2200)
            Catalog.ic_v_fps: 0.0,          # y轴速度 [ft/s] (-2200, 2200)
            Catalog.ic_w_fps: 0.0,          # z轴速度 [ft/s] (-2200, 2200)
            Catalog.ic_p_rad_sec: 0.0,      # 滚转角速度 [rad/s]
            Catalog.ic_q_rad_sec: 0.0,      # 俯仰角速度 [rad/s]
            Catalog.ic_r_rad_sec: 0.0,      # 偏航角速度 [rad/s]
            Catalog.ic_roc_fpm: 0.0,        # 爬升率 [ft/min]
            Catalog.ic_terrain_elevation_ft: 0,  # 地形高度
        }
        for prop, value in default_condition.items():
            self.set_property_value(prop, value)

    def run(self):
        """运行 JSBSim 仿真，更新状态。

        Returns:
            bool: 如果仿真达到 JSBSim 终止条件，返回 False，否则返回 True。
        """
        if self.is_alive:
            if self.bloods <= 0:
                self.shotdown()
            result = self.jsbsim_exec.run()
            if not result:
                logging.error("JSBSim 仿真运行失败")
                raise RuntimeError("JSBSim failed.")
            self._update_properties()
            return result
        else:
            return False  # 非存活状态返回 False，停止仿真

    def close(self):
        """关闭仿真器并清理资源。"""
        if self.jsbsim_exec:
            self.jsbsim_exec = None
        self.partners = []
        self.enemies = []
        logging.info(f"Agent {self.uid} simulator closed")

    def _update_properties(self):
        """更新内部位置、姿态和速度属性。"""
        # 更新位置
        self._geodetic[:] = self.get_property_values([
            Catalog.position_long_gc_deg,
            Catalog.position_lat_geod_deg,
            Catalog.position_h_sl_m
        ])
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
        # 更新姿态
        self._posture[:] = self.get_property_values([
            Catalog.attitude_roll_rad,
            Catalog.attitude_pitch_rad,
            Catalog.attitude_heading_true_rad
        ])
        # 更新速度
        self._velocity[:] = self.get_property_values([
            Catalog.velocities_v_north_mps,
            Catalog.velocities_v_east_mps,
            Catalog.velocities_v_down_mps
        ])
        # v_down -> v_up
        self._velocity[2] = -self._velocity[2]

    def get_sim_time(self):
        """获取仿真时间。

        Returns:
            float: JSBSim 的仿真时间。
        """
        return self.jsbsim_exec.get_sim_time()

    def get_property_values(self, props):
        """获取指定属性的值。

        Args:
            props: 属性列表。

        Returns:
            list: 属性值列表。
        """
        return [self.get_property_value(prop) for prop in props]

    def set_property_values(self, props, values):
        """设置指定属性的值。

        Args:
            props: 属性列表。
            values: 值列表。
        """
        if len(props) != len(values):
            logging.error(f"属性和值的数量不匹配: props={len(props)}, values={len(values)}")
            raise ValueError("mismatch between properties and values size")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def get_property_value(self, prop):
        """从 JSBSim 获取指定属性的值。

        Args:
            prop: 属性对象。

        Returns:
            float: 属性值。
        """
        if isinstance(prop, Property):
            if prop.access == "R":
                if prop.update:
                    prop.update(self)
            return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        else:
            logging.error(f"无效的属性类型: {type(prop)} ({prop})")
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def set_property_value(self, prop, value):
        """设置指定属性的值。

        Args:
            prop: 属性对象。
            value: float 值。
        """
        if isinstance(prop, Property):
            if value < prop.min:
                value = prop.min
            elif value > prop.max:
                value = prop.max
            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)
            if "W" in prop.access:
                if prop.update:
                    prop.update(self)
        else:
            logging.error(f"无效的属性类型: {type(prop)} ({prop})")
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def check_missile_warning(self):
        """检查是否有导弹威胁。

        Returns:
            MissileSimulator or None: 如果存在威胁的导弹，返回该导弹实例，否则返回 None。
        """
        for missile in self.under_missiles:
            if missile.is_alive:
                return missile
        return None


class MissileSimulator(BaseSimulator):
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str, missile_model: str = "AIM-120"):
        assert parent.dt == target.dt, "integration timestep must be same!"
        missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)
        missile.target(target)
        return missile

    def __init__(self, uid="A0101", color="Red", model="AIM-120", dt=1 / 12):
        super().__init__(uid, color, dt)
        self.__status = MissileSimulator.INACTIVE
        self.model = model
        self.parent_aircraft = None
        self.target_aircraft = None
        self.render_explosion = False

        self._g = 9.81
        self._t_max = 120
        self._t_thrust = 5
        self._Isp = 200
        self._Length = 3.66
        self._Diameter = 0.18
        self._cD = 0.3
        self._m0 = 150
        self._dm = 5
        self._K = 4
        self._nyz_max = 40
        self._Rc = 10
        self._v_min = 200

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
        self._velocity[:] = parent.get_velocity()
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

    def target(self, target: AircraftSimulator):
        self.target_aircraft = target
        self.target_aircraft.under_missiles.append(self)

    def run(self):
        self._t += self.dt
        action, distance = self._guidance()
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance
        if distance < self._Rc and self.target_aircraft.is_alive:
            self.__status = MissileSimulator.HIT
            self.target_aircraft.shotdown()
        elif (self._t > self._t_max) or (np.linalg.norm(self.get_velocity()) < self._v_min) or \
             np.sum(self._distance_increment) >= self._distance_increment.maxlen or not self.target_aircraft.is_alive:
            self.__status = MissileSimulator.MISS
        else:
            self._state_trans(action)

    def log(self):
        if self.is_alive:
            return super().log()
        elif self.is_done and not self.render_explosion:
            self.render_explosion = True
            log_msg = f"-{self.uid}\n"
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
            return log_msg
        return None

    def close(self):
        self.target_aircraft = None

    def _guidance(self):
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])
        theta_m = np.arcsin(dz_m / v_m)

        x_t, y_t, z_t = self.target_aircraft.get_position()
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()

        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])
        Rxyz = np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])

        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy ** 2
        deps = ((dz_t - dz_m) * Rxy ** 2 - (z_t - z_m) * (
                (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz ** 2 * Rxy)

        ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
        nz = self.K * v_m / self._g * deps + np.cos(theta_m)

        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), Rxyz

    def _state_trans(self, action):
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = LLA2NEU(*self.get_position(), self.lon0, self.lat0, self.alt0)

        v = np.linalg.norm(self.get_velocity())
        theta, phi = self.get_rpy()[1:]
        T = self._g * self.Isp * self._dm
        D = 0.5 * self._cD * self.S * self.rho * v ** 2
        nx = (T - D) / (self._m * self._g)
        ny, nz = action

        dv = self._g * (nx - np.sin(theta))
        self._dphi = self._g / v * (ny / np.cos(theta))
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
