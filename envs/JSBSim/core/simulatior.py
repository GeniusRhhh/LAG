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
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str, missile_model: str = "AIM-9L"):
        assert parent.dt == target.dt, "integration timestep must be same!" #时间步长
        missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)
        missile.target(target)
        return missile

    def __init__(self,
                 uid="A0101",
                 color="Red",
                 model="AIM-9L",
                 dt=1 / 12):
        super().__init__(uid, color, dt)
        self.__status = MissileSimulator.INACTIVE
        self.model = model
        self.parent_aircraft = None  # type: AircraftSimulator
        self.target_aircraft = None  # type: AircraftSimulator
        self.render_explosion = False

        # 定义导弹的物理参数
        self._g = 9.81  # 重力加速度
        self._t_max = 60  # 导弹最大飞行时间
        self._t_thrust = 3  # 发动机推力时间
        self._Isp = 120  # 平均比冲
        self._Length = 2.87  # 导弹长度
        self._Diameter = 0.127  # 导弹直径
        self._cD = 0.4  # 空气阻力系数
        self._m0 = 84  # 初始质量
        self._dm = 6  # 燃料消耗速率
        self._K = 3  # 比例导航常数
        self._nyz_max = 30  # 最大过载
        self._Rc = 300  # 爆炸半径
        self._v_min = 150  # 最小速度

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
        return self._Isp if self._t < self._t_thrust else 0

    @property
    def K(self):
        """Proportional Guidance Coefficient"""
        # return self._K
        return max(self._K * (self._t_max - self._t) / self._t_max, 0)

    @property
    def S(self):
        """Cross-Sectional area, unit m^2"""
        S0 = np.pi * (self._Diameter / 2)**2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        """Air Density, unit: kg/m^3"""
        # approximate expression
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)
        # exact expression (Reference: https://www.cnblogs.com/pathjh/p/9127352.html)
        rho0, T0, h = 1.225, 288.15, self._geodetic[-1]
        if h <= 11000:  # Troposphere
            T = T0 - 0.0065 * h
            return rho0 * (T / T0)**4.25588
        elif h <= 20000:  # Lower Stratosphere
            T = 216.65
            return 0.36392 * np.exp((11000 - h) / 6341.62)
        else:  # Upper Stratosphere
            T = 216.65 + 0.001 * (h - 20000)
            return 0.088035 * (T / 216.65)**(-35.1632)

    @property
    def target_distance(self) -> float:
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, parent: AircraftSimulator):
        # inherit kinetic parameters from parent aricraft
        self.parent_aircraft = parent
        self.parent_aircraft.launch_missiles.append(self)
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0  # missile's roll remains zero
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
        # init status
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self.__status = MissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(5 / self.dt))  # 5s of distance increment -- can't hit 存储最近 5 秒内的导弹与目标距离增量
        self._left_t = int(1 / self.dt)  # remove missile 1s after its destroying

    def target(self, target: AircraftSimulator):
        self.target_aircraft = target  # TODO: change target? #指定导弹的目标飞机。
        self.target_aircraft.under_missiles.append(self) #将导弹添加到目标飞机的受攻击列表中。

    def run(self):
        self._t += self.dt
        action, distance = self._guidance()
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance
        if distance < self._Rc and self.target_aircraft.is_alive:
            self.__status = MissileSimulator.HIT #如果距离目标小于爆炸半径且目标仍存活，设为 HIT
            #self.hit_reward_flag=True
            self.target_aircraft.shotdown()
            #如果飞行时间超过限制或速度过低等，设为 MISS
        elif (self._t > self._t_max) or (np.linalg.norm(self.get_velocity()) < self._v_min) \
                or np.sum(self._distance_increment) >= self._distance_increment.maxlen or not self.target_aircraft.is_alive:
            self.__status = MissileSimulator.MISS
        else:
            self._state_trans(action)

    def log(self):
        if self.is_alive:
            log_msg = super().log()
        elif self.is_done and (not self.render_explosion):
            self.render_explosion = True
            # remove missile model
            log_msg = f"-{self.uid}\n"
            # add explosion
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
        else:
            log_msg = None
        return log_msg

    def close(self):
        self.target_aircraft = None

    def _guidance(self):  # 实现比例导航制导算法
        x_m, y_m, z_m = self.get_position()  # 获取导弹当前位置的东北天坐标
        dx_m, dy_m, dz_m = self.get_velocity()  # 获取导弹当前速度分量
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])  # 计算导弹的速度大小（标量）
        theta_m = np.arcsin(dz_m / v_m)  # 计算导弹的俯仰角（theta）

        x_t, y_t, z_t = self.target_aircraft.get_position()  # 获取目标飞机的当前位置
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()  # 获取目标飞机的速度分量

        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])  # 计算导弹与目标的水平距离（X-Y 平面投影）
        Rxyz = np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])  # 计算导弹与目标的三维空间距离
        # calculate beta & eps, but no need actually...
        # beta = np.arctan2(y_m - y_t, x_m - x_t)  # relative yaw
        # eps = np.arctan2(z_m - z_t, np.linalg.norm([x_m - x_t, y_m - y_t]))  # relative pitch

        # 计算相对航向角变化率（dbeta）和相对俯仰角变化率（deps）
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy ** 2
        deps = ((dz_t - dz_m) * Rxy ** 2 - (z_t - z_m) * (
                (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz ** 2 * Rxy)

        # 计算导弹的横向过载指令（ny），基于比例导航算法
        ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
        # 计算导弹的纵向过载指令（nz），结合俯仰角的影响
        nz = self.K * v_m / self._g * deps + np.cos(theta_m)

        # 将横向和纵向过载指令限制在最大过载范围内，返回过载指令和当前距离
        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), Rxyz

    def _state_trans(self, action):  # 状态转移函数，更新导弹的运动状态
        # 更新导弹的位置，根据当前速度进行积分
        self._position[:] += self.dt * self.get_velocity()
        # 将更新后的东北天坐标转换为地理坐标（经度、纬度、高度）
        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)

        # 获取当前速度的大小（标量）
        v = np.linalg.norm(self.get_velocity())
        # 获取导弹当前的俯仰角（theta）和偏航角（phi）
        theta, phi = self.get_rpy()[1:]
        # 计算推力（T），基于比冲和质量流失速率
        T = self._g * self.Isp * self._dm
        # 计算空气阻力（D），基于速度平方和导弹的截面积
        D = 0.5 * self._cD * self.S * self.rho * v ** 2
        # 计算导弹沿着 x 方向的加速度分量
        nx = (T - D) / (self._m * self._g)
        # 提取导航指令的横向和纵向过载
        ny, nz = action

        # 更新导弹的速度变化（dv），结合推力、阻力和俯仰角
        dv = self._g * (nx - np.sin(theta))
        # 计算偏航角变化率（dphi），与横向过载和速度相关
        self._dphi = self._g / v * (ny / np.cos(theta))
        # 计算俯仰角变化率（dtheta），与纵向过载和速度相关
        self._dtheta = self._g / v * (nz - np.cos(theta))

        # 更新速度标量（v），结合速度变化率
        v += self.dt * dv
        # 更新偏航角（phi），结合偏航角变化率
        phi += self.dt * self._dphi
        # 更新俯仰角（theta），结合俯仰角变化率
        theta += self.dt * self._dtheta

        # 更新导弹的速度分量（速度在 x、y、z 方向上的投影）
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),  # x 方向速度分量
            v * np.cos(theta) * np.sin(phi),  # y 方向速度分量
            v * np.sin(theta)  # z 方向速度分量
        ])
        # 更新导弹的姿态角（俯仰角和偏航角）
        self._posture[:] = np.array([0, theta, phi])

        # 更新导弹的质量，如果仍在推力阶段（_t < _t_thrust）
        if self._t < self._t_thrust:
            self._m = self._m - self.dt * self._dm  # 减少燃料质量
