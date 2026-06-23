"""
R-27ER导弹模拟器
基于真实R-27ER参数完整模拟敌方导弹
R-27ER是俄罗斯SU-27系列飞机的标准中程空空导弹
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from collections import deque

# 导入基础模拟器
try:
    from envs.JSBSim.core.simulatior import BaseSimulator, AircraftSimulator
except ImportError:
    # 如果导入失败，创建一个基础的基类
    class BaseSimulator:
        def __init__(self, uid: str, color: str, dt: float):
            self.uid = uid
            self.color = color
            self.dt = dt
            self._position = np.zeros(3)
            self._velocity = np.zeros(3)
            self._geodetic = np.zeros(3)
            self._posture = np.zeros(3)
            self._t = 0.0
            self._m = 0.0
            self._dtheta = 0.0
            self._dphi = 0.0
            self.lon0 = 0.0
            self.lat0 = 0.0
            self.alt0 = 0.0

        def get_position(self):
            return self._position

        def get_velocity(self):
            return self._velocity

        def get_geodetic(self):
            return self._geodetic

        def get_rpy(self):
            return self._posture


class R27ERMissileSimulator(BaseSimulator):
    """R-27ER 中程空空导弹模拟器 - 基于真实参数"""

    # 导弹状态
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    # 飞行阶段定义
    BOOST_PHASE = 0      # 助推段
    MIDCOURSE_PHASE = 1  # 中段制导
    TERMINAL_PHASE = 2   # 末段制导

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, uid: str):
        """创建R-27ER导弹实例"""
        assert parent.dt == target.dt, "integration timestep must be same!"
        missile = R27ERMissileSimulator(uid, parent.color, parent.dt)
        missile.launch(parent)
        missile.target(target)
        return missile

    def __init__(self, uid: str = "B1001", color: str = "Blue", dt: float = 1/60):
        super().__init__(uid, color, dt)
        
        # 导弹状态
        self.__status = R27ERMissileSimulator.INACTIVE
        self.model = "R-27ER"
        self.parent_aircraft = None  # 发射平台
        self.guide_aircraft = None   # 当前照射/制导平台（可接力）
        self.guide_agent_id = None
        self.target_aircraft = None  # 目标
        self.launch_target_id = None
        self.launch_target_uid = None
        self.launch_target_real_id = None
        self.render_explosion = False
        self.print_interval = 10  # 每10秒打印一次
        
        # 击中记录
        self._hit_time = None
        self._hit_distance = None
        self._hit_recorded = False
        
        # ===== R-27ER真实参数 =====
        # 物理参数 - 基于真实R-27ER (AA-10 Alamo-C) 技术规格
        self._g = 9.81                    # 重力加速度 m/s²
        self._t_max = 100                  # 最大飞行时间 s (真实R-27ER约60-90秒)
        self._t_boost = 14.0              # 🔥 提升：总助推时间从12.0s增加到14.0s，增加加速时间
        self._t_terminal = 25             # 末段制导开始时间 s

        # 发动机参数 (R-27ER使用双脉冲发动机)
        self._Isp = 255                   # 比冲 s (修正为真实值)
        self._Length = 4.08               # 长度 m (修正为真实R-27ER)
        self._Diameter = 0.23             # 直径 m (真实R-27ER)
        self._cD = 0.3                   # 削弱：阻力系数从0.18增加到0.19，略微增加阻力以降低最高速度
        self._m0 = 253                    # 初始质量 kg (修正为真实R-27ER)
        self._fuel_mass = 100.0           # 🔥 进一步提升：燃料质量从95.0kg增加到100.0kg
        self._dm = self._fuel_mass / self._t_boost  # 质量损失率 kg/s
        self._thrust = 25000              # 削弱：推力从32000N降低到30000N（约6.25%），将最高速度从1122m/s降低到约1070m/s

        # 制导参数
        self._K = 4.2                     # 🔥 加强：大幅提高比例导引系数（从3.8到4.2），显著提高制导精度
        self._nyz_max = 28                # 最大过载 G (修正为真实值)
        self._Rc = 35                     # 爆炸半径 m (修正为真实值)
        self._v_min = 100                 # 最小速度 m/s
        
        # 制导系统参数
        self._phase = R27ERMissileSimulator.BOOST_PHASE
        self._intercept_point = np.zeros(3)
        self._terminal_distance = 15000   # 末段制导启动距离 m (R-27ER半主动雷达，15km更合理)

        # 双脉冲发动机参数 - 基于真实R-27ER双脉冲时序
        self._pulse1_duration = 6.5       # 🔥 提升：第一脉冲持续时间从5.5s增加到6.5s
        self._pulse2_duration = 5.5       # 🔥 提升：第二脉冲持续时间从4.5s增加到5.5s
        self._pulse2_delay = 8.0          # 🔥 优化：第二脉冲延迟从9.0s缩短到8.0s，更快启动第二脉冲
        self._pulse2_thrust_ratio = 0.80   # 🔥 提升：第二脉冲推力比例从0.75增加到0.80，提高第二脉冲推力

        # 制导噪声参数 (R-27ER半主动雷达制导)
        self._guidance_noise = 0.06       # 🔥 加强：大幅降低制导噪声（从0.08到0.06），显著提高命中精度
        self._seeker_range = 25000        # 导引头作用距离 m (半主动雷达限制)
        self._max_guidance_time = 75      # 最大制导时间 s (半主动雷达制导限制)
        self._illumination_grace_s = 18.0  # 照射中断容忍时间（适配机动/接力窗口）
        self._terminal_requires_illumination = True  # 明确保持半主动：末段仍需照射，不允许主动自导
        self._illumination_lost_s = 0.0
        self._terminal_blind_hold_s = 1.5
        self._terminal_switch_settle_s = 2.5
        self._terminal_timeout_extension_s = 8.0
        self._terminal_illumination_bonus_s = 4.0
        self._terminal_distance_increase_hold_s = 4.5
        self._last_guidance_switch_t = -999.0
        self._last_terminal_action = np.zeros(2, dtype=float)
        
        # 飞行监控
        self._phase_changed = False
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(10 / self.dt))
        self._left_t = int(1 / self.dt)

        # 发射安全设置：避免出生点/坐标系异常导致“0.2s内原地爆炸”
        self._arm_time_s = 0.5
        self._spawn_offset_m = 30.0
        self._launch_boresight_limit_rad = np.deg2rad(55.0)
        self._launch_pitch_step_limit_rad = np.deg2rad(18.0)

    @property
    def is_alive(self):
        """导弹是否仍在飞行"""
        return self.__status == R27ERMissileSimulator.LAUNCHED

    @property
    def is_success(self):
        """导弹是否击中目标"""
        return self.__status == R27ERMissileSimulator.HIT

    @property
    def is_done(self):
        """导弹是否已经爆炸"""
        return self.__status == R27ERMissileSimulator.HIT or self.__status == R27ERMissileSimulator.MISS

    @property
    def Isp(self):
        """R-27ER双脉冲发动机比冲模型"""
        if self._t <= self._pulse1_duration:
            # 第一脉冲：全推力
            return self._Isp
        elif self._pulse2_delay <= self._t <= self._pulse2_delay + self._pulse2_duration:
            # 第二脉冲：部分推力
            return self._Isp * self._pulse2_thrust_ratio
        else:
            # 燃料耗尽：无推力
            return 0

    @property
    def K(self):
        """比例导引系数 - R-27ER制导特性"""
        if self._phase == R27ERMissileSimulator.TERMINAL_PHASE:
            # 末段制导时增强机动性，但R-27ER增强幅度较小
            return self._K * 1.3
        return self._K

    @property
    def S(self):
        """横截面积 m²"""
        S0 = np.pi * (self._Diameter / 2) ** 2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        """空气密度 kg/m³ - 标准大气模型"""
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
        """到目标的距离"""
        if not self.target_aircraft:
            return np.inf
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, parent: AircraftSimulator):
        """发射导弹"""
        self.parent_aircraft = parent
        self.guide_aircraft = parent
        self.guide_agent_id = getattr(parent, 'uid', None)
        self.parent_aircraft.launch_missiles.append(self)

        # 设置数据记录所需的属性
        self.launcher_id = parent.uid  # 添加发射者ID属性
        self.parent_uid = parent.uid   # 备用属性名

        # 继承发射平台的运动参数
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0  # 导弹滚转角保持为零
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0

        # 初始化状态
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0  # 角速度初始化为0
        self.__status = R27ERMissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=int(10 / self.dt))
        self._left_t = int(1 / self.dt)
        self._phase = R27ERMissileSimulator.BOOST_PHASE
        self._phase_changed = False
        self._illumination_lost_s = 0.0
        self._last_guidance_switch_t = -999.0
        self._last_terminal_action = np.zeros(2, dtype=float)

        # 导弹仅继承母机速度，不进行任何速度补偿
        print(f"[MISSILE] R-27ER {self.uid} launched: v={np.linalg.norm(self._velocity):.1f}m/s, alt={self._geodetic[2]:.0f}m")

    def set_guidance_aircraft(self, aircraft: AircraftSimulator):
        """设置当前照射/制导平台（用于接力制导）。"""
        previous_guide_id = getattr(self, 'guide_agent_id', None)
        self.guide_aircraft = aircraft
        self.guide_agent_id = getattr(aircraft, 'uid', None)
        if self.guide_agent_id != previous_guide_id:
            self._last_guidance_switch_t = float(self._t)
        self._illumination_lost_s = 0.0

    def _has_illumination(self) -> bool:
        """判断半主动制导照射是否有效。"""
        if self.guide_aircraft is None or self.target_aircraft is None:
            return False
        if not getattr(self.guide_aircraft, 'is_alive', False):
            return False
        if not getattr(self.target_aircraft, 'is_alive', False):
            return False
        try:
            guide_pos = self.guide_aircraft.get_position()
            target_pos = self.target_aircraft.get_position()
            guide_dist = np.linalg.norm(target_pos - guide_pos)
            # 半主动中制导照射距离不应受末段导引头作用距离(25km)限制。
            # 使用稳定的支持窗口（约120km），匹配当前BVR发射几何。
            illumination_support_range = 140000.0
            if guide_dist > illumination_support_range:
                return False

            # 半主动照射几何约束：制导机机动转向过大时视作照射失效，触发接力窗口。
            guide_vel = np.array(self.guide_aircraft.get_velocity(), dtype=float)
            los_vec = np.array(target_pos - guide_pos, dtype=float)
            guide_vel_xy = guide_vel[:2]
            los_xy = los_vec[:2]
            vel_norm = np.linalg.norm(guide_vel_xy)
            los_norm = np.linalg.norm(los_xy)
            if vel_norm < 1e-3 or los_norm < 1e-3:
                return True

            cos_angle = float(np.clip(np.dot(guide_vel_xy, los_xy) / (vel_norm * los_norm), -1.0, 1.0))
            off_boresight_deg = float(np.degrees(np.arccos(cos_angle)))
            # 超过约70度认为已无法稳定照射（与大机动规避场景一致）
            return off_boresight_deg <= 80.0
        except Exception:
            return False

    def target(self, target: AircraftSimulator):
        """设置目标"""
        self.target_aircraft = target
        self.target_aircraft.under_missiles.append(self)

        # 设置数据记录所需的属性
        self.target_id = target.uid    # 添加目标ID属性
        self.target_uid = target.uid   # 备用属性名
        self.launch_target_id = target.uid
        self.launch_target_uid = target.uid
        self.launch_target_real_id = getattr(target, 'real_id', None) or target.uid
        
        # 设置目标后，调整初始姿态对准目标
        velocity_norm = np.linalg.norm(self._velocity)
        if velocity_norm > 0 and self.target_aircraft:
            target_pos = self.target_aircraft.get_position()
            los = target_pos - self._position
            los_norm = np.linalg.norm(los)

            if los_norm > 0:
                los_unit = los / los_norm
                los_theta = np.arcsin(los_unit[2])
                los_phi = np.arctan2(los_unit[1], los_unit[0])
                current_theta = float(self._posture[1])
                current_phi = float(self._posture[2])
                yaw_delta = los_phi - current_phi
                if yaw_delta > np.pi:
                    yaw_delta -= 2 * np.pi
                elif yaw_delta < -np.pi:
                    yaw_delta += 2 * np.pi
                yaw_delta = np.clip(yaw_delta, -self._launch_boresight_limit_rad, self._launch_boresight_limit_rad)
                pitch_delta = np.clip(
                    los_theta - current_theta,
                    -self._launch_pitch_step_limit_rad,
                    self._launch_pitch_step_limit_rad,
                )
                initial_theta = np.clip(current_theta + pitch_delta, -np.pi / 6, np.pi / 6)
                initial_phi = current_phi + yaw_delta
                self._posture[:] = np.array([0, initial_theta, initial_phi])

                forward = np.array([
                    np.cos(initial_theta) * np.cos(initial_phi),
                    np.cos(initial_theta) * np.sin(initial_phi),
                    np.sin(initial_theta),
                ])
                self._velocity[:] = forward * velocity_norm

                try:
                    forward_norm = np.linalg.norm(forward)
                    if forward_norm > 1e-6:
                        self._position[:] = self._position + (forward / forward_norm) * float(self._spawn_offset_m)
                        from envs.JSBSim.core.simulatior import NEU2LLA
                        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)
                except Exception:
                    pass

                print(f"[MISSILE] R-27ER {self.uid} targeting {target.uid}: theta={initial_theta*180/np.pi:.1f}deg, phi={initial_phi*180/np.pi:.1f}deg")

    def run(self):
        """运行导弹仿真"""
        self._t += self.dt

        # 半主动照射状态更新
        if self._phase in (R27ERMissileSimulator.MIDCOURSE_PHASE, R27ERMissileSimulator.TERMINAL_PHASE):
            if self._has_illumination():
                self._illumination_lost_s = 0.0
            else:
                self._illumination_lost_s += self.dt

        # 阶段转换逻辑
        self._update_phase()

        # 根据阶段选择制导律
        action, distance = self._guidance()

        # 距离变化监控
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance

        # 命中判定 - 修复：允许击中已死亡目标（多导弹同时击中场景）
        # 但增加短暂“保险时间”，避免刚发射就因异常距离判定而原地爆炸。
        if distance < self._Rc:
            if self._t < self._arm_time_s:
                # 保险期内不触发命中/爆炸，仅推进状态
                self._state_trans(action)
                return
            # 记录击中时刻和距离
            if self._hit_time is None:
                self._hit_time = self._t
                self._hit_distance = distance
                print(f"[MISSILE] R-27ER {self.uid} HIT target at t={self._hit_time:.1f}s, dist={self._hit_distance:.1f}m")
                
                # ✅ 记录到trace_logger：导弹击中
                try:
                    # 尝试从环境获取trace_logger（如果可用）
                    if hasattr(self, '_env') and self._env is not None:
                        from utils.trace_logger import trace_event
                        target_id = getattr(self.target_aircraft, 'uid', 'Unknown')
                        trace_event(
                            事件="导弹击中",
                            env=self._env,
                            模块="r27er_missile",
                            类型="EVENT",
                            状态="HIT",
                            我机=getattr(self.parent_aircraft, 'uid', 'Unknown') if hasattr(self, 'parent_aircraft') else 'Unknown',
                            敌机=target_id,
                            说明=f"💥 R-27ER {self.uid} HIT target at t={self._hit_time:.1f}s, dist={self._hit_distance:.1f}m",
                            数据={
                                "导弹型号": "R-27ER",
                                "导弹ID": self.uid,
                                "目标": target_id,
                                "击中时间_s": round(self._hit_time, 1),
                                "击中距离_m": round(self._hit_distance, 1),
                            },
                        )
                except Exception:
                    pass  # 如果记录失败，不影响正常流程

            self.__status = R27ERMissileSimulator.HIT
            # 只有在目标仍存活时才调用shotdown()，避免重复击落
            if self.target_aircraft.is_alive:
                self.target_aircraft.shotdown()
        elif self._should_miss():
            if self._t < self._arm_time_s:
                # 保险期内不允许MISS（常见原因：初始数值抖动/速度阈值瞬态）
                self._state_trans(action)
                return
            self.__status = R27ERMissileSimulator.MISS
            miss_reason = self._get_miss_reason()
            if self._t % self.print_interval < self.dt:
                print(f"[MISSILE] R-27ER {self.uid} missed: {miss_reason}, t={self._t:.1f}s, v={np.linalg.norm(self.get_velocity()):.0f}m/s, dist={distance:.0f}m")
        else:
            self._state_trans(action)

    def _update_phase(self):
        """更新飞行阶段"""
        old_phase = self._phase
        distance = self.target_distance

        if self._t <= self._t_boost:
            self._phase = R27ERMissileSimulator.BOOST_PHASE
        elif distance <= self._terminal_distance:
            self._phase = R27ERMissileSimulator.TERMINAL_PHASE
        else:
            self._phase = R27ERMissileSimulator.MIDCOURSE_PHASE

        if old_phase != self._phase and not self._phase_changed:
            phase_names = {0: "boost", 1: "midcourse", 2: "terminal"}
            old_name = phase_names.get(old_phase, "unknown")
            new_name = phase_names.get(self._phase, "unknown")
            print(f"[MISSILE] R-27ER {self.uid} phase change: {old_name} -> {new_name} at t={self._t:.1f}s")
            
            # ✅ 记录到trace_logger：导弹阶段变化
            try:
                if hasattr(self, '_env') and self._env is not None:
                    from utils.trace_logger import trace_event
                    target_id = getattr(self.target_aircraft, 'uid', 'Unknown') if hasattr(self, 'target_aircraft') and self.target_aircraft else 'Unknown'
                    trace_event(
                        事件="导弹阶段变化",
                        env=self._env,
                        模块="r27er_missile",
                        类型="EVENT",
                        状态="PHASE_CHANGE",
                        我机=getattr(self.parent_aircraft, 'uid', 'Unknown') if hasattr(self, 'parent_aircraft') else 'Unknown',
                        敌机=target_id,
                        说明=f"🔄 R-27ER {self.uid} phase change: {old_name} -> {new_name} at t={self._t:.1f}s",
                        数据={
                            "导弹型号": "R-27ER",
                            "导弹ID": self.uid,
                            "目标": target_id,
                            "旧阶段": old_name,
                            "新阶段": new_name,
                            "时间_s": round(self._t, 1),
                        },
                    )
            except Exception:
                pass  # 如果记录失败，不影响正常流程
            
            self._phase_changed = True

    def _is_close_terminal_window(self, distance: Optional[float] = None) -> bool:
        if distance is None:
            distance = self.target_distance
        return (
            self._phase == R27ERMissileSimulator.TERMINAL_PHASE
            and float(distance) <= 12000.0
        )

    def _get_effective_guidance_timeout(self, distance: Optional[float] = None) -> float:
        timeout = float(self._max_guidance_time)
        if self._is_close_terminal_window(distance):
            if distance is not None and float(distance) <= 4000.0:
                timeout += self._terminal_timeout_extension_s + 6.0
            elif distance is not None and float(distance) <= 8000.0:
                timeout += self._terminal_timeout_extension_s + 2.0
            else:
                timeout += max(5.0, self._terminal_timeout_extension_s - 1.0)
        if (
            self._phase == R27ERMissileSimulator.TERMINAL_PHASE
            and (self._t - self._last_guidance_switch_t) <= self._terminal_switch_settle_s
        ):
            timeout += 2.0
        return timeout

    def _get_effective_illumination_grace(self, distance: Optional[float] = None) -> float:
        grace = float(self._illumination_grace_s)
        if self._is_close_terminal_window(distance):
            grace += self._terminal_illumination_bonus_s
            if distance is not None and float(distance) <= 6000.0:
                grace += 2.0
            if distance is not None and float(distance) <= 3000.0:
                grace += 1.5
        if (
            self._phase == R27ERMissileSimulator.TERMINAL_PHASE
            and (self._t - self._last_guidance_switch_t) <= self._terminal_switch_settle_s
        ):
            grace = max(grace, self._illumination_grace_s + 3.5)
        return grace

    def _allow_terminal_blind_hold(self, distance: float) -> bool:
        if self._phase != R27ERMissileSimulator.TERMINAL_PHASE:
            return False
        if float(distance) > 12000.0:
            return False
        blind_hold_s = float(self._terminal_blind_hold_s)
        if float(distance) <= 4000.0:
            blind_hold_s += 1.4
        elif float(distance) <= 8000.0:
            blind_hold_s += 0.8
        if self._illumination_lost_s <= blind_hold_s:
            return True
        recent_switch = (self._t - self._last_guidance_switch_t) <= self._terminal_switch_settle_s
        recent_switch_bonus = 1.6 if float(distance) <= 4000.0 else 1.0
        return recent_switch and self._illumination_lost_s <= (blind_hold_s + recent_switch_bonus)

    def _should_miss(self):
        """判断是否应该脱靶 - 基于真实R-27ER限制"""
        distance = self.target_distance
        # R-27ER的脱靶条件
        if self._t > self._t_max:
            return True

        # 半主动雷达制导时间限制
        if self._t > self._get_effective_guidance_timeout(distance):
            if not (
                self._phase == R27ERMissileSimulator.TERMINAL_PHASE
                and distance <= 4000.0
                and self._illumination_lost_s <= (self._get_effective_illumination_grace(distance) + 1.5)
            ):
                return True

        # 半主动照射丢失过久，制导失效（中段与末段都受约束）
        if self._phase in (R27ERMissileSimulator.MIDCOURSE_PHASE, R27ERMissileSimulator.TERMINAL_PHASE) and self._t > (self._t_boost + 2.0) and self._illumination_lost_s > self._get_effective_illumination_grace(distance):
            if not (self._phase == R27ERMissileSimulator.TERMINAL_PHASE and self._allow_terminal_blind_hold(distance)):
                return True

        # 速度过低
        current_velocity = np.linalg.norm(self.get_velocity())
        if current_velocity < self._v_min:
            return True

        # 距离持续增加判定 - 更严格的条件
        sustained_window = max(int(3.0 / self.dt), 30)
        increase_threshold = int(0.8 * sustained_window)
        if self._phase == R27ERMissileSimulator.TERMINAL_PHASE and distance <= 5000.0:
            sustained_window = max(sustained_window, int((self._terminal_distance_increase_hold_s + 1.8) / self.dt))
            increase_threshold = int(0.95 * sustained_window)
        elif self._phase == R27ERMissileSimulator.TERMINAL_PHASE and distance <= 8000.0:
            sustained_window = max(sustained_window, int(self._terminal_distance_increase_hold_s / self.dt))
            increase_threshold = int(0.92 * sustained_window)
        elif self._phase == R27ERMissileSimulator.TERMINAL_PHASE and distance <= 12000.0:
            sustained_window = max(sustained_window, int(4.5 / self.dt))
            increase_threshold = int(0.88 * sustained_window)
        if len(self._distance_increment) >= sustained_window:
            recent_increase = list(self._distance_increment)[-sustained_window:]
            distance_increase = sum(recent_increase)
            # 只有在数秒内持续明显远离目标时才判定脱靶，避免刚发射后几何轻微发散就被误杀
            if distance_increase >= increase_threshold:
                return True

        # 移除"目标已死亡"检查 - 修复：避免与击中检测冲突
        # 导弹应该能够击中刚被击落的目标（多导弹同时攻击场景）
        if not self.target_aircraft:
            return True

        # 距离过远且无制导能力 - 只在飞行后期检查
        if self._t > 30 and distance > self._seeker_range * 2.5:
            return True

        return False

    def _get_miss_reason(self):
        """获取脱靶原因"""
        distance = self.target_distance
        if self._t > self._t_max:
            return "timeout"
        elif self._t > self._get_effective_guidance_timeout(distance):
            if (
                self._phase == R27ERMissileSimulator.TERMINAL_PHASE
                and distance <= 4000.0
                and self._illumination_lost_s <= (self._get_effective_illumination_grace(distance) + 1.5)
            ):
                pass
            else:
                return "guidance_timeout"
        elif self._phase in (R27ERMissileSimulator.MIDCOURSE_PHASE, R27ERMissileSimulator.TERMINAL_PHASE) and self._t > (self._t_boost + 2.0) and self._illumination_lost_s > self._get_effective_illumination_grace(distance):
            if not (self._phase == R27ERMissileSimulator.TERMINAL_PHASE and self._allow_terminal_blind_hold(distance)):
                return "illumination_lost"
        elif np.linalg.norm(self.get_velocity()) < self._v_min:
            return "low_velocity"
        elif len(self._distance_increment) >= max(int(3.0 / self.dt), 30):
            sustained_window = max(int(3.0 / self.dt), 30)
            increase_threshold = int(0.8 * sustained_window)
            if self._phase == R27ERMissileSimulator.TERMINAL_PHASE and distance <= 5000.0:
                sustained_window = max(sustained_window, int((self._terminal_distance_increase_hold_s + 1.8) / self.dt))
                increase_threshold = int(0.95 * sustained_window)
            elif self._phase == R27ERMissileSimulator.TERMINAL_PHASE and distance <= 8000.0:
                sustained_window = max(sustained_window, int(self._terminal_distance_increase_hold_s / self.dt))
                increase_threshold = int(0.92 * sustained_window)
            elif self._phase == R27ERMissileSimulator.TERMINAL_PHASE and distance <= 12000.0:
                sustained_window = max(sustained_window, int(4.5 / self.dt))
                increase_threshold = int(0.88 * sustained_window)
            if sum(list(self._distance_increment)[-sustained_window:]) >= increase_threshold:
                return "distance_increasing"
        elif not self.target_aircraft or not self.target_aircraft.is_alive:
            return "target_lost"
        elif distance > self._seeker_range * 1.5:
            return "out_of_range"
        else:
            return "unknown"

    def _guidance(self):
        """制导律选择"""
        if self._phase == R27ERMissileSimulator.BOOST_PHASE:
            return self._boost_guidance()
        elif self._phase == R27ERMissileSimulator.MIDCOURSE_PHASE:
            return self._midcourse_guidance()
        else:  # TERMINAL_PHASE
            return self._terminal_guidance()

    def _boost_guidance(self):
        """助推段制导 - 改进的初始指向和能量管理"""
        distance = self.target_distance

        # 计算目标方向
        target_pos = self.target_aircraft.get_position()
        missile_pos = self.get_position()
        direction = target_pos - missile_pos
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1:
            return np.array([0, 0]), distance

        # 计算期望的俯仰角和偏航角
        direction_unit = direction / direction_norm
        target_pitch = np.arcsin(np.clip(direction_unit[2], -1, 1))
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

        # 根据速度调整控制参数
        current_velocity = np.linalg.norm(self.get_velocity())
        if current_velocity < 400:  # 低速时使用更温和的控制
            k_p = 3.0  # 增加比例增益
            max_overload = self._nyz_max * 0.6  # 适度限制过载
        else:
            k_p = 4.0  # 正常比例增益
            max_overload = self._nyz_max * 0.8  # 正常过载限制

        ny = k_p * yaw_error
        nz = k_p * pitch_error + np.cos(current_pitch)  # 保持升力平衡

        return np.clip([ny, nz], -max_overload, max_overload), distance

    def _midcourse_guidance(self):
        """中段制导 - 预测拦截制导"""
        distance = self.target_distance

        # 半主动雷达：若照射不可用，仅惯性飞行，等待接力恢复
        if not self._has_illumination():
            return np.array([0, 0]), distance

        # 计算预测拦截点
        intercept_point = self._calculate_intercept_point()

        # 计算到拦截点的方向
        missile_pos = self.get_position()
        direction = intercept_point - missile_pos
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1:
            return np.array([0, 0]), distance

        # 计算期望速度方向
        direction_unit = direction / direction_norm
        target_pitch = np.arcsin(np.clip(direction_unit[2], -1, 1))
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

        # 添加少量制导噪声（模拟真实制导系统）
        noise_ny = np.random.normal(0, self._guidance_noise * 0.1)
        noise_nz = np.random.normal(0, self._guidance_noise * 0.1)
        ny += noise_ny
        nz += noise_nz

        return np.clip([ny, nz], -self._nyz_max * 0.7, self._nyz_max * 0.7), distance  # 中段制导限制过载

    def _terminal_guidance(self):
        """末段制导 - 比例导引"""
        distance = self.target_distance

        # 半主动雷达：末段同样要求持续照射，不具备主动寻的“射后不管”。
        if self._terminal_requires_illumination and not self._has_illumination():
            if self._allow_terminal_blind_hold(distance):
                return np.array(self._last_terminal_action, dtype=float), distance
            return np.array([0, 0]), distance

        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])

        if v_m < 1:
            return np.array([0, 0]), distance

        theta_m = np.arcsin(np.clip(dz_m / v_m, -1, 1))

        x_t, y_t, z_t = self.target_aircraft.get_position()
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()

        # 相对位置和距离
        rel_x, rel_y, rel_z = x_t - x_m, y_t - y_m, z_t - z_m
        Rxy = np.linalg.norm([rel_x, rel_y])
        Rxyz = np.linalg.norm([rel_x, rel_y, rel_z])

        if Rxyz < 1 or Rxy < 1:
            return np.array([0, 0]), distance

        # 相对速度
        rel_dx, rel_dy, rel_dz = dx_t - dx_m, dy_t - dy_m, dz_t - dz_m

        # 视线角速率计算
        dbeta = (rel_dy * rel_x - rel_dx * rel_y) / Rxy ** 2
        deps = (rel_dz * Rxy ** 2 - rel_z * (rel_x * rel_dx + rel_y * rel_dy)) / (Rxyz ** 2 * Rxy)

        # 比例导引律 - 末段制导时增强机动性
        K_terminal = self.K * 1.5  # 末段制导时增强机动性
        ny = K_terminal * v_m / self._g * np.cos(theta_m) * dbeta
        nz = K_terminal * v_m / self._g * deps + np.cos(theta_m)

        # 添加制导噪声模拟
        guidance_noise = self._guidance_noise * 0.05  # 末段制导噪声最小
        noise_ny = np.random.normal(0, guidance_noise)
        noise_nz = np.random.normal(0, guidance_noise)

        ny += noise_ny
        nz += noise_nz

        action = np.clip([ny, nz], -self._nyz_max, self._nyz_max)
        self._last_terminal_action = np.array(action, dtype=float)
        return action, distance

    def _calculate_intercept_point(self):
        """计算预测拦截点 - R-27ER算法"""
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
        distance = np.linalg.norm(rel_pos)
        
        # R-27ER预测算法 - 相对保守
        if closing_velocity < 0:
            t_intercept = distance / (missile_speed * 0.7)  # 更保守的预测
        else:
            t_intercept = distance / (missile_speed + target_speed * 0.4)

        # 限制预测时间范围
        t_intercept = np.clip(t_intercept, 3.0, 60.0)  # R-27ER预测时间范围更大

        # 计算预测拦截点
        intercept_point = target_pos + target_vel * t_intercept

        return intercept_point

    def _state_trans(self, action):
        """状态转换函数 - 基于真实导弹动力学"""
        # 更新位置
        self._position[:] += self.dt * self.get_velocity()

        # 使用NEU2LLA转换更新地理坐标
        try:
            from envs.JSBSim.core.simulatior import NEU2LLA
            self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)
        except ImportError:
            # 简化处理
            self._geodetic[0] += self._velocity[0] * self.dt / 111000  # 经度
            self._geodetic[1] += self._velocity[1] * self.dt / 111000  # 纬度
            self._geodetic[2] += self._velocity[2] * self.dt  # 高度

        # 当前速度和姿态
        v = np.linalg.norm(self.get_velocity())
        theta, phi = self.get_rpy()[1:]  # pitch, yaw

        # R-27ER双脉冲发动机推力模型
        if self._t <= self._pulse1_duration:
            # 第一脉冲：全推力
            T = self._thrust
        elif self._pulse2_delay <= self._t <= self._pulse2_delay + self._pulse2_duration:
            # 第二脉冲：部分推力
            T = self._thrust * self._pulse2_thrust_ratio
        else:
            # 无推力
            T = 0

        # 阻力
        D = 0.5 * self._cD * self.S * self.rho * v ** 2
        # 轴向过载
        nx = (T - D) / (self._m * self._g) if self._m > 0 else 0
        ny, nz = action
        # 速度变化
        dv = self._g * (nx - np.sin(theta))

        # 角速度 - 使用标准导弹动力学方程
        if v > 1:
            self._dphi = self._g / v * (ny / np.cos(theta)) if abs(np.cos(theta)) > 0.1 else 0
            self._dtheta = self._g / v * (nz - np.cos(theta))
        else:
            self._dphi = 0
            self._dtheta = 0

        # 限制角速度
        max_angular_rate = 2.0  # 增加最大角速度
        self._dphi = np.clip(self._dphi, -max_angular_rate, max_angular_rate)
        self._dtheta = np.clip(self._dtheta, -max_angular_rate, max_angular_rate)

        # 更新速度大小
        v_new = max(v + dv * self.dt, 50)  # 最小速度保护

        # 更新姿态角
        phi += self._dphi * self.dt
        theta += self._dtheta * self.dt

        # 限制俯仰角范围
        theta = np.clip(theta, -np.pi/2, np.pi/2)  # ±90度

        # 更新速度向量（基于新的姿态角和速度大小）
        self._velocity[:] = np.array([
            v_new * np.cos(theta) * np.cos(phi),
            v_new * np.cos(theta) * np.sin(phi),
            v_new * np.sin(theta)
        ])

        # 更新姿态
        self._posture[:] = np.array([0, theta, phi])

        # 更新质量 - R-27ER双脉冲发动机（只在有推力时消耗燃料）
        if T > 0:  # 只有在有推力时才消耗燃料
            fuel_consumption_rate = self._dm if self._t <= self._pulse1_duration else self._dm * self._pulse2_thrust_ratio
            self._m = max(self._m - self.dt * fuel_consumption_rate, self._m0 - self._fuel_mass)

    def log(self):
        """记录导弹状态 - ACMI格式"""
        if self.is_alive:
            # 生成ACMI格式的导弹数据
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Name={self.uid},"
            log_msg += "Type=Air+Missile,"
            log_msg += f"ShortName={self.uid},"
            log_msg += f"Color={self.color}"
            return log_msg
        elif self.is_done and (not self.render_explosion):
            self.render_explosion = True
            # 移除导弹模型
            log_msg = f"-{self.uid}\n"
            # 添加爆炸效果
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
            return log_msg
        else:
            return None

    def close(self):
        """关闭导弹"""
        self.__status = R27ERMissileSimulator.MISS

    def get_hit_info(self):
        """获取击中信息"""
        return {
            'hit_time': self._hit_time,
            'hit_distance': self._hit_distance,
            'missile_id': self.uid,
            'target_id': self.target_aircraft.uid if self.target_aircraft else None
        }

    def should_record_hit_data(self):
        """是否应该记录击中数据"""
        return self._hit_time is not None and not self._hit_recorded

    def mark_hit_recorded(self):
        """标记击中数据已记录"""
        self._hit_recorded = True 
