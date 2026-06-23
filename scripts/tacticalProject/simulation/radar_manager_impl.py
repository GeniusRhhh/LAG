"""
统一雷达管理系统 - 支持多项目复用
包含敌我双方的完整雷达系统实现，支持拖曳射击、钳形夹击等多种战术场景

设计理念：
1. 友方雷达：保持简单的状态转换机制，不影响现有战术逻辑
2. 敌方雷达：实现真实的N001VE雷达物理特性，提供挑战性对抗
3. 统一接口：支持多项目复用，便于未来扩展
4. 真实建模：包含完整的电子战、地形遮蔽、大气干扰等效果
"""

import logging
import numpy as np
import math
import random
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

# 导入JSBSim catalog
try:
    from envs.JSBSim.core.catalog import Catalog as c
except ImportError:
    # 如果导入失败，创建一个基础的catalog类
    class c:
        attitude_psi_rad = "attitude/psi-rad"
        position_h_sl_m = "position/h-sl-m"


class RadarStatus(Enum):
    """雷达状态枚举"""
    SEARCH = "SEARCH"      # 搜索模式
    TRACK = "TRACK"        # 跟踪模式
    LOCK = "LOCK"          # 锁定模式
    STANDBY = "STANDBY"    # 待机模式
    JAMMING = "JAMMING"    # 被干扰状态
    MAINTENANCE = "MAINTENANCE"  # 维护状态


class ECMType(Enum):
    """电子对抗措施类型"""
    NOISE_JAMMING = "NOISE_JAMMING"        # 噪声干扰
    DECEPTION_JAMMING = "DECEPTION_JAMMING" # 欺骗干扰
    CHAFF = "CHAFF"                        # 箔条干扰
    FLARE = "FLARE"                        # 红外诱饵
    FREQUENCY_AGILITY = "FREQUENCY_AGILITY" # 频率捷变
    SIDELOBE_BLANKING = "SIDELOBE_BLANKING" # 旁瓣消隐


@dataclass
class RadarTarget:
    """雷达目标信息"""
    target_id: str
    distance: float
    bearing: float
    elevation: float
    velocity: float
    rcs: float = 5.0                       # 雷达截面积 (m²)
    detection_probability: float = 0.0
    track_quality: float = 0.0
    lock_time: float = 0.0
    last_update: float = 0.0
    doppler_shift: float = 0.0             # 多普勒频移
    snr: float = 0.0                       # 信噪比
    multipath_factor: float = 1.0          # 多径效应因子
    atmospheric_loss: float = 0.0          # 大气损耗

    aspect_angle: float = 0.0              # 视角角度（相对目标机头）
    radial_velocity: float = 0.0           # 径向速度 (m/s)
    in_notch: bool = False                 # 是否在多普勒盲区
    clutter_factor: float = 1.0            # 杂波影响因子
    rwr_threat_level: int = 0              # RWR威胁等级 0-5


@dataclass
class APG68RadarModel:
    """AN/APG-68(V)9雷达模型参数 - F-16C Block 50/52真实雷达规格"""
    # 基本性能参数 - 基于真实AN/APG-68(V)9技术规格
    max_detection_range: float = 200000    # 最大探测距离 200km (对5m² RCS战斗机目标)
    max_detection_range_large: float = 240000  # 对大型目标(轰炸机)探测距离 240km
    max_track_range: float = 180000         # 最大跟踪距离 180km (战斗机目标)
    max_lock_range: float = 130000          # 最大锁定距离 130km (STT模式)
    max_simultaneous_tracks: int = 10      # 同时跟踪目标数 10个 (TWS模式)
    max_simultaneous_engagement: int = 2   # 同时攻击目标数 2个 (支持AIM-120)

    # 波束参数 - 平板裂缝阵列天线特性
    search_beam_width: float = 120.0       # 搜索波束宽度 ±120° (宽扫描范围)
    search_elevation_coverage: float = 60.0  # 搜索仰角覆盖 ±60°
    track_beam_width: float = 2.5          # 跟踪波束宽度 2.5° (窄波束)
    lock_beam_width: float = 0.8           # 锁定波束宽度 0.8° (高精度)

    # 时间参数 - 平板裂缝阵列（准相控阵）
    scan_period: float = 2.0               # 扫描周期 2秒 (比机械扫描快)
    lock_update_rate: float = 0.04         # 锁定更新率 0.04秒 (25Hz)
    track_update_rate: float = 0.3         # 跟踪更新率 0.3秒 (TWS模式)

    # 性能参数 - 90年代末/2000年代技术水平
    detection_probability_base: float = 0.90  # 基础探测概率 (优秀性能)
    track_loss_probability: float = 0.015  # 跟踪丢失概率 (低丢失率)
    lock_loss_probability: float = 0.008   # 锁定丢失概率 (STT稳定)

    # 电子战参数 - 第四代雷达抗干扰能力
    jamming_resistance: float = 0.75       # 抗干扰能力 (强于N001VE)
    eccm_capability: float = 0.82          # 电子反对抗能力 (频率捷变+PRF调制)
    frequency_agility: bool = True         # 频率捷变能力 (快速跳频)
    sidelobe_suppression: float = 0.90     # 旁瓣抑制能力 (优秀)
    lpi_capability: float = 0.65           # 低截获概率能力 (部分LPI特性)

    # 多模式能力
    has_look_down_shoot_down: bool = True  # 下视下射能力 (优秀的地杂波抑制)
    has_synthetic_aperture: bool = True    # 合成孔径雷达模式 (对地)
    has_ground_moving_target: bool = True  # 地面移动目标指示 (GMTI)
    has_sea_surface_search: bool = True    # 海面搜索模式

    # 环境适应参数
    weather_degradation: float = 0.08      # 天气影响因子 (优于N001VE)
    terrain_masking_threshold: float = 300.0  # 地形遮蔽阈值 300m
    atmospheric_absorption: float = 0.0008  # 大气吸收系数 (X波段，优化设计)
    
    # 目标分辨率
    range_resolution: float = 30.0         # 距离分辨率 30m (高分辨率)
    angular_resolution: float = 1.5        # 角度分辨率 1.5° (优秀)
    velocity_resolution: float = 5.0       # 速度分辨率 5 m/s


@dataclass
class N001VERadarModel:

    """N001VE雷达模型参数 - V5优化版（基于Su-27/Su-30MKK实际雷达规格）
    
    V5方案要求：我方雷达最大探测距离200km（与预警机400km配合）
    """
    # 基本性能参数 - V5优化（原90km→200km）
    max_detection_range: float = 200000     # 最大探测距离 200km (V5方案要求)
    max_track_range: float = 170000         # 最大跟踪距离 170km
    max_lock_range: float = 120000          # 最大锁定距离 120km (MTR节点)
    max_simultaneous_tracks: int = 10       # 同时跟踪目标数 10个 (官方规格)
    max_simultaneous_engagement: int = 2    # 同时攻击目标数 2个 (TWS模式)

    # 波束参数 - 机械扫描雷达特性
    search_beam_width: float = 70.0        # 搜索波束宽度 ±70° (总140°扫描范围)
    track_beam_width: float = 3.0          # 跟踪波束宽度 3° (窄波束跟踪)
    lock_beam_width: float = 1.0           # 锁定波束宽度 1° (精确锁定)

    # 时间参数 - 机械扫描限制
    scan_period: float = 3.5               # 扫描周期 3.5秒 (机械扫描雷达，非相控阵)
    lock_update_rate: float = 0.05         # 锁定更新率 0.05秒 (连续波照射)
    track_update_rate: float = 0.5         # 跟踪更新率 0.5秒 (TWS模式)

    # 性能参数 - 90年代雷达技术水平
    detection_probability_base: float = 0.85 # 基础探测概率 (考虑杂波和干扰)
    track_loss_probability: float = 0.02   # 跟踪丢失概率 (机动目标)
    lock_loss_probability: float = 0.01    # 锁定丢失概率 (STT模式)

    # 电子战参数 - 第三代雷达抗干扰能力
    jamming_resistance: float = 0.6        # 抗干扰能力 (较老技术，中等水平)
    eccm_capability: float = 0.7           # 电子反对抗能力 (有频率捷变等手段)
    frequency_agility: bool = True         # 频率捷变能力 (支持跳频)
    sidelobe_suppression: float = 0.85     # 旁瓣抑制能力 (中等水平)

    # 环境适应参数

    weather_degradation: float = 0.1       # 天气影响因子 (降雨/云层衰减)
    terrain_masking_threshold: float = 500.0 # 地形遮蔽阈值 500m
    atmospheric_absorption: float = 0.001   # 大气吸收系数 (X波段)
    # 兼容字段：APG-68代码路径会引用该字段
    has_look_down_shoot_down: bool = False


try:
    from .radar_types import (
        APG68RadarModel as _APG68RadarModel,
        ECMType as _ECMType,
        N001VERadarModel as _N001VERadarModel,
        RadarStatus as _RadarStatus,
        RadarTarget as _RadarTarget,
    )
    from . import radar_refactor_helpers as _rrh
except ImportError:
    from radar_types import (
        APG68RadarModel as _APG68RadarModel,
        ECMType as _ECMType,
        N001VERadarModel as _N001VERadarModel,
        RadarStatus as _RadarStatus,
        RadarTarget as _RadarTarget,
    )
    import radar_refactor_helpers as _rrh

RadarStatus = _RadarStatus
ECMType = _ECMType
RadarTarget = _RadarTarget
APG68RadarModel = _APG68RadarModel
N001VERadarModel = _N001VERadarModel


class UnifiedRadarManager:
    """统一雷达管理系统 - 支持多项目复用"""

    def __init__(self):
        """初始化统一雷达管理系统"""

        # 友方雷达状态 - APG-68(V)9功能级建模
        self.friendly_radar_states = {
            "A0100": RadarStatus.SEARCH,
              "A0200": RadarStatus.SEARCH,
              "A0300": RadarStatus.SEARCH,
              "A0400": RadarStatus.SEARCH
        }

        # 敌方雷达状态 - 真实N001VE雷达系统
        self.enemy_radar_states = {
            "B0100": RadarStatus.SEARCH,
              "B0200": RadarStatus.SEARCH,
              "B0300": RadarStatus.SEARCH,
              "B0400": RadarStatus.SEARCH
        }


        # APG-68雷达模型实例 (F-16C)
        self.apg68_radar = APG68RadarModel()
        
        # N001VE雷达模型实例 (Su-27/Su-30)
        self.n001ve_radar = N001VERadarModel()

        # ✅ 雷达对调：友方使用N001VE，敌方使用APG-68（通过交换引用实现）
        self.apg68_radar, self.n001ve_radar = self.n001ve_radar, self.apg68_radar
        logging.info(
            f"[雷达对调生效] FRIEND(A*)=N001VE(max_det={self.apg68_radar.max_detection_range/1000:.0f}km,"
            f" max_lock={self.apg68_radar.max_lock_range/1000:.0f}km, scan={self.apg68_radar.scan_period:.1f}s) | "
            f"ENEMY(B*)=APG-68(max_det={self.n001ve_radar.max_detection_range/1000:.0f}km,"
            f" max_lock={self.n001ve_radar.max_lock_range/1000:.0f}km, scan={self.n001ve_radar.scan_period:.1f}s)"
        )


        # 友方雷达目标跟踪 (APG-68系统)
        self.friendly_radar_targets = {
            "A0100": {},
              "A0200": {},
              "A0300": {},
              "A0400": {}
        }

        # 友方雷达扫描时间记录
        self.friendly_scan_times = {
            "A0100": 0.0,
              "A0200": 0.0,
              "A0300": 0.0,
              "A0400": 0.0
        }

        # 友方雷达锁定目标
        self.friendly_lock_targets = {
            "A0100": None,
              "A0200": None,
              "A0300": None,
              "A0400": None
        }

        # 敌方雷达目标跟踪
        self.enemy_radar_targets = {
            "B0100": {},
              "B0200": {},
              "B0300": {},
              "B0400": {}
        }

        # 敌方雷达扫描时间记录
        self.enemy_scan_times = {
            "B0100": 0.0,
              "B0200": 0.0,
              "B0300": 0.0,
              "B0400": 0.0
        }

        # 日志节流：ECM激活日志时间戳（每机体）
        self._last_ecm_log_time = {}

        # 敌方雷达锁定目标
        self.enemy_lock_targets = {
            "B0100": None,
              "B0200": None,
              "B0300": None,
              "B0400": None
        }


        # 电子战状态 (双方通用)
        self.ecm_states = {

            "A0100": {"active": False, "type": None, "start_time": 0.0, "duration": 0.0},
            "A0200": {"active": False, "type": None, "start_time": 0.0, "duration": 0.0},
            "B0100": {"active": False, "type": None, "start_time": 0.0, "duration": 0.0},
              "B0200": {"active": False, "type": None, "start_time": 0.0, "duration": 0.0},
              "A0300": {"active": False, "type": None, "start_time": 0.0, "duration": 0.0},
              "A0400": {"active": False, "type": None, "start_time": 0.0, "duration": 0.0},
              "B0300": {"active": False, "type": None, "start_time": 0.0, "duration": 0.0},
              "B0400": {"active": False, "type": None, "start_time": 0.0, "duration": 0.0}
        }

        # 环境参数
        self.environmental_conditions = {
            "weather_factor": 1.0,      # 天气因子
            "terrain_height": 0.0,      # 地形高度
            "atmospheric_density": 1.0,  # 大气密度
            "temperature": 15.0,        # 温度 (°C)
            "humidity": 50.0           # 湿度 (%)
        }


        # RWR（雷达告警接收机）状态
        self.rwr_states = {
            "A0100": {"threat_level": 0, "threat_sources": [], "last_warning": 0.0},
            "A0200": {"threat_level": 0, "threat_sources": [], "last_warning": 0.0},
            "B0100": {"threat_level": 0, "threat_sources": [], "last_warning": 0.0},
              "B0200": {"threat_level": 0, "threat_sources": [], "last_warning": 0.0},
              "A0300": {"threat_level": 0, "threat_sources": [], "last_warning": 0.0},
              "A0400": {"threat_level": 0, "threat_sources": [], "last_warning": 0.0},
              "B0300": {"threat_level": 0, "threat_sources": [], "last_warning": 0.0},
              "B0400": {"threat_level": 0, "threat_sources": [], "last_warning": 0.0}
        }

        # 飞机RCS基准值（正面）
        # 飞机基准RCS（正面雷达散射截面积，单位：m²）
        # 注意：这是正面RCS，侧面/尾部会通过视角因子动态调制（侧面×2.5，尾部×1.2）
        # 修正：基于真实雷达测量数据，正面RCS显著小于平均RCS
        self.aircraft_rcs_baseline = {
            "F16": 1.5,    # F-16C 正面RCS约1.5m²（真实测量：1-2m²，侧面8-15m²）
            "Su27": 6.0    # Su-27 正面RCS约6.0m²（真实测量：5-8m²，侧面25-35m²）
        }

        logging.info("[雷达管理] 统一雷达管理系统初始化完成 (APG-68 + N001VE 完整战术级建模)")
        logging.info("   [雷达特性] RCS动态建模 | 多普勒盲区 | 地面杂波 | RWR系统")

    # ==================== 友方雷达系统方法 (APG-68功能级建模) ====================

    def update_friendly_radar_states(self, env, current_time: float):

        """更新友方雷达状态 - APG-68(V)9完备战术级建模"""
        for agent_id in self.friendly_radar_states.keys():
            if agent_id in env.agents and env.agents[agent_id].is_alive:
                self._update_single_friendly_radar(env, agent_id, current_time)

        
        # 更新RWR状态
        self._update_rwr_states(env, current_time)

    def _update_single_friendly_radar(self, env, agent_id: str, current_time: float):

        """更新单个友方雷达状态 - APG-68(V)9完整功能级建模"""
        try:
            # 1. 执行雷达扫描（基于扫描周期）
            if current_time - self.friendly_scan_times[agent_id] >= self.apg68_radar.scan_period or self.friendly_scan_times[agent_id] == 0.0:
                self._scan_friendly_radar(env, agent_id, current_time)
                self.friendly_scan_times[agent_id] = current_time

            # 2. 更新现有目标跟踪
            self._update_friendly_radar_tracks(env, agent_id, current_time)

            # 3. 更新雷达工作模式
            self._update_friendly_radar_mode(env, agent_id, current_time)

            # 4. 处理电子战影响（友方雷达受敌方ECM干扰）
            self._process_friendly_ecm(env, agent_id, current_time)
            
            # 5. 友方随机激活自己的ECM来干扰敌方雷达
            self._activate_friendly_ecm_if_needed(env, agent_id, current_time)

        except Exception as e:

            logging.error(f"❌ {agent_id} APG-68雷达更新错误: {e}")
            self.friendly_radar_states[agent_id] = RadarStatus.SEARCH


    def _scan_friendly_radar(self, env, agent_id: str, current_time: float):
        """APG-68雷达扫描 - 真实探测模型"""
        try:
            agent = env.agents[agent_id]


            # 获取敌方目标列表
            enemy_ids = ["B0100", "B0200"] if agent_id.startswith("A") else ["A0100", "A0200"]


            logging.debug(f"🔍 {agent_id} APG-68开始扫描，目标列表: {enemy_ids}")

            for target_id in enemy_ids:
                if target_id not in env.agents or not env.agents[target_id].is_alive:
                    # 移除已摧毁的目标
                    if target_id in self.friendly_radar_targets[agent_id]:
                        del self.friendly_radar_targets[agent_id][target_id]
                    continue

                target = env.agents[target_id]
                
                # 计算几何参数
                distance = self._calculate_distance(agent, target)
                bearing = self._calculate_bearing(agent, target)
                elevation = self._calculate_elevation(agent, target)
                velocity = self._calculate_target_velocity(env, target_id)
                
                logging.debug(f"🎯 {agent_id} 扫描目标 {target_id}: 距离={distance/1000:.1f}km, 方位={bearing:.1f}°, 仰角={elevation:.1f}°")

                # 检查探测距离
                if distance > self.apg68_radar.max_detection_range:
                    # 移除超出距离的目标
                    if target_id in self.friendly_radar_targets[agent_id]:
                        del self.friendly_radar_targets[agent_id][target_id]
                    continue

                # 计算完整的战术级探测概率
                detection_prob, radar_data = self._calculate_apg68_detection_probability_complete(
                    agent, target, target_id, distance, bearing, elevation, velocity, current_time)
                
                logging.debug(f"📊 {agent_id} 探测概率: {detection_prob:.3f} "
                            f"RCS={radar_data.get('rcs', 0):.1f}m² "
                            f"径向速度={radar_data.get('radial_velocity', 0):.0f}m/s "
                            f"Notch={radar_data.get('in_notch', False)}")

                # 探测成功
                if random.random() < detection_prob:
                    if target_id not in self.friendly_radar_targets[agent_id]:
                        # 检查是否达到最大跟踪目标数（APG-68限制：10个）
                        current_tracks = len(self.friendly_radar_targets[agent_id])
                        if current_tracks >= self.apg68_radar.max_simultaneous_tracks:
                            # 找到最远的目标并移除
                            farthest_target_id = max(
                                self.friendly_radar_targets[agent_id].items(),
                                key=lambda x: x[1].distance
                            )[0]
                            farthest_distance = self.friendly_radar_targets[agent_id][farthest_target_id].distance
                            
                            # 只有新目标更近时才替换
                            if distance < farthest_distance:
                                del self.friendly_radar_targets[agent_id][farthest_target_id]
                                logging.debug(f"📡 {agent_id} 达到最大跟踪数({self.apg68_radar.max_simultaneous_tracks})，"
                                            f"移除最远目标 {farthest_target_id}")
                            else:
                                continue
                        
                        # 创建新目标
                        self.friendly_radar_targets[agent_id][target_id] = RadarTarget(
                            target_id=target_id,
                            distance=distance,
                            bearing=bearing,
                            elevation=elevation,
                            velocity=velocity,
                            detection_probability=detection_prob,
                            last_update=current_time,
                            doppler_shift=self._calculate_doppler_shift(velocity, bearing),
                            snr=self._calculate_snr(distance, bearing, self.apg68_radar),
                            multipath_factor=self._calculate_multipath_factor(distance, elevation),
                            atmospheric_loss=self._calculate_atmospheric_loss(distance)
                        )
                        logging.debug(f"🎯 {agent_id} APG-68雷达探测到新目标: {target_id} 距离={distance/1000:.1f}km")
                    else:
                        # 更新现有目标
                        target_obj = self.friendly_radar_targets[agent_id][target_id]
                        target_obj.distance = distance
                        target_obj.bearing = bearing
                        target_obj.elevation = elevation
                        target_obj.velocity = velocity
                        target_obj.detection_probability = detection_prob
                        target_obj.last_update = current_time
                        target_obj.doppler_shift = self._calculate_doppler_shift(velocity, bearing)
                        target_obj.snr = self._calculate_snr(distance, bearing, self.apg68_radar)
                        target_obj.multipath_factor = self._calculate_multipath_factor(distance, elevation)
                        target_obj.atmospheric_loss = self._calculate_atmospheric_loss(distance)
                else:
                    # 探测失败，移除目标
                    if target_id in self.friendly_radar_targets[agent_id]:
                        logging.debug(f"📡 {agent_id} APG-68雷达失去目标: {target_id} 距离={distance/1000:.1f}km")
                        del self.friendly_radar_targets[agent_id][target_id]

        except Exception as e:

            logging.error(f"❌ {agent_id} APG-68雷达扫描错误: {e}")

    def _calculate_apg68_detection_probability_complete(self, agent, target, target_id: str,
                                                       distance: float, bearing: float,
                                                       elevation: float, velocity: float,
                                                       current_time: float) -> tuple:
        """
        计算完整的APG-68雷达探测概率 - 完备战术级建模
        
        返回: (detection_probability, radar_data_dict)
        """
        try:
            # 基础距离衰减 - APG-68性能优于N001VE（分段与新参数对齐）
            if distance > self.apg68_radar.max_detection_range:  # > 200km
                return 0.0, {}
            elif distance > 180000:  # 180-200km：边缘探测
                base_prob = 0.22
            elif distance > 130000:  # 130-180km：接近跟踪距离
                base_prob = 0.45
            elif distance > 90000:  # 90-130km：高概率区域
                base_prob = 0.88
            elif distance > 50000:  # 50-90km：最佳探测区域
                base_prob = 0.93
            else:  # < 50km：近距离高概率
                base_prob = 0.96

            # ===== 新增：完备战术级因子 =====
            
            # 1. 动态RCS因子（姿态相关）
            dynamic_rcs = self._calculate_dynamic_rcs(agent, target, target_id)
            rcs_factor = min(1.2, math.log10(dynamic_rcs + 1) / math.log10(13))  # 归一化到0.8-1.2
            
            # 2. 计算径向速度和多普勒盲区
            radial_velocity = self._calculate_radial_velocity(agent, target)
            in_notch = self._check_notch_condition(radial_velocity, self.apg68_radar)
            
            # 多普勒盲区 → 探测概率急剧下降
            if in_notch:
                notch_factor = 0.15  # APG-68有一定的杂波抑制，保留15%
                # logging.warning(f"🎯 {agent.uid} Notch检测: {target_id} 径向速度={radial_velocity:.1f}m/s（盲区）")
            else:
                # 高速接近目标更容易探测
                notch_factor = min(1.25, 1.0 + abs(radial_velocity) / 400.0)
            
            # 3. 地面杂波因子
            try:
                target_altitude = target.get_position()[2]
            except:
                target_altitude = 1000.0  # 默认中高空
            
            clutter_factor = self._calculate_ground_clutter_factor(
                elevation, target_altitude, distance, self.apg68_radar)
            
            # 4. 地形遮蔽检查
            is_terrain_masked = self._check_terrain_masking(
                target_altitude, distance, elevation, self.apg68_radar)
            
            if is_terrain_masked:
                # 目标被地形遮蔽，探测概率为0
                logging.debug(f"🏔️ {agent.uid} 目标 {target_id} 被地形遮蔽："
                            f"高度={target_altitude:.0f}m，距离={distance/1000:.1f}km，仰角={elevation:.1f}°")
                return 0.0, {
                    "rcs": dynamic_rcs,
                    "radial_velocity": radial_velocity,
                    "in_notch": in_notch,
                    "clutter_factor": clutter_factor,
                    "terrain_masked": True,
                    "base_prob": 0.0,
                    "final_prob": 0.0
                }

            # 5. 角度因子 - APG-68宽波束搜索（±120°），简化为固定值
            angle_factor = 0.92

            # 6. 仰角因子 - APG-68优秀的下视能力
            elevation_factor = max(0.80, 1.0 - abs(elevation) / 60.0)

            # 7. 大气衰减因子（APG-68优化的X波段设计）
            atmospheric_factor = max(0.80, 1.0 - (distance / self.apg68_radar.max_detection_range) *
                                   self.apg68_radar.atmospheric_absorption * 500)

            # 8. 天气影响（APG-68对恶劣天气适应性更好）
            weather_factor = self.environmental_conditions["weather_factor"]

            # 9. 下视下射能力加成（APG-68强项）
            look_down_bonus = 1.1 if elevation < -10.0 and self.apg68_radar.has_look_down_shoot_down else 1.0

            # ===== 综合探测概率（APG-68完备模型） =====
            total_prob = (base_prob * rcs_factor * angle_factor * elevation_factor *
                         notch_factor * clutter_factor * atmospheric_factor * 
                         weather_factor * look_down_bonus)

            total_prob = min(1.0, max(0.0, total_prob))
            
            # 返回探测概率和详细数据
            radar_data = {
                "rcs": dynamic_rcs,
                "radial_velocity": radial_velocity,
                "in_notch": in_notch,
                "clutter_factor": clutter_factor,
                "terrain_masked": False,
                "base_prob": base_prob,
                "final_prob": total_prob
            }

            return total_prob, radar_data

        except Exception as e:
            logging.error(f"❌ APG-68完整探测概率计算错误: {e}")
            return 0.5, {}

    def _update_friendly_radar_tracks(self, env, agent_id: str, current_time: float):
        """更新友方雷达目标跟踪 - APG-68 TWS模式"""
        try:
            # 清理过期目标
            expired_targets = []
            for target_id, target in self.friendly_radar_targets[agent_id].items():
                if current_time - target.last_update > 8.0:  # 8秒未更新（APG-68数据链更新快）
                    expired_targets.append(target_id)

            for target_id in expired_targets:
                del self.friendly_radar_targets[agent_id][target_id]
                logging.debug(f"📡 {agent_id} 清理过期目标: {target_id}")

            # 更新跟踪质量
            for target_id, target in self.friendly_radar_targets[agent_id].items():
                # 基于距离和时间更新跟踪质量（APG-68跟踪更稳定）
                time_factor = min(1.0, (current_time - target.last_update) / 4.0)
                distance_factor = max(0.15, 1.0 - target.distance / self.apg68_radar.max_track_range)
                
                # 计算机动因子（公式3-25）- APG-68对机动目标较不敏感
                maneuver_factor = 1.0
                if target_id in env.agents and env.agents[target_id].is_alive:
                    try:
                        # 获取目标滚转角
                        roll_rad = env.agents[target_id].get_property_value(c.attitude_phi_rad)
                        roll_deg = abs(np.rad2deg(roll_rad))
                        # APG-68机动因子：f_maneuver = 1.0 + 0.5 × (|φ_roll| / 90°)
                        maneuver_factor = 1.0 + 0.5 * (roll_deg / 90.0)
                    except:
                        maneuver_factor = 1.0
                
                # 完整的跟踪质量公式：Q = f_distance × (1 - f_time) / f_maneuver
                target.track_quality = distance_factor * (1.0 - time_factor) / maneuver_factor
                
                # APG-68跟踪质量更新
                target.detection_probability *= (1.0 - time_factor * self.apg68_radar.track_loss_probability)

                # 移除跟踪丢失的目标
                if random.random() < self.apg68_radar.track_loss_probability * time_factor:
                    expired_targets.append(target_id)

            # 移除跟踪丢失的目标
            for target_id in expired_targets:
                if target_id in self.friendly_radar_targets[agent_id]:
                    del self.friendly_radar_targets[agent_id][target_id]
                    logging.debug(f"📡 {agent_id} 跟踪丢失目标: {target_id}")

        except Exception as e:
            logging.error(f"❌ {agent_id} APG-68跟踪更新错误: {e}")

    def _update_friendly_radar_mode(self, env, agent_id: str, current_time: float):
        """更新友方雷达工作模式 - APG-68多模式"""
        try:
            targets = self.friendly_radar_targets[agent_id]

            if not targets:
                self.friendly_radar_states[agent_id] = RadarStatus.SEARCH
                self.friendly_lock_targets[agent_id] = None
                return

            # 找到最近的威胁目标
            closest_target = min(targets.items(), key=lambda x: x[1].distance)
            target_id, target_data = closest_target

            # 基于距离和跟踪质量决定雷达模式
            # APG-68性能更强，使用稍低的质量阈值（0.5 vs N001VE的0.6）
            if (target_data.distance <= self.apg68_radar.max_lock_range and
                target_data.track_quality > 0.5):
                # STT锁定模式
                self.friendly_radar_states[agent_id] = RadarStatus.LOCK
                
                # 锁定逻辑
                if self.friendly_lock_targets[agent_id] != target_id:
                    self.friendly_lock_targets[agent_id] = target_id
                    # logging.info(f"🔒 {agent_id} APG-68锁定目标: {target_id} 距离={target_data.distance/1000:.1f}km")
                
                # 锁定丢失检查
                if random.random() < self.apg68_radar.lock_loss_probability:
                    self.friendly_radar_states[agent_id] = RadarStatus.TRACK
                    self.friendly_lock_targets[agent_id] = None
                    logging.debug(f"📡 {agent_id} 锁定丢失: {target_id}")

            elif (target_data.distance <= self.apg68_radar.max_track_range and
                  target_data.track_quality > 0.3):
                # TWS跟踪模式（可跟踪多目标）
                self.friendly_radar_states[agent_id] = RadarStatus.TRACK
                self.friendly_lock_targets[agent_id] = None
                
            else:
                # 搜索模式
                self.friendly_radar_states[agent_id] = RadarStatus.SEARCH
                self.friendly_lock_targets[agent_id] = None

        except Exception as e:
            logging.error(f"❌ {agent_id} APG-68模式更新错误: {e}")
            self.friendly_radar_states[agent_id] = RadarStatus.SEARCH

    def _process_friendly_ecm(self, env, agent_id: str, current_time: float):
        """处理友方雷达受敌方ECM干扰 - APG-68 ECCM能力"""
        try:
            # 遍历友方雷达跟踪的每个敌方目标
            for target_id in list(self.friendly_radar_targets[agent_id].keys()):
                # 检查该敌方目标是否激活了ECM来干扰我方雷达
                enemy_ecm_state = self.ecm_states.get(target_id, {})
                
                if enemy_ecm_state.get("active", False):
                    # 检查ECM是否过期
                    if current_time - enemy_ecm_state["start_time"] > enemy_ecm_state["duration"]:
                        enemy_ecm_state["active"] = False
                        enemy_ecm_state["type"] = None
                        logging.debug(f"🛡️ 敌方目标 {target_id} ECM结束")
                    else:
                        # 敌方ECM激活期间，降低友方雷达的探测概率
                        # APG-68具备更强的ECCM能力，受干扰影响较小
                        target = self.friendly_radar_targets[agent_id][target_id]
                        
                        if enemy_ecm_state["type"] == ECMType.NOISE_JAMMING:
                            target.detection_probability *= 0.5  # APG-68抗噪声干扰能力强
                        elif enemy_ecm_state["type"] == ECMType.DECEPTION_JAMMING:
                            target.detection_probability *= 0.6  # APG-68抗欺骗干扰能力强
                        # 箔条干扰未实现（APG-68的优秀ECCM能力可忽略箔条干扰）

        except Exception as e:
            logging.error(f"❌ {agent_id} 友方雷达受ECM干扰处理错误: {e}")
    
    def _activate_friendly_ecm_if_needed(self, env, agent_id: str, current_time: float):
        """友方威胁驱动ECM激活 - 基于RWR告警等级智能激活"""
        try:
            ecm_state = self.ecm_states[agent_id]
            
            # 如果ECM已激活，无需重复激活
            if ecm_state.get("active", False):
                return
            
            # 获取当前最高威胁等级
            rwr_data = self.rwr_states.get(agent_id, {})
            max_threat_level = rwr_data.get("threat_level", 0)
            
            # 威胁驱动激活策略：
            # - 威胁等级 ≥ 3 (LOCK): 80%概率激活（被锁定，高威胁）
            # - 威胁等级 = 2 (TRACK): 30%概率激活（被跟踪，中等威胁）
            # - 威胁等级 = 1 (SEARCH): 5%概率激活（被搜索，低威胁）
            # - 威胁等级 = 0 (无威胁): 不激活
            
            if max_threat_level >= 3:  # LOCK或导弹威胁
                activation_prob = 0.80
                threat_reason = "雷达锁定/导弹威胁"
            elif max_threat_level == 2:  # TRACK
                activation_prob = 0.30
                threat_reason = "雷达跟踪"
            elif max_threat_level == 1:  # SEARCH
                activation_prob = 0.05
                threat_reason = "雷达搜索"
            else:
                return  # 无威胁，不激活
            
            # 基于威胁等级的概率激活
            if random.random() < activation_prob:
                self._activate_ecm(agent_id, current_time)
                # 🔥 删除：不再打印ECM激活信息（用户要求删除无用打印）
                    
        except Exception as e:
            logging.error(f"❌ {agent_id} 友方ECM激活检查错误: {e}")

    def _calculate_distance(self, agent1, agent2) -> float:
        return _rrh.calculate_distance(self, agent1, agent2)
    

    def _calculate_bearing(self, agent, target) -> float:
        return _rrh.calculate_bearing(self, agent, target)

    def _calculate_elevation(self, agent, target) -> float:
        return _rrh.calculate_elevation(self, agent, target)

    # ==================== 新增：完备战术级雷达建模方法 ====================

    def _calculate_dynamic_rcs(self, agent, target, target_id: str) -> float:
        return _rrh.calculate_dynamic_rcs(self, agent, target, target_id)

    def _calculate_aspect_angle(self, agent, target) -> float:
        return _rrh.calculate_aspect_angle(self, agent, target)

    def _calculate_radial_velocity(self, agent, target) -> float:
        return _rrh.calculate_radial_velocity(self, agent, target)

    def _check_notch_condition(self, radial_velocity: float, radar_model) -> bool:
        return _rrh.check_notch_condition(self, radial_velocity, radar_model)

    def _calculate_ground_clutter_factor(self, elevation: float, altitude: float, distance: float, radar_model) -> float:
        return _rrh.calculate_ground_clutter_factor(self, elevation, altitude, distance, radar_model)

    def _check_terrain_masking(self, target_altitude: float, distance: float, elevation: float, radar_model) -> bool:
        return _rrh.check_terrain_masking(self, target_altitude, distance, elevation, radar_model)

    def _update_rwr_states(self, env, current_time: float):
        """
        更新RWR（雷达告警接收机）状态
        
        威胁等级：
        0 = 无威胁
        1 = SEARCH（搜索模式照射）
        2 = TRACK（跟踪模式照射）
        3 = LOCK（锁定模式照射）
        4 = MISSILE_LAUNCH（检测到导弹发射）
        5 = MISSILE_GUIDANCE（检测到导弹制导雷达）
        """
        try:
            # 友方RWR检测敌方雷达
            for friendly_id in ["A0100", "A0200"]:
                if friendly_id not in env.agents or not env.agents[friendly_id].is_alive:
                    continue
                
                threat_sources = []
                max_threat = 0
                
                for enemy_id in ["B0100", "B0200"]:
                    if enemy_id not in env.agents or not env.agents[enemy_id].is_alive:
                        continue
                    
                    # 检查敌方雷达状态
                    enemy_radar_state = self.enemy_radar_states.get(enemy_id, RadarStatus.SEARCH)
                    enemy_targets = self.enemy_radar_targets.get(enemy_id, {})
                    
                    threat_level = 0
                    if friendly_id in enemy_targets:
                        if enemy_radar_state == RadarStatus.LOCK:
                            threat_level = 3
                            # logging.warning(f"🚨 {friendly_id} RWR: {enemy_id} 雷达锁定告警！")
                        elif enemy_radar_state == RadarStatus.TRACK:
                            threat_level = 2
                            # logging.info(f"⚠️ {friendly_id} RWR: {enemy_id} 雷达跟踪告警")
                        elif enemy_radar_state == RadarStatus.SEARCH:
                            threat_level = 1
                    
                    if threat_level > 0:
                        threat_sources.append({
                            "source": enemy_id,
                            "level": threat_level,
                            "bearing": self._calculate_bearing(
                                env.agents[friendly_id], 
                                env.agents[enemy_id]
                            )
                        })
                        max_threat = max(max_threat, threat_level)
                
                self.rwr_states[friendly_id] = {
                    "threat_level": max_threat,
                    "threat_sources": threat_sources,
                    "last_warning": current_time if max_threat > 0 else self.rwr_states[friendly_id]["last_warning"]
                }
            
            # 敌方RWR检测友方雷达（对称）
            for enemy_id in ["B0100", "B0200"]:
                if enemy_id not in env.agents or not env.agents[enemy_id].is_alive:
                    continue
                
                threat_sources = []
                max_threat = 0
                
                for friendly_id in ["A0100", "A0200"]:
                    if friendly_id not in env.agents or not env.agents[friendly_id].is_alive:
                        continue
                    
                    friendly_radar_state = self.friendly_radar_states.get(friendly_id, RadarStatus.SEARCH)
                    friendly_targets = self.friendly_radar_targets.get(friendly_id, {})
                    
                    threat_level = 0
                    if enemy_id in friendly_targets:
                        if friendly_radar_state == RadarStatus.LOCK:
                            threat_level = 3
                            # logging.warning(f"🚨 {enemy_id} RWR: {friendly_id} 雷达锁定告警！")
                        elif friendly_radar_state == RadarStatus.TRACK:
                            threat_level = 2
                        elif friendly_radar_state == RadarStatus.SEARCH:
                            threat_level = 1
                    
                    if threat_level > 0:
                        threat_sources.append({
                            "source": friendly_id,
                            "level": threat_level,
                            "bearing": self._calculate_bearing(
                                env.agents[enemy_id],
                                env.agents[friendly_id]
                            )
                        })
                        max_threat = max(max_threat, threat_level)
                
                self.rwr_states[enemy_id] = {
                    "threat_level": max_threat,
                    "threat_sources": threat_sources,
                    "last_warning": current_time if max_threat > 0 else self.rwr_states[enemy_id]["last_warning"]
                }

        except Exception as e:
            logging.error(f"❌ RWR更新错误: {e}")
    
    # ==================== 敌方雷达系统方法 ====================

    def update_enemy_radar_states(self, env, current_time: float):

        """更新敌方雷达状态 - 真实N001VE雷达系统（完备战术级建模）"""
        for agent_id in self.enemy_radar_states.keys():
            if agent_id in env.agents and env.agents[agent_id].is_alive:
                self._update_single_enemy_radar(env, agent_id, current_time)

        
        # RWR状态已在友方更新中统一处理
    
    def _update_single_enemy_radar(self, env, agent_id: str, current_time: float):
        """更新单个敌方雷达状态 - 真实N001VE雷达物理特性"""
        try:
            # 检查任务完成状态
            alive_enemies = sum(1 for eid in ["A0100", "A0200"]
                               if eid in env.agents and env.agents[eid].is_alive)

            if alive_enemies == 0:
                self.enemy_radar_states[agent_id] = RadarStatus.STANDBY
                return

            # 更新雷达扫描
            self._update_enemy_radar_scan(env, agent_id, current_time)

            # 更新目标跟踪
            self._update_enemy_target_tracking(env, agent_id, current_time)

            # 更新雷达工作模式
            self._update_enemy_radar_mode(env, agent_id, current_time)

            # 处理电子战效果（敌方雷达受友方ECM干扰）
            self._process_electronic_warfare(env, agent_id, current_time)
            
            # 敌方随机激活自己的ECM来干扰友方雷达
            self._activate_enemy_ecm_if_needed(env, agent_id, current_time)

        except Exception as e:
            logging.error(f"❌ {agent_id} 敌方雷达状态更新错误: {e}")
            self.enemy_radar_states[agent_id] = RadarStatus.SEARCH

    def _update_enemy_radar_scan(self, env, agent_id: str, current_time: float):
        """更新敌方雷达扫描 - 真实物理建模"""
        try:
            # 检查扫描周期

            if current_time - self.enemy_scan_times[agent_id] < self.n001ve_radar.scan_period and self.enemy_scan_times[agent_id] != 0.0:
                return

            self.enemy_scan_times[agent_id] = current_time

            # 扫描友方目标
            friendly_agents = ["A0100", "A0200"]
            for target_id in friendly_agents:
                if target_id not in env.agents or not env.agents[target_id].is_alive:
                    continue

                # 计算目标参数

                agent = env.agents[agent_id]
                target = env.agents[target_id]
                distance = self._calculate_distance(agent, target)
                bearing = self._calculate_bearing(agent, target)
                elevation = self._calculate_elevation(agent, target)
                velocity = self._calculate_target_velocity(env, target_id)

                # 检查探测距离
                if distance > self.n001ve_radar.max_detection_range:
                    # 移除超出距离的目标
                    if target_id in self.enemy_radar_targets[agent_id]:
                        del self.enemy_radar_targets[agent_id][target_id]
                    continue


                # 计算完整的战术级探测概率
                detection_prob, radar_data = self._calculate_n001ve_detection_probability_complete(
                    agent, target, target_id, distance, bearing, elevation, velocity, current_time)

                # 探测成功
                if random.random() < detection_prob:
                    if target_id not in self.enemy_radar_targets[agent_id]:
                        # 创建新目标
                        self.enemy_radar_targets[agent_id][target_id] = RadarTarget(
                            target_id=target_id,
                            distance=distance,
                            bearing=bearing,
                            elevation=elevation,
                            velocity=velocity,
                            detection_probability=detection_prob,
                            last_update=current_time,
                            doppler_shift=self._calculate_doppler_shift(velocity, bearing),
                            snr=self._calculate_snr(distance, bearing, self.n001ve_radar),
                            multipath_factor=self._calculate_multipath_factor(distance, elevation),
                            atmospheric_loss=self._calculate_atmospheric_loss(distance)
                        )
                        logging.debug(f"🎯 {agent_id} N001VE雷达探测到新目标: {target_id} 距离={distance/1000:.1f}km")
                    else:
                        # 更新现有目标
                        target = self.enemy_radar_targets[agent_id][target_id]
                        target.distance = distance
                        target.bearing = bearing
                        target.elevation = elevation
                        target.velocity = velocity
                        target.detection_probability = detection_prob
                        target.last_update = current_time
                        target.doppler_shift = self._calculate_doppler_shift(velocity, bearing)
                        target.snr = self._calculate_snr(distance, bearing, self.n001ve_radar)
                        target.multipath_factor = self._calculate_multipath_factor(distance, elevation)
                        target.atmospheric_loss = self._calculate_atmospheric_loss(distance)
                else:
                    # 探测失败，移除目标
                    if target_id in self.enemy_radar_targets[agent_id]:
                        logging.debug(f"📡 {agent_id} N001VE雷达失去目标: {target_id} 距离={distance/1000:.1f}km")
                        del self.enemy_radar_targets[agent_id][target_id]

        except Exception as e:
            logging.error(f"❌ {agent_id} 敌方雷达扫描错误: {e}")
    

    def _calculate_target_velocity(self, env, target_id: str) -> float:
        return _rrh.calculate_target_velocity(self, env, target_id)

    def _calculate_n001ve_detection_probability_complete(self, agent, target, target_id: str,
                                                        distance: float, bearing: float,
                                                        elevation: float, velocity: float,
                                                        current_time: float) -> tuple:
        """
        计算完整的N001VE雷达探测概率 - 完备战术级建模
        
        返回: (detection_probability, radar_data_dict)
        """
        try:
            # 基础距离衰减 - N001VE分段与当前模型参数对齐
            if distance > self.n001ve_radar.max_detection_range:  # > 200km
                return 0.0, {}
            elif distance > 180000:  # 180-200km：边缘探测
                base_prob = 0.18
            elif distance > 170000:  # 170-180km：中等概率
                base_prob = 0.40
            elif distance > 120000:  # 120-170km：高概率
                base_prob = 0.85
            elif distance > 50000:  # 50-120km：极高概率
                base_prob = 0.92
            else:  # < 50km：最佳探测区
                base_prob = 0.95


            # ===== 新增：完备战术级因子 =====
            
            # 1. 动态RCS因子（姿态相关）
            dynamic_rcs = self._calculate_dynamic_rcs(agent, target, target_id)
            rcs_factor = min(1.15, math.log10(dynamic_rcs + 1) / math.log10(7))  # 归一化到0.7-1.15
            
            # 2. 计算径向速度和多普勒盲区
            radial_velocity = self._calculate_radial_velocity(agent, target)
            in_notch = self._check_notch_condition(radial_velocity, self.n001ve_radar)
            
            # 多普勒盲区 → 探测概率急剧下降（N001VE更严重）
            if in_notch:
                notch_factor = 0.08  # N001VE杂波抑制较差，只保留8%
                # logging.warning(f"🎯 {agent.uid} Notch检测: {target_id} 径向速度={radial_velocity:.1f}m/s（盲区）")
            else:
                # 高速接近目标更容易探测
                notch_factor = min(1.20, 1.0 + abs(radial_velocity) / 450.0)
            
            # 3. 地面杂波因子
            try:
                target_altitude = target.get_position()[2]
            except:
                target_altitude = 1000.0  # 默认中高空
            
            clutter_factor = self._calculate_ground_clutter_factor(
                elevation, target_altitude, distance, self.n001ve_radar)
            
            # 4. 地形遮蔽检查
            is_terrain_masked = self._check_terrain_masking(
                target_altitude, distance, elevation, self.n001ve_radar)
            
            if is_terrain_masked:
                # 目标被地形遮蔽，探测概率为0
                logging.debug(f"🏔️ {agent.uid} 目标 {target_id} 被地形遮蔽："
                            f"高度={target_altitude:.0f}m，距离={distance/1000:.1f}km，仰角={elevation:.1f}°")
                return 0.0, {
                    "rcs": dynamic_rcs,
                    "radial_velocity": radial_velocity,
                    "in_notch": in_notch,
                    "clutter_factor": clutter_factor,
                    "terrain_masked": True,
                    "base_prob": 0.0,
                    "final_prob": 0.0
                }

            # 5. 角度因子 - N001VE机械扫描（±70°），简化为固定值
            angle_factor = 0.90

            # 6. 仰角因子 - 低仰角性能更好（提高最小值）
            elevation_factor = max(0.8, 1.0 - abs(elevation) / 45.0)

            # 7. 大气衰减因子（减少衰减影响）
            atmospheric_factor = max(0.8, 1.0 - (distance / self.n001ve_radar.max_detection_range) *
                                   self.n001ve_radar.atmospheric_absorption * 500)


            # 8. 天气影响
            weather_factor = self.environmental_conditions["weather_factor"]


            # ===== 综合探测概率（N001VE完备模型） =====
            total_prob = (base_prob * rcs_factor * angle_factor * elevation_factor *

                         notch_factor * clutter_factor * atmospheric_factor * weather_factor)

            total_prob = min(1.0, max(0.0, total_prob))
            
            # 返回探测概率和详细数据
            radar_data = {
                "rcs": dynamic_rcs,
                "radial_velocity": radial_velocity,
                "in_notch": in_notch,
                "clutter_factor": clutter_factor,
                "terrain_masked": False,
                "base_prob": base_prob,
                "final_prob": total_prob
            }

            return total_prob, radar_data

        except Exception as e:

            logging.error(f"❌ N001VE完整探测概率计算错误: {e}")
            return 0.5, {}
    
    def _calculate_doppler_shift(self, velocity: float, bearing: float) -> float:
        return _rrh.calculate_doppler_shift(self, velocity, bearing)

    def _calculate_snr(self, distance: float, bearing: float, radar_model=None) -> float:
        return _rrh.calculate_snr(self, distance, bearing, radar_model)

    def _calculate_multipath_factor(self, distance: float, elevation: float) -> float:
        return _rrh.calculate_multipath_factor(self, distance, elevation)

    def _calculate_atmospheric_loss(self, distance: float) -> float:
        return _rrh.calculate_atmospheric_loss(self, distance)

    def _update_enemy_target_tracking(self, env, agent_id: str, current_time: float):
        """更新敌方目标跟踪"""
        try:
            # 清理过期目标
            expired_targets = []
            for target_id, target in self.enemy_radar_targets[agent_id].items():
                if current_time - target.last_update > 10.0:  # 10秒未更新
                    expired_targets.append(target_id)

            for target_id in expired_targets:
                del self.enemy_radar_targets[agent_id][target_id]
                logging.debug(f"📡 {agent_id} 清理过期目标: {target_id}")

            # 更新跟踪质量
            for target_id, target in self.enemy_radar_targets[agent_id].items():
                # 基于距离和时间更新跟踪质量
                time_factor = min(1.0, (current_time - target.last_update) / 5.0)
                distance_factor = max(0.1, 1.0 - target.distance / self.n001ve_radar.max_track_range)
                
                # 计算机动因子（公式3-25）- N001VE对机动目标更敏感
                maneuver_factor = 1.0
                if target_id in env.agents and env.agents[target_id].is_alive:
                    try:
                        # 获取目标滚转角
                        roll_rad = env.agents[target_id].get_property_value(c.attitude_phi_rad)
                        roll_deg = abs(np.rad2deg(roll_rad))
                        # N001VE机动因子：f_maneuver = 1.0 + 0.8 × (|φ_roll| / 90°)
                        maneuver_factor = 1.0 + 0.8 * (roll_deg / 90.0)
                    except:
                        maneuver_factor = 1.0
                
                # 完整的跟踪质量公式：Q = f_distance × (1 - f_time) / f_maneuver
                target.track_quality = distance_factor * (1.0 - time_factor) / maneuver_factor

                # 检查跟踪丢失
                if (target.track_quality < 0.2 or
                    target.distance > self.n001ve_radar.max_track_range or
                    random.random() < self.n001ve_radar.track_loss_probability):
                    expired_targets.append(target_id)

            # 移除跟踪丢失的目标
            for target_id in expired_targets:
                if target_id in self.enemy_radar_targets[agent_id]:
                    del self.enemy_radar_targets[agent_id][target_id]
                    logging.debug(f"📡 {agent_id} 跟踪丢失目标: {target_id}")

        except Exception as e:
            logging.error(f"❌ {agent_id} 目标跟踪更新错误: {e}")
    
    def _update_enemy_radar_mode(self, env, agent_id: str, current_time: float):
        """更新敌方雷达工作模式"""
        try:
            targets = self.enemy_radar_targets[agent_id]

            if not targets:
                self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
                self.enemy_lock_targets[agent_id] = None
                return

            # 找到最高优先级目标
            best_target = None
            best_priority = -1

            for target_id, target in targets.items():
                # 计算目标优先级（距离越近，优先级越高）
                priority = (1.0 / max(target.distance, 1000)) * target.track_quality
                if priority > best_priority:
                    best_priority = priority
                    best_target = target

            if not best_target:
                self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
                self.enemy_lock_targets[agent_id] = None
                return

            # 根据距离和跟踪质量确定雷达模式
            if (best_target.distance <= self.n001ve_radar.max_lock_range and
                best_target.track_quality > 0.6):
                self.enemy_radar_states[agent_id] = RadarStatus.LOCK
                self.enemy_lock_targets[agent_id] = best_target.target_id
                best_target.lock_time = current_time
            elif (best_target.distance <= self.n001ve_radar.max_track_range and
                  best_target.track_quality > 0.3):
                self.enemy_radar_states[agent_id] = RadarStatus.TRACK
                if self.enemy_lock_targets[agent_id] == best_target.target_id:
                    self.enemy_lock_targets[agent_id] = None
            else:
                self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
                self.enemy_lock_targets[agent_id] = None

        except Exception as e:
            logging.error(f"❌ {agent_id} 雷达模式更新错误: {e}")
            self.enemy_radar_states[agent_id] = RadarStatus.SEARCH
    
    def _process_electronic_warfare(self, env, agent_id: str, current_time: float):
        """处理敌方雷达受友方ECM干扰 - N001VE相对较弱的ECCM能力"""
        try:
            # 遍历敌方雷达跟踪的每个友方目标
            for target_id in list(self.enemy_radar_targets[agent_id].keys()):
                # 检查该友方目标是否激活了ECM来干扰敌方雷达
                friendly_ecm_state = self.ecm_states.get(target_id, {})
                
                if friendly_ecm_state.get("active", False):
                    # 检查ECM是否过期
                    if current_time - friendly_ecm_state["start_time"] > friendly_ecm_state["duration"]:
                        friendly_ecm_state["active"] = False
                        friendly_ecm_state["type"] = None
                        logging.debug(f"🛡️ 友方目标 {target_id} ECM结束")
                    else:
                        # 友方ECM激活期间，降低敌方雷达的探测概率
                        # N001VE的ECCM能力较弱，受干扰影响较大
                        target = self.enemy_radar_targets[agent_id][target_id]
                        
                        if friendly_ecm_state["type"] == ECMType.NOISE_JAMMING:
                            target.detection_probability *= 0.3  # N001VE抗噪声干扰能力弱
                        elif friendly_ecm_state["type"] == ECMType.DECEPTION_JAMMING:
                            target.detection_probability *= 0.5  # N001VE抗欺骗干扰能力一般
                        elif friendly_ecm_state["type"] == ECMType.CHAFF:
                            target.detection_probability *= 0.2  # N001VE抗箔条干扰能力弱
                            
        except Exception as e:
            logging.error(f"❌ {agent_id} 敌方雷达受ECM干扰处理错误: {e}")

    def _activate_enemy_ecm_if_needed(self, env, agent_id: str, current_time: float):
        """敌方威胁驱动ECM激活 - 基于RWR告警等级智能激活"""
        try:
            ecm_state = self.ecm_states[agent_id]
            
            # 如果ECM已激活，无需重复激活
            if ecm_state.get("active", False):
                return
            
            # 获取当前最高威胁等级
            rwr_data = self.rwr_states.get(agent_id, {})
            max_threat_level = rwr_data.get("threat_level", 0)
            
            # 威胁驱动激活策略（敌方N001VE雷达性能较弱，更依赖ECM防御）：
            # - 威胁等级 ≥ 3 (LOCK): 90%概率激活（被锁定，立即激活）
            # - 威胁等级 = 2 (TRACK): 40%概率激活（被跟踪，积极防御）
            # - 威胁等级 = 1 (SEARCH): 10%概率激活（被搜索，预防性激活）
            # - 威胁等级 = 0 (无威胁): 不激活
            
            if max_threat_level >= 3:  # LOCK或导弹威胁
                activation_prob = 0.90  # 敌方更积极（ECCM能力弱）
                threat_reason = "雷达锁定/导弹威胁"
            elif max_threat_level == 2:  # TRACK
                activation_prob = 0.40
                threat_reason = "雷达跟踪"
            elif max_threat_level == 1:  # SEARCH
                activation_prob = 0.10
                threat_reason = "雷达搜索"
            else:
                return  # 无威胁，不激活
            
            # 基于威胁等级的概率激活
            if random.random() < activation_prob:
                self._activate_ecm(agent_id, current_time)
                # 🔥 删除：不再打印ECM激活信息（用户要求删除无用打印）
                    
        except Exception as e:
            logging.error(f"❌ {agent_id} 敌方ECM激活检查错误: {e}")

    def _activate_ecm(self, agent_id: str, current_time: float):
        """激活电子对抗措施"""
        try:
            ecm_types = [ECMType.NOISE_JAMMING, ECMType.DECEPTION_JAMMING, ECMType.CHAFF]
            ecm_type = random.choice(ecm_types)
            duration = random.uniform(5.0, 15.0)  # 5-15秒

            self.ecm_states[agent_id] = {
                "active": True,
                "type": ecm_type,
                "start_time": current_time,
                "duration": duration
            }

            logging.debug(f"🛡️ {agent_id} 激活ECM: {ecm_type.value} 持续{duration:.1f}秒")

        except Exception as e:
            logging.error(f"❌ {agent_id} ECM激活错误: {e}")
    
    # ==================== 统一接口方法 ====================

    def get_friendly_radar_state(self, agent_id: str) -> RadarStatus:
        """获取友方雷达状态"""
        return self.friendly_radar_states.get(agent_id, RadarStatus.SEARCH)

    def get_enemy_radar_state(self, agent_id: str) -> RadarStatus:
        """获取敌方雷达状态"""
        return self.enemy_radar_states.get(agent_id, RadarStatus.SEARCH)

    def get_enemy_radar_targets(self, agent_id: str) -> Dict[str, RadarTarget]:
        """获取敌方雷达目标"""
        return self.enemy_radar_targets.get(agent_id, {})

    def get_enemy_lock_target(self, agent_id: str) -> Optional[str]:
        """获取敌方锁定目标"""
        return self.enemy_lock_targets.get(agent_id, None)

    def is_enemy_ecm_active(self, agent_id: str) -> bool:
        """检查敌方ECM是否激活"""
        return self.ecm_states.get(agent_id, {}).get("active", False)

    def get_enemy_ecm_type(self, agent_id: str) -> Optional[ECMType]:
        """获取敌方ECM类型"""
        return self.ecm_states.get(agent_id, {}).get("type", None)
    
    # ==================== RWR系统接口（文档3.7.1节）====================
    
    def get_rwr_threat_level(self, agent_id: str) -> int:
        """
        获取RWR威胁等级（文档表3.2）
        
        Args:
            agent_id: 本机ID
            
        Returns:
            int: 威胁等级 0-5
                0 = 无威胁
                1 = SEARCH（被搜索）
                2 = TRACK（被跟踪）
                3 = LOCK（被锁定）
                4 = MISSILE_LAUNCH（导弹发射）
                5 = MISSILE_GUIDANCE（导弹制导）
        """
        return self.rwr_states.get(agent_id, {}).get("threat_level", 0)
    
    def get_rwr_threat_sources(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        获取RWR探测到的威胁源列表及方位（文档3.7.1节）
        
        Args:
            agent_id: 本机ID
            
        Returns:
            List[Dict]: 威胁源列表，每个元素包含：
                - source: 威胁源ID
                - level: 威胁等级
                - bearing: 方位角（度）
                
        Example:
            [{"source": "B0100", "level": 3, "bearing": 45.2},
             {"source": "B0200", "level": 2, "bearing": 120.5}]
        """
        return self.rwr_states.get(agent_id, {}).get("threat_sources", [])
    
    def get_rwr_max_threat_bearing(self, agent_id: str) -> Optional[float]:
        """
        获取最高威胁源的方位角θ_threat（文档3.7.1节）
        
        Args:
            agent_id: 本机ID
            
        Returns:
            Optional[float]: 最高威胁源的方位角（度），无威胁时返回None
        """
        threat_sources = self.get_rwr_threat_sources(agent_id)
        if not threat_sources:
            return None
        
        # 找到威胁等级最高的源
        max_threat = max(threat_sources, key=lambda x: x["level"])
        return max_threat.get("bearing")
    
    # ==================== 导弹发射集成检查（文档3.7.2节）====================
    
    def check_missile_launch_conditions(self, env, shooter_id: str, target_id: str) -> Dict[str, Any]:
        """
        检查导弹发射条件（文档3.7.2节完整的6项检查）
        
        Args:
            env: 环境对象
            shooter_id: 发射机ID
            target_id: 目标ID
            
        Returns:
            Dict包含：
                - can_launch: bool，是否可以发射
                - conditions: Dict，各项条件的检查结果
                - reason: str，不能发射的原因（如果can_launch=False）
        """
        try:
            # ✅ 复盘日志：记录3.7.2检查耗时与关键中间量（变化触发，避免刷屏）
            try:
                from utils.trace_logger import trace_span
            except Exception:
                trace_span = None

            span_ctx = (
                trace_span(
                    标题="3.7.2发射条件检查",
                    env=env,
                    模块="radar_manager",
                    类型="GATE",
                    我机=str(shooter_id),
                    敌机=str(target_id),
                    说明="雷达侧综合发射条件检查（跟踪列表/模式/Q/距离/Notch/Pd）",
                    log_start=False,
                    slow_ms=80.0,
                )
                if trace_span is not None
                else None
            )

            if span_ctx is not None:
                span_ctx.__enter__()

            # 判断发射机是友方还是敌方
            # 🔥 修复：支持4v4 — A0300/A0400也是友方
            is_friendly = shooter_id.startswith('A')
            
            # ✅ 雷达型号对调（按用户要求）：
            # - 我方(A*) 使用 N001VE 雷达模型参数
            # - 敌方(B*) 使用 APG-68 雷达模型参数
            if is_friendly:
                radar_targets = self.friendly_radar_targets.get(shooter_id, {})
                radar_state = self.friendly_radar_states.get(shooter_id, RadarStatus.SEARCH)
                radar_model = self.n001ve_radar
            else:
                radar_targets = self.enemy_radar_targets.get(shooter_id, {})
                radar_state = self.enemy_radar_states.get(shooter_id, RadarStatus.SEARCH)
                radar_model = self.apg68_radar
            
            # 初始化检查结果
            conditions = {
                "in_track_list": False,
                "radar_mode_ok": False,
                "track_quality_ok": False,
                "in_range": False,
                "not_in_notch": False,
                "detection_prob_ok": False
            }
            
            # 条件1: 目标在雷达跟踪列表中
            if target_id not in radar_targets:
                try:
                    from utils.trace_logger import trace_if_changed
                    trace_if_changed(
                        key=f"radar_gate:{shooter_id}:{target_id}",
                        value=(False, "not_in_track_list", str(radar_state.value)),
                        标题="雷达门限-3.7.2",
                        env=env,
                        模块="radar_manager",
                        类型="GATE",
                        状态="DENY",
                        我机=str(shooter_id),
                        敌机=str(target_id),
                        说明="目标不在雷达跟踪列表，禁止发射",
                        数据={
                            "radar_state": radar_state.value if hasattr(radar_state, 'value') else str(radar_state),
                            "is_friendly": bool(is_friendly),
                            "track_count": int(len(radar_targets)) if isinstance(radar_targets, dict) else None,
                            "conditions": conditions,
                        },
                    )
                except Exception:
                    pass
                return {
                    "can_launch": False,
                    "conditions": conditions,
                    "reason": f"目标{target_id}不在雷达跟踪列表中"
                }
            conditions["in_track_list"] = True
            
            target_data = radar_targets[target_id]
            
            # ===== 门槛参数（友/敌可区分） =====
            # 说明：原逻辑要求 TRACK/LOCK + Q>=0.3 + Pd>=0.2，实战中会导致长时间无法满足从而“几乎不发射”。
            # 这里做“受控放宽”：
            # - TRACK/LOCK 仍视为强条件
            # - 友方允许在 SEARCH 且目标已稳定探测/跟踪时（Q、Pd足够）走“软发射”通道
            # - 敌方略放宽但不允许 SEARCH 软发射（避免过度增强敌方）
            min_track_q = 0.20 if is_friendly else 0.25
            min_detect_p = 0.15 if is_friendly else 0.18

            # 条件2: 雷达处于TRACK或LOCK模式（或友方SEARCH软通道）
            soft_mode_ok = False
            if radar_state in [RadarStatus.TRACK, RadarStatus.LOCK]:
                conditions["radar_mode_ok"] = True
            else:
                # 友方：SEARCH时若目标已较稳定探测，可允许“软发射”（仍受Q/Pd/Notch/距离约束）
                if is_friendly and radar_state == RadarStatus.SEARCH:
                    if target_data.track_quality >= (min_track_q + 0.05) and target_data.detection_probability >= (min_detect_p + 0.10):
                        soft_mode_ok = True
                        conditions["radar_mode_ok"] = True
                if not conditions["radar_mode_ok"]:
                    try:
                        from utils.trace_logger import trace_if_changed
                        trace_if_changed(
                            key=f"radar_gate:{shooter_id}:{target_id}",
                            value=(False, "mode_not_ok", str(radar_state.value)),
                            标题="雷达门限-3.7.2",
                            env=env,
                            模块="radar_manager",
                            类型="GATE",
                            状态="DENY",
                            我机=str(shooter_id),
                            敌机=str(target_id),
                            说明="雷达模式不满足TRACK/LOCK或SEARCH软通道条件",
                            数据={
                                "radar_state": radar_state.value if hasattr(radar_state, 'value') else str(radar_state),
                                "soft_mode_ok": bool(soft_mode_ok),
                                "min_track_q": float(min_track_q),
                                "min_detect_p": float(min_detect_p),
                                "track_quality": float(getattr(target_data, 'track_quality', -1.0)),
                                "detection_probability": float(getattr(target_data, 'detection_probability', -1.0)),
                                "conditions": conditions,
                            },
                        )
                    except Exception:
                        pass
                    return {
                        "can_launch": False,
                        "conditions": conditions,
                        "reason": f"雷达模式为{radar_state.value}，需要TRACK/LOCK（或友方SEARCH软通道）"
                    }

            # 条件3: 跟踪质量达到最低要求
            if target_data.track_quality < min_track_q:
                try:
                    from utils.trace_logger import trace_if_changed
                    trace_if_changed(
                        key=f"radar_gate:{shooter_id}:{target_id}",
                        value=(False, "track_quality_low", str(radar_state.value)),
                        标题="雷达门限-3.7.2",
                        env=env,
                        模块="radar_manager",
                        类型="GATE",
                        状态="DENY",
                        我机=str(shooter_id),
                        敌机=str(target_id),
                        说明="跟踪质量不足，禁止发射",
                        数据={
                            "radar_state": radar_state.value if hasattr(radar_state, 'value') else str(radar_state),
                            "track_quality": float(getattr(target_data, 'track_quality', -1.0)),
                            "min_track_q": float(min_track_q),
                            "conditions": conditions,
                        },
                    )
                except Exception:
                    pass
                return {
                    "can_launch": False,
                    "conditions": conditions,
                    "reason": f"跟踪质量{target_data.track_quality:.2f}低于{min_track_q:.2f}"
                }
            conditions["track_quality_ok"] = True

            # 条件4: 目标在最大跟踪距离内
            # 软通道仍使用 max_track_range（避免在“仅探测到”但无法持续跟踪的极远距离盲发）
            if target_data.distance > radar_model.max_track_range:
                try:
                    from utils.trace_logger import trace_if_changed
                    trace_if_changed(
                        key=f"radar_gate:{shooter_id}:{target_id}",
                        value=(False, "out_of_range", str(radar_state.value)),
                        标题="雷达门限-3.7.2",
                        env=env,
                        模块="radar_manager",
                        类型="GATE",
                        状态="DENY",
                        我机=str(shooter_id),
                        敌机=str(target_id),
                        说明="目标超出最大跟踪距离，禁止发射",
                        数据={
                            "target_distance_km": float(getattr(target_data, 'distance', 0.0)) / 1000.0,
                            "max_track_range_km": float(getattr(radar_model, 'max_track_range', 0.0)) / 1000.0,
                            "conditions": conditions,
                        },
                    )
                except Exception:
                    pass
                return {
                    "can_launch": False,
                    "conditions": conditions,
                    "reason": f"目标距离{target_data.distance/1000:.1f}km超出最大跟踪距离{radar_model.max_track_range/1000:.0f}km"
                }
            conditions["in_range"] = True
            
            # 条件5: 目标不在多普勒盲区
            # 计算径向速度
            radial_velocity = None
            in_notch = None
            if shooter_id in env.agents and target_id in env.agents:
                shooter = env.agents[shooter_id]
                target = env.agents[target_id]
                radial_velocity = self._calculate_radial_velocity(shooter, target)
                
                # 检查是否在Notch盲区
                in_notch = self._check_notch_condition(radial_velocity, radar_model)
                if in_notch:
                    try:
                        from utils.trace_logger import trace_if_changed
                        trace_if_changed(
                            key=f"radar_gate:{shooter_id}:{target_id}",
                            value=(False, "in_notch", str(radar_state.value)),
                            标题="雷达门限-3.7.2",
                            env=env,
                            模块="radar_manager",
                            类型="GATE",
                            状态="DENY",
                            我机=str(shooter_id),
                            敌机=str(target_id),
                            说明="目标处于多普勒盲区(Notch)，禁止发射",
                            数据={
                                "radial_velocity_mps": float(radial_velocity) if radial_velocity is not None else None,
                                "conditions": conditions,
                            },
                        )
                    except Exception:
                        pass
                    return {
                        "can_launch": False,
                        "conditions": conditions,
                        "reason": f"目标在多普勒盲区（径向速度{radial_velocity:.1f}m/s）"
                    }
            conditions["not_in_notch"] = True
            
            # 条件6: 探测概率达到最低要求
            if target_data.detection_probability < min_detect_p:
                try:
                    from utils.trace_logger import trace_if_changed
                    trace_if_changed(
                        key=f"radar_gate:{shooter_id}:{target_id}",
                        value=(False, "pd_low", str(radar_state.value)),
                        标题="雷达门限-3.7.2",
                        env=env,
                        模块="radar_manager",
                        类型="GATE",
                        状态="DENY",
                        我机=str(shooter_id),
                        敌机=str(target_id),
                        说明="探测概率不足，禁止发射",
                        数据={
                            "detection_probability": float(getattr(target_data, 'detection_probability', -1.0)),
                            "min_detect_p": float(min_detect_p),
                            "conditions": conditions,
                        },
                    )
                except Exception:
                    pass
                return {
                    "can_launch": False,
                    "conditions": conditions,
                    "reason": f"探测概率{target_data.detection_probability:.2f}低于{min_detect_p:.2f}"
                }
            conditions["detection_prob_ok"] = True
            
            # 所有条件满足
            if soft_mode_ok:
                try:
                    from utils.trace_logger import trace_if_changed
                    trace_if_changed(
                        key=f"radar_gate:{shooter_id}:{target_id}",
                        value=(True, "soft_mode_ok", str(radar_state.value)),
                        标题="雷达门限-3.7.2",
                        env=env,
                        模块="radar_manager",
                        类型="GATE",
                        状态="ALLOW",
                        我机=str(shooter_id),
                        敌机=str(target_id),
                        说明="友方SEARCH软通道满足，允许发射",
                        数据={
                            "radar_state": radar_state.value if hasattr(radar_state, 'value') else str(radar_state),
                            "soft_mode_ok": True,
                            "min_track_q": float(min_track_q),
                            "min_detect_p": float(min_detect_p),
                            "track_quality": float(getattr(target_data, 'track_quality', -1.0)),
                            "detection_probability": float(getattr(target_data, 'detection_probability', -1.0)),
                            "target_distance_km": float(getattr(target_data, 'distance', 0.0)) / 1000.0,
                            "radial_velocity_mps": float(radial_velocity) if radial_velocity is not None else None,
                            "in_notch": bool(in_notch) if in_notch is not None else None,
                            "conditions": conditions,
                        },
                    )
                except Exception:
                    pass
                return {
                    "can_launch": True,
                    "conditions": conditions,
                    "reason": "友方SEARCH软通道发射条件满足"
                }

            try:
                from utils.trace_logger import trace_if_changed
                trace_if_changed(
                    key=f"radar_gate:{shooter_id}:{target_id}",
                    value=(True, "all_ok", str(radar_state.value)),
                    标题="雷达门限-3.7.2",
                    env=env,
                    模块="radar_manager",
                    类型="GATE",
                    状态="ALLOW",
                    我机=str(shooter_id),
                    敌机=str(target_id),
                    说明="全部条件满足，允许发射",
                    数据={
                        "radar_state": radar_state.value if hasattr(radar_state, 'value') else str(radar_state),
                        "soft_mode_ok": False,
                        "min_track_q": float(min_track_q),
                        "min_detect_p": float(min_detect_p),
                        "track_quality": float(getattr(target_data, 'track_quality', -1.0)),
                        "detection_probability": float(getattr(target_data, 'detection_probability', -1.0)),
                        "target_distance_km": float(getattr(target_data, 'distance', 0.0)) / 1000.0,
                        "radial_velocity_mps": float(radial_velocity) if radial_velocity is not None else None,
                        "in_notch": bool(in_notch) if in_notch is not None else None,
                        "conditions": conditions,
                    },
                )
            except Exception:
                pass

            return {
                "can_launch": True,
                "conditions": conditions,
                "reason": "所有发射条件满足"
            }
            
        except Exception as e:
            logging.error(f"❌ 导弹发射条件检查错误: {e}")
            try:
                from utils.trace_logger import trace_exception
                trace_exception(
                    事件="3.7.2发射条件检查错误",
                    env=env,
                    模块="radar_manager",
                    类型="GATE",
                    我机=str(shooter_id),
                    敌机=str(target_id),
                    异常=e,
                )
            except Exception:
                pass
            return {
                "can_launch": False,
                "conditions": {},
                "reason": f"检查过程出错: {e}"
            }
        finally:
            try:
                if span_ctx is not None:
                    span_ctx.__exit__(None, None, None)
            except Exception:
                pass
    
    def is_being_jammed(self, agent_id: str) -> bool:
        return _rrh.is_being_jammed(self, agent_id)

    def get_jamming_sources(self, agent_id: str) -> Dict[str, ECMType]:
        return _rrh.get_jamming_sources(self, agent_id)

    def update_environmental_conditions(self, weather_factor: float = 1.0, terrain_height: float = 0.0, atmospheric_density: float = 1.0, temperature: float = 15.0, humidity: float = 50.0):
        _rrh.update_environmental_conditions(
            self,
            weather_factor=weather_factor,
            terrain_height=terrain_height,
            atmospheric_density=atmospheric_density,
            temperature=temperature,
            humidity=humidity,
        )

    def get_radar_performance_summary(self) -> Dict[str, Any]:
        return _rrh.get_radar_performance_summary(self)

    def record_radar_data(self, env, current_time: float) -> List[Dict[str, Any]]:
        return _rrh.record_radar_data(self, env, current_time)

    def _find_closest_target_id(self, env, agent_id: str) -> Optional[str]:
        return _rrh.find_closest_target_id(self, env, agent_id)

    def _get_target_distance(self, env, agent_id: str, target_id: str) -> float:
        return _rrh.get_target_distance(self, env, agent_id, target_id)

_unified_radar_manager = None

def get_unified_radar_manager() -> UnifiedRadarManager:
    """获取全局统一雷达管理器实例"""
    global _unified_radar_manager
    if _unified_radar_manager is None:
        _unified_radar_manager = UnifiedRadarManager()
        logging.info("[雷达管理] 全局统一雷达管理器实例已创建")
    return _unified_radar_manager

def update_all_radars(env, current_time: float):
    """更新所有雷达状态 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    radar_manager.update_friendly_radar_states(env, current_time)
    radar_manager.update_enemy_radar_states(env, current_time)

    # ✅ 验证：每60秒输出一次雷达状态快照（SEARCH/TRACK/LOCK + lock目标 + ECM）
    try:
        step = getattr(env, "current_step", None)
        if step is not None and step % 300 == 0:
            for aid in ("A0100", "A0200"):
                st = radar_manager.friendly_radar_states.get(aid, RadarStatus.SEARCH)
                lk = radar_manager.friendly_lock_targets.get(aid, None)
                ecm = radar_manager.ecm_states.get(aid, False)
                logging.info(f"[RADAR-{aid}] state={st.value} lock={lk} ecm={'ON' if ecm else 'OFF'}")
            # 🔥 调试：敌方雷达状态日志不打印
            # for bid in ("B0100", "B0200"):
            #     st = radar_manager.enemy_radar_states.get(bid, RadarStatus.SEARCH)
            #     lk = radar_manager.enemy_lock_targets.get(bid, None)
            #     ecm = radar_manager.ecm_states.get(bid, False)
            #     logging.info(f"📡 [RADAR-{bid}] state={st.value} lock={lk} ecm={'ON' if ecm else 'OFF'}")
    except Exception:
        pass

def get_friendly_radar_state(agent_id: str) -> RadarStatus:
    """获取友方雷达状态 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_friendly_radar_state(agent_id)

def get_enemy_radar_state(agent_id: str) -> RadarStatus:
    """获取敌方雷达状态 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_enemy_radar_state(agent_id)

def get_enemy_radar_targets(agent_id: str) -> Dict[str, RadarTarget]:
    """获取敌方雷达目标 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_enemy_radar_targets(agent_id)

def get_enemy_lock_target(agent_id: str) -> Optional[str]:
    """获取敌方锁定目标 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_enemy_lock_target(agent_id)

def is_enemy_ecm_active(agent_id: str) -> bool:
    """检查敌方ECM是否激活 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.is_enemy_ecm_active(agent_id)

def is_being_jammed(agent_id: str) -> bool:
    """查询本机是否正在受到ECM干扰 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.is_being_jammed(agent_id)

def get_jamming_sources(agent_id: str) -> Dict[str, ECMType]:
    """获取正在干扰本机的敌方列表及干扰类型 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_jamming_sources(agent_id)

def get_radar_performance_summary() -> Dict[str, Any]:
    """获取雷达性能摘要 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_radar_performance_summary()

# ==================== RWR系统全局接口（文档3.7.1节）====================

def get_rwr_threat_level(agent_id: str) -> int:
    """获取RWR威胁等级（文档表3.2）- 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_rwr_threat_level(agent_id)

def get_rwr_threat_sources(agent_id: str) -> List[Dict[str, Any]]:
    """获取RWR威胁源列表及方位 - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_rwr_threat_sources(agent_id)

def get_rwr_max_threat_bearing(agent_id: str) -> Optional[float]:
    """获取最高威胁源方位角θ_threat - 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.get_rwr_max_threat_bearing(agent_id)

# ==================== 导弹发射集成检查全局接口（文档3.7.2节）====================

def check_missile_launch_conditions(env, shooter_id: str, target_id: str) -> Dict[str, Any]:
    """检查导弹发射条件（文档3.7.2节完整6项检查）- 统一接口"""
    radar_manager = get_unified_radar_manager()
    return radar_manager.check_missile_launch_conditions(env, shooter_id, target_id)

# ==================== 向后兼容接口 ====================

# 为了保持向后兼容性，保留旧的接口名称
RadarManager = UnifiedRadarManager
get_radar_manager = get_unified_radar_manager

def get_friendly_radar_states() -> Dict[str, str]:

    """获取我方雷达状态（向后兼容）"""
    radar_manager = get_radar_manager()

    return {agent_id: status.value for agent_id, status in radar_manager.friendly_radar_states.items()}

def get_enemy_radar_states() -> Dict[str, str]:

    """获取敌方雷达状态（向后兼容）"""
    radar_manager = get_radar_manager()

    return {agent_id: status.value for agent_id, status in radar_manager.enemy_radar_states.items()}

def get_enemy_radar_data() -> Dict[str, Dict[str, Any]]:

    """获取敌方雷达数据（向后兼容）"""
    radar_manager = get_radar_manager()

    result = {}
    for agent_id, targets in radar_manager.enemy_radar_targets.items():
        result[agent_id] = {
            'status': radar_manager.enemy_radar_states.get(agent_id, RadarStatus.SEARCH).value,
            'targets': {tid: {'distance': t.distance, 'bearing': t.bearing} for tid, t in targets.items()},
            'lock_target': radar_manager.enemy_lock_targets.get(agent_id, None)
        }
    return result 
