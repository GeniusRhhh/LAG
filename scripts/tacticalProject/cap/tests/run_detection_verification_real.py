"""协同探测验证 - 真实仿真验证脚本

对比要求：3个场景 × (proposed/baseline) = 6次运行。

输出目录: cap_results/Cooperative_detection/
"""
import os
# 修复OpenMP冲突错误 (OMP: Error #15 Initializing libiomp5md.dll, but found libiomp5md.dll already initialized)
# 必须由于numpy等库导入之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import logging
import numpy as np
import argparse
import shutil
import yaml
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
cap_dir = os.path.dirname(current_dir)
tactical_dir = os.path.dirname(cap_dir)
project_root = os.path.dirname(os.path.dirname(tactical_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, tactical_dir)
sys.path.insert(0, cap_dir)

# 飞机模型配置
MY_AIRCRAFT_TYPE = 'su27sk'
ENEMY_AIRCRAFT_TYPE = 'f16'

os.environ["FRIEND_BASELINE_MODEL"] = "SU27" if MY_AIRCRAFT_TYPE == 'su27sk' else "F16"
os.environ["ENEMY_BASELINE_MODEL"] = "SU27" if ENEMY_AIRCRAFT_TYPE == 'su27sk' else "F16"
os.environ.setdefault("CAP_ROOTCAUSE_TRACE", "1")

# Enable stronger flight-envelope protection in JSBSim for verification runs.
# This prevents early termination due to brief low-speed/altitude excursions while
# still keeping the physics realistic enough for Tacview validation.
os.environ.setdefault("CAP_VERIFICATION", "1")

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c

# 导入场景定义
from scenario_generator import (
    ScenarioGenerator, EnemyScenario, FormationType, AltitudeProfile,
    ApproachDirection, SpeedProfile, ManeuverType, AwacsStatus,
    SPEED_NORMAL, SPEED_HIGH, SPEED_LOW
)

# 输出目录
OUTPUT_BASE_DIR = os.path.join(tactical_dir, "cap_results", "Cooperative_detection")

# 坐标系参数
A0100_LON = 120.6757
A0100_LAT = 60.0
DEG_TO_KM = 111.0

# 速度常量 (fps)
FPS_NORMAL = 1251.1   # 普通速度
FPS_LOW = 900.0       # 低速
FPS_HIGH = 1500.0     # 高速


class CAPEnvForVerification(MultipleCombatEnv):
    """验证用CAP仿真环境"""
    def __init__(self, config_name):
        super().__init__(config_name)

    def load_simulator(self):
        for uid, conf in self.config.aircraft_configs.items():
            if uid.startswith('A'):
                conf['model'] = MY_AIRCRAFT_TYPE
            elif uid.startswith('B'):
                conf['model'] = ENEMY_AIRCRAFT_TYPE
        super().load_simulator()
        for uid, sim in self._jsbsims.items():
            sim._display_name = uid


def setup_logging(output_dir: str, scenario_id: str) -> str:
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{timestamp}_{scenario_id}.log")
    rootcause_file = os.path.join(output_dir, f"{timestamp}_{scenario_id}_rootcause.log")
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    file_fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    class _SafeConsoleFormatter(logging.Formatter):
        """Format logs and replace characters not supported by the console encoding.

        This avoids UnicodeEncodeError on Windows (GBK) when messages contain symbols
        like ✅/❌/🛫 that are still useful in UTF-8 log files.
        """

        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            s = super().format(record)
            enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
            try:
                s.encode(enc)
                return s
            except Exception:
                return s.encode(enc, errors='replace').decode(enc, errors='replace')

    class _VerificationNoiseFilter(logging.Filter):
        """Suppress high-frequency debug-tagged logs that can flood output."""

        _DROP_SUBSTRINGS = (
            "[DEBUG normalize_action",
            "[DEBUG _lowlevel_control]",
            "[DEBUG normalize_action V1最终]",
            "[normalize_action]",
            "[航向命令V3]",
        )

        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return not any(substr in msg for substr in self._DROP_SUBSTRINGS)

    class _NoRootCauseFilter(logging.Filter):
        """Drop root-cause trace lines from normal console/main log handlers."""

        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return '根因链-' not in msg and '友机根因记录' not in msg

    class _RootCauseOnlyFilter(logging.Filter):
        """Allow only root-cause trace lines for the dedicated trace file."""

        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            try:
                msg = record.getMessage()
            except Exception:
                return False
            return '根因链-' in msg or '友机根因记录' in msg

    fh = logging.FileHandler(log_file, encoding='utf-8')
    # 验证日志以“过程叙事+关键指标”为主；关闭环境/底层的高频DEBUG刷屏。
    fh.setLevel(logging.INFO)
    fh.setFormatter(file_fmt)
    fh.addFilter(_VerificationNoiseFilter())
    fh.addFilter(_NoRootCauseFilter())

    rh = logging.FileHandler(rootcause_file, encoding='utf-8')
    rh.setLevel(logging.INFO)
    rh.setFormatter(file_fmt)
    rh.addFilter(_RootCauseOnlyFilter())

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_SafeConsoleFormatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    ch.addFilter(_VerificationNoiseFilter())
    ch.addFilter(_NoRootCauseFilter())

    logging.basicConfig(level=logging.INFO, handlers=[fh, rh, ch], force=True)

    # 压制JSBSim环境/任务的调试噪声（这些模块内部有大量debug输出）
    for noisy in (
        'envs',
        'envs.JSBSim',
        'envs.JSBSim.envs',
        'envs.JSBSim.tasks',
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"根因链日志已单独输出: {rootcause_file}")
    
    return log_file


def _safe_append_csv_row(csv_path: str, row: list, *, encoding: str = 'utf-8-sig', retries: int = 8) -> str:
    """Append one CSV row with Windows-friendly retry/fallback.

    On Windows, Excel / Explorer Preview can lock the file and raise PermissionError.
    We retry briefly; if still locked we write to a fallback file so a full run doesn't
    crash and lose results.

    Returns the path that was actually written.
    """
    import csv as _csv

    last_err: Optional[Exception] = None
    for i in range(max(1, int(retries))):
        try:
            with open(csv_path, 'a', newline='', encoding=encoding) as f:
                w = _csv.writer(f)
                w.writerow(row)
            return csv_path
        except PermissionError as e:
            last_err = e
            time.sleep(0.25 + 0.15 * i)

    base, ext = os.path.splitext(csv_path)
    fallback = f"{base}_fallback{ext}"
    with open(fallback, 'a', newline='', encoding=encoding) as f:
        w = _csv.writer(f)
        w.writerow(row)
    logging.warning(f"[CSV] 目标文件被占用，已写入fallback: {fallback}")
    logging.warning("[CSV] 请关闭Excel/资源管理器预览后再重试生成主CSV")
    if last_err is not None:
        logging.warning(f"[CSV] 原始错误: {last_err}")
    return fallback


def _scenario_seed(base_seed: int, scenario_id: str) -> int:
    """为每个场景生成稳定seed（同场景 proposed/baseline 保持一致）。"""
    s = (scenario_id or '').strip().upper()
    # 支持 V01/V02/V03 或 S01 等
    digits = ''.join([ch for ch in s if ch.isdigit()])
    try:
        idx = int(digits) if digits else 0
    except Exception:
        idx = 0
    return int(base_seed) + idx


def battlefield_to_geodetic(x_km: float, y_km: float) -> Tuple[float, float]:
    """战场坐标转地球坐标 (经度, 纬度)"""
    # A0100在战场坐标(75, 0)
    dx = x_km - 75.0
    dy = y_km - 0.0
    lon = A0100_LON + dx / DEG_TO_KM
    lat = A0100_LAT + dy / DEG_TO_KM
    return lon, lat


def generate_scenario_config(scenario: EnemyScenario, base_config_path: str) -> dict:
    """
    根据场景生成敌方配置
    
    Args:
        scenario: 敌方场景配置
        base_config_path: 基础配置文件路径
    
    Returns:
        修改后的配置字典
    """
    # 读取基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 生成敌机初始位置和速度
    positions = scenario.generate_initial_positions(base_y=None)  # 使用场景定义的距离
    headings = scenario.get_initial_headings()
    
    # 速度配置
    if scenario.speed == SpeedProfile.NORMAL:
        fps = FPS_NORMAL
    elif scenario.speed == SpeedProfile.LOW_SPEED:
        fps = FPS_LOW
    elif scenario.speed == SpeedProfile.HIGH_SPEED:
        fps = FPS_HIGH
    else:
        fps = FPS_NORMAL
    
    # 高度转换 (km -> ft)
    KM_TO_FT = 3280.84
    
    # 修改敌机配置
    enemy_ids = ['B0100', 'B0200', 'B0300', 'B0400']
    # print(f"[DEBUG] 场景 {scenario.scenario_id} 使用编队类型: {scenario.formation.value}")
    # print(f"[DEBUG] 敌机初始位置: {positions}")
    for i, eid in enumerate(enemy_ids):
        x_km, y_km, alt_km = positions[i]
        lon, lat = battlefield_to_geodetic(x_km, y_km)
        alt_ft = alt_km * KM_TO_FT
        hdg = headings[i] if i < len(headings) else 180.0
        # print(f"[DEBUG] {eid}: x={x_km:.1f}km y={y_km:.1f}km -> lon={lon:.3f} lat={lat:.3f}")
        
        # 混合速度
        if scenario.speed == SpeedProfile.MIXED:
            current_fps = FPS_HIGH if i < 2 else FPS_LOW
        elif scenario.speed == SpeedProfile.ACCELERATING:
            current_fps = 1100.0  # 起始速度
        else:
            current_fps = fps
        
        config['aircraft_configs'][eid]['init_state'] = {
            'ic_long_gc_deg': lon,
            'ic_lat_geod_deg': lat,
            'ic_h_sl_ft': alt_ft,
            'ic_psi_true_deg': hdg,
            'ic_u_fps': current_fps
        }
    
    # 打印我方飞机位置以便对比 (调试已完成，注释掉)
    # print("[DEBUG] 我方飞机初始位置:")
    # for aid in ['A0100', 'A0200', 'A0300', 'A0400']:
    #     if aid in config['aircraft_configs']:
    #         state = config['aircraft_configs'][aid]['init_state']
    #         lon = state['ic_long_gc_deg']
    #         lat = state['ic_lat_geod_deg']
    #         x_km = (lon - A0100_LON) * DEG_TO_KM + 75.0
    #         y_km = (lat - A0100_LAT) * DEG_TO_KM
    #         print(f"[DEBUG] {aid}: x={x_km:.1f}km y={y_km:.1f}km -> lon={lon:.3f} lat={lat:.3f}")
    
    return config


def write_acmi_header(filepath: str, scenario: EnemyScenario):
    """写入ACMI文件头"""
    with open(filepath, 'w', encoding='utf-8-sig') as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.1\n")
        f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
        f.write(f"0,Comment=Scenario:{scenario.scenario_id} - {scenario.description}\n")


def write_acmi_frame(filepath: str, env, timestamp: float, patrol_task=None):
    """写入ACMI帧数据，包含雷达信息"""
    lines = [f"#{timestamp:.2f}\n"]
    
    # 获取雷达状态（如果有）
    radar_info = {}
    if patrol_task and hasattr(patrol_task, 'cap_radar'):
        radar = patrol_task.cap_radar
        for aid in ['A0100', 'A0200', 'A0300', 'A0400']:
            if hasattr(radar, 'get_mode') and hasattr(radar, 'get_tracks'):
                mode = radar.get_mode(aid)
                tracks = radar.get_tracks(aid)
                # 获取扫描分配 (center, azimuth_range, bars)
                scan_center = 0
                scan_range = 60
                if hasattr(radar, '_scan_assignments') and aid in radar._scan_assignments:
                    sa = radar._scan_assignments[aid]
                    if isinstance(sa, tuple) and len(sa) >= 2:
                        scan_center = sa[0]
                        scan_range = sa[1]
                    elif isinstance(sa, dict):
                        scan_center = sa.get('center', 0)
                        scan_range = sa.get('azimuth_range', 60)
                # 判断雷达模式
                radar_mode = 1 if mode else 0  # 1=开机
                lock_target = None
                # CAP模式使用TWS，显示最近的跟踪目标作为"锁定"目标（用于Tacview可视化）
                # 重要：仅当雷达开机时才输出 LockedTarget，否则Tacview会显示误导性的红线。
                if radar_mode == 1 and tracks:
                    # 找到最近的跟踪目标
                    nearest_track = min(tracks.values(), key=lambda t: t.distance)
                    lock_target = nearest_track.track_id

                # 雷达关机时，不输出波束信息，避免Tacview显示雷达扫描/锁定
                if radar_mode == 0:
                    scan_range = 0
                radar_info[aid] = {
                    'mode': radar_mode,
                    'azimuth': scan_center,
                    'range': 200000,  # 200km
                    'beamwidth': scan_range,
                    'lock_target': lock_target,
                    'tracks': list(tracks.keys()) if tracks else []
                }
    
    for uid, sim in env._jsbsims.items():
        if not sim.is_alive:
            continue
        lon, lat, alt = sim.get_geodetic()
        roll, pitch, yaw = sim.get_rpy() * 180 / np.pi
        model_name = sim.model.upper()
        if model_name == "SU27SK":
            model_name = "SU27"
        
        # 基本属性
        line = f"{uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
        line += f"Name={model_name},Type=Air+FixedWing,ShortName={uid},Color={sim.color}"
        
        # 添加雷达属性（仅我方飞机且雷达开启时）
        if uid in radar_info:
            ri = radar_info[uid]
            # 只有当雷达开启时才输出雷达属性
            if ri['mode'] == 1 and ri['beamwidth'] > 0:
                line += f",RadarMode={ri['mode']}"
                # [Fix] 将相对方位角转换为绝对方位角（Tacview需要绝对方位）
                # yaw是飞机航向，ri['azimuth']是相对于机头的扫描中心
                absolute_azimuth = (yaw + ri['azimuth']) % 360
                line += f",RadarAzimuth={absolute_azimuth:.1f}"
                line += f",RadarRange={ri['range']}"
                line += f",RadarHorizontalBeamwidth={ri['beamwidth']:.1f}"
                if ri['lock_target']:
                    line += f",LockedTarget={ri['lock_target']}"
        
        lines.append(line + "\n")
    
    # 导弹
    for sim in env._tempsims.values():
        if hasattr(sim, 'is_alive') and sim.is_alive:
            lon, lat, alt = sim.get_geodetic()
            roll, pitch, yaw = sim.get_rpy() * 180 / np.pi
            lines.append(f"{sim.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},")
            lines.append(f"Name=AIM120,Type=Air+Missile,ShortName={sim.uid},Color={sim.color}\n")
    
    with open(filepath, 'a', encoding='utf-8-sig') as f:
        f.writelines(lines)


def apply_enemy_maneuver(env, scenario: EnemyScenario, distance: float, time_s: float,
                         enemy_target_headings: dict) -> dict:
    """
    应用敌机机动命令
    
    Args:
        env: 仿真环境
        scenario: 敌方场景配置
        distance: 与我方的距离(km)
        time_s: 仿真时间(秒)
        enemy_target_headings: 敌机目标航向字典 {uid: heading}
    
    Returns:
        更新后的敌机目标航向字典
    """
    enemy_ids = ['B0100', 'B0200', 'B0300', 'B0400']
    
    # 初始化敌机目标航向（只在第一次调用时）
    for eid in enemy_ids:
        if eid not in enemy_target_headings:
            enemy_target_headings[eid] = 180.0  # 初始航向180°向南

    # 维护“基础航向”(用于WEAVE等非累积偏置)
    base_key = '_base_headings'
    if base_key not in enemy_target_headings or not isinstance(enemy_target_headings.get(base_key), dict):
        enemy_target_headings[base_key] = {eid: float(enemy_target_headings.get(eid, 180.0)) for eid in enemy_ids}

    # 维护上次时间戳用于 altitude_rate 积分
    last_t_key = '_last_time_s'
    if last_t_key not in enemy_target_headings:
        enemy_target_headings[last_t_key] = float(time_s)
    dt_s = float(time_s) - float(enemy_target_headings.get(last_t_key, time_s))
    if dt_s < 0:
        dt_s = 0.0
    
    # 兼容：部分验证场景需要多阶段早期机动（偏转->回正），不能被“单次偏转”门控阻断
    multi_stage_turn = scenario.maneuver in (
        ManeuverType.FLANK_TURN,
        ManeuverType.CH6_LOW_RISK_FEINT,
        ManeuverType.CH6_MEDIUM_RISK_PRESSURE,
        ManeuverType.CH6_HIGH_RISK_PENETRATION,
        ManeuverType.EARLY_TURN_BACK,
        ManeuverType.EARLY_SPLIT_TURN_BACK,
        ManeuverType.EARLY_COMPLEX_SEQUENCE,
    )
    
    # 获取机动命令
    maneuver_cmds = scenario.get_maneuver_commands(distance, time_s)

    # 维护“持续高度速率”(km/s)：
    # 重要：scenario_generator 中部分机动(如 V03 POPUP_TURN)只在触发帧给一次 altitude_rate，
    # 但 run_detection 侧的高度链路需要每步积分，所以必须把 altitude_rate 持续保存到被覆盖(如 altitude_rate=0)为止。
    rate_key = '_altitude_rate_km_s'
    if rate_key not in enemy_target_headings or not isinstance(enemy_target_headings.get(rate_key), dict):
        enemy_target_headings[rate_key] = {eid: 0.0 for eid in enemy_ids}

    alt_key = '_target_altitudes_ft'

    # 仅在场景确实给出 altitude_rate 时，才启用“目标高度(ft)”链路。
    # 但：对于 POPUP_TURN/多阶段机动，altitude_rate 可能只在触发帧给一次，之后依赖“持续速率”继续积分。
    # 因此：只要存在未清零的持续速率，或目标高度字典已初始化，就必须保持 has_alt_control=True。
    # 注意：env.agents[*].get_geodetic() 的高度单位在本项目中为米(m)，
    # 因此这里必须使用 JSBSim 的 position/h-sl-ft 读取英尺值，避免单位错配。
    has_alt_control = any(
        isinstance(cmd, dict) and ('altitude_rate' in cmd or 'target_altitude' in cmd)
        for cmd in (maneuver_cmds or [])
    )
    if not has_alt_control:
        try:
            if isinstance(enemy_target_headings.get(alt_key), dict) and len(enemy_target_headings.get(alt_key, {})) > 0:
                has_alt_control = True
            elif isinstance(enemy_target_headings.get(rate_key), dict):
                has_alt_control = any(abs(float(v)) > 0.0 for v in enemy_target_headings[rate_key].values())
        except Exception:
            has_alt_control = False

    if has_alt_control:
        if alt_key not in enemy_target_headings or not isinstance(enemy_target_headings.get(alt_key), dict):
            enemy_target_headings[alt_key] = {}
            for eid in enemy_ids:
                if eid in env.agents and env.agents[eid].is_alive:
                    try:
                        enemy_target_headings[alt_key][eid] = float(env.agents[eid].get_property_value(c.position_h_sl_ft))
                    except Exception:
                        pass
    
    for i, eid in enumerate(enemy_ids):
        if eid not in env.agents or not env.agents[eid].is_alive:
            continue
        
        cmd = maneuver_cmds[i] if i < len(maneuver_cmds) else {}
        
        # 航向变化
        if 'heading_change' in cmd:
            hdg_change = float(cmd['heading_change'])
            base_hdg = float(enemy_target_headings[base_key].get(eid, 180.0))

            # WEAVE 是连续摆动：heading_change 为“相对基础航向的偏置”，每步都刷新目标航向
            if scenario.maneuver == ManeuverType.WEAVE:
                enemy_target_headings[eid] = (base_hdg + hdg_change) % 360.0
            # 多阶段机动：允许多次航向变化（每个阶段都可以改变航向）
            elif multi_stage_turn:
                new_hdg = (base_hdg + hdg_change) % 360.0
                enemy_target_headings[eid] = new_hdg
                enemy_target_headings[base_key][eid] = new_hdg  # 更新基础航向
                logging.info(f"🛫 [敌机机动] {eid} 转向 {hdg_change:+.0f}°, 新航向: {new_hdg:.0f}°")
            else:
                # 其它机动默认视为“单次偏转”（避免每步累积）
                turn_executed_key = '_turn_executed'
                if turn_executed_key not in enemy_target_headings:
                    new_hdg = (base_hdg + hdg_change) % 360.0
                    enemy_target_headings[eid] = new_hdg
                    enemy_target_headings[base_key][eid] = new_hdg  # 更新基础航向
                    logging.info(f"🛫 [敌机机动] {eid} 转向 {hdg_change:+.0f}°, 新航向: {new_hdg:.0f}°")

        # 高度变化（km/s）：把速率积分到“目标高度(ft)”
        if has_alt_control:
            # 优先处理“目标高度”(km)：直接写入目标高度并取消持续速率，避免积分漂移
            if 'target_altitude' in cmd:
                try:
                    target_alt_km = float(cmd['target_altitude'])
                    if isinstance(enemy_target_headings.get(alt_key), dict):
                        enemy_target_headings[alt_key][eid] = target_alt_km * 3280.84
                    enemy_target_headings[rate_key][eid] = 0.0
                except Exception:
                    pass

            # 更新“持续速率”缓存（若本帧给了 altitude_rate 就覆盖；否则沿用上一帧）
            if 'altitude_rate' in cmd:
                enemy_target_headings[rate_key][eid] = float(cmd['altitude_rate'])

            rate_km_s = float(enemy_target_headings[rate_key].get(eid, 0.0))

            if abs(rate_km_s) > 0.0:
                if dt_s > 0 and isinstance(enemy_target_headings.get(alt_key), dict) and eid in enemy_target_headings[alt_key]:
                    # km -> ft
                    enemy_target_headings[alt_key][eid] = float(enemy_target_headings[alt_key][eid]) + rate_km_s * dt_s * 3280.84

                    # 🔥 V03场景专用日志：跟踪爬升过程（已注释，避免日志过多）
                    # if scenario.scenario_id == 'V03' and abs(rate_km_s) > 0.01:
                    #     new_alt_ft = enemy_target_headings[alt_key][eid]
                    #     logging.info(f"🚀 [V03爬升] {eid} 高度变化至 {new_alt_ft:.0f}ft "
                    #                  f"(速率={rate_km_s:.3f}km/s, dt={dt_s:.2f}s, 距离={distance:.1f}km)")
    
    # 如果本次有“单次航向变化”机动命令，标记为已执行
    # 多阶段机动不启用该门控（由场景内部阶段控制，避免重复触发）
    if (scenario.maneuver != ManeuverType.WEAVE) and (not multi_stage_turn):
        turn_executed_key = '_turn_executed'
        if any('heading_change' in (maneuver_cmds[i] if i < len(maneuver_cmds) else {}) for i in range(4)):
            if turn_executed_key not in enemy_target_headings:
                enemy_target_headings[turn_executed_key] = True

    # 更新时间戳
    enemy_target_headings[last_t_key] = float(time_s)
    
    return enemy_target_headings


def run_scenario(scenario: EnemyScenario, max_steps: int = 3000, mode: str = 'proposed', seed: int = 1) -> dict:
    """
    运行单个验证场景
    
    Args:
        scenario: 敌方场景配置
        max_steps: 最大仿真步数
    
    Returns:
        验证结果字典
    """
    mode = (mode or 'proposed').strip().lower()
    if mode not in ('proposed', 'baseline'):
        mode = 'proposed'

    # CRITICAL: scenario_generator 中的机动逻辑会在 EnemyScenario 对象上维护阶段状态。
    # compare 模式下同一个 scenario 会依次用于 proposed 和 baseline；必须在每次仿真
    # 开始前重置，否则两次敌方机动会不一致，导致对比失效。
    try:
        if hasattr(scenario, 'reset_runtime_state'):
            scenario.reset_runtime_state()
    except Exception:
        pass

    timestamp_str = datetime.now().strftime("%m%d_%H%M%S")
    scenario_tag = f"{scenario.scenario_id}_{mode.upper()}"
    log_file = setup_logging(OUTPUT_BASE_DIR, f"{scenario_tag}_{timestamp_str}")
    
    logging.info("=" * 60)
    logging.info(f"场景 {scenario.scenario_id} | 模式 {mode.upper()}: {scenario.description}")
    logging.info(f"难度: {'*' * scenario.difficulty}")
    logging.info("=" * 60)
    
    results = {
        'scenario_id': scenario.scenario_id,
        'mode': mode,
        'seed': int(seed),
        'log_file': log_file,
        'success': False,
        'first_lock_time': None,
        'detection_mode_changes': 0,
        'phase_changes': 0,
        'acmi_path': None,
        # 便于对比输出/落盘
        'total_time_s': None,
        'lock_continuity': None,
        'any_track_continuity_post_activation': None,
        'lock_continuity_all_time': None,
        'activation_mean_s': None,
        'activation_std_s': None,
        'avg_first_detect_delay_s': None,
        'full_detect_delay_s': None,
        'full_detect_time_s': None,
        'activation_times_s': None,
        'mode_time_s': None,
        'phase_time_s': None,
        'a0100_spd_cmd_hist': None,
        'per_target_best_first_detect': None,
        'prelaunch_gate_pass_count': 0,
        'prelaunch_gate_block_count': 0,
        'prelaunch_gate_pass_rate': 0.0,
        'relay_attempt_count': 0,
        'relay_success_count': 0,
        'relay_success_rate': 0.0,
        'relay_success_by_reason': None,
        'active_guided_missile_peak': 0,
    }
    
    # 设置特殊场景的环境变量行为
    if scenario.scenario_id == 'S21':
        os.environ['ENEMY_BEHAVIOR'] = 'POPUP'
    else:
        # 清除可能残留的环境变量
        if 'ENEMY_BEHAVIOR' in os.environ:
            del os.environ['ENEMY_BEHAVIOR']
    
    try:
        # 基础配置路径
        base_config = os.path.join(cap_dir, 'config', 'patrol_config.yaml')
        if not os.path.exists(base_config):
            logging.error(f"配置文件不存在: {base_config}")
            return results
        
        # 生成场景配置
        config_dict = generate_scenario_config(scenario, base_config)

        # 实验对比配置：传递到CAPTask
        config_dict['cap_experiment_mode'] = mode
        config_dict['cap_rng_seed'] = int(seed)
        config_dict['cap_awacs_seed'] = int(seed)
        
        # 打印敌机初始位置信息（调试用）
        logging.info("敌机初始配置:")
        for eid in ['B0100', 'B0200', 'B0300', 'B0400']:
            if eid in config_dict['aircraft_configs']:
                init_state = config_dict['aircraft_configs'][eid]['init_state']
                logging.info(f"  {eid}: 经度={init_state['ic_long_gc_deg']:.4f}, "
                            f"纬度={init_state['ic_lat_geod_deg']:.4f}, "
                            f"高度={init_state['ic_h_sl_ft']:.0f}ft, "
                            f"航向={init_state['ic_psi_true_deg']:.0f}度")
        
        # 写入临时配置文件
        temp_config_name = f'verification_{scenario.scenario_id}'
        jsbsim_config_dir = os.path.join(project_root, 'envs', 'JSBSim', 'configs')
        os.makedirs(jsbsim_config_dir, exist_ok=True)
        temp_config_path = os.path.join(jsbsim_config_dir, f'{temp_config_name}.yaml')
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True)
        
        logging.info(f"生成配置: {temp_config_path}")
        
        # 创建环境
        env = CAPEnvForVerification(temp_config_name)
        env.max_steps = max_steps
        
        # 创建CAP任务
        from cap.cap_task import CAPTask

        # ===== 验证公平性：强制AWACS确定性 =====
        # 目的：保证 proposed/baseline 在同一场景/同一seed 下，AWACS 的“丢失/误差”时间线一致。
        # 否则即使 seed 相同，也可能因为随机数消耗顺序差异、距离门控等因素导致两模式丢失时刻不同。
        os.environ.setdefault('CAP_AWACS_DETERMINISTIC', '1')
        os.environ.setdefault('CAP_AWACS_FIXED_CENTER_KM', '100,0,9')

        patrol_task = CAPTask(env.config)
        env.task = patrol_task
        
        logging.info("使用CAPTask - 我方正常巡逻")
        
        # 重置
        env.reset()
        env.max_steps = max_steps
        
        # 🔥 诊断：打印所有飞机的初始高度和速度
        from envs.JSBSim.core.catalog import Catalog as c_diag
        logging.info("=== 初始化诊断 ===")
        for aid, agent in env.agents.items():
            alt_m = agent.get_property_value(c_diag.position_h_sl_m)
            alt_ft = agent.get_property_value(c_diag.position_h_sl_ft)
            vel_fps = agent.get_property_value(c_diag.velocities_u_fps)
            pitch_deg = np.rad2deg(agent.get_property_value(c_diag.attitude_theta_rad))
            logging.info(f"  {aid}: 高度={alt_m:.0f}m ({alt_ft:.0f}ft) 速度={vel_fps:.0f}fps 俯仰={pitch_deg:.1f}°")
        logging.info("=================")
        
        # ACMI文件
        acmi_path = os.path.join(OUTPUT_BASE_DIR, f'{timestamp_str}_{scenario_tag}.txt.acmi')
        write_acmi_header(acmi_path, scenario)
        results['acmi_path'] = acmi_path
        
        logging.info(f"我方: {MY_AIRCRAFT_TYPE.upper()} x4 (正常巡逻)")
        logging.info(f"敌方: {ENEMY_AIRCRAFT_TYPE.upper()} x4 ({scenario.description})")
        logging.info(f"ACMI: {acmi_path}")
        logging.info(f"实验参数: mode={mode} | seed={seed}")
        logging.info(
            f"场景参数: 编队={scenario.formation.value} | 高度档={scenario.altitude.value} | 来向={scenario.direction.value} | "
            f"速度={scenario.speed.value} | 机动={scenario.maneuver.value} | 初始距离={scenario.initial_distance:.0f}km"
        )
        logging.info("-" * 60)
        
        # 主仿真循环
        step = 0
        dt = env.time_interval
        last_report_time = 0
        prev_detection_mode = None
        prev_phase = None
        first_lock_step = None
        lock_history: List[bool] = []
        mode_time: Dict[str, float] = {}
        phase_time: Dict[str, float] = {}
        a0100_spd_cmds: List[int] = []
        enemy_target_headings = {}  # 敌机目标航向 {uid: heading} + 内部辅助键
        enemy_target_altitudes_ft = {}  # 敌机目标高度(ft)
        
        # 🔥 新增指标：协同探测效能数据收集
        # 记录每个时刻每架飞机探测到的目标列表
        detection_timeline: List[Dict[str, List[str]]] = []  # [{aid: [tid1, tid2, ...]}, ...]
        # 记录每个时刻的目标覆盖情况
        coverage_timeline: List[int] = []  # [num_targets_detected, ...]
        # 记录每步的探测模式与时间戳（用于按模式/窗口统计）
        detection_mode_timeline: List[str] = []
        time_timeline_s: List[float] = []
        
        # 🔥 V6修复：初始化敌机目标高度（从配置读取初始高度作为维持目标）
        # 关键：必须同时初始化enemy_target_headings['_target_altitudes_ft']和enemy_target_altitudes_ft
        # 因为apply_enemy_maneuver使用前者，而CAPTask使用后者
        enemy_ids = ['B0100', 'B0200', 'B0300', 'B0400']
        enemy_target_headings['_target_altitudes_ft'] = {}  # 🔥 关键修复：初始化内部字典
        for eid in enemy_ids:
            if eid in config_dict['aircraft_configs']:
                init_alt_ft = config_dict['aircraft_configs'][eid]['init_state'].get('ic_h_sl_ft', 30000.0)
                enemy_target_altitudes_ft[eid] = init_alt_ft
                enemy_target_headings['_target_altitudes_ft'][eid] = init_alt_ft  # 🔥 关键修复：同步初始化
        
        # V6高度诊断日志（默认关闭；需要时设置环境变量 CAP_ALTITUDE_DEBUG=1）
        altitude_debug = os.environ.get('CAP_ALTITUDE_DEBUG', '').strip() in ('1', 'true', 'True', 'YES', 'yes')
        if altitude_debug:
            logging.info(f"[V6诊断] 初始化完成:")
            logging.info(f"  enemy_target_altitudes_ft = {enemy_target_altitudes_ft}")
            logging.info(
                f"  enemy_target_headings['_target_altitudes_ft'] = {enemy_target_headings.get('_target_altitudes_ft', {})}"
            )
        
        # 将场景信息传递给CAPTask用于敌机控制
        patrol_task.enemy_scenario = scenario
        patrol_task.enemy_target_headings = enemy_target_headings
        patrol_task.enemy_target_altitudes_ft = enemy_target_altitudes_ft
        
        while step < max_steps:
            step += 1
            time_s = step * dt
            
            # 计算敌我距离（用于敌机机动判断）
            my_positions = []
            enemy_positions = []
            for aid, agent in env.agents.items():
                if not agent.is_alive:
                    continue
                lon, lat, _ = agent.get_geodetic()
                x = (lon - A0100_LON) * DEG_TO_KM + 75
                y = (lat - A0100_LAT) * DEG_TO_KM
                if aid.startswith('A'):
                    my_positions.append((x, y))
                else:
                    enemy_positions.append((x, y))
            
            if my_positions and enemy_positions:
                my_center = np.mean(my_positions, axis=0)
                current_distance = min(np.sqrt((ex-my_center[0])**2 + (ey-my_center[1])**2) 
                                       for ex, ey in enemy_positions)
            else:
                current_distance = 999
            
            # 应用敌机机动
            enemy_target_headings = apply_enemy_maneuver(
                env, scenario, current_distance, time_s, enemy_target_headings)
            patrol_task.enemy_target_headings = enemy_target_headings

            # 同步敌机目标高度（从enemy_target_headings内部键导出）
            if isinstance(enemy_target_headings.get('_target_altitudes_ft'), dict):
                enemy_target_altitudes_ft.update(enemy_target_headings['_target_altitudes_ft'])
                patrol_task.enemy_target_altitudes_ft = enemy_target_altitudes_ft
                
                # V6高度同步诊断（默认关闭）
                if altitude_debug and step % 50 == 0 and scenario.scenario_id == 'V03':
                    logging.info(f"[V6诊断 主循环] step={step} distance={current_distance:.1f}km")
                    logging.info(
                        f"  enemy_target_headings['_target_altitudes_ft'] = {enemy_target_headings['_target_altitudes_ft']}"
                    )
                    logging.info(f"  enemy_target_altitudes_ft = {enemy_target_altitudes_ft}")
                    logging.info(f"  patrol_task.enemy_target_altitudes_ft = {patrol_task.enemy_target_altitudes_ft}")
            
            # 构建动作数组：使用CAPTask获取每个飞机的动作
            agent_ids = list(env.agents.keys())
            actions = []
            for aid in agent_ids:
                if env.agents[aid].is_alive:
                    # 从CAPTask获取动作命令索引 (alt_cmd, hdg_cmd, spd_cmd)
                    alt_cmd, hdg_cmd, spd_cmd = patrol_task.get_action(env, aid)
                    if aid == 'A0100':
                        try:
                            a0100_spd_cmds.append(int(spd_cmd))
                        except Exception:
                            pass
                    # 动作数组格式: [alt_cmd, hdg_cmd, spd_cmd, 0] (第4维为射击，始终为0)
                    actions.append([alt_cmd, hdg_cmd, spd_cmd, 0])
                else:
                    actions.append([7, 8, 3, 0])  # 默认保持动作
            
            # 转换为正确的形状 (1, num_agents, 4)
            action_array = np.array(actions, dtype=np.float32).reshape(1, len(agent_ids), 4)
            env.step(action_array)
            
            # 写入ACMI（包含雷达信息）
            write_acmi_frame(acmi_path, env, time_s, patrol_task)
            
            # 获取协同探测状态
            detection_mode = None
            phase = None
            locked_targets = 0
            
            # 从coop_detection获取模式
            if hasattr(patrol_task, 'coop_detection'):
                det = patrol_task.coop_detection
                if hasattr(det, '_mode'):
                    detection_mode = str(det._mode)
                elif hasattr(det, 'mode'):
                    detection_mode = str(det.mode)
            
            # 从cap_radar获取锁定目标数（统计所有我方飞机的航迹）
            if hasattr(patrol_task, 'cap_radar'):
                radar = patrol_task.cap_radar
                all_target_ids = set()
                for aid in ['A0100', 'A0200', 'A0300', 'A0400']:
                    if hasattr(radar, 'get_tracks'):
                        tracks = radar.get_tracks(aid)
                        if tracks:
                            all_target_ids.update(tracks.keys())
                locked_targets = len(all_target_ids)
            
            # 从cap_state_machine获取阶段
            if hasattr(patrol_task, 'cap_state_machine'):
                sm = patrol_task.cap_state_machine
                if hasattr(sm, 'state'):
                    phase = str(sm.state)

            # 指标累积（按上一步状态计时，避免“当前步”切换引入偏差）
            if prev_detection_mode is not None:
                mode_time[prev_detection_mode] = mode_time.get(prev_detection_mode, 0.0) + dt
            if prev_phase is not None:
                phase_time[prev_phase] = phase_time.get(prev_phase, 0.0) + dt

            lock_history.append(locked_targets > 0)
            
            # 🔥 新增：收集协同探测数据
            current_detections = {}  # {aid: [tid1, tid2, ...]}
            detected_targets_set = set()
            if hasattr(patrol_task, 'cap_radar'):
                radar = patrol_task.cap_radar
                for aid in ['A0100', 'A0200', 'A0300', 'A0400']:
                    if hasattr(radar, 'get_tracks'):
                        tracks = radar.get_tracks(aid)
                        if tracks:
                            target_ids = list(tracks.keys())
                            current_detections[aid] = target_ids
                            detected_targets_set.update(target_ids)
            detection_timeline.append(current_detections)
            coverage_timeline.append(len(detected_targets_set))
            detection_mode_timeline.append(detection_mode or '')
            time_timeline_s.append(float(time_s))
            
            # 检测模式/阶段变化
            if detection_mode != prev_detection_mode and prev_detection_mode is not None:
                results['detection_mode_changes'] += 1
                logging.info(f"[{time_s:.0f}s] 探测模式变化: {prev_detection_mode} -> {detection_mode}")
            prev_detection_mode = detection_mode
            
            if phase != prev_phase and prev_phase is not None:
                results['phase_changes'] += 1
                logging.info(f"[{time_s:.0f}s] 阶段变化: {prev_phase} -> {phase}")
            prev_phase = phase
            
            # 首次锁定
            if locked_targets > 0 and first_lock_step is None:
                first_lock_step = step
                results['first_lock_time'] = time_s
                logging.info(f"[{time_s:.0f}s] 首次锁定目标! 锁定数: {locked_targets}")
            
            # 定期状态报告
            if time_s - last_report_time >= 30:
                last_report_time = time_s
                
                # 计算距离
                my_positions = []
                enemy_positions = []
                for aid, agent in env.agents.items():
                    if not agent.is_alive:
                        continue
                    lon, lat, _ = agent.get_geodetic()
                    x = (lon - A0100_LON) * DEG_TO_KM + 75
                    y = (lat - A0100_LAT) * DEG_TO_KM
                    if aid.startswith('A'):
                        my_positions.append((x, y))
                    else:
                        enemy_positions.append((x, y))
                
                if my_positions and enemy_positions:
                    my_center = np.mean(my_positions, axis=0)
                    min_dist = min(np.sqrt((ex-my_center[0])**2 + (ey-my_center[1])**2) 
                                   for ex, ey in enemy_positions)
                else:
                    min_dist = 999
                
                mode_str = detection_mode if detection_mode else "N/A"
                phase_str = phase if phase else "N/A"
                logging.info(f"[{time_s:.0f}s] 距离:{min_dist:.0f}km 模式:{mode_str} "
                            f"阶段:{phase_str} 锁定:{locked_targets}")
            
            # 终止条件
            all_dead = all(not env.agents[aid].is_alive for aid in env.agents)
            if all_dead:
                logging.info(f"[{time_s:.0f}s] 全部飞机被摧毁，结束")
                break
        
        # 结果
        results['success'] = results['first_lock_time'] is not None
        
        logging.info("=" * 60)
        logging.info(f"场景 {scenario.scenario_id} 完成")
        logging.info(f"  首次锁定: {results['first_lock_time']:.1f}秒" if results['first_lock_time'] else "  首次锁定: 失败")
        logging.info(f"  模式切换: {results['detection_mode_changes']}次")
        logging.info(f"  阶段切换: {results['phase_changes']}次")
        logging.info(f"  ACMI: {acmi_path}")

        # 协同跟踪/接力制导验收指标（来自CAPTask内部统计）
        guidance_snapshot = {}
        try:
            if hasattr(patrol_task, 'get_guidance_verification_snapshot'):
                guidance_snapshot = patrol_task.get_guidance_verification_snapshot() or {}
        except Exception:
            guidance_snapshot = {}

        if guidance_snapshot:
            results['prelaunch_gate_pass_count'] = int(guidance_snapshot.get('prelaunch_gate_pass_count', 0))
            results['prelaunch_gate_block_count'] = int(guidance_snapshot.get('prelaunch_gate_block_count', 0))
            results['prelaunch_gate_pass_rate'] = float(guidance_snapshot.get('prelaunch_gate_pass_rate', 0.0))
            results['relay_attempt_count'] = int(guidance_snapshot.get('relay_attempt_count', 0))
            results['relay_success_count'] = int(guidance_snapshot.get('relay_success_count', 0))
            results['relay_success_rate'] = float(guidance_snapshot.get('relay_success_rate', 0.0))
            results['relay_success_by_reason'] = dict(guidance_snapshot.get('relay_success_by_reason', {}))
            results['active_guided_missile_peak'] = int(guidance_snapshot.get('active_guided_missile_peak', 0))

            logging.info("[协同跟踪/接力制导验收]")
            logging.info(
                f"  发射前门禁: pass={results['prelaunch_gate_pass_count']} "
                f"block={results['prelaunch_gate_block_count']} "
                f"pass_rate={results['prelaunch_gate_pass_rate']:.2%}"
            )
            logging.info(
                f"  接力制导: attempts={results['relay_attempt_count']} "
                f"success={results['relay_success_count']} "
                f"success_rate={results['relay_success_rate']:.2%}"
            )
            if results['relay_success_by_reason']:
                logging.info(f"  接力成功原因分布: {results['relay_success_by_reason']}")
            logging.info(f"  中制导活跃峰值导弹数: {results['active_guided_missile_peak']}")

        # ===== 末尾指标汇总（便于报告第4章直接引用） =====
        total_time = float(time_s) if 'time_s' in locals() else float(step * dt)
        # lock_history: 旧实现为“任意目标在轨(>0)占比”，保留为全程诊断信息
        any_track_continuity_all = (sum(lock_history) / len(lock_history)) if lock_history else 0.0
        results['total_time_s'] = total_time
        results['lock_continuity_all_time'] = float(any_track_continuity_all)
        logging.info("-" * 60)
        logging.info("[验证指标汇总]")
        logging.info(f"  总仿真时间: {total_time:.1f}s | 步数: {step}")
        if results.get('first_lock_time') is not None:
            logging.info(f"  首次锁定时间: {results['first_lock_time']:.1f}s")
        else:
            logging.info("  首次锁定时间: 失败")
        logging.info(f"  任意目标连续性(全程, locked_targets>0占比): {any_track_continuity_all:.3f}")
        if mode_time:
            items = ' | '.join([f"{k}:{v:.1f}s" for k, v in sorted(mode_time.items(), key=lambda kv: kv[0])])
            logging.info(f"  模式驻留时间: {items}")
            try:
                results['mode_time_s'] = {str(k): float(v) for k, v in mode_time.items()}
            except Exception:
                pass
        if phase_time:
            items = ' | '.join([f"{k}:{v:.1f}s" for k, v in sorted(phase_time.items(), key=lambda kv: kv[0])])
            logging.info(f"  阶段驻留时间: {items}")
            try:
                results['phase_time_s'] = {str(k): float(v) for k, v in phase_time.items()}
            except Exception:
                pass
        if a0100_spd_cmds:
            cnt = Counter(a0100_spd_cmds)
            accel = sum(v for k, v in cnt.items() if k >= 4)
            decel = sum(v for k, v in cnt.items() if k <= 2)
            keep = cnt.get(3, 0)
            total = len(a0100_spd_cmds)
            logging.info(
                f"  A0100 spd_cmd分布: {dict(sorted(cnt.items()))} | "
                f"加速:{accel/total:.2%} 保持:{keep/total:.2%} 减速:{decel/total:.2%}"
            )
            try:
                results['a0100_spd_cmd_hist'] = {int(k): int(v) for k, v in cnt.items()}
            except Exception:
                pass

        # ===== 统一窗口：仅统计“至少一架我机进入200km范围(activation)”之后的协同探测指标 =====
        expected_targets = ['B0100', 'B0200', 'B0300', 'B0400']
        activation_times = getattr(patrol_task, '_radar_range_entry_times', {}) if 'patrol_task' in locals() else {}
        first_activation = None
        if isinstance(activation_times, dict) and activation_times:
            try:
                first_activation = float(min(activation_times.values()))
            except Exception:
                first_activation = None

        post_activation_indices: List[int] = []
        if first_activation is not None and time_timeline_s:
            for i, t in enumerate(time_timeline_s):
                if t >= first_activation:
                    post_activation_indices.append(i)

        # 连续性：
        # - lock_continuity: 激活后“全覆盖(4/4目标在轨)”占比（更贴近协同优势）
        # - any_track_continuity_post_activation: 激活后“至少1目标在轨”占比（用于诊断黑屏/丢失）
        if coverage_timeline:
            idx = post_activation_indices if post_activation_indices else list(range(len(coverage_timeline)))
            try:
                full_flags = [1.0 if coverage_timeline[i] >= len(expected_targets) else 0.0 for i in idx]
                any_flags = [1.0 if coverage_timeline[i] > 0 else 0.0 for i in idx]
                results['lock_continuity'] = float(np.mean(full_flags)) if full_flags else 0.0
                results['any_track_continuity_post_activation'] = float(np.mean(any_flags)) if any_flags else 0.0
                logging.info(
                    f"  全覆盖连续性(激活后, 4/4在轨占比): {results['lock_continuity']:.3f}"
                    + (f" | 激活起始t={first_activation:.1f}s" if first_activation is not None else "")
                )
                logging.info(
                    f"  任意目标连续性(激活后, >0在轨占比): {results['any_track_continuity_post_activation']:.3f}"
                )
            except Exception:
                pass

        # 探测效率（首探/全探）— 从CAPTask内部追踪变量汇总
        # 注意：activation_times 是“进入200km范围”的时间，不等同于“雷达开机”。
        first_detect = getattr(patrol_task, '_first_radar_detection_time', {}) if 'patrol_task' in locals() else {}
        if isinstance(activation_times, dict) and isinstance(first_detect, dict) and activation_times:
            first_activation = min(activation_times.values())
            try:
                results['activation_times_s'] = {str(aid): float(t) for aid, t in activation_times.items()}
            except Exception:
                pass
            logging.info(
                "  雷达激活时间: "
                + ' | '.join([f"{aid}:{t:.1f}s" for aid, t in sorted(activation_times.items(), key=lambda kv: kv[0])])
            )

            # 到位同步性：激活时间均值/标准差
            try:
                ts = [float(t) for t in activation_times.values()]
                logging.info(f"  雷达激活同步性: mean={np.mean(ts):.1f}s std={np.std(ts):.1f}s")
                results['activation_mean_s'] = float(np.mean(ts))
                results['activation_std_s'] = float(np.std(ts))
            except Exception:
                pass

            # 目标级：最小首探延迟（min over agents that detected the target)
            per_target_best = []  # (tid, delay, aid)
            for tid, dets in first_detect.items():
                if not isinstance(dets, dict) or not dets:
                    continue
                best = None
                for aid, t_det in dets.items():
                    act = activation_times.get(aid)
                    if act is None:
                        continue
                    try:
                        delay = float(t_det) - float(act)
                    except Exception:
                        continue
                    delay = max(0.2, delay)
                    if best is None or delay < best[0]:
                        best = (delay, aid)
                if best is not None:
                    per_target_best.append((tid, best[0], best[1]))

            if per_target_best:
                delays = [d for _, d, _ in per_target_best]
                logging.info(
                    f"  首探延迟(min/目标): 平均={np.mean(delays):.1f}s 最小={np.min(delays):.1f}s 最大={np.max(delays):.1f}s"
                )
                try:
                    results['avg_first_detect_delay_s'] = float(np.mean(delays))
                except Exception:
                    pass
                items = ' | '.join([f"{tid}:{aid[-4:]}:{d:.1f}s" for tid, d, aid in sorted(per_target_best, key=lambda x: x[0])])
                logging.info(f"  各目标首探(min): {items}")
                try:
                    results['per_target_best_first_detect'] = [
                        {'target': str(tid), 'delay_s': float(d), 'agent': str(aid)}
                        for tid, d, aid in sorted(per_target_best, key=lambda x: x[0])
                    ]
                except Exception:
                    pass

                # 联队全探完成时间：每个目标最早被任一我机探测的时刻，取最大者
                expected_targets = ['B0100', 'B0200', 'B0300', 'B0400']
                earliest_by_target = {}
                for tid in expected_targets:
                    dets = first_detect.get(tid, {})
                    if not isinstance(dets, dict) or not dets:
                        continue
                    try:
                        earliest_by_target[tid] = min(float(t) for t in dets.values())
                    except Exception:
                        continue
                if len(earliest_by_target) == len(expected_targets):
                    full_detect_time = max(earliest_by_target.values())
                    full_detect_delay = max(0.0, full_detect_time - float(first_activation))
                    logging.info(f"  联队全探完成: t={full_detect_time:.1f}s | 相对首激活延迟={full_detect_delay:.1f}s")
                    results['full_detect_time_s'] = float(full_detect_time)
                    results['full_detect_delay_s'] = float(full_detect_delay)
                else:
                    missing = [t for t in expected_targets if t not in earliest_by_target]
                    logging.info(f"  联队全探完成: 未完成(缺失: {','.join(missing)})")

                # 首探“延迟”标准差：对4个目标的(min/目标)首探延迟做STD（比绝对时刻STD更可比）
                try:
                    if per_target_best and len(per_target_best) >= 2:
                        delays = [float(d) for _, d, _ in per_target_best]
                        std_delay = float(np.std(delays))
                        results['first_detect_time_std'] = std_delay
                        logging.info(f"  首探延迟标准差(min/目标): {std_delay:.1f}s (越小越均衡)")
                except Exception:
                    pass
        
        # 🔥 新增指标计算：协同探测效能分析
        logging.info("-" * 60)
        logging.info("[协同探测效能指标]")
        
        # 1. 目标分工/负载指标
        def calculate_gini_coefficient(detection_counts: List[int]) -> float:
            """计算基尼系数（0=完全均衡, 1=极端不均衡）。

            注意：这里必须保留0计数，否则“只有1架机负责”的情形会被误判为均衡。
            """
            if not detection_counts or len(detection_counts) == 0:
                return 0.0
            n = len(detection_counts)
            if n == 1:
                return 0.0
            sorted_counts = sorted(max(0.0, float(x)) for x in detection_counts)
            cumsum = 0.0
            for i, count in enumerate(sorted_counts):
                cumsum += (i + 1) * count
            total = sum(sorted_counts)
            if total == 0:
                return 0.0
            return (2 * cumsum) / (n * total) - (n + 1) / n

        def calculate_normalized_entropy(counts: List[int]) -> float:
            """归一化熵(0~1)。1表示最均匀，0表示最集中。"""
            xs = [max(0.0, float(x)) for x in counts]
            s = sum(xs)
            if s <= 0 or len(xs) <= 1:
                return 0.0
            ps = [x / s for x in xs if x > 0]
            if not ps:
                return 0.0
            h = -sum(p * float(np.log(p)) for p in ps)
            h_max = float(np.log(len(xs)))
            return float(h / h_max) if h_max > 0 else 0.0
        
        # 统计每架飞机探测每个目标的总次数
        agent_target_counts = {}  # {aid: {tid: count}}
        expected_targets = ['B0100', 'B0200', 'B0300', 'B0400']
        for aid in ['A0100', 'A0200', 'A0300', 'A0400']:
            agent_target_counts[aid] = {tid: 0 for tid in expected_targets}
        
        for frame_detections in detection_timeline:
            for aid, target_list in frame_detections.items():
                for tid in target_list:
                    if tid in expected_targets:
                        agent_target_counts[aid][tid] += 1
        
        agents = ['A0100', 'A0200', 'A0300', 'A0400']

        # (a) 目标分工集中度(基尼)：对每个目标计算4机探测次数的基尼（含0），再取平均。
        #     越大表示越集中（更像“分工明确/专注”）；越小表示越均匀（更多“重复覆盖”）。
        per_target_ginis = []
        for tid in expected_targets:
            counts_all = [agent_target_counts[aid][tid] for aid in agents]
            if sum(counts_all) <= 0:
                continue
            per_target_ginis.append(calculate_gini_coefficient(counts_all))
        if per_target_ginis:
            avg_gini = float(np.mean(per_target_ginis))
            results['target_allocation_gini'] = avg_gini
            logging.info(f"  目标分工集中度(基尼): {avg_gini:.3f} (越大越集中/分工越明确)")

        # (b) 主责分配均衡(熵)：每个目标由“探测次数最多”的飞机作为主责，统计主责分配的均衡性。
        primary_counts = {aid: 0 for aid in agents}
        for tid in expected_targets:
            counts_all = {aid: agent_target_counts[aid][tid] for aid in agents}
            if sum(counts_all.values()) <= 0:
                continue
            # tie-break: aid排序保证可复现
            best_aid = sorted(counts_all.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            primary_counts[best_aid] += 1
        results['target_primary_entropy'] = float(calculate_normalized_entropy([primary_counts[aid] for aid in agents]))
        logging.info(
            f"  主责分配均衡(熵0~1): {results['target_primary_entropy']:.3f} (越大越均衡)"
        )

        # (c) 探测负载均衡(基尼)：按飞机汇总总探测次数的基尼。
        agent_totals = [sum(agent_target_counts[aid].values()) for aid in agents]
        results['agent_workload_gini'] = float(calculate_gini_coefficient(agent_totals))
        logging.info(
            f"  探测负载不均衡(基尼): {results['agent_workload_gini']:.3f} (越小越均衡)"
        )
        
        # 2. 探测覆盖率（每个时刻有多少目标被至少1架飞机探测）
        # 默认使用“激活后窗口”，并同时给出全程覆盖率作为诊断。
        if coverage_timeline:
            try:
                results['detection_coverage_ratio_all_time'] = float(np.mean(coverage_timeline) / len(expected_targets))
            except Exception:
                results['detection_coverage_ratio_all_time'] = None

            idx = post_activation_indices if post_activation_indices else list(range(len(coverage_timeline)))
            cov_post = [coverage_timeline[i] for i in idx] if idx else coverage_timeline
            avg_coverage = float(np.mean(cov_post) / len(expected_targets)) if cov_post else 0.0
            results['detection_coverage_ratio'] = float(avg_coverage)
            if first_activation is not None:
                logging.info(
                    f"  平均探测覆盖率(激活后): {avg_coverage:.3f} ({avg_coverage*100:.1f}%) | 激活起始t={first_activation:.1f}s"
                )
            else:
                logging.info(f"  平均探测覆盖率: {avg_coverage:.3f} ({avg_coverage*100:.1f}%)")
            if results.get('detection_coverage_ratio_all_time') is not None:
                logging.info(
                    f"  平均探测覆盖率(全程诊断): {results['detection_coverage_ratio_all_time']:.3f} ({results['detection_coverage_ratio_all_time']*100:.1f}%)"
                )
        
        # 3. 覆盖冗余度（多机对同一目标的重叠比例）
        # 说明：冗余并不必然“越低越好”。DIRECTED下适度冗余可提升鲁棒性。
        def _norm_mode(m: str) -> str:
            s = (m or '').upper()
            if 'SWEEP' in s:
                return 'SWEEP'
            if 'DIRECT' in s:
                return 'DIRECTED'
            if 'SEARCH' in s:
                return 'SEARCH'
            return 'UNKNOWN'

        def _calc_redundant_ratio(indices: Optional[List[int]] = None) -> Optional[float]:
            redundant = 0
            total = 0
            if not detection_timeline:
                return None
            idxs = indices if indices is not None else list(range(len(detection_timeline)))
            for i in idxs:
                if i < 0 or i >= len(detection_timeline):
                    continue
                frame_detections = detection_timeline[i]
                target_agent_count = {}
                for _, target_list in frame_detections.items():
                    for tid in target_list:
                        if tid in expected_targets:
                            target_agent_count[tid] = target_agent_count.get(tid, 0) + 1
                for _, count in target_agent_count.items():
                    total += count
                    if count > 1:
                        redundant += (count - 1)
            if total <= 0:
                return None
            return float(redundant) / float(total)

        overall_redundant = _calc_redundant_ratio()
        if overall_redundant is not None:
            results['redundant_detection_ratio'] = float(overall_redundant)
            logging.info(f"  覆盖冗余度(全程): {overall_redundant:.3f} ({overall_redundant*100:.1f}%)")

        # 按模式分解（便于解释DIRECTED的“设计性冗余”）
        try:
            mode_norms = [_norm_mode(m) for m in detection_mode_timeline]
            idx_sweep = [i for i, m in enumerate(mode_norms) if m == 'SWEEP']
            idx_directed = [i for i, m in enumerate(mode_norms) if m == 'DIRECTED']
            r_sweep = _calc_redundant_ratio(idx_sweep)
            r_directed = _calc_redundant_ratio(idx_directed)
            if r_sweep is not None:
                results['redundant_detection_ratio_sweep'] = float(r_sweep)
                logging.info(f"  覆盖冗余度(SWEEP): {r_sweep:.3f} ({r_sweep*100:.1f}%)")
            if r_directed is not None:
                results['redundant_detection_ratio_directed'] = float(r_directed)
                logging.info(f"  覆盖冗余度(DIRECTED): {r_directed:.3f} ({r_directed*100:.1f}%)")
        except Exception:
            pass
        
        # 4. 首探标准差：已在“探测效率(首探/全探)”段落中用“首探延迟STD(min/目标)”替代绝对时间STD。
        
        # 5. 目标分配热力图数据（记录每架飞机探测每个目标的时间分布）
        allocation_heatmap = {}
        for aid in ['A0100', 'A0200', 'A0300', 'A0400']:
            allocation_heatmap[aid] = {}
            for tid in expected_targets:
                count = agent_target_counts[aid][tid]
                total_frames = len(detection_timeline)
                ratio = count / total_frames if total_frames > 0 else 0.0
                allocation_heatmap[aid][tid] = {
                    'count': count,
                    'ratio': float(ratio)
                }
        
        results['allocation_heatmap'] = allocation_heatmap
        logging.info("  目标分配热力图:")
        for aid in ['A0100', 'A0200', 'A0300', 'A0400']:
            items = ' | '.join([f"{tid}:{allocation_heatmap[aid][tid]['ratio']*100:.1f}%" 
                               for tid in expected_targets])
            logging.info(f"    {aid}: {items}")
        
        logging.info("=" * 60)
        logging.info("=" * 60)
        
        env.close()
        
        # 清理临时配置
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
    except Exception as e:
        logging.error(f"场景运行失败: {e}", exc_info=True)
    
    return results


def list_scenarios():
    """列出所有场景"""
    scenarios = ScenarioGenerator.get_core_scenarios()
    print("\n可用场景列表:")
    print("-" * 70)
    for s in scenarios:
        stars = "*" * s.difficulty
        print(f"  {s.scenario_id:6} [{stars:10}] {s.description}")
        print(f"         初始距离: {s.initial_distance:.0f}km")
    print("-" * 70)
    print(f"共 {len(scenarios)} 个场景 (V01, V02, V03)")


def list_verification_scenarios():
    """列出验证场景（与list_scenarios相同）"""
    list_scenarios()


def run_verification(scenario_ids: Optional[List[str]] = None, max_steps: int = 3000):
    """运行验证"""
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    all_scenarios = ScenarioGenerator.get_core_scenarios()
    if scenario_ids:
        scenarios = [s for s in all_scenarios if s.scenario_id in scenario_ids]
        if not scenarios:
            print(f"未找到场景: {scenario_ids}")
            return
    else:
        scenarios = all_scenarios
    
    print("=" * 60)
    print("协同探测V4验证 - 真实仿真")
    print(f"场景数量: {len(scenarios)}")
    print(f"输出目录: {OUTPUT_BASE_DIR}")
    print("=" * 60)
    
    all_results = []
    for i, scenario in enumerate(scenarios):
        print(f"\n进度: {i+1}/{len(scenarios)}")
        # 默认只跑proposed；若compare=True则在main中外层循环处理
        results = run_scenario(scenario, max_steps=max_steps)
        all_results.append(results)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("验证汇总")
    print("-" * 60)
    print(f"{'场景ID':8} {'模式':9} {'锁定':6} {'首锁(s)':10} {'模式切换':10}")
    print("-" * 60)
    for r in all_results:
        lock_str = "Y" if r['success'] else "N"
        first_lock = f"{r['first_lock_time']:.1f}" if r['first_lock_time'] else "-"
        mode = (r.get('mode') or '-').upper()
        print(f"{r['scenario_id']:8} {mode:9} {lock_str:6} {first_lock:10} {r['detection_mode_changes']:10}")
    print("-" * 60)
    
    success_count = sum(1 for r in all_results if r['success'])
    print(f"成功率: {success_count}/{len(all_results)} ({success_count/len(all_results)*100:.1f}%)")


def parse_scenario_id(user_input: str) -> str:
    """智能解析场景ID，支持多种输入格式"""
    s = user_input.strip().upper()
    if not s:
        return 'V01'
    # 如果已经是V开头的格式
    if s.startswith('V') and len(s) >= 2:
        return s
    # 如果是纯数字1-3，转换为V01-V03
    try:
        num = int(s)
        if 1 <= num <= 3:
            return f'V{num:02d}'
    except ValueError:
        pass
    return s


def interactive_menu():
    """交互式控制台菜单"""
    print("\n" + "=" * 60)
    print("    协同探测V4验证 - 真实仿真")
    print("=" * 60)
    print("\n请选择运行模式:")
    print("  [1] 运行单个场景 (输入场景ID，如 V01)")
    print("  [2] 运行多个场景 (输入场景ID，用空格分隔)")
    print("  [3] 运行全部3个验证场景 (V01, V02, V03)")
    print("  [4] 列出所有场景")
    print("  [0] 退出")
    print("-" * 60)
    
    choice = input("请输入选择 [1-4, 0退出]: ").strip()
    
    if choice == '0':
        print("已退出")
        return None, None
    elif choice == '4':
        list_scenarios()
        return interactive_menu()  # 递归再次显示菜单
    elif choice == '1':
        list_scenarios()
        scenario_input = input("\n请输入场景ID (如 V01，直接回车默认V01): ").strip()
        scenario_id = parse_scenario_id(scenario_input) if scenario_input else 'V01'
        print(f"选择场景: {scenario_id}")
        return [scenario_id], get_steps(scenario_id)
    elif choice == '2':
        list_scenarios()
        scenario_input = input("\n请输入多个场景ID (空格分隔，如 V01 V02 V03): ").strip()
        if not scenario_input:
            scenario_input = 'V01'
        ids = [parse_scenario_id(s) for s in scenario_input.split()]
        print(f"选择场景: {ids}")
        return ids, get_steps(ids[0] if ids else None)
    elif choice == '3':
        return ['V01', 'V02', 'V03'], get_steps(None)
    else:
        print("无效选择，默认运行V01")
        return ['V01'], get_steps('V01')


def get_steps(scenario_id=None):
    """获取仿真步数"""
    # 默认最大7分钟（2100步，dt=0.2s），避免验证脚本运行过久
    default_steps = 2100
    default_desc = "2100，约420秒=7分钟"
    
    steps_input = input(f"仿真步数 (默认{default_desc}，直接回车使用默认；建议不超过7分钟): ").strip()
    if steps_input:
        try:
            return int(steps_input)
        except ValueError:
            print(f"无效数字，使用默认{default_steps}步")
            return default_steps
    return default_steps


def main():
    """主函数 - 支持命令行参数和交互式菜单"""
    import sys
    
    # 如果有命令行参数，使用命令行模式
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='协同探测V4真实仿真验证')
        parser.add_argument('--list', action='store_true', help='列出所有场景')
        parser.add_argument('--scenario', type=str, nargs='+', help='指定场景ID', default='V01')
        parser.add_argument('--all', action='store_true', help='运行全部3个验证场景')
        parser.add_argument('--steps', type=int, default=2100, help='最大仿真步数')
        parser.add_argument('--mode', type=str, default='proposed', choices=['proposed', 'baseline'], help='实验模式')
        parser.add_argument('--compare', action='store_true', help='每个场景运行proposed+baseline一对对比(共6次)')
        parser.add_argument('--seed', type=int, default=1, help='随机seed（同场景对比使用相同seed）')
        args = parser.parse_args()
        
        if args.list:
            list_scenarios()
            return
        
        # 组装场景列表
        all_scenarios = ScenarioGenerator.get_core_scenarios()
        if args.all:
            scenarios = all_scenarios
        else:
            scenario_ids = args.scenario if isinstance(args.scenario, list) else [args.scenario]
            scenarios = [s for s in all_scenarios if s.scenario_id in scenario_ids]
            if not scenarios:
                print(f"未找到场景: {scenario_ids}")
                return

        modes = ['proposed', 'baseline'] if args.compare else [args.mode]

        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        print("=" * 60)
        print("协同探测验证 - 真实仿真")
        print(f"场景数量: {len(scenarios)} | 对比: {modes}")
        print(f"输出目录: {OUTPUT_BASE_DIR}")
        print("=" * 60)

        import csv
        timestamp_str = datetime.now().strftime("%m%d_%H%M%S")
        summary_csv = os.path.join(OUTPUT_BASE_DIR, f"compare_summary_{timestamp_str}_seed{args.seed}_steps{args.steps}.csv")
        summary_csv_zh = os.path.join(OUTPUT_BASE_DIR, f"compare_summary_{timestamp_str}_seed{args.seed}_steps{args.steps}_zh.csv")

        # 写入CSV表头（一次性，汇总6次实验/3场景两模式对比）
        with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([
                'scenario_id', 'seed',
                'proposed_success', 'baseline_success',
                'proposed_first_lock_s', 'baseline_first_lock_s', 'delta_first_lock_s',
                'proposed_lock_continuity', 'baseline_lock_continuity', 'delta_lock_continuity',
                'proposed_mode_changes', 'baseline_mode_changes', 'delta_mode_changes',
                'proposed_full_detect_delay_s', 'baseline_full_detect_delay_s', 'delta_full_detect_delay_s',
                'proposed_avg_first_detect_delay_s', 'baseline_avg_first_detect_delay_s', 'delta_avg_first_detect_delay_s',
                'proposed_activation_mean_s', 'baseline_activation_mean_s',
                'proposed_activation_std_s', 'baseline_activation_std_s',
                'proposed_target_allocation_gini', 'baseline_target_allocation_gini', 'delta_target_allocation_gini',
                'proposed_detection_coverage_ratio', 'baseline_detection_coverage_ratio', 'delta_detection_coverage_ratio',
                'proposed_redundant_detection_ratio', 'baseline_redundant_detection_ratio', 'delta_redundant_detection_ratio',
                'proposed_first_detect_time_std', 'baseline_first_detect_time_std', 'delta_first_detect_time_std',
                'proposed_any_track_continuity_post_activation', 'baseline_any_track_continuity_post_activation', 'delta_any_track_continuity_post_activation',
                'proposed_target_primary_entropy', 'baseline_target_primary_entropy', 'delta_target_primary_entropy',
                'proposed_agent_workload_gini', 'baseline_agent_workload_gini', 'delta_agent_workload_gini',
                'proposed_detection_coverage_ratio_all_time', 'baseline_detection_coverage_ratio_all_time', 'delta_detection_coverage_ratio_all_time',
                'proposed_redundant_detection_ratio_sweep', 'baseline_redundant_detection_ratio_sweep', 'delta_redundant_detection_ratio_sweep',
                'proposed_redundant_detection_ratio_directed', 'baseline_redundant_detection_ratio_directed', 'delta_redundant_detection_ratio_directed',
                'steps'
            ])

        # 同时输出中文表头版本，便于人工阅读（不影响英文CSV的下游解析）
        with open(summary_csv_zh, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([
                '场景ID', '随机种子',
                '我方算法_成功', '无算法_成功',
                '我方算法_首次锁定(s)', '无算法_首次锁定(s)', '首次锁定差值(s)(无-有)',
                '我方算法_全覆盖连续性(激活后)', '无算法_全覆盖连续性(激活后)', '全覆盖连续性差值(无-有)',
                '我方算法_模式切换次数', '无算法_模式切换次数', '模式切换差值(无-有)',
                '我方算法_全探延迟(s)', '无算法_全探延迟(s)', '全探延迟差值(s)(无-有)',
                '我方算法_平均首探延迟(s)', '无算法_平均首探延迟(s)', '平均首探延迟差值(s)(无-有)',
                '我方算法_雷达激活均值(s)', '无算法_雷达激活均值(s)',
                '我方算法_雷达激活标准差(s)', '无算法_雷达激活标准差(s)',
                '我方算法_目标分工集中度(基尼)', '无算法_目标分工集中度(基尼)', '目标分工集中度差值(无-有)',
                '我方算法_平均覆盖率(激活后)', '无算法_平均覆盖率(激活后)', '覆盖率差值(无-有)',
                '我方算法_覆盖冗余度(全程)', '无算法_覆盖冗余度(全程)', '覆盖冗余度差值(无-有)',
                '我方算法_首探延迟标准差(s)', '无算法_首探延迟标准差(s)', '首探延迟标准差差值(s)(无-有)',
                '我方算法_任意目标连续性(激活后)', '无算法_任意目标连续性(激活后)', '任意目标连续性差值(无-有)',
                '我方算法_主责分配均衡(熵0~1)', '无算法_主责分配均衡(熵0~1)', '主责均衡差值(无-有)',
                '我方算法_探测负载不均衡(基尼)', '无算法_探测负载不均衡(基尼)', '负载不均衡差值(无-有)',
                '我方算法_平均覆盖率(全程诊断)', '无算法_平均覆盖率(全程诊断)', '全程覆盖率差值(无-有)',
                '我方算法_覆盖冗余度(SWEEP)', '无算法_覆盖冗余度(SWEEP)', 'SWEEP冗余度差值(无-有)',
                '我方算法_覆盖冗余度(DIRECTED)', '无算法_覆盖冗余度(DIRECTED)', 'DIRECTED冗余度差值(无-有)',
                '步数'
            ])

        all_results = []
        run_idx = 0
        total_runs = len(scenarios) * len(modes)

        def _fmt(v, nd=1):
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        # compare 模式：每个场景立刻跑 proposed + baseline，然后立刻做对比并写入CSV
        if args.compare:
            for scenario in scenarios:
                seed = _scenario_seed(args.seed, scenario.scenario_id)
                run_idx += 1
                print(f"\n进度: {run_idx}/{total_runs} | {scenario.scenario_id} PROPOSED seed={seed}")
                r_p = run_scenario(scenario, max_steps=args.steps, mode='proposed', seed=seed)
                all_results.append(r_p)

                run_idx += 1
                print(f"\n进度: {run_idx}/{total_runs} | {scenario.scenario_id} BASELINE seed={seed}")
                r_b = run_scenario(scenario, max_steps=args.steps, mode='baseline', seed=seed)
                all_results.append(r_b)

                # 立刻对比（控制台）
                p_lock = _fmt(r_p.get('first_lock_time'))
                b_lock = _fmt(r_b.get('first_lock_time'))
                p_full = _fmt(r_p.get('full_detect_delay_s'))
                b_full = _fmt(r_b.get('full_detect_delay_s'))
                p_cont = _fmt(r_p.get('lock_continuity'))
                b_cont = _fmt(r_b.get('lock_continuity'))
                p_mc = _fmt(r_p.get('detection_mode_changes'), 0)
                b_mc = _fmt(r_b.get('detection_mode_changes'), 0)
                print("\n" + "-" * 60)
                print(f"[对比] {scenario.scenario_id} (seed={seed}, steps={args.steps})")
                print(f"  首次锁定(s): proposed={p_lock if p_lock is not None else '-'} | baseline={b_lock if b_lock is not None else '-'}")
                if p_lock is not None and b_lock is not None:
                    print(f"  首次锁定差值(s): baseline - proposed = {b_lock - p_lock:+.1f}")
                print(f"  全探延迟(s): proposed={p_full if p_full is not None else '-'} | baseline={b_full if b_full is not None else '-'}")
                if p_full is not None and b_full is not None:
                    print(f"  全探延迟差值(s): baseline - proposed = {b_full - p_full:+.1f}")
                print(f"  锁定连续性: proposed={p_cont if p_cont is not None else '-'} | baseline={b_cont if b_cont is not None else '-'}")
                if p_cont is not None and b_cont is not None:
                    print(f"  连续性差值: proposed - baseline = {p_cont - b_cont:+.3f}")
                print(f"  模式切换: proposed={int(r_p.get('detection_mode_changes') or 0)} | baseline={int(r_b.get('detection_mode_changes') or 0)}")
                print("-" * 60)

                def _d(a, b):
                    if a is None or b is None:
                        return None
                    return float(b) - float(a)

                row = [
                    scenario.scenario_id, seed,
                    int(bool(r_p.get('success'))), int(bool(r_b.get('success'))),
                    p_lock, b_lock, _d(p_lock, b_lock),
                    p_cont, b_cont, _d(p_cont, b_cont),
                    int(r_p.get('detection_mode_changes') or 0), int(r_b.get('detection_mode_changes') or 0),
                    int(r_b.get('detection_mode_changes') or 0) - int(r_p.get('detection_mode_changes') or 0),
                    p_full, b_full, _d(p_full, b_full),
                    _fmt(r_p.get('avg_first_detect_delay_s')), _fmt(r_b.get('avg_first_detect_delay_s')),
                    _d(_fmt(r_p.get('avg_first_detect_delay_s')), _fmt(r_b.get('avg_first_detect_delay_s'))),
                    _fmt(r_p.get('activation_mean_s')), _fmt(r_b.get('activation_mean_s')),
                    _fmt(r_p.get('activation_std_s')), _fmt(r_b.get('activation_std_s')),
                    _fmt(r_p.get('target_allocation_gini'), 3), _fmt(r_b.get('target_allocation_gini'), 3),
                    _d(_fmt(r_p.get('target_allocation_gini'), 3), _fmt(r_b.get('target_allocation_gini'), 3)),
                    _fmt(r_p.get('detection_coverage_ratio'), 3), _fmt(r_b.get('detection_coverage_ratio'), 3),
                    _d(_fmt(r_p.get('detection_coverage_ratio'), 3), _fmt(r_b.get('detection_coverage_ratio'), 3)),
                    _fmt(r_p.get('redundant_detection_ratio'), 3), _fmt(r_b.get('redundant_detection_ratio'), 3),
                    _d(_fmt(r_p.get('redundant_detection_ratio'), 3), _fmt(r_b.get('redundant_detection_ratio'), 3)),
                    _fmt(r_p.get('first_detect_time_std')), _fmt(r_b.get('first_detect_time_std')),
                    _d(_fmt(r_p.get('first_detect_time_std')), _fmt(r_b.get('first_detect_time_std'))),
                    _fmt(r_p.get('any_track_continuity_post_activation'), 3), _fmt(r_b.get('any_track_continuity_post_activation'), 3),
                    _d(_fmt(r_p.get('any_track_continuity_post_activation'), 3), _fmt(r_b.get('any_track_continuity_post_activation'), 3)),
                    _fmt(r_p.get('target_primary_entropy'), 3), _fmt(r_b.get('target_primary_entropy'), 3),
                    _d(_fmt(r_p.get('target_primary_entropy'), 3), _fmt(r_b.get('target_primary_entropy'), 3)),
                    _fmt(r_p.get('agent_workload_gini'), 3), _fmt(r_b.get('agent_workload_gini'), 3),
                    _d(_fmt(r_p.get('agent_workload_gini'), 3), _fmt(r_b.get('agent_workload_gini'), 3)),
                    _fmt(r_p.get('detection_coverage_ratio_all_time'), 3), _fmt(r_b.get('detection_coverage_ratio_all_time'), 3),
                    _d(_fmt(r_p.get('detection_coverage_ratio_all_time'), 3), _fmt(r_b.get('detection_coverage_ratio_all_time'), 3)),
                    _fmt(r_p.get('redundant_detection_ratio_sweep'), 3), _fmt(r_b.get('redundant_detection_ratio_sweep'), 3),
                    _d(_fmt(r_p.get('redundant_detection_ratio_sweep'), 3), _fmt(r_b.get('redundant_detection_ratio_sweep'), 3)),
                    _fmt(r_p.get('redundant_detection_ratio_directed'), 3), _fmt(r_b.get('redundant_detection_ratio_directed'), 3),
                    _d(_fmt(r_p.get('redundant_detection_ratio_directed'), 3), _fmt(r_b.get('redundant_detection_ratio_directed'), 3)),
                    int(args.steps)
                ]
                _safe_append_csv_row(summary_csv, row)
                _safe_append_csv_row(summary_csv_zh, row)

            # 生成详细文本报告（便于直接引用到报告文档）
            report_txt = os.path.join(OUTPUT_BASE_DIR, f"compare_report_{timestamp_str}_seed{args.seed}_steps{args.steps}.txt")
            by_scenario = {}
            for r in all_results:
                sid = r.get('scenario_id')
                if not sid:
                    continue
                by_scenario.setdefault(sid, {})[str(r.get('mode') or '').lower()] = r

            def _f(v, nd=1):
                if v is None:
                    return '-'
                try:
                    if isinstance(v, (int, float)):
                        return f"{float(v):.{nd}f}"
                    return str(v)
                except Exception:
                    return str(v)

            with open(report_txt, 'w', encoding='utf-8') as f:
                f.write("协同探测验证 对比报告（proposed vs baseline）\n")
                f.write(f"时间: {timestamp_str} | seed={args.seed} | steps={args.steps} | dt=0.2s\n")
                f.write("说明：同一场景 proposed/baseline 使用相同 seed；敌方场景参数与机动完全一致。\n")
                f.write("\n")

                for scenario in scenarios:
                    sid = scenario.scenario_id
                    rp = by_scenario.get(sid, {}).get('proposed')
                    rb = by_scenario.get(sid, {}).get('baseline')

                    f.write("=" * 80 + "\n")
                    f.write(f"场景 {sid}\n")
                    f.write(f"描述: {scenario.description}\n")
                    f.write(
                        f"设置: 编队={scenario.formation.value} | 高度档={scenario.altitude.value} | 来向={scenario.direction.value} | "
                        f"速度档={scenario.speed.value} | 机动={scenario.maneuver.value} | 初始距离={scenario.initial_distance:.0f}km\n"
                    )
                    try:
                        pos = scenario.generate_initial_positions(base_y=None)
                        f.write("敌机初始(战场坐标km):\n")
                        for i, eid in enumerate(['B0100','B0200','B0300','B0400']):
                            x,y,z = pos[i]
                            f.write(f"  {eid}: x={x:.1f} y={y:.1f} alt={z:.1f}km\n")
                    except Exception:
                        pass
                    f.write("\n")

                    def _write_mode_block(tag, r):
                        if not r:
                            f.write(f"[{tag}] 无结果\n")
                            return
                        f.write(f"[{tag}] success={r.get('success')} | first_lock={_f(r.get('first_lock_time'))}s | mode_changes={r.get('detection_mode_changes')}\n")
                        f.write(f"  full_coverage_continuity(post-activation)={_f(r.get('lock_continuity'),3)} | any_track_continuity(post-activation)={_f(r.get('any_track_continuity_post_activation'),3)}\n")
                        f.write(f"  any_track_continuity(all-time,diag)={_f(r.get('lock_continuity_all_time'),3)} | total_time={_f(r.get('total_time_s'))}s\n")
                        f.write(f"  activation(mean/std)={_f(r.get('activation_mean_s'))}/{_f(r.get('activation_std_s'))} s\n")
                        f.write(f"  avg_first_detect_delay={_f(r.get('avg_first_detect_delay_s'))} s\n")
                        f.write(f"  full_detect_delay={_f(r.get('full_detect_delay_s'))} s | full_detect_time={_f(r.get('full_detect_time_s'))} s\n")
                        # 新增协同探测效能指标
                        f.write(f"  target_allocation_gini={_f(r.get('target_allocation_gini'), 3)} (越大越集中/分工越明确)\n")
                        f.write(f"  target_primary_entropy={_f(r.get('target_primary_entropy'), 3)} (越大越均衡)\n")
                        f.write(f"  agent_workload_gini={_f(r.get('agent_workload_gini'), 3)} (越小越均衡)\n")
                        f.write(f"  detection_coverage_ratio(post-activation)={_f(r.get('detection_coverage_ratio'), 3)} (越大越好)\n")
                        f.write(f"  detection_coverage_ratio(all-time,diag)={_f(r.get('detection_coverage_ratio_all_time'), 3)}\n")
                        f.write(f"  redundant_detection_ratio(all-time)={_f(r.get('redundant_detection_ratio'), 3)} (重叠/冗余度)\n")
                        f.write(f"  redundant_detection_ratio_sweep={_f(r.get('redundant_detection_ratio_sweep'), 3)}\n")
                        f.write(f"  redundant_detection_ratio_directed={_f(r.get('redundant_detection_ratio_directed'), 3)}\n")
                        f.write(f"  first_detect_delay_std={_f(r.get('first_detect_time_std'))} s (越小越均衡)\n")
                        if r.get('per_target_best_first_detect'):
                            f.write("  per-target best first-detect:\n")
                            for item in r['per_target_best_first_detect']:
                                f.write(f"    {item['target']}: {item['agent']} {float(item['delay_s']):.1f}s\n")
                        if r.get('mode_time_s'):
                            f.write(f"  mode_time_s: {r.get('mode_time_s')}\n")
                        if r.get('phase_time_s'):
                            f.write(f"  phase_time_s: {r.get('phase_time_s')}\n")
                        if r.get('a0100_spd_cmd_hist'):
                            f.write(f"  A0100 spd_cmd_hist: {r.get('a0100_spd_cmd_hist')}\n")
                        if r.get('log_file'):
                            f.write(f"  log: {r.get('log_file')}\n")
                        if r.get('acmi_path'):
                            f.write(f"  acmi: {r.get('acmi_path')}\n")

                    _write_mode_block('PROPOSED', rp)
                    _write_mode_block('BASELINE', rb)

                    # 差值（baseline - proposed）
                    if rp and rb:
                        def _delta(key):
                            a = rp.get(key)
                            b = rb.get(key)
                            if a is None or b is None:
                                return '-'
                            try:
                                dv = float(b) - float(a)
                                k = str(key)
                                # 时间类指标：按 0.1s 输出；其余比例/系数类：按 0.001 输出
                                is_time_like = (
                                    k.endswith('_s')
                                    or 'time' in k
                                    or 'delay' in k
                                    or 'activation' in k
                                )
                                return f"{dv:+.1f}" if is_time_like else f"{dv:+.3f}"
                            except Exception:
                                return '-'

                        f.write("\n[DELTA] baseline - proposed\n")
                        f.write(f"  first_lock: {_delta('first_lock_time')} s\n")
                        f.write(f"  full_detect_delay: {_delta('full_detect_delay_s')} s\n")
                        f.write(f"  avg_first_detect_delay: {_delta('avg_first_detect_delay_s')} s\n")
                        f.write(f"  full_coverage_continuity(post-activation): {_delta('lock_continuity')}\n")
                        f.write(f"  any_track_continuity(post-activation): {_delta('any_track_continuity_post_activation')}\n")
                        f.write(f"  target_allocation_gini: {_delta('target_allocation_gini')} (正值表示baseline更集中/分工更明确)\n")
                        f.write(f"  target_primary_entropy: {_delta('target_primary_entropy')} (正值表示baseline主责更均衡)\n")
                        f.write(f"  agent_workload_gini: {_delta('agent_workload_gini')} (负值表示proposed负载更均衡)\n")
                        f.write(f"  detection_coverage_ratio(post-activation): {_delta('detection_coverage_ratio')} (正值表示baseline覆盖更好)\n")
                        f.write(f"  redundant_detection_ratio(all-time): {_delta('redundant_detection_ratio')} (正值表示baseline重叠更多)\n")
                        f.write(f"  first_detect_delay_std: {_delta('first_detect_time_std')} s (负值表示proposed更均衡)\n")
                        try:
                            f.write(
                                f"  mode_changes: {int(rb.get('detection_mode_changes') or 0) - int(rp.get('detection_mode_changes') or 0):+d}\n"
                            )
                        except Exception:
                            pass
                    f.write("\n")

            print(f"\n详细对比报告已写入: {report_txt}")

            print(f"\n对比汇总已写入: {summary_csv}")

            try:
                print(f"对比汇总(中文表头)已写入: {summary_csv_zh}")
            except Exception:
                pass

        else:
            for scenario in scenarios:
                seed = _scenario_seed(args.seed, scenario.scenario_id)
                for mode in modes:
                    run_idx += 1
                    print(f"\n进度: {run_idx}/{total_runs} | {scenario.scenario_id} {mode.upper()} seed={seed}")
                    all_results.append(run_scenario(scenario, max_steps=args.steps, mode=mode, seed=seed))

        # 汇总
        print("\n" + "=" * 60)
        print("验证汇总")
        print("-" * 60)
        print(f"{'场景ID':8} {'模式':9} {'锁定':6} {'首锁(s)':10} {'模式切换':10}")
        print("-" * 60)
        for r in all_results:
            lock_str = "Y" if r['success'] else "N"
            first_lock = f"{r['first_lock_time']:.1f}" if r['first_lock_time'] else "-"
            mode = (r.get('mode') or '-').upper()
            print(f"{r['scenario_id']:8} {mode:9} {lock_str:6} {first_lock:10} {r['detection_mode_changes']:10}")
        print("-" * 60)

        success_count = sum(1 for r in all_results if r.get('success'))
        print(f"成功率: {success_count}/{len(all_results)} ({success_count/len(all_results)*100:.1f}%)")
        return
    else:
        # 无命令行参数，使用交互式菜单
        scenario_ids, max_steps = interactive_menu()
        if scenario_ids is not None or max_steps is not None:
            run_verification(scenario_ids=scenario_ids, max_steps=max_steps)


if __name__ == "__main__":
    main()
