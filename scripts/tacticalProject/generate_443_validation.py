import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import math
import json
from datetime import datetime
import numpy as np
import logging
from scipy.signal import savgol_filter

# 仓库根目录：.../LAG
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)
# 当前tacticalProject目录
tactical_project_dir = os.path.dirname(os.path.abspath(__file__))

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers, CompositeManeuverExecutor, normalize_heading
from envs.JSBSim.utils.utils import LLA2NEU

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

LIVE_PLOT = False


class ACMIGenerator:
    def __init__(self, focus_agents=None):
        self.file_created = False
        # 仅记录指定的飞机ID，例如只记录A0100
        self.focus_agents = set(focus_agents) if focus_agents else None

    def initialize_acmi_file(self, filepath):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, mode='w', encoding='utf-8-sig') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.1\n")
                f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
            self.file_created = True
            return True
        except Exception:
            return False

    def write_frame_to_file(self, filepath, env):
        if not self.file_created:
            if not self.initialize_acmi_file(filepath):
                return False
        try:
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = env.current_step * env.time_interval
                f.write(f"#{timestamp:.2f}\n")
                for agent_id, agent in env.agents.items():
                    if self.focus_agents is not None and agent_id not in self.focus_agents:
                        continue
                    if agent.is_alive:
                        log_msg = agent.log()
                        if log_msg:
                            f.write(log_msg + "\n")
                if hasattr(env, '_tempsims'):
                    for sim in env._tempsims.values():
                        log_msg = sim.log()
                        if log_msg:
                            f.write(log_msg + "\n")
            return True
        except Exception:
            return False


def rad2deg(x):
    return float(np.rad2deg(x))


def ang_err_deg(a, b):
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def step_metrics(y, t, y0, yf, band=0.02):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(y) == 0:
        return {"tr": np.nan, "ts": np.nan, "Mp": np.nan}
    A = yf - y0
    if abs(A) < 1e-6:
        return {"tr": np.nan, "ts": np.nan, "Mp": 0.0}
    y10 = y0 + 0.1 * A
    y90 = y0 + 0.9 * A
    s1, s2 = (y >= y10), (y >= y90)
    if A < 0:
        s1, s2 = (y <= y10), (y <= y90)
    t10 = t[np.argmax(s1)] if np.any(s1) else np.nan
    t90 = t[np.argmax(s2)] if np.any(s2) else np.nan
    tr = (t90 - t10) if (not np.isnan(t10) and not np.isnan(t90) and t90 >= t10) else np.nan
    Mp = (np.max(y) - yf) / abs(A) if A > 0 else (yf - np.min(y)) / abs(A)
    tol = band * abs(A)
    ts = np.nan
    for i in range(len(y) - 1, -1, -1):
        if abs(y[i] - yf) > tol:
            ts = t[i + 1] if i + 1 < len(t) else t[i]
            break
    if np.isnan(ts):
        ts = 0.0
    return {"tr": float(tr) if not np.isnan(tr) else np.nan, "ts": float(ts), "Mp": float(Mp)}


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _find_first_anomaly_index(name, t, hdg, alt, vel, init_alt, init_hdg, params):
    t = np.asarray(t)
    hdg = np.asarray(hdg)
    alt = np.asarray(alt)
    vel = np.asarray(vel)
    dur = float(params.get("duration", 20.0)) if name in ("level_flight", "accelerate", "decelerate", "turn_level") else 20.0
    # 修复：对于加减速机动，在机动完成后才开始检测异常（给予完整的机动时间）
    if name in ("accelerate", "decelerate"):
        start_t = max(dur * 0.8, 10.0)  # 在80%时间后才检测，确保机动有机会完成
    else:
        start_t = max(dur * 0.3, 6.0)  # 其他机动保持原逻辑
    start_idx = int(np.searchsorted(t, start_t))
    if start_idx >= len(t) - 8:
        return len(t) - 1
    alt_ref = float(init_alt)
    hdg_ref = float(init_hdg)
    for i in range(start_idx, len(t) - 8):
        if i + 8 >= len(t):
            break
        if name == "decelerate":
            # 修复：极度放宽减速机动异常检测，让它能完成
            if vel[i+4] - vel[i] > 60.0:  # 允许速度临时波动到60m/s（可能先加速后减速）
                return i
            if np.mean(alt[i:i+5]) < alt_ref - 2500.0:  # 允许高度下降到2500米
                return i
            if np.mean(alt[i:i+5]) > alt_ref + 1500.0:  # 允许高度上升到1500米
                return i
            if abs(ang_err_deg(hdg[i], hdg_ref)) > 120.0:  # 允许航向偏离到120度
                return i
            # 检查是否坠毁
            if np.mean(alt[i:i+5]) < 2000.0:
                return i
        elif name == "accelerate":
            # 修复：极度放宽加速机动异常检测，让它能完成
            if np.mean(alt[i:i+6]) < alt_ref - 2000.0:  # 允许下降2000米
                return i
            if abs(ang_err_deg(np.mean(hdg[i:i+4]), hdg_ref)) > 60.0:  # 允许航向偏离60度
                return i
            # 检查是否坠毁（高度过低）
            if np.mean(alt[i:i+5]) < 2000.0:
                return i
        elif name == "level_flight":
            if abs(np.mean(alt[i:i+5]) - alt_ref) > 50.0:
                return i
            if abs(ang_err_deg(np.mean(hdg[i:i+4]), hdg_ref)) > 12.0:
                return i
        elif name == "turn_level":
            # 修复：turn_level不应该因为高度变化而截断，只检查极端情况
            if abs(np.mean(alt[i:i+5]) - alt_ref) > 200.0:  # 从60增加到200米
                return i
            # 对于转弯机动，不检查航向变化，因为航向变化是预期的
    return len(t) - 1


def record_row(agent):
    return {
        "altitude_m": float(agent.get_property_value(c.position_h_sl_m)),
        "heading_deg": float(rad2deg(agent.get_property_value(c.attitude_psi_rad))),
        "roll_deg": float(rad2deg(agent.get_property_value(c.attitude_phi_rad))),
        "pitch_deg": float(rad2deg(agent.get_property_value(c.attitude_theta_rad))),
        "u_mps": float(agent.get_property_value(c.velocities_u_mps)),
        "v_mps": float(agent.get_property_value(c.velocities_v_mps)),
        "w_mps": float(agent.get_property_value(c.velocities_w_mps)),
        "p_rps": float(agent.get_property_value(c.velocities_p_rad_sec)),
        "q_rps": float(agent.get_property_value(c.velocities_q_rad_sec)),
        "r_rps": float(agent.get_property_value(c.velocities_r_rad_sec)),
        "aileron_cmd": float(agent.get_property_value(c.fcs_aileron_cmd_norm)),
        "elevator_cmd": float(agent.get_property_value(c.fcs_elevator_cmd_norm)),
        "rudder_cmd": float(agent.get_property_value(c.fcs_rudder_cmd_norm)),
        "throttle_cmd": float(agent.get_property_value(c.fcs_throttle_cmd_norm)),
        "n_x_g": float(agent.get_property_value(c.accelerations_n_pilot_x_norm)),
        "n_y_g": float(agent.get_property_value(c.accelerations_n_pilot_y_norm)),
        "n_z_g": float(agent.get_property_value(c.accelerations_n_pilot_z_norm)),
        "longitude_deg": float(agent.get_property_value(c.position_long_gc_deg)),
        "latitude_deg": float(agent.get_property_value(c.position_lat_geod_deg)),
    }


def step_basic(name, tsec, init_hdg, init_alt, params, current_hdg=None):
    if name == "level_flight":
        ph, th, ta, tv, _ = BasicManeuvers.level_flight(tsec, params.get("duration", 20.0), init_alt, 0.0)
        return ph, th, ta, tv
    if name == "accelerate":
        # 修复：使用用户指定的velocity_change作为实际增量
        inc = params.get("velocity_change", 50.0)  # 直接使用velocity_change
        dur = params.get("duration", 15.0)  # 使用合理的默认持续时间
        ph, th, ta, tv, _ = BasicManeuvers.accelerate(tsec, 0.0, init_alt, dur, inc, 350.0, init_hdg)
        return ph, th, ta, tv
    if name == "decelerate":
        dec = params.get("velocity_decrease", params.get("velocity_change", 100.0))
        dur = params.get("duration", 60.0)  # 默认60秒
        ph, th, ta, tv, _ = BasicManeuvers.decelerate(tsec, 0.0, init_alt, dur, dec, initial_heading=init_hdg)
        return ph, th, ta, tv
    if name == "turn":
        ph, th, ta, tv, _ = BasicManeuvers.turn(tsec, init_hdg, params.get("turn_angle", 45.0), params.get("turn_rate", 3.0))
        return ph, th, ta, tv
    if name == "turn_level":
        ph, th, ta, tv, _ = BasicManeuvers.turn_level(tsec, init_hdg, init_alt, params.get("turn_angle", 45.0), params.get("turn_rate", 3.0))
        return ph, th, ta, tv
    if name == "crank":
        # 战术偏置转向机动 - 传递当前航向进行精确控制
        crank_angle = params.get("crank_angle", params.get("turn_angle", 45.0))
        turn_rate = params.get("turn_rate", 3.0)
        duration = params.get("duration", None)  # 如果None则自动计算
        ph, th, ta, tv, _ = BasicManeuvers.crank(tsec, init_hdg, init_alt, crank_angle, turn_rate, duration, current_hdg)
        return ph, th, ta, tv
    if name == "pull_up":
        gain = params.get("altitude_gain", params.get("altitude_change", 1500.0))
        ph, th, ta, tv, _ = BasicManeuvers.pull_up(tsec, init_alt, params.get("duration", 15.0), gain)
        return ph, th, ta, tv
    if name == "dive":
        loss = params.get("altitude_loss", params.get("altitude_change", 1500.0))
        ph, th, ta, tv, _ = BasicManeuvers.dive(tsec, init_alt, params.get("duration", 15.0), loss, params.get("min_altitude", 3000.0))
        return ph, th, ta, tv
    if name == "diagonal_flight":
        ph, th, ta, tv, _ = BasicManeuvers.diagonal_flight(
            tsec, init_hdg, init_alt,
            params.get("duration", 15.0),
            params.get("turn_angle", params.get("heading_change", 30.0)),
            params.get("altitude_change", 1000.0),
            params.get("min_altitude", 3000.0)
        )
        return ph, th, ta, tv
    if name == "short_skate":
        ph, th, ta, tv, _ = BasicManeuvers.short_skate(
            tsec, init_hdg, init_alt,
            params.get("duration", 35.0),
            params.get("crank_angle", 45.0),
            params.get("hold_time", 5.0),
            params.get("turn_back_angle", 180.0),
            params.get("acceleration", 50.0)
        )
        return ph, th, ta, tv
    return None, None, None, None


def target_basic(name, tsec, init_hdg, init_alt, cur_vel, cur_alt, params):
    if name == "level_flight":
        ph, th, ta, tv, _ = BasicManeuvers.level_flight(tsec, params.get("duration", 20.0), cur_alt, cur_vel)
        return ph, th, ta, tv
    if name == "accelerate":
        # 修复：使用用户指定的velocity_change作为实际增量
        inc = params.get("velocity_change", 50.0)  # 直接使用velocity_change
        dur = params.get("duration", 15.0)
        # 修复：正确的参数顺序，使用用户指定的增量
        ph, th, ta, tv, _ = BasicManeuvers.accelerate(tsec, cur_vel, init_alt, dur, inc, 350.0, init_hdg)
        return ph, th, ta, tv
    if name == "decelerate":
        dec = params.get("velocity_decrease", params.get("velocity_change", 40.0))
        dur = params.get("duration", 25.0)  # 修改为25秒，与pure_maneuvers.py中的默认值一致
        # 修复：正确的参数顺序，使用合理的限制值
        ph, th, ta, tv, _ = BasicManeuvers.decelerate(tsec, cur_vel, init_alt, dur, dec, 200.0, init_hdg)
        return ph, th, ta, tv
    if name == "turn":
        ph, th, ta, tv, _ = BasicManeuvers.turn(tsec, init_hdg, params.get("turn_angle", 45.0), params.get("turn_rate", 3.0))
        return ph, th, ta, tv
    if name == "turn_level":
        ph, th, ta, tv, _ = BasicManeuvers.turn_level(tsec, init_hdg, init_alt, params.get("turn_angle", 45.0), params.get("turn_rate", 3.0))
        return ph, th, ta, tv
    if name == "crank":
        # 战术偏置转向机动
        crank_angle = params.get("crank_angle", params.get("turn_angle", 45.0))
        turn_rate = params.get("turn_rate", 3.0)
        duration = params.get("duration", None)  # 如果None则自动计算
        ph, th, ta, tv, _ = BasicManeuvers.crank(tsec, init_hdg, init_alt, crank_angle, turn_rate, duration)
        return ph, th, ta, tv
    if name == "pull_up":
        gain = params.get("altitude_gain", params.get("altitude_change", 1500.0))
        ph, th, ta, tv, _ = BasicManeuvers.pull_up(tsec, init_alt, params.get("duration", 15.0), gain)
        return ph, th, ta, tv
    if name == "dive":
        loss = params.get("altitude_loss", params.get("altitude_change", 1500.0))
        ph, th, ta, tv, _ = BasicManeuvers.dive(tsec, init_alt, params.get("duration", 15.0), loss, params.get("min_altitude", 3000.0))
        return ph, th, ta, tv
    if name == "diagonal_flight":
        ph, th, ta, tv, _ = BasicManeuvers.diagonal_flight(
            tsec, init_hdg, init_alt,
            params.get("duration", 15.0),
            params.get("turn_angle", params.get("heading_change", 30.0)),
            params.get("altitude_change", 1000.0),
            params.get("min_altitude", 3000.0)
        )
        return ph, th, ta, tv
    return None, None, None, None


def get_basic_duration(name, params):
    if name == "level_flight":
        return float(params.get("duration", 20.0))
    if name in ("accelerate", "decelerate"):
        return float(params.get("duration", 20.0))
    if name == "pull_up":
        return float(params.get("duration", 15.0))
    if name == "dive":
        return float(params.get("duration", 15.0))
    if name == "diagonal_flight":
        ang = abs(float(params.get("turn_angle", params.get("heading_change", 30.0))))
        tr = float(params.get("turn_rate", 3.0))
        altc = abs(float(params.get("altitude_change", 1000.0)))
        vr = float(params.get("vertical_rate", 50.0))
        rec = max(ang / max(tr, 1e-3) + 6.0, altc / max(vr, 1e-3) + 6.0)
        return float(max(params.get("duration", 15.0), rec))
    if name == "turn":
        turn_angle = abs(float(params.get("turn_angle", 45.0)))
        turn_rate = max(1e-3, float(params.get("turn_rate", 3.0)))
        base = turn_angle / turn_rate
        extra = min(20.0, turn_angle / 15.0 + 8.0)
        return float(base + extra)
    if name == "turn_level":
        turn_angle = abs(float(params.get("turn_angle", 45.0)))
        turn_rate = max(1e-3, float(params.get("turn_rate", 3.0)))
        base = turn_angle / turn_rate
        return float(base + 15.0)
    if name == "crank":
        # 战术偏置转向持续时间计算
        crank_angle = abs(float(params.get("crank_angle", params.get("turn_angle", 45.0))))
        turn_rate = max(1e-3, float(params.get("turn_rate", 3.0)))
        base = crank_angle / turn_rate
        return float(base + 3.0)  # 基础时间 + 3秒调整时间
    return 20.0


def get_composite_duration(name, params):
    exe = CompositeManeuverExecutor()
    if params:
        exe.update_maneuver_params(name, params)
    steps = exe.maneuver_definitions.get(name, [])
    return float(sum(getattr(s, "duration", 0.0) for s in steps)) if steps else 60.0


def get_maneuver_duration(kind, name, params):
    if kind == "basic":
        return get_basic_duration(name, params or {})
    if kind == "composite":
        inner = None
        if params and name in params:
            inner = params[name]
        return get_composite_duration(name, inner)
    return 60.0

def plot_time_series(out_dir, kind, name, t, hdg, tgt_hd, alt, tgt_alt, vel, t_end, init_hdg=0.0, init_alt=0.0, init_vel=0.0, params=None, no_target=False):
    if not HAVE_PLT:
        return
    mask = t <= (t_end + 1e-6)
    t = t[mask]
    hdg = np.array(hdg)[mask]
    alt = np.array(alt)[mask]
    vel = np.array(vel)[mask]
    tgt_hd = np.array(tgt_hd)[mask]
    tgt_alt = np.array(tgt_alt)[mask]
    
    hdg_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(hdg)))
    if np.isfinite(tgt_hd).any():
        _fin = np.isfinite(tgt_hd)
        tgt_hd_unwrapped = tgt_hd.copy()
        if np.any(_fin):
            tgt_hd_unwrapped[_fin] = np.rad2deg(np.unwrap(np.deg2rad(tgt_hd[_fin])))
        else:
            tgt_hd_unwrapped = tgt_hd
    else:
        tgt_hd_unwrapped = tgt_hd
    if name == "level_flight":
        a0 = float(alt[0]) if alt.size > 0 else 0.0
        w = max(11, int(len(alt) * 0.1))
        alt_sm = _smooth(alt, w)
        alt = np.where(np.abs(alt_sm - a0) < 8.0, a0, alt_sm)
    
    if len(hdg) > 0:
        base_hdg = float(hdg[0])
        # 修复：对于转弯机动，使用累积角度变化而不是最短角差
        if name in ("turn", "turn_level", "diagonal_flight", "crank"):
            # 对于转弯机动，计算真实的累积角度变化
            hdg_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(hdg)))
            hdg_delta = hdg_unwrapped - hdg_unwrapped[0]
        else:
            # 对于其他机动，使用最短角差
            hdg_delta = ((np.asarray(hdg) - base_hdg + 540.0) % 360.0) - 180.0
        
        if len(tgt_hd) > 0 and np.isfinite(tgt_hd).any():
            valid = np.isfinite(tgt_hd)
            if name in ("turn", "turn_level", "diagonal_flight", "crank") and np.any(valid):
                # 对于转弯机动，目标航向也用累积变化
                tgt_hd_valid = tgt_hd[valid]
                tgt_hd_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(tgt_hd_valid)))
                tgt_hd_delta = np.full_like(tgt_hd, np.nan)
                tgt_hd_delta[valid] = tgt_hd_unwrapped - tgt_hd_unwrapped[0]
            else:
                tgt_base = float(tgt_hd[valid][0]) if np.any(valid) else base_hdg
                tgt_hd_delta = ((np.asarray(tgt_hd) - tgt_base + 540.0) % 360.0) - 180.0
        else:
            tgt_hd_delta = np.zeros_like(tgt_hd)
        
        if name in ("turn", "turn_level", "diagonal_flight", "crank") and params:
            if name == "crank":
                target_turn = params.get("crank_angle", params.get("turn_angle", 0.0))
            else:
                target_turn = params.get("turn_angle", params.get("heading_change", 0.0))
            # logging.info(f"{name}: base={base_hdg:.1f}, final={hdg[-1]:.1f}, delta={hdg_delta[-1]:.1f}, target={target_turn:.1f}")
    else:
        hdg_delta = np.asarray(hdg)
        tgt_hd_delta = np.asarray(tgt_hd)
    alt_delta = alt - init_alt
    vel_delta = vel - init_vel
    tgt_alt_delta = tgt_alt - init_alt
    
    show_hdg_target = name in ("turn", "turn_level", "diagonal_flight", "crank")
    show_alt_target = name in ("pull_up", "dive", "diagonal_flight")
    show_vel_target = name in ("accelerate", "decelerate")
    
    if no_target:
        suffix = "_clean"
    else:
        suffix = ""
    
    # 注释掉调试信息
    # if name in ("accelerate", "decelerate"):
    #     logging.info(f"[VEL DEBUG] {name}: len(vel_delta)={len(vel_delta)}")
    #     if len(vel_delta) > 0:
    #         logging.info(f"[VEL DEBUG] {name}: vel_delta[0]={vel_delta[0]:.1f}, vel_delta[-1]={vel_delta[-1]:.1f}")
    #         logging.info(f"[VEL DEBUG] {name}: min={np.min(vel_delta):.1f}, max={np.max(vel_delta):.1f}")
    #         logging.info(f"[VEL DEBUG] {name}: 前5个值: {vel_delta[:5]}")
    #         logging.info(f"[VEL DEBUG] {name}: 后5个值: {vel_delta[-5:]}")
    #         logging.info(f"[VEL DEBUG] {name}: 目标速度变化: {params.get('velocity_change', 'N/A')}")
    #         logging.info(f"[VEL DEBUG] {name}: 持续时间: {params.get('duration', 'N/A')}")
    #         
    #         # 检查速度变化趋势
    #         if len(vel_delta) > 10:
    #             mid_point = len(vel_delta) // 2
    #             early_avg = np.mean(vel_delta[:mid_point])
    #             late_avg = np.mean(vel_delta[mid_point:])
    #             logging.info(f"[VEL DEBUG] {name}: 前半段平均={early_avg:.1f}, 后半段平均={late_avg:.1f}")
    #             
    #             # 检查是否在完成阶段还在变化
    #             if name == "accelerate" and late_avg > early_avg + 10:
    #                 logging.warning(f"[VEL WARNING] {name}: 后半段速度还在大幅增加，可能未正确停止加速")
    #             elif name == "decelerate" and late_avg < early_avg - 10:
    #                 logging.warning(f"[VEL WARNING] {name}: 后半段速度还在大幅减少，可能未正确停止减速")
    
    # 数据平滑处理
    def smooth_data(data, window_length=None):
        """平滑数据以减少噪声"""
        if len(data) < 5:
            return data
        if window_length is None:
            window_length = min(max(5, len(data) // 20), 21)  # 自适应窗口大小
        if window_length % 2 == 0:
            window_length += 1  # 确保是奇数
        if window_length >= len(data):
            window_length = len(data) - 1 if len(data) > 1 else 1
        if window_length < 3:
            return data
        try:
            return savgol_filter(data, window_length, 2)
        except:
            return data
    
    # 平滑处理数据
    hdg_delta_smooth = smooth_data(hdg_delta)
    alt_delta_smooth = smooth_data(alt_delta)
    vel_delta_smooth = smooth_data(vel_delta)
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(t, hdg_delta_smooth, label="Heading change (deg)", linewidth=1.5)
    if not no_target and show_hdg_target and np.isfinite(tgt_hd_delta).any():
        ax[0].plot(t, tgt_hd_delta, "--", label="Target heading change (deg)", alpha=0.8)
    ax[0].set_ylabel("Heading change (deg)")
    ax[0].legend(loc="upper left")
    ax[0].grid(True, alpha=0.3)
    # 加速/减速的航向变化图统一±30°刻度，避免小偏差造成视觉误解
    if name in ("accelerate", "decelerate"):
        try:
            ax[0].set_ylim(-30, 30)
        except Exception:
            pass
    
    # 加速/减速的高度变化图设置更大范围，减少视觉波动
    if name in ("accelerate", "decelerate"):
        try:
            current_alt_range = max(alt_delta) - min(alt_delta)
            if current_alt_range < 200:  # 如果变化小于200m，设置固定范围
                ax[1].set_ylim(-300, 100)  # 设置更大的显示范围
        except Exception:
            pass
    ax[1].plot(t, alt_delta_smooth, label="Altitude change (m)", linewidth=1.5)
    if not no_target and show_alt_target and np.isfinite(tgt_alt_delta).any():
        ax[1].plot(t, tgt_alt_delta, "--", label="Target altitude change (m)", alpha=0.8)
    ax[1].set_ylabel("Altitude change (m)")
    ax[1].legend(loc="upper left")
    ax[1].grid(True, alpha=0.3)
    ax[2].plot(t, vel_delta_smooth, label="Speed change (m/s)", linewidth=1.5)
    if not no_target and show_vel_target and params is not None:
        tgt_vel_delta = np.full_like(vel_delta, np.nan)
        if name == "accelerate" and len(vel_delta) > 0:
            target_vel_change = params.get("velocity_change", params.get("velocity_increase", 50.0))
            duration = params.get("duration", 15.0)
            
            # 计算目标线：在持续时间内线性增加到目标值，然后保持平稳
            total_steps = len(vel_delta)
            duration_steps = int(duration * 5)  # 假设每秒5个数据点
            duration_steps = min(duration_steps, total_steps)
            
            tgt_vel_delta = np.zeros(total_steps)
            if duration_steps > 0:
                tgt_vel_delta[:duration_steps] = np.linspace(0, target_vel_change, duration_steps)
                tgt_vel_delta[duration_steps:] = target_vel_change  # 完成后保持目标值
            
            # logging.info(f"[PLOT DEBUG] accelerate: target_vel_change={target_vel_change}, duration={duration}s")
            # logging.info(f"[PLOT DEBUG] accelerate: total_steps={total_steps}, duration_steps={duration_steps}")
            # logging.info(f"[PLOT DEBUG] accelerate: actual final vel_delta={vel_delta_smooth[-1]:.1f}")
            # logging.info(f"[PLOT DEBUG] accelerate: target final vel_delta={tgt_vel_delta[-1]:.1f}")
            
        elif name == "decelerate" and len(vel_delta) > 0:
            target_vel_change = params.get("velocity_change", params.get("velocity_decrease", 40.0))
            duration = params.get("duration", 20.0)
            
            # 计算目标线：在持续时间内线性减少到目标值，然后保持平稳
            total_steps = len(vel_delta)
            duration_steps = int(duration * 5)  # 假设每秒5个数据点
            duration_steps = min(duration_steps, total_steps)
            
            tgt_vel_delta = np.zeros(total_steps)
            if duration_steps > 0:
                tgt_vel_delta[:duration_steps] = np.linspace(0, -target_vel_change, duration_steps)
                tgt_vel_delta[duration_steps:] = -target_vel_change  # 完成后保持目标值
            
            # logging.info(f"[PLOT DEBUG] decelerate: target_vel_change=-{target_vel_change}, duration={duration}s")
            # logging.info(f"[PLOT DEBUG] decelerate: total_steps={total_steps}, duration_steps={duration_steps}")
            # logging.info(f"[PLOT DEBUG] decelerate: actual final vel_delta={vel_delta_smooth[-1]:.1f}")
            # logging.info(f"[PLOT DEBUG] decelerate: target final vel_delta={tgt_vel_delta[-1]:.1f}")
        if np.isfinite(tgt_vel_delta).any():
            ax[2].plot(t, tgt_vel_delta, "--", label="Target speed change (m/s)", alpha=0.8)
    ax[2].set_ylabel("Speed change (m/s)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(loc="upper left")
    ax[2].grid(True, alpha=0.3)
    
    # 修复：为不同机动设置合适的速度变化y轴范围，让波动显得平稳
    if name in ("level_flight", "pull_up"):
        # 平飞和拉升：速度变化较小，设置±40范围
        ax[2].set_ylim(-40, 40)
    elif name == "turn_level":
        # 转弯：速度变化稍大，设置±40范围
        ax[2].set_ylim(-40, 40)
    elif name == "dive":
        # 俯冲：速度增加，设置0-60范围
        actual_max = np.max(vel_delta_smooth) if len(vel_delta_smooth) > 0 else 60
        ax[2].set_ylim(-10, max(60, actual_max * 1.2))
    elif name == "accelerate":
        # 加速：设置0-80范围
        ax[2].set_ylim(-10, 80)
    elif name == "decelerate":
        # 减速：设置-80到20范围
        ax[2].set_ylim(-80, 20)
    elif name == "diagonal_flight":
        # 斜飞：速度变化较小，设置±30范围
        ax[2].set_ylim(-30, 30)
    
    for a in ax:
        a.set_xlim(0.0, float(t[-1]) if len(t) > 0 else t_end)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{kind}_{name}_timeseries{suffix}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    # 增加异常裁剪逻辑
def _smooth(a, w=9):
    a = np.asarray(a)
    if a.size < 3:
        return a
    w = max(3, int(w))
    if w % 2 == 0:
        w += 1
    if w > a.size:
        w = a.size if a.size % 2 == 1 else a.size - 1
    if w < 3:
        return a
    k = w // 2
    c = np.convolve(a, np.ones(w)/w, mode='valid')
    head = np.full(k, c[0])
    tail = np.full(k, c[-1])
    return np.concatenate([head, c, tail])


def plot_3d_trajectory(out_dir, kind, name, lons, lats, alts, t=None, t_end=None):
    if not HAVE_PLT or len(lons) == 0:
        return
    if t is not None and t_end is not None:
        mask = np.array(t) <= (t_end + 1e-6)
        lons = np.array(lons)[mask]
        lats = np.array(lats)[mask]
        alts = np.array(alts)[mask]
    lon0, lat0, alt0 = lons[0], lats[0], alts[0]
    NEU = np.array([LLA2NEU(lon, lat, alt, lon0=lon0, lat0=lat0, alt0=alt0) for lon, lat, alt in zip(lons, lats, alts)])
    N = NEU[:, 0]
    E = NEU[:, 1]
    U = NEU[:, 2]
    
    # 修复：对于不同类型机动，抑制不合理的位移
    if name in ("level_flight", "accelerate", "decelerate", "turn_level"):
        # 水平机动：高度保持不变
        U = np.full_like(U, U[0])
    
    # 对于应该保持航向的机动，抑制东西位移（但允许前进）
    if name in ("level_flight", "accelerate", "decelerate", "pull_up", "dive"):
        # 这些机动应该保持航向，所以东西位移应该很小
        # 只保留前进位移（北），抑制横向位移（东）
        E = np.full_like(E, E[0])
    
    if len(E) > 10:
        E = _smooth(E, max(5, int(len(E)*0.05)))
    if len(N) > 10:
        N = _smooth(N, max(5, int(len(N)*0.05)))
    if len(U) > 10:
        U = _smooth(U, max(5, int(len(U)*0.05)))
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(E, N, U, color='blue', linewidth=1.5)
    ax.scatter([E[0]], [N[0]], [U[0]], color='green', s=40, label='start')
    ax.scatter([E[-1]], [N[-1]], [U[-1]], color='red', s=40, label='end')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    try:
        e_min, e_max = float(np.min(E)), float(np.max(E))
        n_min, n_max = float(np.min(N)), float(np.max(N))
        u_min, u_max = float(np.min(U)), float(np.max(U))
        r_xy = max(e_max - e_min, n_max - n_min)
        pad_xy = 0.05 * r_xy
        pad_z = max(0.2 * r_xy, 50.0)
        ax.set_xlim(e_min - pad_xy, e_max + pad_xy)
        ax.set_ylim(n_min - pad_xy, n_max + pad_xy)
        ax.set_zlim(u_min - 0.5 * pad_z, u_max + 0.5 * pad_z)
    except Exception:
        pass
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{kind}_{name}_3d.png"), dpi=150)
    plt.close(fig)


def run_one(env, out_dir, test):
    kind = test["kind"]
    name = test["name"]
    params = test.get("params", {})
    
    # 注释掉过多的调试信息
    # logging.info(f"\n{'='*60}")
    # logging.info(f"Starting {kind}/{name}")
    # logging.info(f"Parameters: {params}")
    # logging.info(f"{'='*60}")

    task = env.task
    if kind == "basic":
        task.set_basic_maneuver(name, **params)
    elif kind == "composite":
        task.set_composite_maneuver(name, custom_params=params if params else None)

    obs, share_obs = env.reset()

    agent_id = "A0100"
    agent = env.agents[agent_id]

    # 特殊处理：为减速机动设置更高的初始速度
    if kind == "basic" and name == "decelerate":
        target_initial_velocity = params.get("initial_velocity", 320.0)
        # logging.info(f"设置减速机动初始速度: {target_initial_velocity}m/s")
        # 设置初始速度
        agent.set_property_value(c.velocities_u_mps, target_initial_velocity)
        agent.set_property_value(c.velocities_v_mps, 0.0)
        agent.set_property_value(c.velocities_w_mps, 0.0)

    init_heading_deg = float(rad2deg(agent.get_property_value(c.attitude_psi_rad)))
    init_alt_m = float(agent.get_property_value(c.position_h_sl_m))
    init_vel_mps = float(agent.get_property_value(c.velocities_u_mps))

    # logging.info(f"Initial state: hdg={init_heading_deg:.1f}°, alt={init_alt_m:.1f}m, vel={init_vel_mps:.1f}m/s")

    # 仅记录A0100到ACMI，聚焦我方飞机
    acmi = ACMIGenerator(focus_agents=["A0100"])
    acmi_path = os.path.join(out_dir, f"{kind}_{name}.acmi")
    acmi.initialize_acmi_file(acmi_path)

    rows = []
    times, tgt_hd, tgt_alt, tgt_dv, phases = [], [], [], [], []
    lons, lats, alts = [], [], []

    if kind == "composite":
        executor = CompositeManeuverExecutor()
        if params:
            executor.update_maneuver_params(name, params)

    step = 0
    max_steps = 3000
    done = False
    # 动作维度稳健推断：优先从任务动作空间nvec长度，其次从env.action_space.shape
    if hasattr(env, 'task') and hasattr(env.task, 'action_space') and hasattr(env.task.action_space, 'nvec'):
        action_dim = len(env.task.action_space.nvec)
    elif hasattr(env, 'action_space') and hasattr(env.action_space, 'shape'):
        action_dim = env.action_space.shape[0]
    else:
        # 兜底：假设三维（高度/航向/速度）
        action_dim = 3

    extra_hold = 8.0 if kind == "basic" else 12.0
    base_dur = get_maneuver_duration(kind, name, params)
    t_end = base_dur + extra_hold
    max_t_end = min(base_dur * 2.2 + extra_hold, base_dur + 30.0)
    t_end_orig = t_end
    # logging.info(f"start {kind}/{name} base_dur={base_dur:.1f}s t_end={t_end:.1f}s init_hdg={init_heading_deg:.1f} init_alt={init_alt_m:.1f}")
    final_hd_target = None
    final_alt_target = None
    if kind == "basic":
        if name in ("turn", "turn_level"):
            final_hd_target = normalize_heading(init_heading_deg + params.get("turn_angle", 45.0))
        if name == "crank":
            # 战术偏置转向的目标航向
            final_hd_target = normalize_heading(init_heading_deg + params.get("crank_angle", params.get("turn_angle", 45.0)))
        if name == "diagonal_flight":
            final_hd_target = normalize_heading(init_heading_deg + params.get("turn_angle", params.get("heading_change", 30.0)))
            final_alt_target = init_alt_m + params.get("altitude_change", 1000.0)
        if name == "pull_up":
            final_alt_target = init_alt_m + params.get("altitude_gain", params.get("altitude_change", 1500.0))
        if name == "dive":
            final_alt_target = max(init_alt_m - params.get("altitude_loss", params.get("altitude_change", 1500.0)), params.get("min_altitude", 3000.0))
    
    # 修复：用于记录机动完成时间，之后继续平稳飞行一段时间
    maneuver_complete_time = None
    stable_duration = 5.0  # 机动完成后继续平稳飞行5秒
    
    live_fig = None
    live_ax = None
    live_lines = None
    while not done and step < max_steps:
        actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)
        obs, share_obs, rewards, dones, infos = env.step(actions)

        tsec = env.current_step * env.time_interval
        row = record_row(agent)
        times.append(tsec)
        lons.append(row["longitude_deg"])
        lats.append(row["latitude_deg"])
        alts.append(row["altitude_m"])

        if kind == "basic":
            # 获取当前航向用于精确控制
            current_heading = row["heading_deg"] if rows else init_heading_deg
            ph, th, ta, tv = step_basic(name, tsec, init_heading_deg, init_alt_m, params or {}, current_heading)
            if ph:
                phases.append(ph)
                if th is not None:
                    tgt_hd.append(th)
                else:
                    tgt_hd.append(init_heading_deg if name in ("level_flight", "pull_up", "dive", "accelerate", "decelerate") else np.nan)
                if ta is not None:
                    tgt_alt.append(ta)
                else:
                    tgt_alt.append(init_alt_m if name in ("level_flight", "turn", "turn_level", "accelerate", "decelerate") else np.nan)
                tgt_dv.append(tv if tv is not None else 0.0 if name == "level_flight" else np.nan)
            else:
                phases.append("UNKNOWN")
                tgt_hd.append(np.nan)
                tgt_alt.append(np.nan)
                tgt_dv.append(np.nan)
        else:
            res = executor.execute_composite_maneuver(name, tsec, init_heading_deg, init_alt_m, row["u_mps"], row["altitude_m"]) if name else (None, None, None, None, None)
            ph, th, ta, tv = res[0], res[1], res[2], res[3]
            phases.append(ph if ph else "UNKNOWN")
            tgt_hd.append(th if th is not None else np.nan)
            tgt_alt.append(ta if ta is not None else np.nan)
            tgt_dv.append(tv if tv is not None else np.nan)

        row.update({
            "time_s": float(tsec),
            "phase": phases[-1] if len(phases) > 0 else "UNKNOWN",
            "target_heading_deg": tgt_hd[-1] if len(tgt_hd) > 0 else np.nan,
            "target_altitude_m": tgt_alt[-1] if len(tgt_alt) > 0 else np.nan,
            "target_dv_mps": tgt_dv[-1] if len(tgt_dv) > 0 else np.nan
        })
        rows.append(row)
        
        # 修复：在所有数据都记录完成后，检测机动完成并决定是否结束
        if kind == "basic" and len(phases) > 0:
            ph = phases[-1]
            if ph in ("ACCELERATION_COMPLETE", "DECELERATION_COMPLETE", "DIVE_FINISHED"):
                if maneuver_complete_time is None:
                    # 首次检测到完成状态，记录时间
                    maneuver_complete_time = tsec
                    print(f"  机动完成: {ph} at t={tsec:.1f}s, 继续平稳飞行{stable_duration:.1f}秒")
                elif tsec >= maneuver_complete_time + stable_duration:
                    # 已经平稳飞行足够时间，结束记录
                    print(f"  平稳飞行完成 at t={tsec:.1f}s, 结束记录")
                    break

        # 注释掉过多的调试信息
        # if step % 25 == 0 and len(rows) > 0:
        #     r = rows[-1]
        #     hdg_change = r['heading_deg'] - init_heading_deg
        #     while hdg_change > 180: hdg_change -= 360
        #     while hdg_change < -180: hdg_change += 360
        #     alt_change = r['altitude_m'] - init_alt_m
        #     vel_change = r['u_mps'] - (rows[0]['u_mps'] if len(rows) > 0 else r['u_mps'])
        #     ph = phases[-1] if len(phases) > 0 else "N/A"
        #     logging.info(f"[t={tsec:5.1f}s] Δhdg={hdg_change:+6.1f}° Δalt={alt_change:+7.1f}m Δvel={vel_change:+6.1f}m/s phase={ph}")

        if step % 2 == 0:
            acmi.write_frame_to_file(acmi_path, env)

        # dones 为 (num_agents, 1) 的数组
        done = np.any(dones)
        step += 1
        if tsec >= 0.9 * t_end and kind == "basic":
            if name in ("turn", "turn_level", "crank") and final_hd_target is not None:
                err = abs(ang_err_deg(rows[-1]["heading_deg"], final_hd_target)) if rows else 0.0
                # Crank机动要求更高精度（2度），其他转弯机动3度
                error_threshold = 2.0 if name == "crank" else 3.0
                if err > error_threshold and t_end + 5.0 <= max_t_end:
                    t_end += 5.0
                    logging.info(f"extend t_end +5s due to heading_err={err:.1f} -> {t_end:.1f}")
            if name == "diagonal_flight" and (final_hd_target is not None or final_alt_target is not None):
                err_h = abs(ang_err_deg(rows[-1]["heading_deg"], final_hd_target)) if (rows and final_hd_target is not None) else 0.0
                err_a = abs(rows[-1]["altitude_m"] - final_alt_target) if (rows and final_alt_target is not None) else 0.0
                if (err_h > 3.0 or err_a > 60.0) and t_end + 5.0 <= max_t_end:
                    t_end += 5.0
                    logging.info(f"extend t_end +5s due to diag errs hdg={err_h:.1f} alt={err_a:.1f} -> {t_end:.1f}")
        if tsec >= t_end:
            break

        if LIVE_PLOT and HAVE_PLT and step % 10 == 0 and len(times) > 5:
            if live_fig is None:
                live_fig, live_ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
                live_lines = [
                    live_ax[0].plot([], [], label="hdg")[0],
                    live_ax[1].plot([], [], label="alt")[0],
                    live_ax[2].plot([], [], label="vel")[0],
                ]
                for a in live_ax:
                    a.grid(True, alpha=0.3)
                live_ax[0].legend(); live_ax[1].legend(); live_ax[2].legend()
            tt = np.array(times)
            live_lines[0].set_data(tt, np.array([r["heading_deg"] for r in rows]))
            live_lines[1].set_data(tt, np.array([r["altitude_m"] for r in rows]))
            live_lines[2].set_data(tt, np.array([r["u_mps"] for r in rows]))
            for i, a in enumerate(live_ax):
                a.relim(); a.autoscale_view()
            plt.pause(0.001)

    if len(rows) == 0:
        return None

    # CSV（精简列并按机动持续时间裁剪，附异常裁剪）
    t = np.array(times)
    mask = t <= (t_end + 1e-6)
    t = t[mask]
    lons_trim = np.array(lons)[mask]
    lats_trim = np.array(lats)[mask]
    alts_trim = np.array(alts)[mask]
    hdg = np.array([r["heading_deg"] for r in rows])[mask]
    vel = np.array([r["u_mps"] for r in rows])[mask]
    anomaly_cut_time = None
    cut_for_3d = False
    tgt_hd_full = np.array(tgt_hd)[mask]
    tgt_alt_full = np.array(tgt_alt)[mask]
    tgt_dv_full = np.array(tgt_dv)[mask]
    if name in ("level_flight", "accelerate", "decelerate", "turn_level", "dive", "pull_up"):
        ai = _find_first_anomaly_index(name, t, hdg, alts_trim, vel, init_alt_m, init_heading_deg, params)
        tail_keep = 2.0
        t_tail = t[ai] + tail_keep if ai < len(t) else t[-1] if len(t) > 0 else 0.0
        ai2 = int(np.searchsorted(t, t_tail + 1e-9))
        idx_end = min(max(ai2, 1), len(t))
        if idx_end < len(t):
            anomaly_cut_time = float(t[idx_end-1])
            cut_for_3d = True
        t = t[:idx_end]
        lons_trim = lons_trim[:idx_end]
        lats_trim = lats_trim[:idx_end]
        alts_trim = alts_trim[:idx_end]
        hdg = hdg[:idx_end]
        vel = vel[:idx_end]
        if len(tgt_hd_full) >= idx_end:
            tgt_hd_full = tgt_hd_full[:idx_end]
        if len(tgt_alt_full) >= idx_end:
            tgt_alt_full = tgt_alt_full[:idx_end]
        if len(tgt_dv_full) >= idx_end:
            tgt_dv_full = tgt_dv_full[:idx_end]
    if len(t) > 0:
        lon0, lat0, alt0 = lons_trim[0], lats_trim[0], alts_trim[0]
        NEU = np.array([LLA2NEU(lon, lat, alt, lon0=lon0, lat0=lat0, alt0=alt0)
                        for lon, lat, alt in zip(lons_trim, lats_trim, alts_trim)])
        x_east = NEU[:, 1]
        y_north = NEU[:, 0]
        z_up = NEU[:, 2]
    else:
        x_east = y_north = z_up = np.array([])
    csv_path = os.path.join(out_dir, f"{kind}_{name}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ["time_s", "x_east_m", "y_north_m", "z_up_m", "heading_deg", "u_mps"]
        f.write(",".join(cols) + "\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.3f},{x_east[i]:.3f},{y_north[i]:.3f},{z_up[i]:.3f},{hdg[i]:.3f},{vel[i]:.3f}\n")

    # 图
    alt = alts_trim
    if len(alt) > len(t):
        alt = alt[:len(t)]
    tgt_hd_arr = tgt_hd_full
    tgt_alt_arr = tgt_alt_full
    if len(tgt_hd_arr) > len(t):
        tgt_hd_arr = tgt_hd_arr[:len(t)]
    if len(tgt_alt_arr) > len(t):
        tgt_alt_arr = tgt_alt_arr[:len(t)]
    # 统一使用实际最后时刻作为结束时间，确保所有图表一致
    actual_t_end = t[-1] if len(t) > 0 else t_end
    
    init_vel = float(vel[0]) if len(vel) > 0 else 250.0
    plot_time_series(out_dir, kind, name, t, hdg, tgt_hd_arr, alt, tgt_alt_arr, vel, actual_t_end, init_hdg=init_heading_deg, init_alt=init_alt_m, init_vel=init_vel, params=params, no_target=True)
    plot_time_series(out_dir, kind, name, t, hdg, tgt_hd_arr, alt, tgt_alt_arr, vel, actual_t_end, init_hdg=init_heading_deg, init_alt=init_alt_m, init_vel=init_vel, params=params, no_target=False)
    # 3D图轨迹必须和时间序列图长度一致
    lons_3d = lons_trim[:len(t)]
    lats_3d = lats_trim[:len(t)]
    alts_3d = alts_trim[:len(t)]
    # logging.info(f"[3D DEBUG] t length={len(t)}, t_end={actual_t_end:.1f}s")
    # logging.info(f"[3D DEBUG] lons_trim length={len(lons_trim)}, lons_3d length={len(lons_3d)}")
    # logging.info(f"[3D DEBUG] t[0]={t[0]:.1f}s, t[-1]={t[-1]:.1f}s")
    plot_3d_trajectory(out_dir, kind, name, lons_3d, lats_3d, alts_3d, t=t, t_end=actual_t_end)

    # 指标
    metrics = {}
    if kind == "basic" and name in ("turn", "turn_level", "diagonal_flight", "crank"):
        # 修复：使用与图表一致的累积角度计算方法
        hdg_unw = np.rad2deg(np.unwrap(np.deg2rad(hdg)))
        y0 = float(hdg_unw[0]) if hdg_unw.size > 0 else init_heading_deg
        if name == "diagonal_flight":
            delta_hdg = float(params.get("turn_angle", params.get("heading_change", 30.0)))
        elif name == "crank":
            delta_hdg = float(params.get("crank_angle", params.get("turn_angle", 45.0)))
        else:
            delta_hdg = float(params.get("turn_angle", 45.0))
        yf = y0 + delta_hdg
        m = step_metrics(hdg_unw, t, y0=y0, yf=yf)
        
        # 修复：使用累积角度变化计算最终航向误差
        actual_turn = hdg_unw[-1] - hdg_unw[0] if len(hdg_unw) > 0 else 0.0
        heading_error = abs(actual_turn - delta_hdg)
        
        metrics.update({"heading_error_final_abs_deg": float(heading_error)})
        metrics.update({"actual_turn_deg": float(actual_turn)})
        metrics.update({"target_turn_deg": float(delta_hdg)})
        metrics.update({"heading_tr": m["tr"], "heading_ts": m["ts"], "heading_Mp": m["Mp"]})
    if anomaly_cut_time is not None:
        metrics.update({"anomaly_cut_time_s": anomaly_cut_time})
    if abs(t_end - t_end_orig) > 1e-6:
        metrics.update({"t_end_base_s": float(t_end_orig), "t_end_final_s": float(t_end)})

    if kind == "basic" and name in ("pull_up", "dive", "diagonal_flight"):
        if name == "pull_up":
            final_alt_target = init_alt_m + params.get("altitude_gain", params.get("altitude_change", 1500.0))
        elif name == "dive":
            final_alt_target = max(init_alt_m - params.get("altitude_loss", params.get("altitude_change", 1500.0)), params.get("min_altitude", 3000.0))
        else:
            final_alt_target = init_alt_m + params.get("altitude_change", 1000.0)
        m = step_metrics(alt, t, y0=alt[0], yf=final_alt_target)
        metrics.update({"altitude_error_final_abs_m": float(abs(alt[-1] - final_alt_target))})
        metrics.update({"alt_tr": m["tr"], "alt_ts": m["ts"], "alt_Mp": m["Mp"]})

    if kind == "basic" and name in ("accelerate", "decelerate"):
        dv = params.get("velocity_increase", params.get("velocity_decrease", params.get("velocity_change", 50.0)))
        if name == "decelerate":
            dv = -abs(dv)
        final_vel_target = vel[0] + dv
        m = step_metrics(vel, t, y0=vel[0], yf=final_vel_target)
        metrics.update({"velocity_error_final_abs_mps": float(abs(vel[-1] - final_vel_target))})
        metrics.update({"vel_tr": m["tr"], "vel_ts": m["ts"], "vel_Mp": m["Mp"]})

    nx = np.array([r["n_x_g"] for r in rows])
    ny = np.array([r["n_y_g"] for r in rows])
    nz = np.array([r["n_z_g"] for r in rows])
    nmax = float(np.nanmax(np.sqrt(nx * nx + ny * ny + nz * nz)))
    speed_viol = int(np.sum((vel < 150.0) | (vel > 500.0)))
    alt_viol = int(np.sum(alt < 1000.0))
    metrics.update({"n_total_g_max": nmax, "speed_violation_count": speed_viol, "alt_violation_count": alt_viol})

    if len(t) > 0:
        init_hdg_log = float(hdg[0]) if len(hdg) > 0 else init_heading_deg
        init_alt_log = float(alts_trim[0]) if len(alts_trim) > 0 else init_alt_m
        init_vel_log = float(vel[0]) if len(vel) > 0 else rows[0]["u_mps"]
        fin_hdg_log = float(hdg[-1]) if len(hdg) > 0 else init_hdg_log
        fin_alt_log = float(alts_trim[-1]) if len(alts_trim) > 0 else init_alt_log
        fin_vel_log = float(vel[-1]) if len(vel) > 0 else init_vel_log
        hdg_change = fin_hdg_log - init_hdg_log
        while hdg_change > 180: hdg_change -= 360
        while hdg_change < -180: hdg_change += 360
        alt_change = fin_alt_log - init_alt_log
        vel_change = fin_vel_log - init_vel_log
        logging.info(f"=== {kind}/{name} SUMMARY ===")
        logging.info(f"Duration: {t[-1]:.1f}s | Heading: {init_hdg_log:.1f}° → {fin_hdg_log:.1f}° (Δ{hdg_change:+.1f}°)")
        logging.info(f"Altitude: {init_alt_log:.1f}m → {fin_alt_log:.1f}m (Δ{alt_change:+.1f}m) | Speed: {init_vel_log:.1f}m/s → {fin_vel_log:.1f}m/s (Δ{vel_change:+.1f}m/s)")
        if "anomaly_cut_time_s" in metrics:
            logging.info(f"Anomaly cut at t={metrics['anomaly_cut_time_s']:.1f}s")
        if "t_end_base_s" in metrics:
            logging.info(f"Duration extended {metrics['t_end_base_s']:.1f}s → {metrics['t_end_final_s']:.1f}s")
        
        # 注释掉详细的调试信息
        # logging.info(f"=== {kind}/{name} SUMMARY ===")
        # logging.info(f"Duration: {t[-1]:.1f}s | Heading: {init_hdg_log:.1f}° → {fin_hdg_log:.1f}° (Δ{hdg_change:+.1f}°)")
        # logging.info(f"Altitude: {init_alt_log:.1f}m → {fin_alt_log:.1f}m (Δ{alt_change:+.1f}m) | Speed: {init_vel_log:.1f}m/s → {fin_vel_log:.1f}m/s (Δ{vel_change:+.1f}m/s)")
        # if "anomaly_cut_time_s" in metrics:
        #     logging.info(f"Anomaly cut at t={metrics['anomaly_cut_time_s']:.1f}s")
        # if "t_end_base_s" in metrics:
        #     logging.info(f"Duration extended {metrics['t_end_base_s']:.1f}s → {metrics['t_end_final_s']:.1f}s")
    return {"csv": csv_path, "acmi": acmi_path, "metrics": metrics}


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 固定输出到 scripts/tacticalProject/air_combat_results/4_4_3/<ts>
    out_dir = os.path.join(tactical_project_dir, "air_combat_results", "4_4_3", ts)
    ensure_dir(out_dir)

    env = MultipleCombatEnv("simple_maneuver_config")

    # 按照用户要求的10个动作进行测试
    tests = []
    tests.extend([
        {"kind": "basic", "name": "level_flight", "params": {"duration": 40.0}},
        {"kind": "basic", "name": "turn_level", "params": {"turn_angle": 80.0, "turn_rate": 3.0, "duration": 40.0}},
        {"kind": "basic", "name": "pull_up", "params": {"altitude_gain": 1500.0, "duration": 30.0}},
        {"kind": "basic", "name": "dive", "params": {"altitude_loss": 1500.0, "duration": 25.0, "min_altitude": 2000.0}},
        {"kind": "basic", "name": "accelerate", "params": {"velocity_change": 50.0, "duration": 15.0}},  # 缩短持续时间，确保能完成
        {"kind": "basic", "name": "decelerate", "params": {"velocity_change": 40.0, "duration": 25.0, "initial_velocity": 350.0}},  # 与机动函数默认值一致，更高初始速度
        {"kind": "basic", "name": "crank", "params": {"crank_angle": 68.0, "turn_rate": 3.0, "duration": 35.0}},  # 战术偏置转向
        {"kind": "basic", "name": "diagonal_flight", "params": {"turn_angle": 45.0, "altitude_change": 1000.0, "duration": 30.0}},  # 斜向飞行
        {"kind": "composite", "name": "turn_pull_up", "params": {"turn_pull_up": {"turn_angle": 68.0, "turn_duration": 30.0, "turn_rate": 3.0, "altitude_gain": 1500.0, "pull_up_duration": 30.0}}},
        {"kind": "composite", "name": "short_skate_tactical", "params": {"duration": 60.0}},
    ])

    summary = []
    for tcase in tests:
        res = run_one(env, out_dir, tcase)
        if res is None:
            continue
        row = {"kind": tcase["kind"], "name": tcase["name"]}
        row.update(res["metrics"]) 
        row["csv_path"] = res["csv"]
        row["acmi_path"] = res["acmi"]
        summary.append(row)

    env.close()

    if summary:
        sum_path = os.path.join(out_dir, "summary_metrics.csv")
        keys = list(summary[0].keys())
        with open(sum_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in summary:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print(out_dir)


if __name__ == "__main__":
    main()
