#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2v2拖曳射击战术仿真主运行脚本
基于现有的MultipleCombatEnv架构
"""

import os
import sys
import logging
import time
import numpy as np
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from envs.JSBSim.core.simulatior import MissileSimulator

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.tasks.multiplecombat_task import HierarchicalMultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
import torch
from enum import Enum


class TacticalPhase(Enum):
    """拖曳射击战术阶段"""
    NLT_MELD = "NLT_MELD"    # 90-81km
    MELD_MTR = "MELD_MTR"    # 81-45km
    MTR_TR = "MTR_TR"        # 45-41km
    TR_DOR = "TR_DOR"        # 41-19.6km
    DOR_DR = "DOR_DR"        # 19.6-14.5km


# 全局变量
current_phase = TacticalPhase.NLT_MELD
missile_launched = {"A0100": False, "A0200": False, "B0100": False, "B0200": False}
tactical_distances = {
    'NLT_MELD_min': 81000,   # 81km
    'MELD_MTR_min': 45000,   # 45km
    'MTR_TR_min': 41000,     # 41km
    'TR_DOR_min': 19600,     # 19.6km
    'DOR_DR_min': 14500,     # 14.5km
}

# 加载baseline模型
baseline_policy = None
rnn_states = {}

def load_baseline_model():
    """加载baseline模型 - 参考pure_maneuver_task实现"""
    global baseline_policy
    try:
        baseline_policy = BaselineActor()
        model_path = get_root_dir() + '/model/baseline_model.pt'
        baseline_policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        baseline_policy.eval()

        # 初始化指令数组 - 参考pure_maneuver_task
        global norm_delta_altitude, norm_delta_heading, norm_delta_velocity
        norm_delta_altitude = np.array([-1000, -500, -200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200, 500, 1000]) / 1000.0
        norm_delta_heading = np.array([-180, -90, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 90, 180]) / 180.0
        norm_delta_velocity = np.array([-50, -20, -10, 0, 10, 20, 50]) / 50.0

        logging.info("Successfully loaded baseline model")
        return True
    except Exception as e:
        logging.error(f"Failed to load baseline model: {e}")
        return False

# 全局变量
norm_delta_altitude = None
norm_delta_heading = None
norm_delta_velocity = None


def execute_drag_shoot_tactics(env, current_time: float):
    """执行拖曳射击战术"""
    global current_phase, missile_launched

    # 更新战术阶段
    update_tactical_phase(env, current_time)

    # 执行战术动作
    for agent_id, aircraft in env._jsbsims.items():
        if not aircraft.is_alive:
            continue

        # 根据阶段和角色执行不同的战术行为
        if agent_id == "A0100":  # 己方长机
            execute_leader_tactics(aircraft, current_time)
        elif agent_id == "A0200":  # 己方僚机
            execute_wingman_tactics(aircraft, current_time)
        elif agent_id.startswith("B"):  # 敌方
            execute_enemy_tactics(aircraft, current_time)

    # 处理导弹发射
    handle_missile_launch(env, current_time)


def update_tactical_phase(env, current_time: float):
    """更新战术阶段"""
    global current_phase

    # 获取主要对抗双方
    leader_red = env._jsbsims.get("A0100")
    leader_blue = env._jsbsims.get("B0100")

    if not leader_red or not leader_blue or not leader_red.is_alive or not leader_blue.is_alive:
        return

    # 计算距离
    distance = calculate_distance(leader_red, leader_blue)

    # 确定当前阶段
    new_phase = get_phase_by_distance(distance)

    if new_phase != current_phase:
        logging.info(f"Phase transition: {current_phase.value} -> {new_phase.value} "
                    f"at t={current_time:.1f}s, distance={distance/1000:.1f}km")
        current_phase = new_phase


def get_phase_by_distance(distance: float) -> TacticalPhase:
    """根据距离确定战术阶段"""
    if distance > tactical_distances['NLT_MELD_min']:
        return TacticalPhase.NLT_MELD
    elif distance > tactical_distances['MELD_MTR_min']:
        return TacticalPhase.MELD_MTR
    elif distance > tactical_distances['MTR_TR_min']:
        return TacticalPhase.MTR_TR
    elif distance > tactical_distances['TR_DOR_min']:
        return TacticalPhase.TR_DOR
    else:
        return TacticalPhase.DOR_DR


def calculate_distance(aircraft1, aircraft2) -> float:
    """计算两架飞机之间的距离"""
    pos1 = aircraft1.get_position()
    pos2 = aircraft2.get_position()
    return np.linalg.norm(pos1 - pos2)


def execute_leader_tactics(aircraft, current_time: float):
    """执行长机战术 - 使用pure_maneuvers机动"""
    agent_id = aircraft.uid

    if current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
        # 平稳飞行 (航向180°)
        execute_maneuver_command(aircraft, "level_flight", target_heading=180.0)
    elif current_phase == TacticalPhase.TR_DOR:
        # 左侧short_skate (turn_angle=-45.0)
        execute_maneuver_command(aircraft, "turn_left", turn_angle=-45.0)
    elif current_phase == TacticalPhase.DOR_DR:
        # 返航 (航向0°)
        execute_maneuver_command(aircraft, "level_flight", target_heading=0.0)


def execute_wingman_tactics(aircraft, current_time: float):
    """执行僚机战术 - 使用pure_maneuvers机动"""
    agent_id = aircraft.uid

    if current_phase == TacticalPhase.NLT_MELD:
        # 右侧crank (turn_angle=30.0)
        execute_maneuver_command(aircraft, "turn_right", turn_angle=30.0)
    elif current_phase == TacticalPhase.MELD_MTR:
        # 左侧crank (turn_angle=-30.0, 调整至180°)
        execute_maneuver_command(aircraft, "turn_left", turn_angle=-30.0)
    elif current_phase in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR]:
        # 平稳飞行 (航向180°)
        execute_maneuver_command(aircraft, "level_flight", target_heading=180.0)
    elif current_phase == TacticalPhase.DOR_DR:
        # 左侧short_skate后返航
        execute_maneuver_command(aircraft, "turn_left", turn_angle=-45.0)


def execute_enemy_tactics(aircraft, current_time: float):
    """执行敌方战术 - 平稳飞行"""
    # 敌方始终平稳飞行 (航向0°)
    execute_maneuver_command(aircraft, "level_flight", target_heading=0.0)


def execute_maneuver_command(aircraft, maneuver_type: str, **params):
    """执行机动指令 - 参考pure_maneuvers"""
    try:
        # 获取当前状态
        current_heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
        current_altitude = aircraft.get_property_value(c.position_h_sl_ft) * 0.3048  # 转换为米
        current_velocity = aircraft.get_property_value(c.velocities_u_fps) * 0.3048  # 转换为m/s

        # 目标状态
        target_altitude = 6096.0  # 20000ft
        target_velocity = 243.84  # 800fps

        if maneuver_type == "level_flight":
            target_heading = params.get("target_heading", current_heading)
            # 平稳飞行 - 保持当前状态，只调整航向
            execute_high_level_command(aircraft, target_heading, target_altitude, target_velocity)

        elif maneuver_type == "turn_left":
            turn_angle = params.get("turn_angle", -30.0)
            target_heading = current_heading + turn_angle
            while target_heading < 0: target_heading += 360
            while target_heading >= 360: target_heading -= 360
            execute_high_level_command(aircraft, target_heading, target_altitude, target_velocity)

        elif maneuver_type == "turn_right":
            turn_angle = params.get("turn_angle", 30.0)
            target_heading = current_heading + turn_angle
            while target_heading < 0: target_heading += 360
            while target_heading >= 360: target_heading -= 360
            execute_high_level_command(aircraft, target_heading, target_altitude, target_velocity)

        else:
            # 默认平稳飞行
            execute_high_level_command(aircraft, current_heading, target_altitude, target_velocity)

    except Exception as e:
        logging.warning(f"Failed to execute maneuver {maneuver_type} for {aircraft.uid}: {e}")
        # 回退到基础控制模式
        set_simple_control(aircraft, current_heading, target_altitude, target_velocity)


def execute_high_level_command(aircraft, target_heading: float, target_altitude: float, target_velocity: float):
    """执行高层指令 - 完全参考pure_maneuver_task实现"""
    global baseline_policy, rnn_states, norm_delta_altitude, norm_delta_heading, norm_delta_velocity

    if baseline_policy is None or norm_delta_altitude is None:
        logging.warning("Baseline policy not loaded, using simple control")
        set_simple_control(aircraft, target_heading, target_altitude, target_velocity)
        return

    try:
        agent_id = aircraft.uid

        # 获取当前状态
        current_heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
        current_altitude = aircraft.get_property_value(c.position_h_sl_ft) * 0.3048  # 转换为米
        current_velocity = aircraft.get_property_value(c.velocities_u_fps) * 0.3048  # 转换为m/s

        # 计算高层指令差值
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        altitude_diff = target_altitude - current_altitude
        velocity_diff = target_velocity - current_velocity

        # 转换为指令索引 - 参考pure_maneuver_task
        altitude_cmd_id = convert_altitude_to_index(altitude_diff)
        heading_cmd_id = convert_heading_to_index(heading_diff)
        velocity_cmd_id = convert_velocity_to_index(velocity_diff)

        # 使用baseline模型 - 完全参考pure_maneuver_task
        action = use_lowlevel_policy(aircraft, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        # 应用控制
        aircraft.set_property_value(c.fcs_aileron_cmd_norm, action[0])
        aircraft.set_property_value(c.fcs_elevator_cmd_norm, action[1])
        aircraft.set_property_value(c.fcs_rudder_cmd_norm, action[2])
        aircraft.set_property_value(c.fcs_throttle_cmd_norm, action[3])

    except Exception as e:
        logging.warning(f"Failed to execute high level command for {aircraft.uid}: {e}")
        # 回退到基础控制模式
        set_simple_control(aircraft, target_heading, target_altitude, target_velocity)


def convert_altitude_to_index(altitude_diff: float) -> int:
    """转换高度差为索引 - 参考pure_maneuver_task"""
    global norm_delta_altitude
    altitude_values = norm_delta_altitude * 1000.0  # 反归一化
    distances = np.abs(altitude_values - altitude_diff)
    return np.argmin(distances)


def convert_heading_to_index(heading_diff: float) -> int:
    """转换航向差为索引 - 参考pure_maneuver_task"""
    global norm_delta_heading
    heading_values = norm_delta_heading * 180.0  # 反归一化
    distances = np.abs(heading_values - heading_diff)
    return np.argmin(distances)


def convert_velocity_to_index(velocity_diff: float) -> int:
    """转换速度差为索引 - 参考pure_maneuver_task"""
    global norm_delta_velocity
    velocity_values = norm_delta_velocity * 50.0  # 反归一化
    distances = np.abs(velocity_values - velocity_diff)
    return np.argmin(distances)


def use_lowlevel_policy(aircraft, altitude_cmd_id: int, heading_cmd_id: int, velocity_cmd_id: int):
    """使用低级策略网络 - 完全参考pure_maneuver_task"""
    global baseline_policy, rnn_states, norm_delta_altitude, norm_delta_heading, norm_delta_velocity

    agent_id = aircraft.uid

    try:
        # 获取原始观测 - 需要实现get_obs方法
        raw_obs = get_aircraft_obs(aircraft)
        input_obs = np.zeros(12)

        # 安全索引访问
        altitude_cmd_id = min(altitude_cmd_id, len(norm_delta_altitude) - 1)
        heading_cmd_id = min(heading_cmd_id, len(norm_delta_heading) - 1)
        velocity_cmd_id = min(velocity_cmd_id, len(norm_delta_velocity) - 1)

        # 构建输入观测 - 参考pure_maneuver_task
        input_obs[0] = norm_delta_altitude[altitude_cmd_id]
        input_obs[1] = norm_delta_heading[heading_cmd_id]
        input_obs[2] = norm_delta_velocity[velocity_cmd_id]
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.nan_to_num(input_obs, nan=0.0)
        input_obs = np.expand_dims(input_obs, axis=0)

        # 初始化RNN状态 - 正确的3D张量
        if agent_id not in rnn_states:
            rnn_states[agent_id] = np.zeros((1, 1, 128))

        # 调用baseline模型
        _action, _rnn_states = baseline_policy(
            torch.FloatTensor(input_obs),
            torch.FloatTensor(rnn_states[agent_id])
        )

        action_output = _action.detach().cpu().numpy().squeeze(0)
        rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

        # 动作归一化 - 参考pure_maneuver_task
        norm_act = np.zeros(4)
        norm_act[0] = action_output[0] / 20 - 1.  # aileron
        norm_act[1] = action_output[1] / 20 - 1.  # elevator
        norm_act[2] = action_output[2] / 20 - 1.  # rudder
        norm_act[3] = action_output[3] / 58 + 0.4  # throttle

        # 高度安全检查
        current_alt = aircraft.get_position()[2]
        if current_alt < 1000:
            norm_act[1] = max(norm_act[1], 0.0)  # 拉升
            norm_act[3] = max(norm_act[3], 0.8)  # 增加油门

        return norm_act

    except Exception as e:
        logging.error(f"低级策略错误: {e}")
        return np.array([0.0, 0.0, 0.0, 0.7])  # 默认动作


def get_aircraft_obs(aircraft):
    """获取飞机观测 - 简化版本"""
    try:
        # 基本状态
        position = aircraft.get_position()
        velocity = aircraft.get_velocity()

        # 姿态角
        roll = aircraft.get_property_value(c.attitude_phi_rad)
        pitch = aircraft.get_property_value(c.attitude_theta_rad)
        yaw = aircraft.get_property_value(c.attitude_psi_rad)

        # 构建9维观测向量
        obs = np.array([
            position[2] / 10000.0,  # 归一化高度
            np.linalg.norm(velocity) / 300.0,  # 归一化速度
            roll,  # 滚转角
            pitch,  # 俯仰角
            yaw,  # 偏航角
            velocity[0] / 300.0,  # x速度
            velocity[1] / 300.0,  # y速度
            velocity[2] / 300.0,  # z速度
            0.0  # 填充
        ])

        return obs

    except Exception as e:
        logging.warning(f"Failed to get aircraft obs: {e}")
        return np.zeros(9)


def set_simple_control(aircraft, target_heading: float, target_altitude: float, target_velocity: float):
    """基础控制逻辑 - 作为baseline模型的备用方案"""
    try:
        current_heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
        current_altitude = aircraft.get_property_value(c.position_h_sl_ft) * 0.3048
        current_velocity = aircraft.get_property_value(c.velocities_u_fps) * 0.3048

        # 航向控制
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        # 高度控制
        altitude_diff = target_altitude - current_altitude

        # 速度控制
        velocity_diff = target_velocity - current_velocity

        # 简化的PID控制
        aileron_cmd = np.clip(heading_diff / 90.0, -0.3, 0.3)
        rudder_cmd = np.clip(heading_diff / 180.0, -0.2, 0.2)
        elevator_cmd = np.clip(altitude_diff / 500.0, -0.3, 0.3)
        throttle_cmd = np.clip(0.6 + velocity_diff / 100.0, 0.3, 0.9)

        # 应用控制
        aircraft.set_property_value(c.fcs_aileron_cmd_norm, aileron_cmd)
        aircraft.set_property_value(c.fcs_elevator_cmd_norm, elevator_cmd)
        aircraft.set_property_value(c.fcs_rudder_cmd_norm, rudder_cmd)
        aircraft.set_property_value(c.fcs_throttle_cmd_norm, throttle_cmd)

    except Exception as e:
        logging.warning(f"Failed to set simple control: {e}")


def handle_missile_launch(env, current_time: float):
    """处理导弹发射"""
    global missile_launched

    for agent_id, aircraft in env._jsbsims.items():
        if not aircraft.is_alive or missile_launched[agent_id]:
            continue

        # 检查导弹数量
        if aircraft.num_missiles <= 0:
            continue

        # 找到目标
        target = find_target(env, agent_id)
        if not target:
            continue

        distance = calculate_distance(aircraft, target)

        # 根据需求确定发射条件
        should_launch = False

        if agent_id == "A0100":  # 己方长机45km发射
            should_launch = (current_phase == TacticalPhase.MTR_TR and
                           44000 <= distance <= 46000)
        elif agent_id == "A0200":  # 己方僚机41km发射
            should_launch = (current_phase == TacticalPhase.TR_DOR and
                           40000 <= distance <= 42000)
        elif agent_id == "B0100":  # 敌方长机45km发射
            should_launch = (current_phase == TacticalPhase.MTR_TR and
                           44000 <= distance <= 46000)
        elif agent_id == "B0200":  # 敌方僚机41km发射
            should_launch = (current_phase == TacticalPhase.TR_DOR and
                           40000 <= distance <= 42000)

        if should_launch:
            launch_missile(env, agent_id, target, current_time)


def find_target(env, agent_id: str):
    """找到目标"""
    for enemy_id, enemy in env._jsbsims.items():
        if is_enemy(agent_id, enemy_id) and enemy.is_alive:
            return enemy
    return None


def is_enemy(agent_id1: str, agent_id2: str) -> bool:
    """判断是否为敌方"""
    return (agent_id1.startswith('A') and agent_id2.startswith('B')) or \
           (agent_id1.startswith('B') and agent_id2.startswith('A'))


def launch_missile(env, agent_id: str, target, current_time: float):
    """发射导弹"""
    global missile_launched

    try:
        aircraft = env._jsbsims[agent_id]

        # 创建导弹
        missile_uid = f"M{agent_id}_{aircraft.num_missiles}"
        missile = MissileSimulator.create(
            parent=aircraft,
            target=target,
            uid=missile_uid
        )

        # 添加到环境
        env.add_temp_simulator(missile)

        # 初始化导弹记录系统
        if not hasattr(env, '_missile_records'):
            env._missile_records = {}
        
        # 记录导弹信息
        env._missile_records[missile_uid] = {
            'launcher': agent_id,
            'target': target.uid,
            'type': 'AIM-120C-7',
            'status': 'LAUNCHED',
            'launch_time': current_time,
            'launch_position': aircraft.get_position().copy(),
            'launch_velocity': aircraft.get_velocity().copy()
        }

        # 更新状态
        aircraft.num_missiles -= 1
        missile_launched[agent_id] = True

        logging.info(f"MISSILE LAUNCH: {agent_id} -> {target.uid} at t={current_time:.1f}s, "
                    f"distance={calculate_distance(aircraft, target)/1000:.1f}km")

    except Exception as e:
        logging.warning(f"Failed to launch missile from {agent_id}: {e}")


def check_termination(env) -> bool:
    """检查终止条件 - 只有双方全灭才终止"""
    # 检查是否有飞机存活
    alive_aircraft = [a for a in env._jsbsims.values() if a.is_alive]
    if len(alive_aircraft) == 0:
        logging.info("All aircraft destroyed - terminating")
        return True

    # 检查双方存活情况 - 只有当一方全灭时才终止
    red_alive = [aid for aid, a in env._jsbsims.items() if aid.startswith('A') and a.is_alive]
    blue_alive = [aid for aid, a in env._jsbsims.items() if aid.startswith('B') and a.is_alive]

    # 只有当一方全灭时才终止
    if len(red_alive) == 0:
        logging.info("Red team eliminated - terminating")
        return True
    elif len(blue_alive) == 0:
        logging.info("Blue team eliminated - terminating")
        return True

    # 继续仿真 - 不因为单架飞机被击落而终止
    return False

def record_simulation_data(env, current_time, trajectory_data, radar_data, missile_data):
    """记录仿真数据到CSV格式 - 使用统一数据记录器"""
    # 使用统一数据记录器
    from unified_data_recorder import UnifiedDataRecorder

    # 创建临时记录器实例
    temp_recorder = UnifiedDataRecorder("drag_shoot")

    # 记录所有数据
    temp_recorder.record_aircraft_trajectory(env, current_time)
    temp_recorder.record_radar_data(env, current_time)
    temp_recorder.record_missile_data(env, current_time)

    # 将数据添加到现有列表中
    trajectory_data.extend(temp_recorder.trajectory_data)
    radar_data.extend(temp_recorder.radar_data)
    missile_data.extend(temp_recorder.missile_data)


def save_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data, simulation_log=None):
    """保存CSV数据文件 - 使用统一数据记录器"""
    from unified_data_recorder import UnifiedDataRecorder

    # 创建统一数据记录器
    recorder = UnifiedDataRecorder("drag_shoot")

    # 将数据添加到记录器
    recorder.trajectory_data = trajectory_data
    recorder.radar_data = radar_data
    recorder.missile_data = missile_data

    # 使用统一格式保存文件 - 传入仿真日志
    saved_files = recorder.save_csv_files(output_dir, timestamp, simulation_log)

    return saved_files


def get_missile_final_status_and_time(missile_traj):
    """获取导弹的真正最终状态和结束时间"""
    # 按时间排序确保顺序正确
    missile_traj = missile_traj.sort_values('Time_s')

    # 查找HIT状态
    hit_records = missile_traj[missile_traj['Status'] == 'HIT']
    if not hit_records.empty:
        # 如果有HIT记录，使用第一个HIT记录的时间
        hit_time = hit_records['Time_s'].iloc[0]
        return 'HIT', hit_time

    # 查找MISS状态
    miss_records = missile_traj[missile_traj['Status'] == 'MISS']
    if not miss_records.empty:
        # 如果有MISS记录，使用第一个MISS记录的时间
        miss_time = miss_records['Time_s'].iloc[0]
        return 'MISS', miss_time

    # 查找其他终止状态
    terminal_states = ['TIMEOUT', 'DESTROYED', 'LOST']
    for state in terminal_states:
        state_records = missile_traj[missile_traj['Status'] == state]
        if not state_records.empty:
            state_time = state_records['Time_s'].iloc[0]
            return state, state_time

    # 如果没有找到明确的终止状态，检查是否有状态变化
    # 从活跃状态（BOOST, MIDCOURSE, TERMINAL, ACTIVE）变为非活跃状态
    active_states = ['BOOST', 'MIDCOURSE', 'TERMINAL', 'ACTIVE']

    # 找到最后一个活跃状态的记录
    active_records = missile_traj[missile_traj['Status'].isin(active_states)]
    if not active_records.empty:
        last_active_time = active_records['Time_s'].iloc[-1]
        last_active_status = active_records['Status'].iloc[-1]

        # 检查在最后活跃时间之后是否有其他记录
        after_active = missile_traj[missile_traj['Time_s'] > last_active_time]
        if not after_active.empty:
            # 有后续记录，使用后续记录的第一个状态和时间
            next_status = after_active['Status'].iloc[0]
            next_time = after_active['Time_s'].iloc[0]
            return next_status, next_time
        else:
            # 没有后续记录，导弹可能仍在飞行，使用最后记录
            return last_active_status, last_active_time

    # 如果都没有，使用最后一条记录
    final_status = missile_traj['Status'].iloc[-1]
    final_time = missile_traj['Time_s'].iloc[-1]
    return final_status, final_time


def generate_missile_trajectory_analysis(missile_df, output_dir, timestamp):
    """生成导弹轨迹分析数据"""
    if missile_df.empty:
        print("没有导弹数据，跳过轨迹分析")
        return

    # 只处理轨迹数据，过滤掉状态数据
    trajectory_df = missile_df[missile_df.get('Data_Type', '') == 'Trajectory']
    if trajectory_df.empty:
        print("没有导弹轨迹数据，跳过轨迹分析")
        return

    # 检查是否有必要的列
    required_columns = ['Missile_ID', 'Launcher_ID', 'Target_ID', 'Time_s', 'Velocity_m_s', 'X_m', 'Y_m', 'Z_m', 'Status']
    missing_columns = [col for col in required_columns if col not in trajectory_df.columns]
    if missing_columns:
        print(f"导弹轨迹数据缺少必要列: {missing_columns}，跳过轨迹分析")
        return

    missile_analysis = []

    # 按导弹ID分组分析
    for missile_id in trajectory_df['Missile_ID'].unique():
        missile_traj = trajectory_df[trajectory_df['Missile_ID'] == missile_id]
        
        # 检查导弹轨迹是否为空
        if missile_traj.empty:
            continue

        try:
            # 获取基本信息
            launcher_id = missile_traj['Launcher_ID'].iloc[0]
            target_id = missile_traj['Target_ID'].iloc[0]

            # 获取发射时间
            launch_time = missile_traj['Time_s'].iloc[0]

            # 获取真正的最终状态和结束时间
            final_status, final_time = get_missile_final_status_and_time(missile_traj)
            flight_duration = final_time - launch_time

            # 计算速度统计
            velocities = missile_traj['Velocity_m_s'].values
            max_velocity = velocities.max()
            avg_velocity = velocities.mean()

            # 计算总飞行距离
            positions = missile_traj[['X_m', 'Y_m', 'Z_m']].values
            total_distance = 0.0
            for i in range(1, len(positions)):
                total_distance += np.linalg.norm(positions[i] - positions[i - 1])

            # 计算最小距离到目标
            if 'Range_to_Target_km' in missile_traj.columns:
                min_distance_to_target = missile_traj['Range_to_Target_km'].min()
            else:
                min_distance_to_target = 0.0

            missile_analysis.append({
                'Missile_ID': missile_id,
                'Launcher_ID': launcher_id,
                'Target_ID': target_id,
                'Launch_Time_s': launch_time,
                'Final_Time_s': final_time,
                'Flight_Duration_s': flight_duration,
                'Max_Velocity_m_s': max_velocity,
                'Average_Velocity_m_s': avg_velocity,
                'Total_Distance_m': total_distance,
                'Min_Distance_to_Target_km': min_distance_to_target,
                'Final_Status': final_status
            })
        except Exception as e:
            print(f"处理导弹 {missile_id} 轨迹时出错: {e}")
            continue

    # 保存导弹轨迹分析
    if missile_analysis:
        analysis_df = pd.DataFrame(missile_analysis)
        analysis_file = os.path.join(output_dir, f"missile_trajectory_analysis_{timestamp}.csv")
        analysis_df.to_csv(analysis_file, index=False)
        print(f"导弹轨迹分析已保存: {analysis_file}")

        # 生成导弹轨迹摘要报告
        generate_missile_summary_report(analysis_df, output_dir, timestamp)
    else:
        print("没有有效的导弹轨迹数据可分析")

def generate_missile_summary_report(analysis_df, output_dir, timestamp):
    """生成导弹轨迹摘要报告"""
    if analysis_df.empty:
        return
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("导弹轨迹分析摘要报告")
    report_lines.append("=" * 60)
    report_lines.append(f"生成时间: {timestamp}")
    report_lines.append(f"总导弹数量: {len(analysis_df)}")
    report_lines.append("")
    
    # 统计信息
    hit_missiles = len(analysis_df[analysis_df['Final_Status'] == 'HIT'])
    miss_missiles = len(analysis_df[analysis_df['Final_Status'] == 'MISS'])
    active_missiles = len(analysis_df[analysis_df['Final_Status'] == 'ACTIVE'])
    other_missiles = len(analysis_df) - hit_missiles - miss_missiles - active_missiles

    report_lines.append(f"导弹击中目标: {hit_missiles}")
    report_lines.append(f"导弹未击中目标: {miss_missiles}")
    report_lines.append(f"仍在飞行导弹: {active_missiles}")
    if other_missiles > 0:
        report_lines.append(f"其他状态导弹: {other_missiles}")
    report_lines.append("")

    # 计算命中率
    if hit_missiles + miss_missiles > 0:
        hit_rate = hit_missiles / (hit_missiles + miss_missiles) * 100
        report_lines.append(f"导弹命中率: {hit_rate:.1f}% ({hit_missiles}/{hit_missiles + miss_missiles})")
        report_lines.append("")
    
    # 性能统计
    if len(analysis_df) > 0:
        avg_flight_duration = analysis_df['Flight_Duration_s'].mean()
        avg_max_velocity = analysis_df['Max_Velocity_m_s'].mean()
        avg_distance = analysis_df['Total_Distance_km'].mean()
        
        report_lines.append("性能统计:")
        report_lines.append(f"  平均飞行时间: {avg_flight_duration:.2f}秒")
        report_lines.append(f"  平均最大速度: {avg_max_velocity:.1f}m/s")
        report_lines.append(f"  平均飞行距离: {avg_distance:.2f}km")
        report_lines.append("")
    
    # 详细导弹信息
    report_lines.append("详细导弹信息:")
    report_lines.append("-" * 60)
    
    for _, missile in analysis_df.iterrows():
        report_lines.append(f"导弹 {missile['Missile_ID']}:")
        report_lines.append(f"  发射器: {missile['Launcher_ID']} -> 目标: {missile['Target_ID']}")
        report_lines.append(f"  飞行时间: {missile['Flight_Duration_s']:.2f}秒")
        report_lines.append(f"  最大速度: {missile['Max_Velocity_m_s']:.1f}m/s")
        report_lines.append(f"  飞行距离: {missile['Total_Distance_km']:.2f}km")
        report_lines.append(f"  最小距离到目标: {missile['Min_Distance_to_Target_km']:.2f}km")
        report_lines.append(f"  最终状态: {missile['Final_Status']}")
        report_lines.append("")
    
    # 保存报告
    report_file = os.path.join(output_dir, f"missile_summary_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"导弹摘要报告已保存: {report_file}")

def setup_logging(output_dir: str) -> str:
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"simulation_log_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    return log_file


def print_banner():
    """打印横幅"""
    print("=" * 80)
    print("2v2拖曳射击战术仿真系统")
    print("基于现有MultipleCombatEnv架构 - 正确实现版本")
    print("=" * 80)


def print_status(env, step_interval: int = 50):
    """打印仿真状态"""
    if env.current_step % step_interval == 0:
        # 计算距离
        leader_red = env._jsbsims.get("A0100")
        leader_blue = env._jsbsims.get("B0100")
        distance = 0.0
        if leader_red and leader_blue and leader_red.is_alive and leader_blue.is_alive:
            pos1 = leader_red.get_position()
            pos2 = leader_blue.get_position()
            distance = np.linalg.norm(pos1 - pos2) / 1000.0

        # 统计存活飞机和导弹
        alive_aircraft = sum(1 for a in env._jsbsims.values() if a.is_alive)
        active_missiles = len([m for m in env._tempsims.values() if m.is_alive])

        current_time = env.current_step * env.time_interval
        phase_name = env.task.current_phase.value if hasattr(env.task, 'current_phase') else "UNKNOWN"
        print(f"t={current_time:6.1f}s | step={env.current_step:4d} | "
              f"phase={phase_name:8s} | distance={distance:5.1f}km | "
              f"aircraft={alive_aircraft} | missiles={active_missiles}")


def run_simulation():
    """运行仿真"""
    import io
    import sys

    # 捕获仿真输出
    captured_output = io.StringIO()
    original_stdout = sys.stdout

    try:
        # 配置文件名称 - 不需要路径和后缀
        config_name = "drag_shoot_tactical"

        # 检查配置文件是否存在 - 使用本地configs目录
        config_file_path = os.path.join(current_dir, 'configs', f'{config_name}.yaml')
        if not os.path.exists(config_file_path):
            print(f"❌ 配置文件不存在: {config_file_path}")
            return False

        # 设置输出目录
        output_dir = os.path.join(os.path.dirname(__file__), "air_combat_results")
        log_file = setup_logging(output_dir)

        print_banner()
        print(f"配置文件: {config_file_path}")
        print(f"输出目录: {output_dir}")
        print(f"日志文件: {log_file}")
        print()
        # 加载baseline模型
        print("加载baseline模型...")
        if not load_baseline_model():
            print("⚠️ Baseline模型加载失败，使用基础控制备用方案")
        else:
            print("✅ Baseline模型加载成功")

        # 创建仿真环境 - 使用MultipleCombatEnv + 拖曳射击任务
        print("初始化仿真环境...")
        from drag_shoot_tactical_task import DragShootTacticalTask

        # 使用现有的配置文件
        env = MultipleCombatEnv(config_name)

        # 强制设置1500步（300秒）
        env.max_steps = 1500

        # 替换任务为拖曳射击任务
        env.task = DragShootTacticalTask(env.config)
        
        # 重置环境
        obs = env.reset()
        
        print("仿真环境初始化完成")
        print(f"飞机数量: {len(env.agents)}")
        print(f"时间步长: {env.time_interval}秒")
        print(f"最大步数: {env.max_steps}")
        print()
        
        # 打印初始状态
        print("初始飞机状态:")
        for agent_id, aircraft in env._jsbsims.items():
            pos = aircraft.get_position()
            print(f"  {agent_id}: 位置({pos[0]/1000:6.1f}, {pos[1]/1000:6.1f}, {pos[2]/1000:6.1f})km, "
                  f"导弹{aircraft.num_missiles}")
        print()
        
        # 开始仿真
        print("开始仿真...")
        print("时间(s) | 步数 | 战术阶段 | 距离(km) | 飞机 | 导弹")
        print("-" * 60)
        
        start_time = time.time()

        # 准备ACMI文件路径和数据记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        acmi_filepath = os.path.join(output_dir, f"air_combat_2v2_{timestamp}.acmi")

        # 初始化CSV数据记录
        trajectory_data = []
        radar_data = []
        missile_data = []

        # 仿真循环
        step_count = 0
        while step_count < env.max_steps:
            current_time = step_count * env.time_interval

            # 创建虚拟动作 - 因为任务类会在normalize_action中处理战术逻辑
            num_agents = len(env._jsbsims)
            actions = np.zeros((1, num_agents, 4))  # 虚拟动作，不会被使用

            # 使用环境的step方法 - 任务类会在normalize_action中处理战术
            obs, share_obs, rewards, dones, info = env.step(actions)

            step_count += 1

            # 每步都渲染ACMI - 这是关键！
            try:
                env.render(mode="txt", filepath=acmi_filepath)
            except Exception as e:
                logging.warning(f"Failed to render step {step_count}: {e}")

            # 记录数据
            record_simulation_data(env, current_time, trajectory_data, radar_data, missile_data)

            # 打印状态
            print_status(env)
            
            # 检查终止条件 - 只使用我们自定义的终止逻辑，忽略环境的dones
            if check_termination(env):
                print(f"\n仿真终止于步数 {step_count}")
                break
                
            # 检查飞机存活状态 - 减少冗余输出
            alive_count = sum(1 for aircraft in env.agents.values() if aircraft.is_alive)
            if alive_count < len(env.agents) and step_count % 50 == 0:  # 每50步输出一次
                print(f"\n飞机被击落，存活数量: {alive_count}")
        
        end_time = time.time()
        simulation_duration = end_time - start_time
        
        print("-" * 60)
        print("仿真完成!")
        print()
        
        # 计算最终状态
        current_time = env.current_step * env.time_interval
        leader_red = env._jsbsims.get("A0100")
        leader_blue = env._jsbsims.get("B0100")
        final_distance = 0.0
        if leader_red and leader_blue and leader_red.is_alive and leader_blue.is_alive:
            pos1 = leader_red.get_position()
            pos2 = leader_blue.get_position()
            final_distance = np.linalg.norm(pos1 - pos2) / 1000.0

        alive_aircraft = sum(1 for a in env._jsbsims.values() if a.is_alive)
        active_missiles = len(getattr(env, '_missile_records', {}))
        print("最终状态:")
        print(f"  仿真时间: {current_time:.1f}秒")
        print(f"  总步数: {env.current_step}")
        print(f"  最终阶段: {env.task.current_phase.value}")
        print(f"  最终距离: {final_distance:.1f}km")
        print(f"  存活飞机: {alive_aircraft}")
        print(f"  活跃导弹: {active_missiles}")
        print(f"  计算耗时: {simulation_duration:.2f}秒")
        print()
        
        # 打印最终飞机状态
        print("最终飞机状态:")
        for agent_id, aircraft in env._jsbsims.items():
            pos = aircraft.get_position()
            status = "存活" if aircraft.is_alive else "被击落"
            print(f"  {agent_id}: {status}, "
                  f"位置({pos[0]/1000:6.1f}, {pos[1]/1000:6.1f}, {pos[2]/1000:6.1f})km, "
                  f"剩余导弹{aircraft.num_missiles}")
        print()
        
        # 恢复标准输出并获取捕获的内容
        sys.stdout = original_stdout
        simulation_log = captured_output.getvalue()

        # 保存CSV数据 - 传入仿真日志
        save_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data, simulation_log)

        # ACMI文件已在每步生成
        print(f"ACMI文件已生成: {acmi_filepath}")
        print("CSV数据文件已保存")

        print("=" * 80)
        print("仿真成功完成!")
        print(f"数据文件保存在: {output_dir}")
        print(f"ACMI文件: {acmi_filepath}")
        print("可以运行数据分析脚本查看结果")
        print("=" * 80)
        
        return True

    except KeyboardInterrupt:
        sys.stdout = original_stdout  # 恢复输出
        print("\n仿真被用户中断")
        return False

    except Exception as e:
        sys.stdout = original_stdout  # 恢复输出
        print(f"\n❌ 仿真运行失败: {e}")
        logging.error(f"Simulation failed: {e}", exc_info=True)
        return False

    finally:
        # 确保输出总是被恢复
        sys.stdout = original_stdout


def main():
    """主函数"""
    success = run_simulation()
    
    if success:
        print("\n🎉 仿真成功完成!")
        print("下一步可以:")
        print("1. 查看生成的CSV数据文件")
        print("2. 使用现有的TacView查看ACMI文件")
        print("3. 运行数据分析脚本生成图表")
    else:
        print("\n❌ 仿真失败，请检查配置和日志")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
