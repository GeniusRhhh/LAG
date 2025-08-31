#!/usr/bin/env python3
"""
并排射击战术仿真运行脚本
基于拖曳射击项目的成功架构，实现并排射击战术的完整仿真
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)  # 添加当前目录到路径

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.tasks.multiplecombat_task import HierarchicalMultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
import torch
from enum import Enum


class TacticalPhase(Enum):
    """并排射击战术阶段"""
    NLT_MELD = "NLT_MELD"    # 90-81km
    MELD_MTR = "MELD_MTR"    # 81-45km
    MTR_TR = "MTR_TR"        # 45-41km
    TR_DOR = "TR_DOR"        # 41-19.6km
    DOR_DR = "DOR_DR"        # 19.6-14.5km


def setup_logging(output_dir: str) -> str:
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"side_by_side_shooting_simulation_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("🚀 并排射击战术仿真开始")
    logging.info(f"📝 日志文件: {log_file}")
    return log_file


def print_simulation_header():
    """打印仿真标题"""
    print("\n" + "="*80)
    print("🎯 并排射击战术仿真系统")
    print("="*80)
    print("战术特点:")
    print("  • 编队保持平行航向接敌，水平间距1-3海里")
    print("  • 长机和僚机在TR-DOR阶段末期同时发射导弹")
    print("  • DOR-DR阶段执行short_skate返航，长机左转，僚机右转")
    print("  • 火力密度高，适合正面交会场景")
    print("="*80)


def print_phase_info():
    """打印战术阶段信息"""
    print("\n📋 战术阶段说明:")
    print("  NLT-MELD (90-81km): 长机僚机保持编队间距，平稳飞行")
    print("  MELD-MTR (81-45km): 继续保持编队间距，平稳飞行")
    print("  MTR-TR   (45-41km): 继续保持编队间距，平稳飞行")
    print("  TR-DOR   (41-19.6km): 保持编队间距，阶段末期同时发射导弹")
    print("  DOR-DR   (19.6-14.5km): 执行short_skate返航机动")
    print()


def calculate_distance(agent1, agent2):
    """计算两个智能体之间的距离"""
    pos1 = agent1.get_position()
    pos2 = agent2.get_position()
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)


def get_current_phase(env):
    """获取当前战术阶段"""
    # 计算双方最近距离
    min_distance = float('inf')
    for friendly_id in ["A0100", "A0200"]:
        if friendly_id not in env.agents or not env.agents[friendly_id].is_alive:
            continue
        for enemy_id in ["B0100", "B0200"]:
            if enemy_id not in env.agents or not env.agents[enemy_id].is_alive:
                continue
            distance = calculate_distance(env.agents[friendly_id], env.agents[enemy_id])
            min_distance = min(min_distance, distance)

    if min_distance == float('inf'):
        return TacticalPhase.NLT_MELD

    # 根据距离确定阶段
    if min_distance > 81000:
        return TacticalPhase.NLT_MELD
    elif min_distance > 45000:
        return TacticalPhase.MELD_MTR
    elif min_distance > 41000:
        return TacticalPhase.MTR_TR
    elif min_distance > 19600:
        return TacticalPhase.TR_DOR
    else:
        return TacticalPhase.DOR_DR


def print_status(env, step, current_time):
    """打印仿真状态"""
    if step % 25 != 0:  # 每5秒打印一次
        return
        
    current_phase = get_current_phase(env)
    
    print(f"\n⏰ 时间: {current_time:6.1f}s | 步数: {step:4d} | 阶段: {current_phase.value}")
    
    # 打印飞机状态
    for agent_id in ["A0100", "A0200", "B0100", "B0200"]:
        if agent_id in env.agents and env.agents[agent_id].is_alive:
            agent = env.agents[agent_id]
            pos = agent.get_position()
            heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
            altitude = agent.get_property_value(c.position_h_sl_m)
            velocity = agent.get_property_value(c.velocities_u_mps)
            
            role = "长机" if agent_id.endswith("100") else "僚机"
            side = "己方" if agent_id.startswith("A") else "敌方"
            
            print(f"✈️  {side}{role}({agent_id}): "
                  f"位置({pos[0]/1000:6.1f}, {pos[1]/1000:6.1f}, {pos[2]/1000:6.1f})km, "
                  f"航向{heading:6.1f}°, 高度{altitude/1000:5.1f}km, "
                  f"速度{velocity:3.0f}m/s, 导弹{agent.num_missiles}枚")
    
    # 打印编队间距（己方）
    if ("A0100" in env.agents and env.agents["A0100"].is_alive and
        "A0200" in env.agents and env.agents["A0200"].is_alive):
        distance = calculate_distance(env.agents["A0100"], env.agents["A0200"])
        distance_nm = distance / 1852.0  # 转换为海里
        print(f"📐 己方编队间距: {distance_nm:.1f}海里 ({distance/1000:.1f}km)")
    
    # 打印双方距离
    min_distance = float('inf')
    for friendly_id in ["A0100", "A0200"]:
        if friendly_id not in env.agents or not env.agents[friendly_id].is_alive:
            continue
        for enemy_id in ["B0100", "B0200"]:
            if enemy_id not in env.agents or not env.agents[enemy_id].is_alive:
                continue
            distance = calculate_distance(env.agents[friendly_id], env.agents[enemy_id])
            min_distance = min(min_distance, distance)
    
    if min_distance != float('inf'):
        print(f"🎯 双方最近距离: {min_distance/1000:.1f}km")


def check_simulation_end(env):
    """检查仿真是否应该结束"""
    # 统计存活飞机
    friendly_alive = sum(1 for agent_id in ["A0100", "A0200"] 
                        if agent_id in env.agents and env.agents[agent_id].is_alive)
    enemy_alive = sum(1 for agent_id in ["B0100", "B0200"] 
                     if agent_id in env.agents and env.agents[agent_id].is_alive)
    
    if friendly_alive == 0:
        print("\n💥 己方全部被击落，仿真结束")
        return True
    elif enemy_alive == 0:
        print("\n🎉 敌方全部被击落，仿真结束")
        return True
    
    return False


def record_simulation_data(env, current_time, trajectory_data, radar_data, missile_data, data_recorder):
    """记录仿真数据到CSV格式 - 使用统一数据记录器"""
    # 记录当前数据长度，用于确定新增数据
    prev_traj_len = len(data_recorder.trajectory_data)
    prev_radar_len = len(data_recorder.radar_data)
    prev_missile_len = len(data_recorder.missile_data)

    # 使用统一数据记录器记录数据
    data_recorder.record_all_data(env, current_time)

    # 将新增数据添加到本地列表
    trajectory_data.extend(data_recorder.trajectory_data[prev_traj_len:])
    radar_data.extend(data_recorder.radar_data[prev_radar_len:])
    missile_data.extend(data_recorder.missile_data[prev_missile_len:])


def save_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data, simulation_log=None):
    """保存CSV数据文件 - 使用统一数据记录器"""
    from unified_data_recorder import UnifiedDataRecorder

    # 创建统一数据记录器
    recorder = UnifiedDataRecorder("side_by_side_shooting")

    # 设置数据
    recorder.trajectory_data = trajectory_data
    recorder.radar_data = radar_data
    recorder.missile_data = missile_data

    # 使用统一格式保存文件 - 传入仿真日志
    saved_files = recorder.save_csv_files(output_dir, timestamp, simulation_log)

    return saved_files


def run_side_by_side_shooting_simulation():
    """运行并排射击战术仿真"""
    try:
        # 设置输出目录
        output_dir = os.path.join(os.path.dirname(__file__), "side_by_side_shooting_results")
        log_file = setup_logging(output_dir)

        # 打印标题信息
        print_simulation_header()
        print_phase_info()

        print(f"输出目录: {output_dir}")
        print(f"日志文件: {log_file}")
        print()

        # 配置文件路径
        config_name = "side_by_side_shooting_tactical"
        
        print("初始化仿真环境...")
        from side_by_side_shooting_tactical_task import SideBySideShootingTacticalTask
        
        # 使用配置文件创建环境
        env = MultipleCombatEnv(config_name)
        
        # 强制设置1500步（300秒）
        env.max_steps = 1500
        
        # 替换任务为并排射击任务
        env.task = SideBySideShootingTacticalTask(env.config)
        
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
        
        # 开始仿真循环
        print("🚀 开始并排射击战术仿真...")
        start_time = time.time()

        # 准备ACMI文件路径和数据记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        acmi_filepath = os.path.join(output_dir, f"side_by_side_shooting_{timestamp}.acmi")

        # 初始化CSV数据记录
        trajectory_data = []
        radar_data = []
        missile_data = []

        # 创建持久的数据记录器实例
        from unified_data_recorder import UnifiedDataRecorder
        data_recorder = UnifiedDataRecorder("side_by_side_shooting")

        for step in range(env.max_steps):
            current_time = step * env.time_interval
            
            # 打印状态
            print_status(env, step, current_time)
            
            # 检查仿真结束条件
            if check_simulation_end(env):
                break
            
            # 执行一步仿真
            try:
                # 生成动作 - 调用任务的normalize_action方法
                actions = []
                for agent_id in ["A0100", "A0200", "B0100", "B0200"]:
                    if agent_id in env.agents and env.agents[agent_id].is_alive:
                        # 调用任务的normalize_action方法生成4维底层控制动作
                        high_level_action = [7, 8, 3]  # 占位符高层动作
                        low_level_action = env.task.normalize_action(env, agent_id, high_level_action)
                        actions.append(low_level_action)
                    else:
                        # 死亡智能体的默认动作
                        actions.append([0.0, 0.0, 0.0, 0.8])

                # 转换为正确的格式：[n_rollout_threads, n_agents, action_dim]
                actions = np.array(actions).reshape(1, 4, 4)  # 1个线程，4个智能体，4维动作

                # 执行环境步骤 - 环境返回5个值
                step_result = env.step(actions)
                if len(step_result) == 5:
                    obs, share_obs, rewards, dones, infos = step_result
                else:
                    obs, rewards, dones, infos = step_result

                # 每步都渲染ACMI - 这是关键！
                try:
                    env.render(mode="txt", filepath=acmi_filepath)
                except Exception as e:
                    logging.warning(f"Failed to render step {step}: {e}")

                # 记录数据 - 使用持久的数据记录器
                record_simulation_data(env, current_time, trajectory_data, radar_data, missile_data, data_recorder)
                
                # 检查是否有飞机被击落
                if isinstance(dones, dict):
                    for agent_id, done in dones.items():
                        if done and agent_id in env.agents:
                            if not env.agents[agent_id].is_alive:
                                role = "长机" if agent_id.endswith("100") else "僚机"
                                side = "己方" if agent_id.startswith("A") else "敌方"
                                print(f"💥 {side}{role}({agent_id}) 被击落！")
                else:
                    # dones是数组格式，检查所有智能体
                    for i, agent_id in enumerate(["A0100", "A0200", "B0100", "B0200"]):
                        if i < len(dones) and dones[i] and agent_id in env.agents:
                            if not env.agents[agent_id].is_alive:
                                role = "长机" if agent_id.endswith("100") else "僚机"
                                side = "己方" if agent_id.startswith("A") else "敌方"
                                print(f"💥 {side}{role}({agent_id}) 被击落！")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                logging.error(f"❌ 仿真步骤 {step} 执行错误: {e}")
                logging.error(f"❌ 详细错误信息:\n{error_details}")
                print(f"❌ 仿真步骤 {step} 执行错误: {e}")
                print(f"❌ 详细错误信息:\n{error_details}")
                break
        
        # 仿真结束
        end_time = time.time()
        simulation_time = end_time - start_time
        
        print(f"\n🏁 并排射击战术仿真完成")
        print(f"⏱️  仿真用时: {simulation_time:.2f}秒")
        print(f"📊 总步数: {step + 1}")
        print(f"🕐 仿真时间: {(step + 1) * env.time_interval:.1f}秒")
        
        # 最终统计
        friendly_alive = sum(1 for agent_id in ["A0100", "A0200"] 
                           if agent_id in env.agents and env.agents[agent_id].is_alive)
        enemy_alive = sum(1 for agent_id in ["B0100", "B0200"] 
                         if agent_id in env.agents and env.agents[agent_id].is_alive)
        
        print(f"📈 最终结果: 己方存活{friendly_alive}架，敌方存活{enemy_alive}架")

        if friendly_alive > enemy_alive:
            print("🎉 己方获胜！")
        elif enemy_alive > friendly_alive:
            print("💔 敌方获胜！")
        else:
            print("🤝 平局！")

        # 保存CSV数据
        print("\n💾 保存仿真数据...")
        save_csv_data(output_dir, timestamp, trajectory_data, radar_data, missile_data)

        # ACMI文件已在每步生成
        print(f"📊 ACMI文件已生成: {acmi_filepath}")
        print("📈 CSV数据文件已保存")

        print("=" * 80)
        print("🎉 并排射击战术仿真成功完成!")
        print(f"📁 数据文件保存在: {output_dir}")
        print(f"🎬 ACMI文件: {acmi_filepath}")
        print("📊 可以运行数据分析脚本查看结果")
        print("=" * 80)

        return True
        
    except Exception as e:
        logging.error(f"❌ 仿真运行错误: {e}")
        print(f"❌ 仿真运行错误: {e}")
        return False


if __name__ == "__main__":
    print("🎯 并排射击战术仿真系统")
    print("基于拖曳射击项目架构，实现编队平行接敌、同时发射的战术")
    print()
    
    success = run_side_by_side_shooting_simulation()
    
    if success:
        print("\n✅ 仿真成功完成")
    else:
        print("\n❌ 仿真执行失败")
    
    input("\n按回车键退出...")
