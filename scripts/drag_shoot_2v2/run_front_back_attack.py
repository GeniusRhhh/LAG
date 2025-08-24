#!/usr/bin/env python3
"""
前后攻击战术运行脚本
基于拖曳射击的成功架构，实现前后攻击战术
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

# 导入必要的模块
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.tasks.multiplecombat_task import HierarchicalMultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from front_back_attack_tactical_task import FrontBackAttackTacticalTask, DragShootTermination
import pandas as pd
import torch

def setup_logging():
    """设置日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"front_back_attack_results/front_back_attack_log_{timestamp}.log"
    
    # 创建结果目录
    os.makedirs("front_back_attack_results", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return timestamp

def run_front_back_attack_simulation():
    """运行前后攻击战术仿真 - 简化版本"""
    timestamp = setup_logging()
    logging.info("🎯 开始前后攻击战术仿真")

    try:
        # 创建环境配置 - 基于成功的拖曳射击配置
        config_path = '2v2/ShootMissile/HierarchySelfplay'

        # 创建前后攻击战术任务
        task = FrontBackAttackTacticalTask(config_path)

        # 创建环境并设置任务
        env = MultipleCombatEnv(config_path)
        env.task = task

        # 重置环境
        obs = env.reset()
        logging.info("✅ 环境初始化完成")

        # 仿真循环
        step_count = 0
        trajectory_data = []

        while True:
            step_count += 1
            current_time = step_count * env.time_interval

            # 使用任务的step方法
            obs, share_obs, rewards, dones, infos = task.step(env)

            # 记录轨迹数据
            if step_count % 5 == 0:  # 每1秒记录一次
                for agent_id in env.agents:
                    if env.agents[agent_id].is_alive:
                        agent = env.agents[agent_id]
                        trajectory_data.append({
                            'Time': current_time,
                            'Agent': agent_id,
                            'X': agent.get_property_value(c.position_long_gc_deg) * 111320,  # 转换为米
                            'Y': agent.get_property_value(c.position_lat_geod_deg) * 111320,  # 转换为米
                            'Altitude_m': agent.get_property_value(c.position_h_sl_m),
                            'Heading_deg': np.rad2deg(agent.get_property_value(c.attitude_psi_rad)),
                            'Speed_mps': agent.get_property_value(c.velocities_u_mps),
                            'Missiles': agent.num_missiles,
                            'Status': 'ALIVE'
                        })

            # 检查终止条件
            if isinstance(dones, dict):
                all_done = all(dones.values())
            else:
                all_done = all(dones)

            if all_done or step_count >= 2500:
                logging.info(f"🏁 仿真结束: 步数={step_count}, 时间={current_time:.1f}s")
                break

            # 定期打印状态
            if step_count % 50 == 0:  # 每10秒打印一次
                alive_count = sum(1 for agent in env.agents.values() if agent.is_alive)
                logging.info(f"⏱️  t={current_time:.1f}s, 步数={step_count}, 存活={alive_count}/4, "
                           f"阶段={task.current_phase.value}")

        # 保存结果
        save_results(trajectory_data, [], timestamp)

        # 打印最终统计
        print_final_statistics(env, task, current_time)

        logging.info("🎯 前后攻击战术仿真完成")

    except Exception as e:
        logging.error(f"❌ 仿真运行失败: {e}")
        import traceback
        traceback.print_exc()



def save_results(trajectory_data, missile_data, timestamp):
    """保存仿真结果"""
    try:
        # 保存轨迹数据
        if trajectory_data:
            df_trajectory = pd.DataFrame(trajectory_data)
            trajectory_file = f"front_back_attack_results/front_back_attack_trajectory_{timestamp}.csv"
            df_trajectory.to_csv(trajectory_file, index=False, encoding='utf-8')
            logging.info(f"✅ 轨迹数据已保存: {trajectory_file}")

        # 保存导弹数据
        if missile_data:
            df_missile = pd.DataFrame(missile_data)
            missile_file = f"front_back_attack_results/front_back_attack_missile_analysis_{timestamp}.csv"
            df_missile.to_csv(missile_file, index=False, encoding='utf-8')
            logging.info(f"✅ 导弹数据已保存: {missile_file}")

    except Exception as e:
        logging.error(f"❌ 保存结果失败: {e}")

def print_final_statistics(env, task, final_time):
    """打印最终统计信息"""
    try:
        logging.info("=" * 60)
        logging.info("🎯 前后攻击战术仿真结果统计")
        logging.info("=" * 60)
        
        # 存活统计
        alive_agents = [agent_id for agent_id, agent in env.agents.items() if agent.is_alive]
        logging.info(f"📊 最终存活: {len(alive_agents)}/4")
        for agent_id in alive_agents:
            logging.info(f"   ✅ {agent_id}: 存活")
        
        dead_agents = [agent_id for agent_id, agent in env.agents.items() if not agent.is_alive]
        for agent_id in dead_agents:
            logging.info(f"   ❌ {agent_id}: 被击落")
        
        # 导弹发射统计
        logging.info(f"🚀 导弹发射统计:")
        for agent_id, count in task.missile_launch_count.items():
            logging.info(f"   {agent_id}: {count}枚导弹")
        
        # 战术阶段统计
        logging.info(f"⏱️  仿真时长: {final_time:.1f}秒")
        logging.info(f"🎯 最终阶段: {task.current_phase.value}")
        
        # 前后攻击特有统计
        if task.formation_states["A0200"]["formation_established"]:
            logging.info("✅ 僚机后方队形建立成功")
        else:
            logging.info("❌ 僚机后方队形建立失败")
            
        if task.attack_timeline["leader_missile_launched"]:
            logging.info("✅ 长机先行攻击成功")
        else:
            logging.info("❌ 长机先行攻击失败")
            
        if task.attack_timeline["wingman_attack_ready"]:
            logging.info("✅ 僚机后方攻击就绪")
        else:
            logging.info("❌ 僚机后方攻击未就绪")
        
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"❌ 统计信息打印失败: {e}")

if __name__ == "__main__":
    run_front_back_attack_simulation()
