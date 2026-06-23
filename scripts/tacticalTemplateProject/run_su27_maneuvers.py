#!/usr/bin/env python3
"""
SU27 Baseline模型机动测试
使用Pure_maneuver_task测试SU27模型，生成ACMI文件在Tacview中查看

使用方法:
  python run_su27_maneuvers.py
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from envs.JSBSim.envs import MultipleCombatEnv
from envs.JSBSim.tasks.pure_maneuver_task import PureManeuverTask


def setup_logging():
    """配置日志"""
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'su27_maneuver_test_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file


def run_basic_maneuver(maneuver_name, params, max_steps=3000):
    """测试基础机动"""
    logging.info(f"\n{'='*80}")
    logging.info(f"测试机动: {maneuver_name}")
    logging.info(f"参数: {params}")
    logging.info(f"{'='*80}\n")
    
    # 创建环境 - 使用2v2/NoWeapon/Selfplay配置
    env = MultipleCombatEnv("2v2/NoWeapon/Selfplay")
    
    # 修改环境配置，使用pure_maneuver任务
    env.config.task = "pure_maneuver"
    
    # 重新加载任务
    env.load_task()
    
    # 设置基础机动
    env.task.set_basic_maneuver(maneuver_name, **params)
    
    # 重置环境
    obs = env.reset()
    
    # 启用ACMI记录
    acmi_dir = os.path.join(project_root, 'acmi_output')
    os.makedirs(acmi_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    acmi_file = os.path.join(acmi_dir, f'su27_{maneuver_name}_{timestamp}.acmi')
    
    # 初始化ACMI文件
    env.renderer.initialize_acmi(acmi_file)
    
    # 运行仿真
    for step in range(max_steps):
        # 构建动作：(n_rollout_threads, num_agents, action_dim)
        # Pure_maneuver_task会在normalize_action中自动处理，这里传入dummy动作
        num_agents = len(env.agents)
        action_dim = env.action_space.shape[0]
        actions = np.zeros((env.n_rollout_threads, num_agents, action_dim))
        
        # Step环境 - MultipleCombatEnv返回5个值
        obs, share_obs, rewards, dones, infos = env.step(actions)
        
        # 渲染到ACMI（filepath参数会被忽略，使用initialize_acmi设置的路径）
        env.renderer.render(env, mode='txt')
        
        # 检查是否所有飞机都终止
        # dones可能是字典或数组，需要处理
        if isinstance(dones, dict):
            all_done = all([dones[aid][0] if isinstance(dones[aid], list) else dones[aid] 
                           for aid in dones.keys()])
        else:
            # 如果是数组，检查所有元素
            all_done = np.all(dones)
        
        if all_done:
            logging.info(f"所有飞机终止，步数: {step}")
            break
        
        # 每100步打印一次状态
        if step % 100 == 0:
            if isinstance(infos, dict):
                for agent_id, info in infos.items():
                    if 'altitude' in info:
                        logging.info(f"步数 {step} - {agent_id}: "
                                   f"高度={info['altitude']:.1f}m, "
                                   f"航向={info['heading']:.1f}°, "
                                   f"速度={info['velocity']:.1f}m/s")
            else:
                logging.info(f"步数 {step} 完成")
    
    # 关闭ACMI文件
    env.renderer.close()
    logging.info(f"✅ ACMI文件已保存: {acmi_file}")
    
    env.close()
    return acmi_file


def main():
    """主函数"""
    log_file = setup_logging()
    logging.info(f"日志文件: {log_file}")
    
    # 定义要测试的机动
    test_cases = [
        # 基础机动
        ("level_flight", {"duration": 30.0}),
        ("turn", {"turn_angle": 45.0, "turn_rate": 3.0}),
        ("turn", {"turn_angle": -45.0, "turn_rate": 3.0}),
        ("pull_up", {"altitude_gain": 500.0, "duration": 15.0}),
        ("dive", {"altitude_loss": 200.0, "duration": 10.0, "min_altitude": 3000.0}),
        ("accelerate", {"velocity_increase": 50.0, "duration": 15.0}),
        ("decelerate", {"velocity_decrease": 50.0, "duration": 15.0}),
    ]
    
    acmi_files = []
    
    for maneuver_name, params in test_cases:
        try:
            acmi_file = run_basic_maneuver(maneuver_name, params)
            acmi_files.append(acmi_file)
        except Exception as e:
            logging.error(f"测试 {maneuver_name} 失败: {e}", exc_info=True)
    
    # 打印总结
    logging.info(f"\n{'='*80}")
    logging.info("测试完成！")
    logging.info(f"{'='*80}")
    logging.info(f"生成的ACMI文件:")
    for acmi_file in acmi_files:
        logging.info(f"  - {acmi_file}")
    logging.info(f"\n请在Tacview中打开这些文件查看机动效果")


if __name__ == "__main__":
    main()
