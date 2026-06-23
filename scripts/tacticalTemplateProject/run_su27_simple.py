#!/usr/bin/env python3
"""
SU27 Baseline模型简单测试
每个机动单独测试，生成独立的ACMI文件

使用方法:
  python run_su27_simple.py
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


def setup_logging():
    """配置日志"""
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'su27_test_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file


def run_single_maneuver(maneuver_name, params, max_steps=1500):
    """测试单个机动"""
    logging.info(f"\n{'='*80}")
    logging.info(f"开始测试: {maneuver_name}")
    logging.info(f"参数: {params}")
    logging.info(f"{'='*80}\n")
    
    # 创建环境
    env = MultipleCombatEnv("2v2/NoWeapon/Selfplay")
    env.config.task = "pure_maneuver"
    env.load_task()
    
    # 设置机动
    env.task.set_basic_maneuver(maneuver_name, **params)
    
    # 重置环境
    env.reset()
    
    # 记录初始状态
    initial_state = None
    final_state = None
    
    # 设置ACMI文件
    acmi_dir = os.path.join(project_root, 'acmi_output')
    os.makedirs(acmi_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    acmi_file = os.path.join(acmi_dir, f'{maneuver_name}_{timestamp}.acmi')
    
    # 初始化ACMI
    env.renderer.initialize_acmi(acmi_file)
    logging.info(f"ACMI文件: {acmi_file}")
    
    # 运行仿真
    completed = False
    total_steps = 0
    
    for step in range(max_steps):
        # 构建动作
        num_agents = len(env.agents)
        action_dim = env.action_space.shape[0]
        actions = np.zeros((env.n_rollout_threads, num_agents, action_dim))
        
        # Step
        obs, share_obs, rewards, dones, infos = env.step(actions)
        
        # 记录初始和最终状态
        if step == 0 and isinstance(infos, dict) and 'A0100' in infos:
            info = infos['A0100']
            if 'altitude' in info:
                initial_state = {
                    'altitude': info['altitude'],
                    'heading': info['heading'],
                    'velocity': info['velocity']
                }
        
        if isinstance(infos, dict) and 'A0100' in infos:
            info = infos['A0100']
            if 'altitude' in info:
                final_state = {
                    'altitude': info['altitude'],
                    'heading': info['heading'],
                    'velocity': info['velocity']
                }
        
        # 渲染
        env.renderer.render(env, mode='txt')
        
        # 检查终止
        if isinstance(dones, dict):
            all_done = all([dones[aid][0] if isinstance(dones[aid], list) else dones[aid] 
                           for aid in dones.keys()])
        else:
            all_done = np.all(dones)
        
        total_steps = step + 1
        
        if all_done:
            logging.info(f"✅ 机动完成，步数: {step}")
            completed = True
            break
        
        # 每200步打印状态
        if step % 200 == 0 and isinstance(infos, dict):
            for agent_id, info in infos.items():
                if agent_id == 'A0100' and 'altitude' in info:
                    logging.info(f"步数 {step}: "
                               f"高度={info['altitude']:.1f}m, "
                               f"航向={info['heading']:.1f}°, "
                               f"速度={info['velocity']:.1f}m/s")
    
    if not completed:
        logging.warning(f"⚠️  达到最大步数 {max_steps}，机动未完成")
    
    # 关闭
    env.renderer.close()
    env.close()
    
    logging.info(f"✅ ACMI已保存: {acmi_file}\n")
    
    # 返回结果
    return {
        'acmi_file': acmi_file,
        'completed': completed,
        'steps': total_steps,
        'initial': initial_state,
        'final': final_state,
        'params': params
    }


def main():
    """主函数"""
    log_file = setup_logging()
    logging.info(f"日志文件: {log_file}\n")
    
    # 定义测试用例 - 增大参数幅度
    test_cases = [
        ("level_flight", {"duration": 40.0}, "平稳飞行40秒"),
        ("turn_right_90", {"turn_angle": 90.0, "turn_rate": 3.0}, "右转90度"),
        ("turn_left_90", {"turn_angle": -90.0, "turn_rate": 3.0}, "左转90度"),
        ("pull_up_1500", {"altitude_gain": 1500.0, "duration": 20.0}, "爬升1500米"),
        ("dive_1000", {"altitude_loss": 1000.0, "duration": 15.0, "min_altitude": 2000.0}, "下降1000米"),
        ("accelerate_100", {"velocity_increase": 100.0, "duration": 20.0}, "加速100m/s"),
        ("decelerate_100", {"velocity_decrease": 100.0, "duration": 20.0}, "减速100m/s"),
    ]
    
    results = []
    success_count = 0
    
    for maneuver_name, params, description in test_cases:
        try:
            result = run_single_maneuver(maneuver_name, params)
            result['name'] = maneuver_name
            result['description'] = description
            results.append(result)
            if result['completed']:
                success_count += 1
        except Exception as e:
            logging.error(f"❌ 测试 {maneuver_name} 失败: {e}", exc_info=True)
            results.append({
                'name': maneuver_name,
                'description': description,
                'completed': False,
                'error': str(e)
            })
    
    # 详细总结报告
    logging.info(f"\n{'='*80}")
    logging.info(f"SU27 BASELINE模型机动测试报告")
    logging.info(f"{'='*80}")
    logging.info(f"测试完成: {success_count}/{len(test_cases)} 成功\n")
    
    for i, result in enumerate(results, 1):
        logging.info(f"{i}. {result['description']} ({result['name']})")
        
        if 'error' in result:
            logging.info(f"   ❌ 失败: {result['error']}\n")
            continue
        
        if not result['completed']:
            logging.info(f"   ⚠️  未完成（超时）\n")
            continue
        
        initial = result['initial']
        final = result['final']
        
        if initial and final:
            # 计算变化
            alt_change = final['altitude'] - initial['altitude']
            hdg_change = final['heading'] - initial['heading']
            # 处理航向跨越360度的情况
            if hdg_change > 180:
                hdg_change -= 360
            elif hdg_change < -180:
                hdg_change += 360
            vel_change = final['velocity'] - initial['velocity']
            
            logging.info(f"   ✅ 完成 (步数: {result['steps']})")
            logging.info(f"   初始: 高度{initial['altitude']:.0f}m, 航向{initial['heading']:.0f}°, 速度{initial['velocity']:.0f}m/s")
            logging.info(f"   最终: 高度{final['altitude']:.0f}m, 航向{final['heading']:.0f}°, 速度{final['velocity']:.0f}m/s")
            logging.info(f"   变化: 高度{alt_change:+.0f}m, 航向{hdg_change:+.0f}°, 速度{vel_change:+.0f}m/s")
            
            # 文件信息
            file_size = os.path.getsize(result['acmi_file']) / 1024
            logging.info(f"   文件: {os.path.basename(result['acmi_file'])} ({file_size:.1f} KB)\n")
        else:
            logging.info(f"   ⚠️  状态数据不完整\n")
    
    logging.info(f"{'='*80}")
    logging.info(f"📁 ACMI目录: {os.path.join(project_root, 'acmi_output')}")
    logging.info(f"请在Tacview中打开这些文件查看机动效果")
    logging.info(f"{'='*80}")


if __name__ == "__main__":
    main()
