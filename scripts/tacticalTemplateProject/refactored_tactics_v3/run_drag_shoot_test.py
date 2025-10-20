#!/usr/bin/env python3
"""
拖曳射击战术测试脚本 - 重构版V3
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, project_root)

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

# 导入重构版战术任务
from drag_shoot_tactical_task_v3 import DragShootTacticalTask

# 导入敌方AI和数据记录器
from drag_shoot_enemy_ai_adapter import create_drag_shoot_enemy_ai_integration
from unified_data_recorder import UnifiedDataRecorder


def setup_logging(output_dir="drag_shoot_v3_output"):
    """设置日志"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"simulation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file, output_dir


def main():
    """主函数"""
    # 设置日志和输出目录
    log_file, output_dir = setup_logging()
    
    print("\n" + "=" * 80)
    print("🚀 拖曳射击战术 - 重构版V3测试")
    print("=" * 80)
    
    # ACMI文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acmi_filepath = os.path.join(output_dir, f"air_combat_{timestamp}.acmi")
    
    # 创建环境配置
    env_config = {
        'num_agents': 4,
        'scenario': 'drag_shoot',
        'render_mode': 'txt',
        'acmi_filepath': acmi_filepath
    }
    
    # 创建战术任务
    task = DragShootTacticalTask()
    
    # 集成统一敌方AI
    task = create_drag_shoot_enemy_ai_integration(task)
    
    # 创建环境
    env = MultipleCombatEnv(task=task)
    
    # 创建数据记录器
    recorder = UnifiedDataRecorder()
    
    print("\n" + "=" * 80)
    print("🎮 开始仿真")
    print("=" * 80)
    
    try:
        # 重置环境
        obs = env.reset()
        done = False
        step_count = 0
        max_steps = 2500
        
        while not done and step_count < max_steps:
            # 获取动作（战术任务内部生成）
            actions = {}
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    actions[agent_id] = np.zeros(4)  # 占位，实际由战术任务生成
            
            # 执行步骤
            obs, rewards, dones, info = env.step(actions)
            
            # 记录数据
            recorder.record_step(env, step_count)
            
            # 渲染ACMI
            try:
                env.render(mode="txt", filepath=acmi_filepath)
            except Exception as e:
                if step_count == 0:
                    logging.warning(f"ACMI渲染失败: {e}")
            
            # 检查是否结束
            done = all(dones.values())
            step_count += 1
            
            # 每100步打印一次进度
            if step_count % 100 == 0:
                alive_red = sum(1 for aid in ["A0100", "A0200"] 
                              if aid in env.agents and env.agents[aid].is_alive)
                alive_blue = sum(1 for aid in ["B0100", "B0200"] 
                               if aid in env.agents and env.agents[aid].is_alive)
                print(f"步骤 {step_count}: 红方{alive_red}/2, 蓝方{alive_blue}/2")
        
        print(f"\n🏁 仿真终止于第 {step_count} 步 ({step_count * 0.2:.1f}s)")
        
        # 统计结果
        print("\n" + "=" * 80)
        print("✅ 仿真完成")
        print("=" * 80)
        print(f"总步数: {step_count}")
        print(f"仿真时间: {step_count * 0.2:.1f}秒")
        
        # 统计存活情况
        red_alive = sum(1 for aid in ["A0100", "A0200"] 
                       if aid in env.agents and env.agents[aid].is_alive)
        blue_alive = sum(1 for aid in ["B0100", "B0200"] 
                        if aid in env.agents and env.agents[aid].is_alive)
        
        print("\n" + "=" * 80)
        print("📊 仿真结果")
        print("=" * 80)
        print(f"红方存活: {red_alive}/2")
        print(f"蓝方存活: {blue_alive}/2")
        
        if red_alive > blue_alive:
            print("🏆 红方获胜")
        elif blue_alive > red_alive:
            print("🏆 蓝方获胜")
        else:
            print("🤝 平局")
        
        # 统计导弹发射
        friendly_launches = sum(1 for mid in env.agents.keys() 
                               if mid.startswith('A') and 'missile' in mid.lower())
        enemy_launches = sum(1 for mid in env.agents.keys() 
                            if mid.startswith('B') and 'missile' in mid.lower())
        
        print(f"\n导弹发射总数: {friendly_launches + enemy_launches}")
        print(f"  友方: {friendly_launches} | 敌方: {enemy_launches}")
        
        # 保存数据
        print("\n" + "=" * 80)
        print("💾 保存数据")
        print("=" * 80)
        
        saved_files = recorder.save_csv_files(output_dir, timestamp)
        print("✅ 数据已保存:")
        for file_type, filepath in saved_files.items():
            print(f"  - {file_type}: {filepath}")
        
        print("\n" + "=" * 80)
        print("🎉 测试完成！")
        print("=" * 80)
        print(f"\n输出目录: {output_dir}/")
        print(f"  - 日志文件: {log_file}")
        print(f"  - ACMI文件: {acmi_filepath}")
        print(f"  - CSV数据文件: {len(saved_files)} 个")
        
    except Exception as e:
        logging.error(f"仿真过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭环境
        env.close()
        logging.info("环境已关闭")


if __name__ == "__main__":
    main()
