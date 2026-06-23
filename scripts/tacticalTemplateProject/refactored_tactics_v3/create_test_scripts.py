#!/usr/bin/env python3
"""
自动生成所有战术的测试脚本
"""

import os

# 战术配置
TACTICS = [
    {
        'name': 'pincer_attack',
        'display_name': '钳形夹击',
        'class_name': 'PincerAttackTacticalTask',
        'file_name': 'pincer_attack_tactical_task_v3',
        'adapter': 'pincer_enemy_ai_adapter',
        'adapter_func': 'create_pincer_enemy_ai_integration'
    },
    {
        'name': 'front_back_attack',
        'display_name': '前后攻击',
        'class_name': 'FrontBackAttackFinalTask',
        'file_name': 'front_back_attack_v3',
        'adapter': 'front_back_enemy_ai_adapter',
        'adapter_func': 'create_front_back_enemy_ai_integration'
    },
    {
        'name': 'high_low_attack',
        'display_name': '上下夹击',
        'class_name': 'HighLowAttackTacticalTask',
        'file_name': 'high_low_attack_v3',
        'adapter': 'high_low_enemy_ai_adapter',
        'adapter_func': 'create_high_low_enemy_ai_integration'
    },
    {
        'name': 'side_by_side',
        'display_name': '并排射击',
        'class_name': 'SideBySideShootingTacticalTask',
        'file_name': 'side_by_side_shooting_v3',
        'adapter': 'side_by_side_enemy_ai_adapter',
        'adapter_func': 'create_side_by_side_enemy_ai_integration'
    },
    {
        'name': 'turn_around',
        'display_name': '掉头射击',
        'class_name': 'TurnAroundShootingTacticalTask',
        'file_name': 'turn_around_shooting_v3',
        'adapter': 'turn_around_enemy_ai_adapter',
        'adapter_func': 'create_turn_around_enemy_ai_integration'
    }
]

TEMPLATE = '''#!/usr/bin/env python3
"""
{display_name}战术测试脚本 - 重构版V3
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
from {file_name} import {class_name}

# 导入敌方AI和数据记录器
from {adapter} import {adapter_func}
from unified_data_recorder import UnifiedDataRecorder


def setup_logging(output_dir="{name}_v3_output"):
    """设置日志"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"simulation_{{timestamp}}.log")
    
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
    log_file, output_dir = setup_logging()
    
    print("\\n" + "=" * 80)
    print("🚀 {display_name}战术 - 重构版V3测试")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acmi_filepath = os.path.join(output_dir, f"air_combat_{{timestamp}}.acmi")
    
    # 创建战术任务
    task = {class_name}()
    
    # 集成统一敌方AI
    task = {adapter_func}(task)
    
    # 创建环境
    env = MultipleCombatEnv(task=task)
    
    # 创建数据记录器
    recorder = UnifiedDataRecorder()
    
    print("\\n" + "=" * 80)
    print("🎮 开始仿真")
    print("=" * 80)
    
    try:
        obs = env.reset()
        done = False
        step_count = 0
        max_steps = 2500
        
        while not done and step_count < max_steps:
            actions = {{}}
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    actions[agent_id] = np.zeros(4)
            
            obs, rewards, dones, info = env.step(actions)
            recorder.record_step(env, step_count)
            
            try:
                env.render(mode="txt", filepath=acmi_filepath)
            except Exception as e:
                if step_count == 0:
                    logging.warning(f"ACMI渲染失败: {{e}}")
            
            done = all(dones.values())
            step_count += 1
            
            if step_count % 100 == 0:
                alive_red = sum(1 for aid in ["A0100", "A0200"] 
                              if aid in env.agents and env.agents[aid].is_alive)
                alive_blue = sum(1 for aid in ["B0100", "B0200"] 
                               if aid in env.agents and env.agents[aid].is_alive)
                print(f"步骤 {{step_count}}: 红方{{alive_red}}/2, 蓝方{{alive_blue}}/2")
        
        print(f"\\n🏁 仿真终止于第 {{step_count}} 步 ({{step_count * 0.2:.1f}}s)")
        
        # 统计结果
        print("\\n" + "=" * 80)
        print("✅ 仿真完成")
        print("=" * 80)
        
        red_alive = sum(1 for aid in ["A0100", "A0200"] 
                       if aid in env.agents and env.agents[aid].is_alive)
        blue_alive = sum(1 for aid in ["B0100", "B0200"] 
                        if aid in env.agents and env.agents[aid].is_alive)
        
        print(f"红方存活: {{red_alive}}/2")
        print(f"蓝方存活: {{blue_alive}}/2")
        
        if red_alive > blue_alive:
            print("🏆 红方获胜")
        elif blue_alive > red_alive:
            print("🏆 蓝方获胜")
        else:
            print("🤝 平局")
        
        # 保存数据
        print("\\n" + "=" * 80)
        print("💾 保存数据")
        print("=" * 80)
        
        saved_files = recorder.save_csv_files(output_dir, timestamp)
        print("✅ 数据已保存:")
        for file_type, filepath in saved_files.items():
            print(f"  - {{file_type}}: {{filepath}}")
        
        print("\\n" + "=" * 80)
        print("🎉 测试完成！")
        print("=" * 80)
        print(f"\\n输出目录: {{output_dir}}/")
        print(f"  - 日志文件: {{log_file}}")
        print(f"  - ACMI文件: {{acmi_filepath}}")
        
    except Exception as e:
        logging.error(f"仿真过程出错: {{e}}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        logging.info("环境已关闭")


if __name__ == "__main__":
    main()
'''

def create_test_scripts():
    """生成所有测试脚本"""
    for tactic in TACTICS:
        filename = f"run_{tactic['name']}_test.py"
        content = TEMPLATE.format(**tactic)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 创建: {filename}")

if __name__ == "__main__":
    print("🚀 开始生成测试脚本...")
    create_test_scripts()
    print("\\n🎉 所有测试脚本创建完成！")
