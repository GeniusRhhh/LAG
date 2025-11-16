#!/usr/bin/env python3
"""
测试导弹发射修复
"""

import os
import sys
sys.path.append('.')
sys.path.append('scripts/tacticalProject')

from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
from algorithms.sac.sac_trainer import SacTrainer
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def test_missile_launch():
    """测试导弹发射是否正常工作"""
    
    print("=" * 50)
    print("🚀 测试导弹发射修复")
    print("=" * 50)
    
    # 创建环境
    env = SingleControlEnv()
    env.reset()
    
    # 创建SAC训练器（以便获得tactical_task）
    trainer = SacTrainer(env, project_name="missile_test")
    tactical_task = trainer.tactical_task
    
    print(f"✅ 环境初始化完成")
    print(f"✅ 战术任务初始化完成")
    
    # 模拟600步（约300秒）
    for step in range(600):
        # 设置A0200导弹发射标志
        if step == 100:
            tactical_task.state_manager.missile_launched['A0200'] = True
            print(f"步骤{step}: 设置A0200导弹发射标志 = True")
        
        # 获取所有智能体的动作
        actions = {}
        for agent_id in ['A0100', 'A0200', 'B0100', 'B0200']:
            if agent_id in env.agents and env.agents[agent_id].is_alive:
                try:
                    action = tactical_task.get_action(env, agent_id)
                    actions[agent_id] = action
                except Exception as e:
                    print(f"⚠️ {agent_id} 获取动作失败: {e}")
                    actions[agent_id] = (7, 8, 3)  # 默认动作
        
        # 执行环境步进
        env.step(actions)
        
        # 检查导弹发射状态
        if step % 100 == 0:
            print(f"步骤{step}:")
            for agent_id in ['A0100', 'A0200']:
                if agent_id in env.agents:
                    remaining = env.agents[agent_id].num_missiles
                    launched_flag = tactical_task.state_manager.missile_launched.get(agent_id, False)
                    fired_count = tactical_task.state_manager.missiles_fired.get(agent_id, 0)
                    print(f"  {agent_id}: 剩余导弹={remaining}, 发射标志={launched_flag}, 已发射={fired_count}")
    
    print("\n" + "=" * 50)
    print("🎯 最终导弹状态:")
    for agent_id in ['A0100', 'A0200']:
        if agent_id in env.agents:
            remaining = env.agents[agent_id].num_missiles
            fired_count = tactical_task.state_manager.missiles_fired.get(agent_id, 0)
            print(f"  {agent_id}: 剩余导弹={remaining}, 已发射数量={fired_count}")
    print("=" * 50)

if __name__ == "__main__":
    test_missile_launch()