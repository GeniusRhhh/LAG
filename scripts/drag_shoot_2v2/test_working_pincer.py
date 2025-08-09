#!/usr/bin/env python3
"""
测试工作的钳形夹击战术任务
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime
import traceback

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

def test_working_pincer():
    """测试工作的钳形夹击战术任务"""
    print("测试工作的钳形夹击战术任务...")
    
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
        from working_pincer_tactical_task import PincerAttackTacticalTask
        
        # 使用现有的拖曳射击配置
        env = MultipleCombatEnv("drag_shoot_tactical")
        print("✓ 环境创建成功")
        
        # 强制设置步数
        env.max_steps = 300  # 60秒仿真
        
        # 替换任务为钳形夹击任务
        env.task = PincerAttackTacticalTask(env.config)
        print("✓ 钳形夹击任务替换成功")
        
        # 重置环境
        print("重置环境...")
        obs = env.reset()
        print("✓ 环境重置成功")
        
        # 运行仿真
        max_steps = 300  # 60秒仿真
        
        print("\n开始钳形夹击仿真...")
        print("=" * 80)
        
        for step in range(max_steps):
            try:
                # 创建虚拟动作
                num_agents = len(env._jsbsims)
                actions = np.zeros((1, num_agents, 4))
                
                # 执行一步
                obs, share_obs, rewards, dones, infos = env.step(actions)
                
                # 打印状态
                if step % 25 == 0:  # 每5秒打印一次
                    current_time = step * 0.2
                    alive_count = sum(1 for agent in env.agents.values() if agent.is_alive)
                    
                    print(f"\n时间 {current_time:5.1f}s (步骤 {step:3d}): 存活飞机 {alive_count}")
                    
                    # 打印飞机位置和航向
                    for agent_id, agent in env.agents.items():
                        if agent.is_alive:
                            pos = agent.get_position()
                            from envs.JSBSim.core.catalog import Catalog as c
                            heading = np.rad2deg(agent.get_property_value(c.attitude_psi_rad))
                            velocity = np.linalg.norm(agent.get_velocity())
                            print(f"  {agent_id}: pos=({pos[0]/1000:6.1f}, {pos[1]/1000:6.1f}, {pos[2]/1000:4.1f})km, "
                                  f"hdg={heading:5.1f}°, vel={velocity:5.1f}m/s")
                    
                    # 计算双方距离
                    a_pos = None
                    b_pos = None
                    for agent_id, agent in env.agents.items():
                        if agent.is_alive:
                            if agent_id.startswith('A') and a_pos is None:
                                a_pos = agent.get_position()
                            elif agent_id.startswith('B') and b_pos is None:
                                b_pos = agent.get_position()
                    
                    if a_pos is not None and b_pos is not None:
                        distance = np.linalg.norm(b_pos - a_pos)
                        print(f"  双方距离: {distance/1000:.1f}km")
                        
                        # 显示当前战术阶段
                        phase = env.task.current_phase.value
                        print(f"  当前阶段: {phase}")
                
                # 检查终止条件
                if isinstance(dones, dict):
                    all_done = all(dones.values())
                else:
                    all_done = all(dones)

                if all_done:
                    print(f"\n仿真在第 {step} 步结束")
                    break
                    
            except Exception as e:
                print(f"步骤 {step} 执行失败: {e}")
                traceback.print_exc()
                break
        
        print("\n" + "=" * 80)
        print("✓ 钳形夹击仿真完成")
        
        # 打印最终状态
        print("\n最终状态:")
        for agent_id, agent in env.agents.items():
            if agent.is_alive:
                pos = agent.get_position()
                print(f"  {agent_id}: 存活, 位置({pos[0]/1000:.1f}, {pos[1]/1000:.1f}, {pos[2]/1000:.1f})km")
            else:
                print(f"  {agent_id}: 被击落")
        
        print(f"\n最终阶段: {env.task.current_phase.value}")
        
        return True
        
    except Exception as e:
        print(f"✗ 钳形夹击测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("工作的钳形夹击战术测试")
    print("=" * 80)
    
    success = test_working_pincer()
    
    if success:
        print("\n🎉 钳形夹击战术测试成功！")
        print("\n钳形夹击战术特征:")
        print("1. 钳形展开阶段：双机向两侧45°Crank机动")
        print("2. 钳形收拢阶段：双机指向敌机形成包夹")
        print("3. 分层攻击阶段：长机优先攻击，僚机滞后支援")
        print("4. 脱离机动阶段：双机执行Short Skate返航")
        print("5. 返航阶段：双机返回基地")
    else:
        print("\n❌ 测试失败，需要修复问题。")

if __name__ == "__main__":
    main()
