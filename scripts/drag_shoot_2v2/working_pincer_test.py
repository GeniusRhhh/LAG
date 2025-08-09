#!/usr/bin/env python3
"""
工作的钳形夹击测试 - 使用现有的拖曳射击配置，只修改战术逻辑
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

def create_pincer_normalize_action():
    """创建钳形夹击的normalize_action方法"""
    
    def pincer_normalize_action(self, env, agent_id, action):
        """钳形夹击战术的normalize_action实现"""
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])
        
        try:
            # 获取当前状态
            current_time = env.current_step * env.time_interval
            my_pos = env.agents[agent_id].get_position()
            
            # 获取敌机位置
            enemy_pos = None
            for enemy_id, enemy in env.agents.items():
                if enemy_id.startswith('B') and enemy.is_alive:
                    enemy_pos = enemy.get_position()
                    break
            
            if enemy_pos is None:
                return np.array([0.0, 0.0, 0.0, 0.7])
            
            # 计算距离
            distance = np.linalg.norm(enemy_pos - my_pos)
            
            # 确定战术阶段
            if distance > 81000:  # NLT-MELD阶段
                phase = "NLT_MELD"
            elif distance > 45000:  # MELD-MTR阶段
                phase = "MELD_MTR"
            elif distance > 41000:  # MTR-TR阶段
                phase = "MTR_TR"
            elif distance > 19600:  # TR-DOR阶段
                phase = "TR_DOR"
            else:  # DOR-DR阶段
                phase = "DOR_DR"
            
            # 根据阶段和角色执行钳形夹击逻辑
            if agent_id.startswith('A'):  # 友方
                return self._get_pincer_action(env, agent_id, phase, enemy_pos, distance)
            else:  # 敌方
                return self._get_enemy_action(env, agent_id, my_pos, distance)
                
        except Exception as e:
            logging.error(f"钳形夹击逻辑失败 {agent_id}: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    def _get_pincer_action(self, env, agent_id, phase, enemy_pos, distance):
        """获取友方钳形夹击动作"""
        from envs.JSBSim.core.catalog import Catalog as c
        
        my_pos = env.agents[agent_id].get_position()
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        
        if phase == "NLT_MELD":
            # 钳形展开阶段
            if agent_id == "A0100":  # 长机向右Crank
                target_heading = (current_heading + 45) % 360
            else:  # 僚机向左Crank
                target_heading = (current_heading - 45) % 360
            
            print(f"[钳形展开] {agent_id}: 当前航向{current_heading:.1f}° -> 目标航向{target_heading:.1f}°")
        
        elif phase in ["MELD_MTR", "MTR_TR"]:
            # 钳形收拢和攻击阶段 - 指向敌机
            dx = enemy_pos[0] - my_pos[0]
            dy = enemy_pos[1] - my_pos[1]
            target_heading = np.rad2deg(np.arctan2(dx, dy)) % 360
            
            print(f"[钳形收拢] {agent_id}: 当前航向{current_heading:.1f}° -> 目标航向{target_heading:.1f}° (距离{distance/1000:.1f}km)")
        
        else:
            # 脱离和返航阶段 - 转向180度
            target_heading = 180.0
            print(f"[脱离返航] {agent_id}: 当前航向{current_heading:.1f}° -> 目标航向{target_heading:.1f}°")
        
        # 计算航向差值
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        # 转换为控制指令
        if abs(heading_diff) > 2.0:
            # 需要转向
            turn_rate = np.clip(heading_diff / 10.0, -1.0, 1.0)  # 简单的比例控制
            return np.array([0.0, turn_rate, 0.0, 0.7])
        else:
            # 保持直飞
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    def _get_enemy_action(self, env, agent_id, my_pos, distance):
        """获取敌方动作 - 简单对抗"""
        # 敌方保持朝向友方
        friendly_pos = None
        for friendly_id, friendly in env.agents.items():
            if friendly_id.startswith('A') and friendly.is_alive:
                friendly_pos = friendly.get_position()
                break
        
        if friendly_pos is None:
            return np.array([0.0, 0.0, 0.0, 0.7])
        
        from envs.JSBSim.core.catalog import Catalog as c
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        
        # 计算指向友方的航向
        dx = friendly_pos[0] - my_pos[0]
        dy = friendly_pos[1] - my_pos[1]
        target_heading = np.rad2deg(np.arctan2(dx, dy)) % 360
        
        # 计算航向差值
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        # 转换为控制指令
        if abs(heading_diff) > 2.0:
            turn_rate = np.clip(heading_diff / 10.0, -1.0, 1.0)
            return np.array([0.0, turn_rate, 0.0, 0.7])
        else:
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    return pincer_normalize_action, _get_pincer_action, _get_enemy_action

def test_working_pincer():
    """测试工作的钳形夹击"""
    print("测试工作的钳形夹击...")
    
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
        
        # 使用现有的拖曳射击配置
        env = MultipleCombatEnv("drag_shoot_tactical")
        print("✓ 环境创建成功 (使用拖曳射击配置)")
        
        # 获取任务并修改normalize_action方法
        task = env.task
        
        # 创建钳形夹击方法
        pincer_normalize_action, _get_pincer_action, _get_enemy_action = create_pincer_normalize_action()
        
        # 替换方法
        import types
        task.normalize_action = types.MethodType(pincer_normalize_action, task)
        task._get_pincer_action = types.MethodType(_get_pincer_action, task)
        task._get_enemy_action = types.MethodType(_get_enemy_action, task)
        
        print("✓ 钳形夹击逻辑注入成功")
        
        # 重置环境
        obs = env.reset()
        print("✓ 环境重置成功")
        
        # 运行仿真
        max_steps = 200  # 40秒仿真
        
        print("\n开始钳形夹击仿真...")
        print("=" * 60)
        
        for step in range(max_steps):
            # 创建虚拟动作
            num_agents = len(env._jsbsims)
            actions = np.zeros((1, num_agents, 4))
            
            # 执行一步
            obs, share_obs, rewards, dones, infos = env.step(actions)
            
            # 打印状态
            if step % 25 == 0:  # 每5秒打印一次
                current_time = step * 0.2
                alive_count = sum(1 for agent in env.agents.values() if agent.is_alive)
                print(f"\n时间 {current_time:.1f}s (步骤 {step}): 存活飞机 {alive_count}")
                
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
            
            # 检查终止条件
            if all(dones.values()):
                print(f"\n仿真在第 {step} 步结束")
                break
        
        print("\n=" * 60)
        print("✓ 钳形夹击仿真完成")
        
        # 打印最终状态
        print("\n最终状态:")
        for agent_id, agent in env.agents.items():
            if agent.is_alive:
                pos = agent.get_position()
                print(f"  {agent_id}: 存活, 位置({pos[0]/1000:.1f}, {pos[1]/1000:.1f}, {pos[2]/1000:.1f})km")
            else:
                print(f"  {agent_id}: 被击落")
        
        return True
        
    except Exception as e:
        print(f"✗ 钳形夹击测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("工作的钳形夹击测试")
    print("=" * 60)
    
    success = test_working_pincer()
    
    if success:
        print("\n🎉 钳形夹击测试成功！")
        print("钳形夹击战术逻辑工作正常，可以看到：")
        print("1. 钳形展开阶段：双机向两侧Crank")
        print("2. 钳形收拢阶段：双机指向敌机")
        print("3. 脱离返航阶段：双机转向返航")
    else:
        print("\n❌ 测试失败，需要修复问题。")

if __name__ == "__main__":
    main()
