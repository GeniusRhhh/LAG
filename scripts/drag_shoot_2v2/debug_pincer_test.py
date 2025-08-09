#!/usr/bin/env python3
"""
调试钳形夹击测试 - 详细错误信息
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

def create_debug_pincer_normalize_action():
    """创建调试版本的钳形夹击normalize_action方法"""
    
    def debug_pincer_normalize_action(self, env, agent_id, action):
        """调试版本的钳形夹击战术normalize_action实现"""
        try:
            # 基本检查
            if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                return np.array([0.0, 0.0, 0.0, 0.7])
            
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
            if distance > 81000:
                phase = "NLT_MELD"
            elif distance > 45000:
                phase = "MELD_MTR"
            elif distance > 41000:
                phase = "MTR_TR"
            elif distance > 19600:
                phase = "TR_DOR"
            else:
                phase = "DOR_DR"
            
            # 根据阶段和角色执行钳形夹击逻辑
            if agent_id.startswith('A'):  # 友方
                return self._get_debug_pincer_action(env, agent_id, phase, enemy_pos, distance)
            else:  # 敌方
                return self._get_debug_enemy_action(env, agent_id, my_pos, distance)
                
        except Exception as e:
            print(f"ERROR in normalize_action for {agent_id}: {e}")
            traceback.print_exc()
            # 返回安全的默认动作
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    def _get_debug_pincer_action(self, env, agent_id, phase, enemy_pos, distance):
        """获取调试版本的友方钳形夹击动作"""
        try:
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
                turn_rate = np.clip(heading_diff / 10.0, -1.0, 1.0)
                return np.array([0.0, turn_rate, 0.0, 0.7])
            else:
                # 保持直飞
                return np.array([0.0, 0.0, 0.0, 0.7])
                
        except Exception as e:
            print(f"ERROR in _get_debug_pincer_action for {agent_id}: {e}")
            traceback.print_exc()
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    def _get_debug_enemy_action(self, env, agent_id, my_pos, distance):
        """获取调试版本的敌方动作"""
        try:
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
                
        except Exception as e:
            print(f"ERROR in _get_debug_enemy_action for {agent_id}: {e}")
            traceback.print_exc()
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    return debug_pincer_normalize_action, _get_debug_pincer_action, _get_debug_enemy_action

def test_debug_pincer():
    """测试调试版本的钳形夹击"""
    print("测试调试版本的钳形夹击...")
    
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
        
        # 使用现有的拖曳射击配置
        env = MultipleCombatEnv("drag_shoot_tactical")
        print("✓ 环境创建成功")
        
        # 获取任务
        task = env.task
        print(f"✓ 任务类型: {type(task).__name__}")
        
        # 保存原始的normalize_action方法
        original_normalize_action = task.normalize_action
        
        # 创建调试版本的钳形夹击方法
        debug_pincer_normalize_action, _get_debug_pincer_action, _get_debug_enemy_action = create_debug_pincer_normalize_action()
        
        # 替换方法
        import types
        task.normalize_action = types.MethodType(debug_pincer_normalize_action, task)
        task._get_debug_pincer_action = types.MethodType(_get_debug_pincer_action, task)
        task._get_debug_enemy_action = types.MethodType(_get_debug_enemy_action, task)
        
        print("✓ 调试钳形夹击逻辑注入成功")
        
        # 重置环境
        print("重置环境...")
        obs = env.reset()
        print("✓ 环境重置成功")
        
        # 运行仿真
        max_steps = 10  # 只运行10步进行调试
        
        print("\n开始调试钳形夹击仿真...")
        print("=" * 60)
        
        for step in range(max_steps):
            print(f"\n--- 步骤 {step} ---")
            
            try:
                # 创建虚拟动作
                num_agents = len(env._jsbsims)
                actions = np.zeros((1, num_agents, 4))
                print(f"创建动作数组: shape={actions.shape}")
                
                # 调用task.step前先检查
                print("调用 task.step...")
                step_result = task.step(env)
                print(f"task.step 返回: {type(step_result)}")
                
                if step_result is None:
                    print("ERROR: task.step 返回 None!")
                    break
                
                # 检查返回值的结构
                if isinstance(step_result, tuple):
                    print(f"返回元组长度: {len(step_result)}")
                    obs, share_obs, rewards, dones, infos = step_result
                    print("✓ 成功解包返回值")
                else:
                    print(f"ERROR: 返回值不是元组: {type(step_result)}")
                    break
                
                # 打印状态
                alive_count = sum(1 for agent in env.agents.values() if agent.is_alive)
                print(f"存活飞机: {alive_count}")
                
                # 检查终止条件
                if all(dones.values()):
                    print(f"仿真在第 {step} 步结束")
                    break
                    
            except Exception as e:
                print(f"步骤 {step} 执行失败: {e}")
                traceback.print_exc()
                # 恢复原始方法
                task.normalize_action = original_normalize_action
                break
        
        print("\n=" * 60)
        print("✓ 调试钳形夹击仿真完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 调试钳形夹击测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("调试钳形夹击测试")
    print("=" * 60)
    
    success = test_debug_pincer()
    
    if success:
        print("\n🎉 调试测试完成！")
    else:
        print("\n❌ 调试测试失败。")

if __name__ == "__main__":
    main()
