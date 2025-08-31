#!/usr/bin/env python3
"""
运行并排射击战术仿真 - 完全基于拖曳射击项目的成功架构

使用完全移植的并排射击战术任务类，确保所有核心功能都来自已验证的拖曳射击实现
"""

import os
import sys
import logging
import datetime
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置工作目录
os.chdir(project_root)

# 导入必要的模块
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.utils.utils import load_config
from side_by_side_shooting_tactical_task_new import SideBySideShootingTacticalTask

def setup_logging():
    """设置日志系统 - 完全复制拖曳射击项目的日志配置"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("scripts/drag_shoot_2v2/side_by_side_shooting_results")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"side_by_side_shooting_simulation_{timestamp}.log"
    
    # 配置日志格式
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

def create_environment():
    """创建仿真环境 - 完全复制拖曳射击项目的环境配置"""
    try:
        # 加载配置
        config_path = "envs/JSBSim/configs/multiplecombat_config.yaml"
        config = load_config(config_path)
        
        # 创建环境
        env = MultipleCombatEnv(config)
        
        # 设置并排射击战术任务
        task = SideBySideShootingTacticalTask(config)
        env.task = task
        
        logging.info("✅ 仿真环境创建成功")
        logging.info(f"📊 智能体数量: {len(env.agents)}")
        logging.info(f"🎯 战术任务: 并排射击战术")
        
        return env, task
        
    except Exception as e:
        logging.error(f"❌ 环境创建失败: {e}")
        raise

def run_simulation(env, task, max_steps=1500):
    """运行仿真 - 完全复制拖曳射击项目的仿真循环"""
    try:
        # 重置环境
        obs = env.reset()
        logging.info("🔄 环境重置完成")
        
        # 仿真循环
        step_count = 0
        done = False
        
        while not done and step_count < max_steps:
            # 生成动作（使用任务的战术逻辑）
            actions = {}
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    # 使用任务的normalize_action方法生成战术动作
                    action = task.normalize_action(env, agent_id, None)
                    actions[agent_id] = action
                else:
                    # 死亡的智能体使用零动作
                    actions[agent_id] = np.zeros(3)  # 修复：使用正确的动作维度
            
            # 执行步骤
            obs, rewards, dones, infos = env.step(actions)
            
            # 检查终止条件
            done = any(dones.values())
            step_count += 1
            
            # 每100步打印进度
            if step_count % 100 == 0:
                current_time = step_count * env.time_interval
                alive_count = sum(1 for agent in env.agents.values() if agent.is_alive)
                logging.info(f"⏱️ 步骤: {step_count}, 时间: {current_time:.1f}s, "
                           f"存活飞机: {alive_count}/4, 阶段: {task.current_phase.value}")
        
        # 仿真结束
        final_time = step_count * env.time_interval
        logging.info(f"🏁 仿真结束")
        logging.info(f"📊 总步数: {step_count}")
        logging.info(f"⏱️ 总时间: {final_time:.1f}秒")
        logging.info(f"🎯 最终阶段: {task.current_phase.value}")
        
        # 统计存活情况
        friendly_alive = sum(1 for aid in env.agents.keys() 
                           if aid.startswith('A') and env.agents[aid].is_alive)
        enemy_alive = sum(1 for aid in env.agents.keys() 
                        if aid.startswith('B') and env.agents[aid].is_alive)
        
        logging.info(f"✈️ 友方存活: {friendly_alive}/2")
        logging.info(f"✈️ 敌方存活: {enemy_alive}/2")
        
        # 判断结果
        if friendly_alive > enemy_alive:
            logging.info("🎉 友方胜利！")
        elif enemy_alive > friendly_alive:
            logging.info("💥 敌方胜利！")
        else:
            logging.info("🤝 平局！")
            
        return True
        
    except Exception as e:
        logging.error(f"❌ 仿真运行失败: {e}")
        import traceback
        logging.error(f"详细错误信息: {traceback.format_exc()}")
        return False

def main():
    """主函数"""
    try:
        # 设置日志
        log_file = setup_logging()
        
        # 创建环境
        env, task = create_environment()
        
        # 运行仿真
        success = run_simulation(env, task)
        
        if success:
            logging.info("✅ 并排射击战术仿真成功完成")
        else:
            logging.error("❌ 并排射击战术仿真失败")
            
        # 关闭环境
        env.close()
        logging.info("🔚 环境已关闭")
        
    except Exception as e:
        logging.error(f"❌ 主程序执行失败: {e}")
        import traceback
        logging.error(f"详细错误信息: {traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
