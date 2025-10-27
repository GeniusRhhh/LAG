"""
战术仿真运行脚本
支持5种战术的完整2v2空战仿真
"""
import os
import sys
import logging
import argparse
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.utils.utils import parse_config
from tactical_task import TacticalTask
from core import TacticalDecisionManager
import shutil


def setup_logging(output_dir: str, tactic_name: str = "tactical") -> str:
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{tactic_name}_simulation_{timestamp}.log")
    
    # 文件和控制台都记录INFO级别
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    logging.info("=" * 80)
    logging.info("[LAUNCH] 战术决策系统仿真开始")
    logging.info("=" * 80)
    logging.info(f"日志文件: {log_file}")
    return log_file


def print_banner(tactic_name: str = "通用战术"):
    """打印仿真标题"""
    print("\n" + "=" * 80)
    print(f"目标 {tactic_name}仿真")
    print("=" * 80)
    print("我方: F-16C Block 50/52 x2 (蓝方)")
    print("  - 长机: A0100")
    print("  - 僚机: A0200")
    print("  - 雷达: AN/APG-68(V)9")
    print("  - 导弹: AIM-120C AMRAAM")
    print()
    print("敌方: SU-27 Flanker x2 (红方)")
    print("  - 敌机1: B0100")
    print("  - 敌机2: B0200")
    print("  - 雷达: N001VE")
    print("  - 导弹: R-27ER")
    print()
    print("战术系统:")
    print("  - 威胁值计算")
    print("  - 战术选择 (5种进攻战术)")
    print("  - 控制距离节点 (NLT/MELD/MTR/LR/TR/DOR/DR/MAR)")
    print("  - 协同决策")
    print("=" * 80)
    print()


def run_simulation(
    tactic_type: str = "front_back",  # 修改默认战术为前后攻击
    our_intent: str = "conservative_clear",
    max_steps: int = 1650,
    output_dir: str = "./tactical_simulation_results"
):
    """
    运行战术仿真
    
    Args:
        tactic_type: 战术类型 (auto/drag_shoot/pincer/high_low/front_back/side_by_side)
        our_intent: 我方意图 (conservative_clear/aggressive_clear/defensive)
        max_steps: 最大步数
        output_dir: 输出目录
    """
    # 设置日志
    log_file = setup_logging(output_dir, tactic_type)
    
    # 打印标题
    tactic_names = {
        "auto": "自动战术选择",
        "drag_shoot": "拖曳射击战术",
        "pincer": "钳形攻势战术",
        "high_low": "上下夹击战术",
        "front_back": "前后攻击战术",
        "side_by_side": "并排射击战术"
    }
    print_banner(tactic_names.get(tactic_type, "通用战术"))
    
    try:
        # 创建环境 - 使用本地BVR配置
        print("初始化JSBSim环境...")
        
        # 统一使用tactical_bvr配置
        config_name = "tactical_bvr"
        
        # 本地配置文件路径
        local_config_file = os.path.join(current_dir, 'configs', f'{config_name}.yaml')
        
        if not os.path.exists(local_config_file):
            logging.error(f"配置文件不存在: {local_config_file}")
            print(f"❌ 配置文件不存在: {local_config_file}")
            return
        
        # 临时复制配置文件到JSBSim configs目录
        jsbsim_config_dir = os.path.join(project_root, 'envs', 'JSBSim', 'configs')
        jsbsim_config_file = os.path.join(jsbsim_config_dir, f'{config_name}.yaml')
        
        # 复制配置文件
        shutil.copy2(local_config_file, jsbsim_config_file)
        
        logging.info(f"使用配置文件: {local_config_file}")
        print(f"使用配置: {config_name}")
        
        env = MultipleCombatEnv(config_name)
        env.max_steps = max_steps
        print("JSBSim环境创建成功")
        
        # 创建决策管理器
        print(f"初始化战术决策系统 (意图: {our_intent})...")
        
        # 根据战术类型创建不同的决策管理器
        if tactic_type == "drag_shoot":
            from tactics.drag_shoot import DragShootDecisionManager
            decision_manager = DragShootDecisionManager(our_intent_type=our_intent)
        elif tactic_type == "pincer":
            from tactics.pincer_attack import PincerAttackDecisionManager
            decision_manager = PincerAttackDecisionManager(our_intent_type=our_intent)
        elif tactic_type == "high_low":
            from tactics.high_low_attack import HighLowAttackDecisionManager
            decision_manager = HighLowAttackDecisionManager(our_intent_type=our_intent)
        elif tactic_type == "front_back":
            from tactics.sequential_attack import SequentialAttackTactic
            # 前后攻击使用sequential_attack，暂时用通用决策管理器
            decision_manager = TacticalDecisionManager(our_intent_type=our_intent)
        elif tactic_type == "side_by_side":
            from tactics.side_by_side import SideBySideDecisionManager
            decision_manager = SideBySideDecisionManager(our_intent_type=our_intent)
        else:
            # 自动选择战术
            decision_manager = TacticalDecisionManager(our_intent_type=our_intent)
        
        # 创建战术任务
        tactical_task = TacticalTask(env.config, decision_manager=decision_manager)
        env.task = tactical_task
        
        # 重置环境
        print("重置环境...")
        obs = env.reset()
        print("战术系统初始化完成")
        
        # 准备数据记录和ACMI文件
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_data = []
        tactical_data = []
        
        # 设置ACMI文件路径
        acmi_filepath = os.path.join(output_dir, f'tactical_2v2_{timestamp}.txt.acmi')
        
        # 开始仿真
        print("\n" + "=" * 80)
        print("仿真开始")
        print("=" * 80)
        print(f"{'时间(s)':<10} {'步数':<8} {'距离(km)':<12} {'我方':<8} {'敌方':<8} {'战术':<15}")
        print("-" * 80)
        
        step = 0
        dt = env.time_interval
        
        while step < max_steps:
            step += 1
            current_time = step * dt
            
            # 环境步进
            import numpy as np
            num_agents = len(env.agents)
            dummy_actions = np.zeros((1, num_agents, 4))
            obs, share_obs, rewards, dones, info = env.step(dummy_actions)
            
            # 渲染ACMI文件（每步都记录）
            env.render(mode="txt", filepath=acmi_filepath)
            
            # 记录轨迹数据
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    pos = env.agents[agent_id].get_position()
                    vel = env.agents[agent_id].get_velocity()
                    trajectory_data.append({
                        'time': current_time,
                        'agent_id': agent_id,
                        'x': pos[0], 'y': pos[1], 'z': pos[2],
                        'vx': vel[0], 'vy': vel[1], 'vz': vel[2],
                    })
            
            # 每10秒打印一次详细状态
            if step % 120 == 0:  # 120步 ≈ 10秒
                # 打印飞机位置和速度
                for agent_id in env.agents.keys():
                    if env.agents[agent_id].is_alive:
                        pos = env.agents[agent_id].get_position()
                        vel = env.agents[agent_id].get_velocity()
                        speed = np.linalg.norm(vel)
                        alt = pos[2]
                        logging.info(f"[{agent_id}] 位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}m) "
                                   f"速度: {speed:.1f}m/s 高度: {alt:.1f}m")
            
            # 每5秒打印一次状态
            if step % 60 == 0:  # 60步 ≈ 5秒
                # 计算双方最近距离
                min_dist = float('inf')
                for aid1 in ["A0100", "A0200"]:
                    if aid1 not in env.agents or not env.agents[aid1].is_alive:
                        continue
                    for aid2 in ["B0100", "B0200"]:
                        if aid2 not in env.agents or not env.agents[aid2].is_alive:
                            continue
                        pos1 = env.agents[aid1].get_position()
                        pos2 = env.agents[aid2].get_position()
                        dist = ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)**0.5
                        min_dist = min(min_dist, dist)
                
                # 统计存活
                friendly_alive = sum(1 for aid in ["A0100", "A0200"] 
                                   if aid in env.agents and env.agents[aid].is_alive)
                enemy_alive = sum(1 for aid in ["B0100", "B0200"] 
                                if aid in env.agents and env.agents[aid].is_alive)
                
                # 获取当前战术
                tactic = tactical_task.selected_tactic if hasattr(tactical_task, 'selected_tactic') and tactical_task.selected_tactic else 'N/A'
                
                # 战术名称映射
                tactic_names_short = {
                    'DRAG_SHOOT': '拖曳射击',
                    'PINCER_ATTACK': '钳形攻势',
                    'HIGH_LOW_ATTACK': '上下夹击',
                    'FRONT_BACK': '前后攻击',
                    'SIDE_BY_SIDE': '并排射击',
                    'N/A': '未决策'
                }
                tactic_str = tactic_names_short.get(tactic, str(tactic))
                
                print(f"{current_time:<10.1f} {step:<8} {min_dist/1000:<12.1f} "
                      f"{friendly_alive}/2{'':<4} {enemy_alive}/2{'':<4} {tactic_str:<15}")
            
            # 检查终止
            if isinstance(dones, np.ndarray):
                if dones.all():
                    break
            elif isinstance(dones, dict):
                if all(dones.values()):
                    break
        
        # 仿真结束
        print("\n" + "=" * 80)
        print("仿真结束")
        print("=" * 80)
        
        # 统计结果
        friendly_alive = sum(1 for aid in ["A0100", "A0200"] 
                           if aid in env.agents and env.agents[aid].is_alive)
        enemy_alive = sum(1 for aid in ["B0100", "B0200"] 
                        if aid in env.agents and env.agents[aid].is_alive)
        
        print(f"总步数: {step}")
        print(f"总时间: {step * dt:.1f}秒")
        print(f"我方存活: {friendly_alive}/2")
        print(f"敌方存活: {enemy_alive}/2")
        
        if friendly_alive > enemy_alive:
            print("\n我方获胜!")
        elif enemy_alive > friendly_alive:
            print("\n敌方获胜!")
        else:
            print("\n平局")
        
        print(f"\n详细日志: {log_file}")
        
        # 保存数据
        if trajectory_data:
            import pandas as pd
            traj_file = os.path.join(output_dir, f'trajectory_{timestamp}.csv')
            pd.DataFrame(trajectory_data).to_csv(traj_file, index=False)
            print(f"轨迹数据: {traj_file}")
        
        # 显示ACMI文件
        if os.path.exists(acmi_filepath):
            print(f"ACMI文件: {acmi_filepath}")
        else:
            print(f"ACMI文件未生成")
        
        print("=" * 80)
        
        # 关闭环境
        env.close()
        
    except Exception as e:
        logging.error(f"仿真错误: {e}", exc_info=True)
        print(f"\n仿真错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='战术仿真系统')
    parser.add_argument('--tactic', type=str, default='front_back',  # 默认使用前后攻击战术
                       choices=['auto', 'drag_shoot', 'pincer', 'high_low', 'front_back', 'side_by_side'],
                       help='战术类型')
    parser.add_argument('--intent', type=str, default='conservative_clear',
                       choices=['conservative_clear', 'aggressive_clear', 'defensive'],
                       help='我方意图')
    parser.add_argument('--steps', type=int, default=1650,
                       help='最大步数')
    parser.add_argument('--output', type=str, default='./tactical_simulation_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    run_simulation(
        tactic_type=args.tactic,
        our_intent=args.intent,
        max_steps=args.steps,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
