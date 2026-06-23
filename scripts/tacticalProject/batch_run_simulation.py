"""
批量运行仿真脚本 - 用于生成意图识别数据集
运行100次仿真：60次conservative_clear + 40次aggressive_clear
只生成ACMI文件和意图识别轨迹CSV文件a
"""
import os
import sys

# 关键修复：必须在导入任何模块之前设置环境变量！
# 因为 tactical_task.py 在模块导入时就读取环境变量
os.environ["FRIEND_BASELINE_MODEL"] = "SU27"
os.environ["ENEMY_BASELINE_MODEL"] = "F16"

import logging
from datetime import datetime
import time
from typing import List, Tuple

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

# 配置日志 - 只输出关键信息
logging.basicConfig(
    level=logging.WARNING,  # 只显示WARNING及以上级别
    format='%(message)s'
)

# 导入必要的模块
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.utils.utils import parse_config
from tactical_task import TacticalTask
from envs.JSBSim.core.catalog import Catalog as c

# 导入数据记录器
sys.path.insert(0, os.path.join(current_dir, '..', 'tacticalTemplateProject'))
# noinspection PyUnresolvedReferences
from unified_data_recorder import UnifiedDataRecorder


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _build_run_plan() -> List[Tuple[int, str]]:
    start_run_id = max(1, _env_int("INTENT_RUN_START", 1))
    total_runs = max(1, _env_int("INTENT_RUN_COUNT", 100))
    conservative_runs = _env_int("INTENT_CONSERVATIVE_COUNT", 60)
    aggressive_runs = _env_int("INTENT_AGGRESSIVE_COUNT", 40)

    if conservative_runs < 0:
        conservative_runs = 0
    if aggressive_runs < 0:
        aggressive_runs = 0

    if conservative_runs + aggressive_runs <= 0:
        conservative_runs = total_runs
        aggressive_runs = 0

    requested_total = conservative_runs + aggressive_runs
    if requested_total != total_runs:
        if requested_total < total_runs:
            conservative_runs += (total_runs - requested_total)
        else:
            overflow = requested_total - total_runs
            reduce_aggressive = min(aggressive_runs, overflow)
            aggressive_runs -= reduce_aggressive
            overflow -= reduce_aggressive
            if overflow > 0:
                conservative_runs = max(0, conservative_runs - overflow)

    plan: List[Tuple[int, str]] = []
    run_id = start_run_id
    for _ in range(conservative_runs):
        plan.append((run_id, 'conservative_clear'))
        run_id += 1
    for _ in range(aggressive_runs):
        plan.append((run_id, 'aggressive_clear'))
        run_id += 1
    return plan


def run_single_simulation(run_id: int, intent: str, output_base_dir: str):
    """
    运行单次仿真
    
    Args:
        run_id: 运行编号 (1-100)
        intent: 意图类型 ('conservative_clear' 或 'aggressive_clear')
        output_base_dir: 输出基础目录
    """
    try:
        # 环境变量已在文件开头设置，这里不需要再设置
        
        # 选择配置文件（不包含.yaml后缀，parse_config会自动添加）
        if intent == 'conservative_clear':
            config_file = 'tactical_bvr'
        else:  # aggressive_clear
            config_file = 'tactical_bvr_aggressive_clear'
        
        # 解析配置
        config = parse_config(config_file)
        
        # 创建环境
        env = MultipleCombatEnv(config_file)
        
        # 创建战术任务
        tactical_task = TacticalTask(config, force_tactic=None)
        env.task = tactical_task
        
        # 修复问题3：初始化雷达管理器
        # 注意：不要清空字典！雷达管理器需要保持内部状态
        try:
            from simulation.radar_manager import get_unified_radar_manager
            radar_manager = get_unified_radar_manager()
            
            # 临时调整N001VE雷达参数（仅在batch_run_simulation中生效）
            # 目标：让我方雷达比敌方弱，更真实的对抗场景
            # 注意：apg68_radar实际上是N001VE（因为第197行做了对调）
            if hasattr(radar_manager, 'apg68_radar'):
                # 降低我方(N001VE)的探测/跟踪/锁定距离
                radar_manager.apg68_radar.max_detection_range = 90000   # 90km (比敌方APG-68的105km弱)
                radar_manager.apg68_radar.max_track_range = 75000       # 75km (比敌方85km弱)
                radar_manager.apg68_radar.max_lock_range = 60000        # 60km (比敌方70km弱)
                
                if run_id == 1:
                    print(f"[OK] 雷达参数已调整 (仅batch_run_simulation):")
                    print(f"   我方N001VE: 探测{radar_manager.apg68_radar.max_detection_range/1000:.0f}km, "
                          f"跟踪{radar_manager.apg68_radar.max_track_range/1000:.0f}km, "
                          f"锁定{radar_manager.apg68_radar.max_lock_range/1000:.0f}km")
                    print(f"   敌方APG-68: 探测{radar_manager.n001ve_radar.max_detection_range/1000:.0f}km, "
                          f"跟踪{radar_manager.n001ve_radar.max_track_range/1000:.0f}km, "
                          f"锁定{radar_manager.n001ve_radar.max_lock_range/1000:.0f}km")
            
            env.task.radar_manager = radar_manager
            if run_id == 1:
                print("[OK] 雷达管理器已初始化")
        except ImportError as e:
            if run_id == 1:
                print(f"[WARN] 雷达管理器初始化失败: {e}")
        
        # 重置环境
        env.reset()
        tactical_task.reset(env)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录（统一放在一个文件夹）
        os.makedirs(output_base_dir, exist_ok=True)
        
        # ACMI文件路径
        acmi_filename = f"run_{run_id:03d}_{timestamp}_{intent}.txt.acmi"
        acmi_filepath = os.path.join(output_base_dir, acmi_filename)
        
        # 创建数据记录器
        data_recorder = UnifiedDataRecorder(
            project_name=f"run_{run_id:03d}_{intent}",
            trajectory_mode="intent_pairwise_frame",
        )
        
        # 验证数据记录器是否真正导入和创建成功
        if run_id == 1:  # 只在第一次运行时验证
            print(f"[OK] 数据记录器验证:")
            print(f"   - 类型: {type(data_recorder)}")
            print(f"   - 模块: {data_recorder.__class__.__module__}")
            print(f"   - 项目名称: {data_recorder.project_name}")
            # 验证关键方法是否存在
            assert hasattr(data_recorder, 'record_aircraft_trajectory'), "缺少 record_aircraft_trajectory 方法"
            assert hasattr(data_recorder, 'save_csv_files'), "缺少 save_csv_files 方法"
            print(f"   - 方法验证: record_aircraft_trajectory [OK], save_csv_files [OK]")
            print(f"   - 数据记录器已就绪，可以正常使用")
            
            # 验证飞机类型配置
            print(f"\n[OK] 飞机类型配置验证:")
            print(f"   - 环境变量 FRIEND_BASELINE_MODEL: {os.environ.get('FRIEND_BASELINE_MODEL', 'NOT_SET')}")
            print(f"   - 环境变量 ENEMY_BASELINE_MODEL: {os.environ.get('ENEMY_BASELINE_MODEL', 'NOT_SET')}")
            if hasattr(tactical_task, 'friend_lowlevel_type'):
                print(f"   - tactical_task.friend_lowlevel_type: {tactical_task.friend_lowlevel_type}")
            else:
                print(f"   - tactical_task.friend_lowlevel_type: NOT_SET")
            if hasattr(tactical_task, 'enemy_lowlevel_type'):
                print(f"   - tactical_task.enemy_lowlevel_type: {tactical_task.enemy_lowlevel_type}")
            else:
                print(f"   - tactical_task.enemy_lowlevel_type: NOT_SET")
            
            # 验证 env.task 引用是否正确
            print(f"\n[OK] env.task 引用验证:")
            print(f"   - env.task is tactical_task: {env.task is tactical_task}")
            print(f"   - hasattr(env.task, 'friend_lowlevel_type'): {hasattr(env.task, 'friend_lowlevel_type')}")
            print(f"   - hasattr(env.task, 'enemy_lowlevel_type'): {hasattr(env.task, 'enemy_lowlevel_type')}")
            if hasattr(env.task, 'friend_lowlevel_type'):
                print(f"   - env.task.friend_lowlevel_type: {env.task.friend_lowlevel_type}")
            if hasattr(env.task, 'enemy_lowlevel_type'):
                print(f"   - env.task.enemy_lowlevel_type: {env.task.enemy_lowlevel_type}")
            
            print(f"\n[OK] 预期雷达类型:")
            print(f"   - A0100/A0200 (我方): N001VE (SU27)")
            print(f"   - B0100/B0200 (敌方): APG-68 (F16)")
        
        # 运行仿真
        max_steps = 3300  # 11分钟
        step = 0
        
        print(f"[{run_id:3d}/100] 运行中: {intent:20s} | ", end='', flush=True)
        
        while step < max_steps:
            step += 1
            current_time = step * env.time_interval
            
            # 环境步进
            import numpy as np
            num_agents = len(env.agents)
            dummy_actions = np.zeros((1, num_agents, 4))
            obs, share_obs, rewards, dones, info = env.step(dummy_actions)
            
            # 修复问题3：更新雷达状态
            try:
                if hasattr(env.task, 'radar_manager') and env.task.radar_manager:
                    env.task.radar_manager.update_friendly_radar_states(env, current_time)
                    env.task.radar_manager.update_enemy_radar_states(env, current_time)
            except Exception as e:
                if step == 1 and run_id == 1:  # 只在第一次运行的第一步报告错误
                    print(f"\n警告: 雷达状态更新失败 - {e}")
            
            # 渲染ACMI文件
            env.render(mode="txt", filepath=acmi_filepath)
            
            # 记录意图识别数据
            try:
                data_recorder.record_aircraft_trajectory(env, current_time, tactical_task)
            except Exception as e:
                if step == 1:  # 只在第一步报告错误
                    print(f"\n警告: 数据记录失败 - {e}")
            
            # 检查终止条件
            red_alive = sum(1 for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive)
            blue_alive = sum(1 for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive)
            
            if red_alive == 0 or blue_alive == 0:
                break
        
        # 修复问题2：保存CSV文件时只保存trajectory文件
        csv_files = data_recorder.save_csv_files(output_base_dir, timestamp, save_only_trajectory=True)
        
        # 不再需要手动删除文件，因为save_only_trajectory=True已经跳过了其他文件的生成
        
        # 计算仿真时长。
        sim_time = step * env.time_interval
        
        # 获取战术信息
        tactic = getattr(tactical_task, 'selected_tactic', 'Unknown') or 'Unknown'
        
        print(f"完成 | 战术: {tactic:15s} | 时长: {sim_time:5.1f}s | 步数: {step:4d}")
        
        return True
        
    except Exception as e:
        print(f"\n错误: 运行 {run_id} 失败 - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数：批量运行100次仿真"""
    run_plan = _build_run_plan()
    total_runs = len(run_plan)
    conservative_runs = sum(1 for _, intent in run_plan if intent == 'conservative_clear')
    aggressive_runs = sum(1 for _, intent in run_plan if intent == 'aggressive_clear')
    
    # 首次验证：确保数据记录器可以正常导入a
    print("="*80)
    print("预检查：验证数据记录器导入...")
    print("="*80)
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(current_dir, '..', 'tacticalTemplateProject'))
        # noinspection PyUnresolvedReferences
        from unified_data_recorder import UnifiedDataRecorder
        
        # 创建测试实例
        test_recorder = UnifiedDataRecorder(
            project_name="precheck_test",
            trajectory_mode="intent_pairwise_frame",
        )
        
        # 验证关键方法
        assert hasattr(test_recorder, 'record_aircraft_trajectory'), "缺少 record_aircraft_trajectory 方法"
        assert hasattr(test_recorder, 'save_csv_files'), "缺少 save_csv_files 方法"
        
        print(f"[OK] 数据记录器预检查通过:")
        print(f"   - 类型: {type(test_recorder)}")
        print(f"   - 模块: {test_recorder.__class__.__module__}")
        print(f"   - 关键方法: record_aircraft_trajectory [OK], save_csv_files [OK]")
        print(f"   - 数据记录器可以正常使用")
        print("="*80)
        print()
    except Exception as e:
        print(f"[ERROR] 数据记录器预检查失败: {e}")
        print(f"   请检查 unified_data_recorder.py 是否存在")
        print(f"   路径: scripts/tacticalTemplateProject/unified_data_recorder.py")
        print("="*80)
        return
    
    print("="*80)
    print("批量仿真运行 - 意图识别数据集生成")
    print("="*80)
    print(f"总运行次数: {total_runs}")
    print(f"  - conservative_clear: {conservative_runs}次")
    print(f"  - aggressive_clear:   {aggressive_runs}次")
    print(f"输出文件:")
    print(f"  - ACMI文件: run_XXX_YYYYMMDD_HHMMSS_intent.txt.acmi")
    print(f"  - CSV文件:  run_XXX_intent_trajectory_YYYYMMDD_HHMMSS.csv")
    print(f"  - 所有文件统一放在同一个文件夹")
    print("="*80)
    print()
    
    # 创建输出目录
    output_base_dir = os.getenv("INTENT_OUTPUT_BASE_DIR", "").strip()
    if not output_base_dir:
        output_base_dir = os.path.join(current_dir, 'intent_recognition_dataset')
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行计数
    success_count = 0
    fail_count = 0
    
    current_stage = None
    for run_id, intent in run_plan:
        if intent != current_stage:
            current_stage = intent
            stage_count = conservative_runs if intent == 'conservative_clear' else aggressive_runs
            print(f"\n阶段: 运行{intent} ({stage_count}次)")
            print("-"*80)
        if run_single_simulation(run_id, intent, output_base_dir):
            success_count += 1
        else:
            fail_count += 1
    
    # 计算总时长
    total_time = time.time() - start_time
    avg_time = total_time / max(total_runs, 1)
    
    # 打印总结
    print()
    print("="*80)
    print("批量运行完成")
    print("="*80)
    print(f"成功: {success_count}/{total_runs}")
    print(f"失败: {fail_count}/{total_runs}")
    print(f"总时长: {total_time/60:.1f}分钟")
    print(f"平均时长: {avg_time:.1f}秒/次")
    print(f"输出目录: {output_base_dir}")
    print("="*80)
    
    # 统计文件数量
    acmi_files = len([f for f in os.listdir(output_base_dir) if f.endswith('.acmi')]) if os.path.exists(output_base_dir) else 0
    csv_files = len([f for f in os.listdir(output_base_dir) if f.endswith('.csv')]) if os.path.exists(output_base_dir) else 0
    
    print(f"\n生成文件统计:")
    print(f"  ACMI文件: {acmi_files}")
    print(f"  CSV文件:  {csv_files}")
    print("="*80)


if __name__ == "__main__":
    main()
