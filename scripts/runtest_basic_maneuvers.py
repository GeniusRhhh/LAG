import os
import sys
import logging
import numpy as np
import math
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('basic_maneuver_test.log', encoding='utf-8')
    ]
)


class ACMIGenerator:
    """ACMI文件生成器"""

    def __init__(self):
        self.trajectory_data = []
        self.file_created = False

    def initialize_acmi_file(self, filepath):
        """初始化ACMI文件"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, mode='w', encoding='utf-8-sig') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.1\n")
                f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
            self.file_created = True
            logging.info(f"ACMI文件初始化成功: {filepath}")
            return True
        except Exception as e:
            logging.error(f"ACMI文件初始化失败: {e}")
            return False

    def write_frame_to_file(self, filepath, env):
        """实时写入一帧到文件"""
        if not self.file_created:
            self.initialize_acmi_file(filepath)
        try:
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = env.current_step * env.time_interval
                f.write(f"#{timestamp:.2f}\n")
                for agent_id, agent in env.agents.items():
                    if agent.is_alive:
                        log_msg = agent.log()
                        if log_msg:
                            f.write(log_msg + "\n")
                for sim in env._tempsims.values():
                    log_msg = sim.log()
                    if log_msg:
                        f.write(log_msg + "\n")
            return True
        except Exception as e:
            logging.error(f"写入ACMI帧失败: {e}")
            return False


def runtest_basic_maneuver(maneuver_name, maneuver_index, **params):
    """测试单个基础机动"""
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
        logging.info(f"测试基础机动: {maneuver_name} (编号: {maneuver_index})")
        logging.info(f"参数: {params}")
        config_name = "simple_maneuver_config"
        env = MultipleCombatEnv(config_name)
        task = env.task
        task.set_basic_maneuver(maneuver_name, **params)
        logging.info(f"基础机动设置完成: {maneuver_name}")
        acmi_generator = ACMIGenerator()
        timestamp = datetime.now().strftime("%m%d%H%M")
        maneuver_results_dir = os.path.join(project_root, "maneuver_results")
        acmi_filename = f"Basic_{maneuver_index}_{maneuver_name}_{timestamp}.acmi"
        acmi_filepath = os.path.join(maneuver_results_dir, acmi_filename)
        obs, share_obs = env.reset()
        if not acmi_generator.initialize_acmi_file(acmi_filepath):
            return False
        step = 0
        done = False
        frame_count = 0
        max_steps = 800
        logging.info(f"开始记录{maneuver_name}机动ACMI数据到: {acmi_filepath}")
        logging.info(f"最大步数: {max_steps}")
        while not done and step < max_steps:
            action_dim = env.action_space.shape[0]
            actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)
            obs, share_obs, rewards, dones, infos = env.step(actions)
            if step % 2 == 0:
                acmi_generator.write_frame_to_file(acmi_filepath, env)
                frame_count += 1
            step += 1
            done = np.any(dones)
            # 动态日志记录频率：前80秒详细记录，后面间隔久一点
            current_time = env.current_step * env.time_interval
            if current_time <= 80.0:
                # 前80秒：每50步记录一次
                if step % 50 == 0:
                    logging.info(f"[详细] 步骤 {step}, 时间: {current_time:.1f}s")
            else:
                # 80秒后：每200步记录一次
                if step % 200 == 0:
                    logging.info(f"[概要] 步骤 {step}, 时间: {current_time:.1f}s")
        env.close()
        if os.path.exists(acmi_filepath):
            file_size = os.path.getsize(acmi_filepath)
            logging.info(f"{maneuver_name} ACMI文件生成成功")
            logging.info(f"文件: {os.path.basename(acmi_filepath)}")
            logging.info(f"大小: {file_size} 字节, 帧数: {frame_count}")
            return acmi_filepath
        else:
            logging.error(f"{maneuver_name} ACMI文件未生成")
            return None
    except Exception as e:
        logging.error(f"{maneuver_name}测试失败: {e}", exc_info=True)
        return None


def runtest_composite_maneuver(maneuver_name, maneuver_index, custom_params=None):
    """测试组合机动"""
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
        logging.info(f"测试组合机动: {maneuver_name} (编号: {maneuver_index})")
        logging.info(f"参数: {custom_params}")
        config_name = "simple_maneuver_config"
        env = MultipleCombatEnv(config_name)
        task = env.task
        task.set_composite_maneuver(maneuver_name, custom_params=custom_params)
        logging.info(f"组合机动设置完成: {maneuver_name}")
        acmi_generator = ACMIGenerator()
        timestamp = datetime.now().strftime("%m%d%H%M")
        maneuver_results_dir = os.path.join(project_root, "maneuver_results")
        acmi_filename = f"Composite_{maneuver_index}_{maneuver_name}_{timestamp}.acmi"
        acmi_filepath = os.path.join(maneuver_results_dir, acmi_filename)
        obs, share_obs = env.reset()
        if not acmi_generator.initialize_acmi_file(acmi_filepath):
            return None
        step = 0
        done = False
        frame_count = 0
        max_steps = 2000
        logging.info(f"开始记录{maneuver_name}组合机动ACMI数据到: {acmi_filepath}")
        logging.info(f"最大步数: {max_steps}")
        while not done and step < max_steps:
            action_dim = env.action_space.shape[0]
            actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)
            obs, share_obs, rewards, dones, infos = env.step(actions)
            if step % 2 == 0:
                acmi_generator.write_frame_to_file(acmi_filepath, env)
                frame_count += 1
            step += 1
            done = np.any(dones)
            if step % 100 == 0:
                current_time = env.current_step * env.time_interval
                logging.info(f"步骤 {step}, 时间: {current_time + current_time:.1f}")
        env.close()
        if os.path.exists(acmi_filepath):
            file_size = os.path.getsize(acmi_filepath)
            logging.info(f"{maneuver_name} ACMI文件生成成功")
            logging.info(f"文件: {os.path.basename(acmi_filepath)}")
            logging.info(f"大小: {file_size} 字节, 帧数: {frame_count}")
            return acmi_filepath
        else:
            logging.error(f"{maneuver_name} ACMI文件未生成")
            return None
    except Exception as e:
        logging.error(f"{maneuver_name}测试失败: {e}", exc_info=True)
        return None


def run_all_basic_maneuvers():
    """运行所有基础机动测试"""
    logging.info("=== 基础机动综合测试 ===")
    basic_maneuvers = [
        # 核心飞行机动
        ("level_flight", {"duration": 40.0}),
        ("accelerate", {"velocity_change": 50.0, "duration": 25.0}),
        ("decelerate", {"velocity_change": 50.0, "duration": 25.0}),
        ("turn", {"turn_angle": 80.0, "turn_rate": 3.0}),
        ("turn_level", {"turn_angle": 80.0, "turn_rate": 3.0}),  # 新增：保持高度转弯
        ("pull_up", {"altitude_gain": 1500.0, "duration": 20.0}),
        ("dive", {"altitude_loss": 1500.0, "duration": 20.0, "min_altitude": 2000.0}),
        ("diagonal_flight", {"turn_angle": 70.0, "altitude_change": 1000.0, "duration": 20.0}),
        ("maintain_heading_flight", {"duration": 40.0}),  # 修正名称

        # 高级机动
        ("accelerate_escape", {"acceleration": 100.0, "duration": 20.0}),
        ("vertical_loop", {"loop_type": "half", "g_force": 6.0, "duration": 15.0}),
    ]

    results = []
    failed_tests = []

    for i, (maneuver_name, params) in enumerate(basic_maneuvers, 1):
        logging.info(f"测试 {i}/{len(basic_maneuvers)}: {maneuver_name}")
        logging.info(f"参数: {params}")
        try:
            file_path = runtest_basic_maneuver(maneuver_name, i, **params)
            if file_path:
                results.append((f"{maneuver_name}_{i}", file_path))
                logging.info(f"✅ {maneuver_name} 测试成功")
            else:
                failed_tests.append((maneuver_name, "文件生成失败"))
                logging.warning(f"⚠️ {maneuver_name} 测试失败：文件生成失败")
        except Exception as e:
            failed_tests.append((maneuver_name, str(e)))
            logging.error(f"❌ {maneuver_name} 测试异常：{e}")

    # 输出测试总结
    logging.info(f"\n=== 基础机动测试总结 ===")
    logging.info(f"总测试数量: {len(basic_maneuvers)}")
    logging.info(f"成功测试: {len(results)}")
    logging.info(f"失败测试: {len(failed_tests)}")

    if failed_tests:
        logging.warning("失败的测试:")
        for name, error in failed_tests:
            logging.warning(f"  - {name}: {error}")

    return results, failed_tests


def run_all_composite_maneuvers():
    """运行所有组合机动测试"""
    logging.info("组合机动测试")
    composite_configs = [
        ("turn_pull_up", {
            "turn_pull_up": {
                "turn_angle": 90.0,
                "turn_duration": 35.0,
                "turn_rate": 3.0,
                "altitude_gain": 2000.0,
                "pull_up_duration": 20.0
            }
        }),
        ("turn_dive", {
            "turn_dive": {
                "turn_angle": 90.0,
                "turn_duration": 40.0,
                "turn_rate": 4.0,
                "altitude_loss": 1500.0,
                "dive_duration": 20.0,
                "min_altitude": 2000.0
            }
        }),
        ("spiral_climb", {}),
    ]
    results = []
    for i, (maneuver_name, params) in enumerate(composite_configs, 1):
        logging.info(f"测试 {i}/{len(composite_configs)}: {maneuver_name}")
        logging.info(f"参数: {params}")
        file_path = runtest_composite_maneuver(maneuver_name, i, custom_params=params)
        if file_path:
            results.append((f"{maneuver_name}_{i}", file_path))
    return results


def run_all_tactical_maneuvers_simple():
    """运行新增的战术机动测试，包括Short Skate"""
    logging.info("=== 战术机动综合测试 ===")
    tactical_configs = [
        ("crank_tactical", {}),
        ("beam_tactical", {}),
        ("notch_tactical", {}),
        ("short_skate_tactical", {}),
        ("banzai_tactical", {}),  # 新增：包含adaptive_crank和adaptive_turn_to_enemy
        ("sliceback_tactical", {}),  # 新增：包含vertical_loop
    ]

    results = []
    failed_tests = []

    for i, (maneuver_name, params) in enumerate(tactical_configs, 1):
        logging.info(f"测试 {i}/{len(tactical_configs)}: {maneuver_name}")
        logging.info(f"参数: {params}")
        try:
            file_path = runtest_composite_maneuver(maneuver_name, i, custom_params=params)
            if file_path:
                results.append((f"{maneuver_name}_{i}", file_path))
                logging.info(f"✅ {maneuver_name} 战术测试成功")
            else:
                failed_tests.append((maneuver_name, "文件生成失败"))
                logging.warning(f"⚠️ {maneuver_name} 战术测试失败：文件生成失败")
        except Exception as e:
            failed_tests.append((maneuver_name, str(e)))
            logging.error(f"❌ {maneuver_name} 战术测试异常：{e}")

    # 输出测试总结
    logging.info(f"\n=== 战术机动测试总结 ===")
    logging.info(f"总测试数量: {len(tactical_configs)}")
    logging.info(f"成功测试: {len(results)}")
    logging.info(f"失败测试: {len(failed_tests)}")

    if failed_tests:
        logging.warning("失败的战术测试:")
        for name, error in failed_tests:
            logging.warning(f"  - {name}: {error}")

    return results, failed_tests


def run_comprehensive_maneuver_tests():
    """运行全面的机动测试 - 包括所有基础、组合和战术机动"""
    logging.info("🚀 开始全面机动测试")
    logging.info("=" * 80)

    all_results = {}
    all_failed = {}

    # 1. 基础机动测试
    logging.info("第一阶段：基础机动测试")
    basic_results, basic_failed = run_all_basic_maneuvers()
    all_results['basic'] = basic_results
    all_failed['basic'] = basic_failed

    # 2. 组合机动测试
    logging.info("\n第二阶段：组合机动测试")
    composite_results = run_all_composite_maneuvers()
    all_results['composite'] = composite_results

    # 3. 战术机动测试
    logging.info("\n第三阶段：战术机动测试")
    tactical_results, tactical_failed = run_all_tactical_maneuvers_simple()
    all_results['tactical'] = tactical_results
    all_failed['tactical'] = tactical_failed

    # 生成综合报告
    generate_comprehensive_test_report(all_results, all_failed)

    return all_results, all_failed


def generate_comprehensive_test_report(all_results, all_failed):
    """生成综合测试报告"""
    logging.info("\n" + "=" * 80)
    logging.info("📊 全面机动测试综合报告")
    logging.info("=" * 80)

    total_tests = 0
    total_success = 0
    total_failed = 0

    for category, results in all_results.items():
        if isinstance(results, tuple):  # 有失败信息的结果
            success_count = len(results)
            failed_count = len(all_failed.get(category, []))
        else:  # 只有成功结果的列表
            success_count = len(results)
            failed_count = 0

        category_total = success_count + failed_count
        total_tests += category_total
        total_success += success_count
        total_failed += failed_count

        logging.info(f"\n{category.upper()}机动测试:")
        logging.info(f"  总数: {category_total}")
        logging.info(f"  成功: {success_count}")
        logging.info(f"  失败: {failed_count}")
        logging.info(f"  成功率: {(success_count/category_total*100):.1f}%" if category_total > 0 else "  成功率: N/A")

    logging.info(f"\n总体统计:")
    logging.info(f"  总测试数: {total_tests}")
    logging.info(f"  总成功数: {total_success}")
    logging.info(f"  总失败数: {total_failed}")
    logging.info(f"  总成功率: {(total_success/total_tests*100):.1f}%" if total_tests > 0 else "  总成功率: N/A")

    # 分析失败的机动
    if total_failed > 0:
        logging.info(f"\n❌ 失败机动分析:")
        for category, failed_list in all_failed.items():
            if failed_list:
                logging.info(f"  {category.upper()}机动失败:")
                for name, error in failed_list:
                    logging.info(f"    - {name}: {error}")

    # 生成建议
    logging.info(f"\n💡 优化建议:")

    if total_failed == 0:
        logging.info("  - 所有机动测试通过，代码库状态良好")
    elif total_failed < total_tests * 0.1:
        logging.info("  - 大部分机动正常工作，少数失败可能需要修复")
    else:
        logging.info("  - 较多机动测试失败，建议进行系统性检查和修复")

    logging.info("=" * 80)


def run_new_maneuvers_test():
    """运行新增机动测试"""
    logging.info("=" * 60)
    logging.info("新增机动测试系统")
    logging.info("=" * 60)

    # 新增基础机动
    new_basic_maneuvers = [
        ("accelerate_escape", "加速逃离 - 保持航向并加速"),
        ("vertical_loop", "垂直回旋 - 上下方向的高G机动")
    ]

    # 新增战术机动
    new_tactical_maneuvers = [
        ("banzai_tactical", "Banzai机动 - 发射后决策战术"),
        ("sliceback_tactical", "Sliceback机动 - 水平滚转+垂直回旋")
    ]

    logging.info("可用的新增机动:")
    logging.info("基础机动:")
    for i, (name, desc) in enumerate(new_basic_maneuvers, 1):
        logging.info(f"  {i} - {name}: {desc}")

    logging.info("战术机动:")
    for i, (name, desc) in enumerate(new_tactical_maneuvers, len(new_basic_maneuvers) + 1):
        logging.info(f"  {i} - {name}: {desc}")

    total_options = len(new_basic_maneuvers) + len(new_tactical_maneuvers)
    logging.info(f"  {total_options + 1} - 测试所有新增机动")

    while True:
        try:
            choice = int(input(f"请选择机动编号 (1-{total_options + 1}): ").strip())

            if 1 <= choice <= len(new_basic_maneuvers):
                # 测试单个基础机动
                maneuver_name, desc = new_basic_maneuvers[choice - 1]
                logging.info(f"测试基础机动: {maneuver_name} - {desc}")

                file_path = runtest_basic_maneuver(maneuver_name, choice)
                success = file_path is not None

            elif len(new_basic_maneuvers) + 1 <= choice <= total_options:
                # 测试单个战术机动
                tactical_index = choice - len(new_basic_maneuvers) - 1
                maneuver_name, desc = new_tactical_maneuvers[tactical_index]
                logging.info(f"测试战术机动: {maneuver_name} - {desc}")

                file_path = runtest_composite_maneuver(maneuver_name, choice)
                success = file_path is not None

            elif choice == total_options + 1:
                # 测试所有新增机动
                logging.info("测试所有新增机动...")
                results = []

                # 测试所有新增基础机动
                for i, (name, desc) in enumerate(new_basic_maneuvers, 1):
                    logging.info(f"测试基础机动 {i}/{len(new_basic_maneuvers)}: {name}")
                    file_path = runtest_basic_maneuver(name, i)
                    if file_path:
                        results.append(f"basic_{name}")

                # 测试所有新增战术机动
                for i, (name, desc) in enumerate(new_tactical_maneuvers, 1):
                    logging.info(f"测试战术机动 {i}/{len(new_tactical_maneuvers)}: {name}")
                    file_path = runtest_composite_maneuver(name, i + len(new_basic_maneuvers))
                    if file_path:
                        results.append(f"tactical_{name}")

                success = len(results) > 0
                if success:
                    logging.info(f"成功测试 {len(results)} 个新增机动")

            else:
                logging.warning(f"请输入1到{total_options + 1}之间的数字")
                continue

            break

        except ValueError:
            logging.warning("请输入有效的数字")

    return success


if __name__ == "__main__":
    logging.info("基础机动和组合机动测试系统")
    logging.info("可用测试选项:")
    logging.info("  1 - 单个基础机动测试")
    logging.info("  2 - 所有基础机动测试")
    logging.info("  3 - 单个组合机动测试")
    logging.info("  4 - 所有组合机动测试")
    logging.info("  5 - 新增战术机动测试")
    logging.info("  6 - 全部测试")
    logging.info("  7 - 新增机动专项测试 (基础机动+战术机动)")
    logging.info("  8 - 🚀 全面综合测试 (推荐 - 包含所有机动和详细分析)")

    while True:
        try:
            choice = input("请输入数字 (1-8): ").strip()
            choice = int(choice)
            if choice not in [1, 2, 3, 4, 5, 6, 7, 8]:
                logging.warning("请输入1-8之间的数字")
                continue
            break
        except ValueError:
            logging.warning("请输入有效的数字")

    success = False
    basic_maneuvers = [
        ("level_flight", {"duration": 40.0}),
        ("accelerate", {"velocity_change": 50.0, "duration": 25.0}),
        ("decelerate", {"velocity_change": 50.0, "duration": 25.0}),
        ("turn", {"turn_angle": 80.0, "turn_rate": 3.0}),
        ("pull_up", {"altitude_gain": 1500.0, "duration": 20.0}),
        ("dive", {"altitude_loss": 1500.0, "duration": 20.0, "min_altitude": 2000.0}),
        ("diagonal_flight", {"turn_angle": 70.0, "altitude_change": 1000.0, "duration": 20.0}),
        ("level_flight_maintain_heading", {"duration": 40.0}),
    ]
    composite_configs = [
        ("turn_pull_up", {
            "turn_pull_up": {
                "turn_angle": 90.0,
                "turn_duration": 35.0,
                "turn_rate": 3.0,
                "altitude_gain": 2000.0,
                "pull_up_duration": 20.0
            }
        }),
        ("turn_dive", {
            "turn_dive": {
                "turn_angle": 90.0,
                "turn_duration": 40.0,
                "turn_rate": 4.0,
                "altitude_loss": 1500.0,
                "dive_duration": 20.0,
                "min_altitude": 2000.0
            }
        }),
        ("spiral_climb", {}),
    ]

    if choice == 1:
        logging.info("可用的基础机动:")
        for i, (name, _) in enumerate(basic_maneuvers, 1):
            logging.info(f"  {i} - {name}")
        while True:
            try:
                maneuver_choice = int(input("请选择基础机动编号: ").strip())
                if 1 <= maneuver_choice <= len(basic_maneuvers):
                    maneuver_name, params = basic_maneuvers[maneuver_choice - 1]
                    logging.info(f"执行 {maneuver_name}，参数: {params}")
                    success = runtest_basic_maneuver(maneuver_name, maneuver_choice, **params) is not None
                    break
                else:
                    logging.warning(f"请输入1到{len(basic_maneuvers)}之间的数字")
            except ValueError:
                logging.warning("请输入有效的数字")

    elif choice == 2:
        logging.info("运行所有基础机动测试")
        results = run_all_basic_maneuvers()
        success = len(results) > 0
        if success:
            logging.info(f"成功生成 {len(results)} 个基础机动ACMI文件")

    elif choice == 3:
        logging.info("可用的组合机动:")
        for i, (name, _) in enumerate(composite_configs, 1):
            logging.info(f"  {i} - {name}")
        while True:
            try:
                maneuver_choice = int(input("请选择组合机动编号: ").strip())
                if 1 <= maneuver_choice <= len(composite_configs):
                    maneuver_name, params = composite_configs[maneuver_choice - 1]
                    logging.info(f"执行 {maneuver_name}，参数: {params}")
                    success = runtest_composite_maneuver(maneuver_name, maneuver_choice, custom_params=params) is not None
                    break
                else:
                    logging.warning(f"请输入1到{len(composite_configs)}之间的数字")
            except ValueError:
                logging.warning("请输入有效的数字")

    elif choice == 4:
        logging.info("运行所有组合机动测试")
        results = run_all_composite_maneuvers()
        success = len(results) > 0
        if success:
            logging.info(f"成功生成 {len(results)} 个组合机动ACMI文件")

    elif choice == 5:
        logging.info("运行新增战术机动测试")
        results = run_all_tactical_maneuvers_simple()
        success = len(results) > 0
        if success:
            logging.info(f"成功生成 {len(results)} 个战术机动ACMI文件")

    elif choice == 6:
        logging.info("运行全部测试")
        basic_results = run_all_basic_maneuvers()
        composite_results = run_all_composite_maneuvers()
        tactical_results = run_all_tactical_maneuvers_simple()
        success = len(basic_results) > 0 or len(composite_results) > 0 or len(tactical_results) > 0
        if success:
            logging.info(f"成功生成 {len(basic_results)} 个基础机动 + {len(composite_results)} 个组合机动 + {len(tactical_results)} 个战术机动ACMI文件")

    elif choice == 7:
        logging.info("运行新增机动专项测试")
        success = run_new_maneuvers_test()

    elif choice == 8:
        logging.info("🚀 运行全面综合测试")
        all_results, all_failed = run_comprehensive_maneuver_tests()

        # 判断成功标准：至少有一些测试通过
        total_success = sum(len(results) if isinstance(results, list) else len(results)
                          for results in all_results.values())
        success = total_success > 0

        if success:
            logging.info(f"✅ 全面综合测试完成！共有 {total_success} 个测试通过")
        else:
            logging.error("❌ 全面综合测试失败，所有测试都未通过")

    if success:
        logging.info("测试成功完成")
        logging.info("ACMI文件已保存到 maneuver_results/ 目录")
    else:
        logging.error("测试失败，请检查日志")
        sys.exit(1)

    logging.info("测试完成")


def run_remaining_maneuvers():
    """运行剩余的机动测试 - 自适应和特殊机动"""
    results = {}
    failed = {}

    logging.info("🚀 开始运行剩余机动测试")

    # 特殊机动测试
    special_maneuvers = [
        ("vertical_loop", {"loop_type": "half", "g_force": 6.0, "duration": 15.0}),
        ("accelerate_escape", {"acceleration": 100.0, "duration": 20.0}),
    ]

    all_maneuvers = special_maneuvers

    for i, (maneuver_name, params) in enumerate(all_maneuvers, 11):  # 从11开始编号
        logging.info(f"测试 {i-10}/{len(all_maneuvers)}: {maneuver_name}")
        try:
            result = runtest_basic_maneuver(maneuver_name, i, custom_params=params)
            if result:
                results[f"Remaining_{i}_{maneuver_name}"] = result
                logging.info(f"✅ {maneuver_name} 测试成功")
            else:
                failed[maneuver_name] = "测试失败"
                logging.warning(f"⚠️ {maneuver_name} 测试失败")
        except Exception as e:
            failed[maneuver_name] = str(e)
            logging.error(f"❌ {maneuver_name} 测试异常：{e}")

    return results, failed


if __name__ == "__main__":
    # 如果直接运行此脚本，执行剩余机动测试
    if len(sys.argv) > 1 and sys.argv[1] == "remaining":
        logging.info("🚀 执行剩余机动测试")
        results, failed = run_remaining_maneuvers()

        logging.info(f"\n=== 剩余机动测试总结 ===")
        logging.info(f"成功测试: {len(results)}")
        logging.info(f"失败测试: {len(failed)}")

        if failed:
            logging.warning("失败的测试:")
            for name, error in failed.items():
                logging.warning(f"  - {name}: {error}")

        if len(results) > 0:
            logging.info(f"✅ 剩余机动测试完成！共有 {len(results)} 个测试通过")
        else:
            logging.error("❌ 剩余机动测试失败，所有测试都未通过")
    else:
        # 原有的交互式测试
        main()


