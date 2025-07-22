# scripts/runtest_basic_maneuvers.py
import os
import sys
import logging
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 修正日志编码问题
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

            # 写入文件头
            with open(filepath, mode='w', encoding='utf-8-sig') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.1\n")
                f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")

            self.file_created = True
            print(f"ACMI文件初始化成功: {filepath}")
            return True

        except Exception as e:
            print(f"ACMI文件初始化失败: {e}")
            return False

    def write_frame_to_file(self, filepath, env):
        """实时写入一帧到文件"""
        if not self.file_created:
            self.initialize_acmi_file(filepath)

        try:
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = env.current_step * env.time_interval
                f.write(f"#{timestamp:.2f}\n")

                # 写入所有飞机的日志
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
            print(f"写入ACMI帧失败: {e}")
            return False


def runtest_basic_maneuver(maneuver_name, **params):
    """测试单个基础机动"""
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

        print(f"测试基础机动: {maneuver_name}")
        print("=" * 60)

        # 创建环境
        config_name = "simple_maneuver_config"
        env = MultipleCombatEnv(config_name)
        task = env.task

        # 设置基础机动
        task.set_basic_maneuver(maneuver_name, **params)
        print(f"基础机动设置完成: {maneuver_name}")

        # 创建ACMI生成器
        acmi_generator = ACMIGenerator()

        # 准备ACMI文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        maneuver_results_dir = os.path.join(project_root, "maneuver_results")
        acmi_filename = f"Basic_{maneuver_name}_{timestamp}.acmi"
        acmi_filepath = os.path.join(maneuver_results_dir, acmi_filename)

        # 重置环境
        obs, share_obs = env.reset()

        # 初始化ACMI文件
        if not acmi_generator.initialize_acmi_file(acmi_filepath):
            return False

        # 运行测试
        step = 0
        done = False
        frame_count = 0
        max_steps = 600  # 基础机动通常较短

        print(f"开始记录{maneuver_name}机动ACMI数据到: {acmi_filepath}")

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
                print(f"步骤 {step}, 时间: {current_time:.1f}s")

        env.close()

        if os.path.exists(acmi_filepath):
            file_size = os.path.getsize(acmi_filepath)
            print(f"{maneuver_name} ACMI文件生成成功！")
            print(f"文件: {os.path.basename(acmi_filepath)}")
            print(f"大小: {file_size} 字节, 帧数: {frame_count}")
            return acmi_filepath
        else:
            return None

    except Exception as e:
        print(f"{maneuver_name}测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def runtest_composite_maneuver(maneuver_name, custom_params=None):
    """测试组合机动 - 添加自定义参数支持"""
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

        print(f"测试组合机动: {maneuver_name}")
        print("=" * 60)

        # 创建环境
        config_name = "simple_maneuver_config"
        env = MultipleCombatEnv(config_name)
        task = env.task

        # 设置组合机动，支持参数
        task.set_composite_maneuver(maneuver_name, custom_params=custom_params)
        print(f"组合机动设置完成: {maneuver_name} with params {custom_params}")

        # 创建ACMI生成器
        acmi_generator = ACMIGenerator()

        # 准备ACMI文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        maneuver_results_dir = os.path.join(project_root, "maneuver_results")
        acmi_filename = f"Composite_{maneuver_name}_{timestamp}.acmi"
        acmi_filepath = os.path.join(maneuver_results_dir, acmi_filename)

        # 重置环境
        obs, share_obs = env.reset()

        # 初始化ACMI文件
        if not acmi_generator.initialize_acmi_file(acmi_filepath):
            return None  # 修改为返回None以匹配原有逻辑

        # 运行测试
        step = 0
        done = False
        frame_count = 0
        max_steps = 1200  # 组合机动需要更多时间

        print(f"开始记录{maneuver_name}组合机动ACMI数据到: {acmi_filepath}")

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
                print(f"步骤 {step}, 时间: {current_time:.1f}s")

        env.close()

        if os.path.exists(acmi_filepath):
            file_size = os.path.getsize(acmi_filepath)
            print(f"{maneuver_name}组合机动ACMI文件生成成功！")
            print(f"文件: {os.path.basename(acmi_filepath)}")
            print(f"大小: {file_size} 字节, 帧数: {frame_count}")
            return acmi_filepath
        else:
            return None

    except Exception as e:
        print(f"{maneuver_name}组合机动测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_basic_maneuvers():
    """运行所有基础机动测试"""
    print("基础机动测试系统")
    print("=" * 80)

    # 基础机动测试配置
    basic_maneuvers = [
        ("level_flight", {}),#平飞
        ("accelerate", {"velocity_change": 100.0, "duration": 15.0}),#加速
        ("decelerate", {"velocity_change": 100.0, "duration": 15.0}),#减速
        ("turn", {"turn_angle": 90.0, "turn_rate": 5.0}),#转弯
        ("pull_up", {"altitude_change": 1500.0, "duration": 30.0}),#拉起
        ("dive", {"altitude_change": 1500.0, "duration": 30.0}),#俯冲
        ("diagonal_flight", {"turn_angle": 45.0, "altitude_change": 1500.0, "duration": 30.0}),#斜直飞
        ("roll", {"duration": 4.0}),#滚转
        ("turn_pull_up", {"turn_angle": 60.0, "turn_rate": 3.0, "altitude_change": 1500.0}),#转弯拉起
        ("turn_dive", {"turn_angle": 60.0, "turn_rate": 3.0, "altitude_change": 1500.0})#转弯俯冲
    ]

    results = []

    for i, (maneuver_name, params) in enumerate(basic_maneuvers, 1):
        print(f"\n【{i}/{len(basic_maneuvers)}】测试{maneuver_name}")
        file_path = runtest_basic_maneuver(maneuver_name, **params)
        if file_path:
            results.append((maneuver_name, file_path))

    return results


def run_all_composite_maneuvers():
    """运行所有组合机动测试 - 支持参数和批量扫描"""
    print("组合机动测试系统")
    print("=" * 80)

    composite_configs = [
        ("escape", {"turn_angle": 120.0, "accel_velocity": 150.0}),
        ("attack", {"side_climb_duration": 10.0, "dive_velocity": 60.0}),
        ("defense", {"dive_alt": -1500.0, "turn_angle": 120.0}),
        ("scissor", {"turn_angle": 60.0}),  # 新组合
        ("high_yoyo", {"climb_alt": 2000.0})  # 新组合
    ]
    results = []

    for i, (maneuver_name, params) in enumerate(composite_configs, 1):
        print(f"\n【{i}/{len(composite_configs)}】测试{maneuver_name}组合机动")
        file_path = runtest_composite_maneuver(maneuver_name, custom_params=params)
        if file_path:
            results.append((maneuver_name, file_path))

    return results

if __name__ == "__main__":
    print("基础机动和组合机动测试系统")
    print("\n请选择要运行的测试：")
    print("  1 - 单个基础机动测试")
    print("  2 - 所有基础机动测试")
    print("  3 - 单个组合机动测试")
    print("  4 - 所有组合机动测试")
    print("  5 - 全部测试")

    while True:
        try:
            choice = input("请输入数字 (1-5): ").strip()
            choice = int(choice)
            if choice not in [1, 2, 3, 4, 5]:
                print("请输入1、2、3、4或5")
                continue
            break
        except ValueError:
            print("请输入有效的数字")

    success = False

    if choice == 1:
        print("\n可用的基础机动:")
        basic_maneuvers = ["level_flight", "accelerate", "decelerate", "turn", "pull_up",
                           "dive", "diagonal_flight", "roll", "turn_pull_up", "turn_dive"]
        for i, name in enumerate(basic_maneuvers, 1):
            print(f"  {i} - {name}")

        while True:
            try:
                maneuver_choice = input("请选择基础机动编号: ").strip()
                maneuver_choice = int(maneuver_choice)
                if 1 <= maneuver_choice <= len(basic_maneuvers):
                    maneuver_name = basic_maneuvers[maneuver_choice - 1]
                    success = runtest_basic_maneuver(maneuver_name) is not None
                    break
                else:
                    print(f"请输入1到{len(basic_maneuvers)}之间的数字")
            except ValueError:
                print("请输入有效的数字")

    elif choice == 2:
        print("\n运行所有基础机动测试...")
        results = run_all_basic_maneuvers()
        success = len(results) > 0
        if success:
            print(f"\n成功生成 {len(results)}/{len(results)} 个基础机动ACMI文件")

    elif choice == 3:
        print("\n可用的组合机动:")
        composite_maneuvers = ["escape", "attack", "defense", "scissor", "high_yoyo"]
        for i, name in enumerate(composite_maneuvers, 1):
            print(f"  {i} - {name}")

        while True:
            try:
                maneuver_choice = input("请选择组合机动编号: ").strip()
                maneuver_choice = int(maneuver_choice)
                if 1 <= maneuver_choice <= len(composite_maneuvers):
                    maneuver_name = composite_maneuvers[maneuver_choice - 1]
                    success = runtest_composite_maneuver(maneuver_name) is not None
                    break
                else:
                    print(f"请输入1到{len(composite_maneuvers)}之间的数字")
            except ValueError:
                print("请输入有效的数字")

    elif choice == 4:
        print("\n运行所有组合机动测试...")
        results = run_all_composite_maneuvers()
        success = len(results) > 0
        if success:
            print(f"\n成功生成 {len(results)}/{len(results)} 个组合机动ACMI文件")

    elif choice == 5:
        print("\n运行全部测试...")
        basic_results = run_all_basic_maneuvers()
        composite_results = run_all_composite_maneuvers()
        success = len(basic_results) > 0 or len(composite_results) > 0
        if success:
            print(f"\n成功生成 {len(basic_results)} 个基础机动 + {len(composite_results)} 个组合机动ACMI文件")

    if success:
        print("\n🎉 测试成功完成！")
        print("ACMI文件已保存到 maneuver_results/ 目录")
        print("\n使用Tacview查看ACMI文件:")
        print("1. 下载并安装Tacview: https://tacview.net/")
        print("2. 打开Tacview软件")
        print("3. 拖拽ACMI文件到Tacview窗口")
        print("4. 使用鼠标和键盘控制视角")
    else:
        print("测试失败，请检查日志")
        sys.exit(1)

    print("\n测试完成！")