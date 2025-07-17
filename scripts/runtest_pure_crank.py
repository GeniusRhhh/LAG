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
        logging.FileHandler('pure_maneuver_test.log', encoding='utf-8')
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


def runtest_crank_maneuver():
    """测试纯Crank机动"""
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

        print("纯Crank机动测试")
        print("=" * 60)

        # 使用配置文件名
        config_name = "simple_maneuver_config"
        print(f"使用配置文件: {config_name}")

        # 创建环境
        env = MultipleCombatEnv(config_name)
        task = env.task

        # 设置Crank机动和参数
        task.set_maneuver_type("crank")
        task.set_crank_params(angle_deg=45.0, turn_rate_deg_per_sec=3.0, hold_time_sec=40.0)
        print("环境创建成功，Crank参数设置完成")

        # 创建ACMI生成器
        acmi_generator = ACMIGenerator()

        # 准备ACMI文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        maneuver_results_dir = os.path.join(project_root, "maneuver_results")
        acmi_filename = f"PureCrank_{timestamp}.acmi"
        acmi_filepath = os.path.join(maneuver_results_dir, acmi_filename)

        # 重置环境
        obs, share_obs = env.reset()
        print(f"环境重置成功")

        # 初始化ACMI文件
        if not acmi_generator.initialize_acmi_file(acmi_filepath):
            return False

        # 运行测试
        step = 0
        done = False
        frame_count = 0

        print(f"开始记录Crank机动ACMI数据到: {acmi_filepath}")

        while not done and step < 1000:
            # 创建正确格式的动作数组
            action_dim = env.action_space.shape[0]
            actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)

            # 执行步骤
            obs, share_obs, rewards, dones, infos = env.step(actions)

            # 每隔几步记录一次数据
            if step % 2 == 0:  # 每2步记录一次
                acmi_generator.write_frame_to_file(acmi_filepath, env)
                frame_count += 1

            step += 1
            done = np.any(dones)

            if step % 100 == 0:
                current_time = env.current_step * env.time_interval
                print(f"步骤 {step}, 时间: {current_time:.1f}s, 已记录帧数: {frame_count}")

                # 显示飞机状态
                for agent_id in env.agents.keys():
                    if env.agents[agent_id].is_alive:
                        agent = env.agents[agent_id]
                        lon, lat, alt = agent.get_geodetic()
                        roll, pitch, yaw = agent.get_rpy() * 180 / np.pi
                        print(f"   - {agent_id}: 高度={alt:.1f}m, 航向={yaw:.1f}°")

        # 关闭环境
        env.close()
        print("环境关闭成功")

        # 验证ACMI文件
        if os.path.exists(acmi_filepath):
            file_size = os.path.getsize(acmi_filepath)
            print(f"Crank ACMI文件生成成功！")
            print(f"文件路径: {acmi_filepath}")
            print(f"文件大小: {file_size} 字节")
            print(f"记录帧数: {frame_count}")
            return acmi_filepath
        else:
            print(f"Crank ACMI文件未生成")
            return None

    except Exception as e:
        print(f"Crank测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def runtest_beam_maneuver():
    """测试纯Beam机动"""
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

        print("纯Beam机动测试")
        print("=" * 60)

        # 创建环境
        config_name = "simple_maneuver_config"
        env = MultipleCombatEnv(config_name)
        task = env.task

        # 设置Beam机动和参数
        task.set_maneuver_type("beam")
        task.set_beam_params(angle_deg=90.0, turn_rate_deg_per_sec=5.0, hold_time_sec=15.0)
        print("环境创建成功，Beam参数设置完成")

        # 创建ACMI生成器
        acmi_generator = ACMIGenerator()

        # 准备ACMI文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        maneuver_results_dir = os.path.join(project_root, "maneuver_results")
        acmi_filename = f"PureBeam_{timestamp}.acmi"
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

        print(f"开始记录Beam机动ACMI数据到: {acmi_filepath}")

        while not done and step < 800:  # Beam机动相对较快
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
                print(f"Beam步骤 {step}, 时间: {current_time:.1f}s")

        env.close()

        if os.path.exists(acmi_filepath):
            file_size = os.path.getsize(acmi_filepath)
            print(f"Beam ACMI文件生成成功！")
            print(f"文件: {os.path.basename(acmi_filepath)}")
            print(f"大小: {file_size} 字节, 帧数: {frame_count}")
            return acmi_filepath
        else:
            return None

    except Exception as e:
        print(f"Beam测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def runtest_notch_maneuver():
    """测试纯Notch机动"""
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

        print("纯Notch机动测试")
        print("=" * 60)

        # 创建环境
        config_name = "simple_maneuver_config"
        env = MultipleCombatEnv(config_name)
        task = env.task

        # 设置Notch机动和安全参数
        task.set_maneuver_type("notch")
        task.set_notch_params(
            angle_deg=90.0,
            turn_rate_deg_per_sec=4.0,
            descent_rate_ft_per_sec=60.0,
            descent_time_sec=5.0,
            hold_time_sec=15.0
        )
        print("环境创建成功，Notch参数设置完成")

        # 创建ACMI生成器
        acmi_generator = ACMIGenerator()

        # 准备ACMI文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        maneuver_results_dir = os.path.join(project_root, "maneuver_results")
        acmi_filename = f"PureNotch_{timestamp}.acmi"
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

        print(f"开始记录Notch机动ACMI数据到: {acmi_filepath}")

        while not done and step < 1500:  # Notch机动需要更多时间
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
                print(f"Notch步骤 {step}, 时间: {current_time:.1f}s")

                # 显示高度变化（Notch的关键特征）
                test_agent = env.agents.get("A0100")
                if test_agent and test_agent.is_alive:
                    lon, lat, alt = test_agent.get_geodetic()
                    roll, pitch, yaw = test_agent.get_rpy() * 180 / np.pi
                    print(f"   - A0100(Notch): 高度={alt:.1f}m, 航向={yaw:.1f}°")

        env.close()

        if os.path.exists(acmi_filepath):
            file_size = os.path.getsize(acmi_filepath)
            print(f"Notch ACMI文件生成成功！")
            print(f"文件: {os.path.basename(acmi_filepath)}")
            print(f"大小: {file_size} 字节, 帧数: {frame_count}")
            print(f"Notch特点: 下降→转弯→低空保持→爬升")
            return acmi_filepath
        else:
            return None

    except Exception as e:
        print(f"Notch测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_maneuvers():
    """运行所有三种机动测试"""
    print("全机动测试系统")
    print(f"当前工作目录: {os.getcwd()}")
    print("=" * 80)

    results = []

    # 1. 测试Crank机动
    print("\n【1/3】Crank机动测试")
    crank_file = runtest_crank_maneuver()
    if crank_file:
        results.append(("Crank", crank_file))

    # 2. 测试Beam机动
    print("\n【2/3】Beam机动测试")
    beam_file = runtest_beam_maneuver()
    if beam_file:
        results.append(("Beam", beam_file))

    # 3. 测试Notch机动
    print("\n【3/3】Notch机动测试")
    notch_file = runtest_notch_maneuver()
    if notch_file:
        results.append(("Notch", notch_file))

    # 输出总结
    print("\n" + "=" * 80)
    print("全机动测试完成总结")
    print("=" * 80)

    if results:
        print(f"成功生成 {len(results)}/3 个ACMI文件:")
        for i, (maneuver_type, filepath) in enumerate(results, 1):
            print(f"   {i}. {maneuver_type}机动: {os.path.basename(filepath)}")
        return True
    else:
        print("所有测试都失败了")
        return False


if __name__ == "__main__":
    print("开始纯机动测试系统")
    print("\n请选择要运行的测试：")
    print("  1 - Crank机动")
    print("  2 - Beam机动")
    print("  3 - Notch机动")
    print("  4 - 全部机动")
    while True:
        try:
            choice = input("请输入数字 (1-4): ").strip()
            choice = int(choice)
            if choice not in [1, 2, 3, 4]:
                print("请输入1、2、3或4")
                continue
            break
        except ValueError:
            print("请输入有效的数字")

    success = False
    if choice == 1:
        print("\n运行Crank机动测试...")
        success = runtest_crank_maneuver() is not None
    elif choice == 2:
        print("\n运行Beam机动测试...")
        success = runtest_beam_maneuver() is not None
    elif choice == 3:
        print("\n运行Notch机动测试...")
        success = runtest_notch_maneuver() is not None
    elif choice == 4:
        print("\n运行所有机动测试...")
        success = run_all_maneuvers()

    if success:
        print("\n🎉 测试成功完成！")
        print("ACMI文件已保存到 maneuver_results/ 目录")
    else:
        print("测试失败，请检查日志")
        sys.exit(1)

    print("\n测试完成！")