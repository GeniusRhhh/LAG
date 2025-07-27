import os
import sys
import logging
import numpy as np
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
            if step % 100 == 0:
                current_time = env.current_step * env.time_interval
                logging.info(f"步骤 {step}, 时间: {current_time:.1f}s")
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
    logging.info("基础机动测试")
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
    results = []
    for i, (maneuver_name, params) in enumerate(basic_maneuvers, 1):
        logging.info(f"测试 {i}/{len(basic_maneuvers)}: {maneuver_name}")
        logging.info(f"参数: {params}")
        file_path = runtest_basic_maneuver(maneuver_name, i, **params)
        if file_path:
            results.append((f"{maneuver_name}_{i}", file_path))
    return results


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
    """运行新增的3个战术机动测试"""
    logging.info("新增战术机动测试")
    tactical_configs = [
        ("crank_tactical", {}),
        ("beam_tactical", {}),
        ("notch_tactical", {})
    ]
    results = []
    for i, (maneuver_name, params) in enumerate(tactical_configs, 1):
        logging.info(f"测试 {i}/{len(tactical_configs)}: {maneuver_name}")
        logging.info(f"参数: {params}")
        file_path = runtest_composite_maneuver(maneuver_name, i, custom_params=params)
        if file_path:
            results.append((f"{maneuver_name}_{i}", file_path))
    return results


if __name__ == "__main__":
    logging.info("基础机动和组合机动测试系统")
    logging.info("可用测试选项:")
    logging.info("  1 - 单个基础机动测试")
    logging.info("  2 - 所有基础机动测试")
    logging.info("  3 - 单个组合机动测试")
    logging.info("  4 - 所有组合机动测试")
    logging.info("  5 - 新增战术机动测试")
    logging.info("  6 - 全部测试")

    while True:
        try:
            choice = input("请输入数字 (1-6): ").strip()
            choice = int(choice)
            if choice not in [1, 2, 3, 4, 5, 6]:
                logging.warning("请输入1、2、3、4、5或6")
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

    if success:
        logging.info("测试成功完成")
        logging.info("ACMI文件已保存到 maneuver_results/ 目录")
    else:
        logging.error("测试失败，请检查日志")
        sys.exit(1)

    logging.info("测试完成")


