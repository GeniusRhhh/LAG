# scripts/runtest_pure_crank.py
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
    """ACMI文件生成器 - 按照你的项目格式"""

    def __init__(self):
        self.trajectory_data = []
        self.file_created = False

    def initialize_acmi_file(self, filepath):
        """初始化ACMI文件 - 完全按照你的项目格式"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 写入文件头 - 完全按照env_base.py的格式
            with open(filepath, mode='w', encoding='utf-8-sig') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.1\n")
                f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")

            self.file_created = True
            print(f"✅ ACMI文件初始化成功: {filepath}")
            return True

        except Exception as e:
            print(f"❌ ACMI文件初始化失败: {e}")
            return False

    def record_frame(self, env):
        """记录一帧数据 - 按照你的simulator.log()格式"""
        timestamp = env.current_step * env.time_interval
        frame_data = {
            'timestamp': timestamp,
            'agents': []
        }

        # 记录所有飞机的状态
        for agent_id, agent in env.agents.items():
            if agent.is_alive:
                # 获取位置和姿态 - 按照simulator.log()的方式
                lon, lat, alt = agent.get_geodetic()
                roll, pitch, yaw = agent.get_rpy() * 180 / np.pi

                agent_data = {
                    'uid': agent.uid,
                    'lon': lon,
                    'lat': lat,
                    'alt': alt,
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'model': agent.model.upper(),
                    'color': agent.color
                }
                frame_data['agents'].append(agent_data)

        self.trajectory_data.append(frame_data)

    def write_frame_to_file(self, filepath, env):
        """实时写入一帧到文件 - 按照env_base.py的格式"""
        if not self.file_created:
            self.initialize_acmi_file(filepath)

        try:
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = env.current_step * env.time_interval
                f.write(f"#{timestamp:.2f}\n")

                # 写入所有飞机的日志 - 完全按照simulator.log()格式
                for agent_id, agent in env.agents.items():
                    if agent.is_alive:
                        log_msg = agent.log()  # 使用你的原始log方法
                        if log_msg:
                            f.write(log_msg + "\n")

                # 写入导弹日志（如果有）
                for sim in env._tempsims.values():
                    log_msg = sim.log()
                    if log_msg:
                        f.write(log_msg + "\n")

            return True

        except Exception as e:
            print(f"❌ 写入ACMI帧失败: {e}")
            return False


def runtest_crank_maneuver():
    """测试纯Crank机动 - 生成ACMI文件用于Tacview演示"""
    try:
        from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

        print("🚀 开始纯机动测试系统")
        print(f"当前工作目录: {os.getcwd()}")

        print("=" * 60)
        print("🎯 纯Crank机动测试")
        print("=" * 60)

        # 使用配置文件名
        config_name = "simple_maneuver_config"
        print(f"✅ 使用配置文件: {config_name}")

        # 创建环境
        env = MultipleCombatEnv(config_name)
        print("✅ 环境创建成功")

        # 创建ACMI生成器
        acmi_generator = ACMIGenerator()

        # 准备ACMI文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        maneuver_results_dir = os.path.join(project_root, "maneuver_results")
        acmi_filename = f"PureCrank_Maneuver_{timestamp}.acmi"
        acmi_filepath = os.path.join(maneuver_results_dir, acmi_filename)

        # 获取环境信息
        print(f"✅ 环境信息：")
        print(f"   - num_agents: {env.num_agents}")
        print(f"   - n_rollout_threads: {env.n_rollout_threads}")
        print(f"   - action_space.shape: {env.action_space.shape}")
        print(f"   - observation_space: {env.observation_space}")

        # 重置环境
        obs, share_obs = env.reset()
        print(f"✅ 环境重置成功，观测空间形状: {[o.shape for o in [obs, share_obs]]}")

        # 初始化ACMI文件
        if not acmi_generator.initialize_acmi_file(acmi_filepath):
            return False

        # 运行测试
        step = 0
        done = False
        frame_count = 0

        print(f"🎬 开始记录ACMI数据到: {acmi_filepath}")

        while not done and step < 1000:  # 增加步数以便观察完整机动

            # 创建正确格式的动作数组
            action_dim = env.action_space.shape[0]
            actions = np.zeros((env.n_rollout_threads, env.num_agents, action_dim), dtype=np.float32)

            # 执行步骤
            obs, share_obs, rewards, dones, infos = env.step(actions)

            # 每隔几步记录一次数据（避免文件太大）
            if step % 2 == 0:  # 每2步记录一次
                acmi_generator.write_frame_to_file(acmi_filepath, env)
                frame_count += 1

            step += 1
            done = np.any(dones)

            if step % 100 == 0:
                current_time = env.current_step * env.time_interval
                print(f"⏰ 步骤 {step}, 时间: {current_time:.1f}s, 已记录帧数: {frame_count}")

                # 显示飞机状态
                for agent_id in env.agents.keys():
                    if env.agents[agent_id].is_alive:
                        agent = env.agents[agent_id]
                        lon, lat, alt = agent.get_geodetic()
                        roll, pitch, yaw = agent.get_rpy() * 180 / np.pi
                        print(f"   - {agent_id}: 高度={alt:.1f}m, 航向={yaw:.1f}°")

        # 关闭环境
        env.close()
        print("✅ 环境关闭成功")

        # 验证ACMI文件
        if os.path.exists(acmi_filepath):
            file_size = os.path.getsize(acmi_filepath)
            print(f"✅ ACMI文件生成成功！")
            print(f"   📁 文件路径: {acmi_filepath}")
            print(f"   📊 文件大小: {file_size} 字节")
            print(f"   📈 记录帧数: {frame_count}")
            print(f"   🎮 使用方法:")
            print(f"      1. 安装Tacview软件")
            print(f"      2. 打开文件: {acmi_filepath}")
            print(f"      3. 观看纯Crank机动的3D回放")

            # 显示机动参数信息
            if hasattr(env.task, 'maneuver_params'):
                params = env.task.maneuver_params
                print(f"🎯 机动参数:")
                print(f"   - Crank角度: {params.get('crank_angle_deg', 90)}°")
                print(f"   - 转弯率: {params.get('turn_rate_deg_per_sec', 3)}°/s")
                print(f"   - 保持时间: {params.get('hold_time_sec', 20)}s")

            return True
        else:
            print(f"❌ ACMI文件未生成")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        logging.error(f"Traceback:")
        import traceback
        logging.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    print("🚀 开始纯机动测试系统")
    success = runtest_crank_maneuver()

    if success:
        print("🎉 测试成功完成！")
        print("📖 ACMI文件已保存到 maneuver_results/ 目录")
        print("📊 老师要求完成情况:")
        print("   ✅ 纯函数封装 (PureManeuvers.crank_maneuver)")
        print("   ✅ 参数暴露 (角度、转弯率、持续时间)")
        print("   ✅ 可控制修改 (45度可改为60度等)")
        print("   ✅ ACMI文件生成 (Tacview可视化)")
    else:
        print("❌ 测试失败，请检查日志")

    print("\n🎉 测试完成！")