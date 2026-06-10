import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# 加入项目根路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 导入环境和控制器
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import get_root_dir

# 创建环境
def make_env():
    config_path = os.path.abspath("vsBaseline_revised")
    if not config_path.endswith(".yaml"):
        config_path += ".yaml"

    env = MultipleCombatEnv(config_path)
    print("🛠 当前加载的任务类型：", env.config.task)
    env.reset()
    return env


# 执行俯冲任务
def run_rule_dive(agent, baseline_actor, target_dive_km=2.0, max_steps=20000):
    log = []
    acmi_log = []

    # 控制器参数
    k_p = 1.2  # 比例增益（越大越快）
    k_i = 0.00  # 积分增益（累积误差的影响）
    k_d = 0.8  # 微分增益（抑制震荡）
    deadband = 1  # 死区范围：误差小于这个就不动
    prev_error = 0
    integral_error = 0

    for step in range(max_steps):
        # 获取当前俯仰角（rad -> deg）
        pitch = agent.get_property_value(c.attitude_pitch_rad)
        pitch_deg = np.rad2deg(pitch)

        # 设置目标角度
        target_pitch_deg = -30.0

        # 计算误差与变化量
        error = target_pitch_deg - pitch_deg
        delta_error = error - prev_error
        prev_error = error
        # 累积误差
        integral_error += error * agent.dt

        # 生成舵面档位
        if abs(error) < deadband:
            elevator_index = 20  # 死区内不调整，保持中性
        else:
            elevator_index = int(round(20 - k_p * error - k_i * integral_error - k_d * delta_error))
            elevator_index = np.clip(elevator_index, 0, 40)

        # 设置舵面动作：保持中性副翼和方向舵，最大俯冲、低油门
        aileron_index = 20  # 中性副翼（0.0）
        #elevator_index = 40  # 最大俯冲（-1.0）
        rudder_index = 20  # 中性方向舵（0.0）
        throttle_index = 10  # 低油门（约 0.57）

        # 映射为控制量（与训练阶段一致）
        aileron = aileron_index / 20.0 - 1.0
        elevator = elevator_index / 20.0 - 1.0
        rudder = rudder_index / 20.0 - 1.0
        throttle = throttle_index / 58.0 + 0.4

        # 设置控制输入
        agent.set_property_value(c.fcs_aileron_cmd_norm, aileron)
        agent.set_property_value(c.fcs_elevator_cmd_norm, elevator)
        agent.set_property_value(c.fcs_rudder_cmd_norm, rudder)
        agent.set_property_value(c.fcs_throttle_cmd_norm, throttle)

        # 推进一帧仿真
        agent.run()
        agent._update_properties()
        acmi_line = agent.log()
        acmi_log.append(acmi_line)  # 保存每一帧状态

        # 记录状态
        alt = agent.get_property_value(c.position_h_sl_m) / 1000
        pitch = agent.get_property_value(c.attitude_pitch_rad)
        print(f"当前高度: {alt:.2f} km, 俯仰角: {np.rad2deg(pitch):.1f}°")
        log.append((step, alt, pitch))
        print(f"[DEBUG] Agent {agent_id} ego_pos: {env.agents[agent_id].get_position()}")

        # # 终止条件
        # if np.rad2deg(pitch) <= -80:
        #     print("达到俯仰角，结束控制")
        #     break
        if(alt<3):
            break

    return log, acmi_log


if __name__ == "__main__":
    # ✅ 初始化环境 & 获取飞机对象
    env = make_env()
    agent_id = list(env.agents.keys())[0]
    agent = env.agents[agent_id]

    # ✅ 加载已训练的底层控制器
    controller_path = os.path.join(get_root_dir(), "model/baseline_model.pt")
    baseline_actor = BaselineActor()
    baseline_actor.load_state_dict(torch.load(controller_path, map_location="cpu", weights_only=True))
    baseline_actor.eval()

    print("🚀 开始俯冲测试：目标下降 2km")
    log, acmi_log = run_rule_dive(agent, baseline_actor)

    # for step, alt, pitch in log[:10]:
    #     print(f"Step {step:3d} | Altitude: {alt:.2f} km | Pitch: {np.rad2deg(pitch):.1f}°")

    # ✅ 可视化结果
    steps = [s for s, _, _ in log]
    altitudes = [z for _, z, _ in log]
    pitch_deg = [np.rad2deg(p) for _, _, p in log]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(steps, altitudes)
    plt.xlabel("Step")
    plt.ylabel("Altitude (km)")
    plt.title("Altitude over Time")

    plt.subplot(1, 2, 2)
    plt.plot(steps, pitch_deg)
    plt.xlabel("Step")
    plt.ylabel("Pitch (deg)")
    plt.title("Pitch Angle over Time")

    plt.tight_layout()
    plt.show()

    # ✅ 生成 TacView .acmi 文件（用于轨迹回放）
    acmi_lines = []

    # ✅ 添加 TacView 文件头（固定格式）
    acmi_lines.append("FileType=text/acmi/tacview")
    acmi_lines.append("FileVersion=2.1")
    acmi_lines.append("")  # 分隔符
    acmi_lines.append("0,ReferenceTime=2020-01-01T00:00:00Z")

    # ✅ 添加对象定义（一次）
    uid = agent.uid  # 如 A0100
    color = agent.color  # 如 Red
    acmi_lines.append(f"{uid},Type=Aircraft,Name=Agent-0,Color={color}")

    for step, acmi_line in enumerate(acmi_log):  # ✅ 不再用 agent.log()
        timestamp = step * agent.dt
        acmi_lines.append(f"#{timestamp:.2f}")
        acmi_lines.append(acmi_line)

    # ✅ 保存到当前目录
    acmi_path = os.path.join(os.getcwd(), "dive_output.acmi")
    with open(acmi_path, "w") as f:
        for line in acmi_lines:
            f.write(line + "\n")

    print(f"📁 TacView 路径：{acmi_path}")
