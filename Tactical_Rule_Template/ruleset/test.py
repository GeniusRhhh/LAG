import os
import numpy as np
import torch
from envs.JSBSim.utils.utils import parse_config
from config import get_config
from envs.JSBSim.core.catalog import Catalog as c

# 导入环境与策略相关模块
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.core.radar import RadarModel
from algorithms.ppo.ppo_policy import PPOPolicy
from envs.JSBSim.utils.utils import parse_config
# 导入自定义战术模版（如Crank机动）
import tactical_templates
import tactical_templates_v2

def main():
    # === 设定运行设备：使用GPU（如果可用），否则使用CPU ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config_path = os.path.join(project_root, "envs/JSBSim/configs/1v1/ShootMissile/HierarchySelfplay.yaml")

    # 1. 加载 YAML 配置（环境类用这个）
    config = parse_config(config_path)

    # 2. 加载训练用参数（策略类用这个）
    parser = get_config()
    train_args = parser.parse_args([])  # 不加 yaml_args，保持默认训练参数

    # 用 config 创建环境（包含 aircraft_configs 等）
    env = SingleCombatEnv(config_path)
    env.reset()

    # 用 train_args 初始化 PPOPolicy
    policy = PPOPolicy(
        args=train_args,
        obs_space=env.observation_space,
        act_space=env.action_space,
        device=device
    )

    # === 加载 actor 模型权重 ===
    actor_path = os.path.join(
        project_root,
         "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/shoot_missile_train/run13/actor_latest.pt"
    )
    policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
    policy.actor.to(device)     # 模型放入 GPU
    policy.actor.eval()         # 设置为推理模式

    # === 创建雷达模型供Crank使用 ===
    radar = RadarModel()

    # === 初始化RNN状态（仅对使用RNN的actor有效） ===
    rnn_state = torch.zeros(1, 1, 128).to(device)

    acmi_log = []

    # ★ 每个阵营一份共享上下文（回合开始时创建；不要放在循环里反复重建）
    ctx_A = {}
    ctx_B = {}
    for step in range(4000):
        # === 获取所有 agent 的观测 ===
        obs = env.get_obs()
        actions = {}

        # === 遍历每个智能体 ===
        for i, (agent_id, agent) in enumerate(env.agents.items()):
            if agent_id.startswith("A"):
                # 我方 Crank，走高层动作 -> 归一化
                # high_level_action = tactical_templates_v2.Launch_and_Leave(env, agent_id,radar)
                if(step<300):
                    high_level_action = tactical_templates_v2._dispatch_template("T1_LAL", env, agent_id, radar, ctx=ctx_A)
                elif(300<=step<600):
                    high_level_action = tactical_templates_v2._dispatch_template("T3_DEF", env, agent_id, radar, ctx=ctx_A)
                elif (600 <= step):
                    high_level_action = tactical_templates_v2._dispatch_template("T2_LAD", env, agent_id, radar, ctx=ctx_A)

                actions[agent_id] = high_level_action

                # print(
                #     f"[{agent_id}] 红方输出: {high_level_action}"
                # )
                # status = env.agents[agent_id]._AircraftSimulator__status
                # print(f"[{status}] 红方飞机状态")
                # if env.agents[agent_id].is_alive:
                #     print(f"[{env.agents[agent_id].uid}] 红方飞机存活")
                # else:
                #     print(f"[{env.agents[agent_id].uid}] 红方飞机已击毁")
                # pos = agent.get_position()  # 返回 (x, y, z)
                # print(f"[{agent_id}] 位置: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

            elif agent_id.startswith("B"):
                # 敌方 RL，走连续动作，直接使用
                obs_tensor = torch.tensor(obs[agent_id], dtype=torch.float32).unsqueeze(0).to(device)
                masks = torch.ones(1, 1).to(device)
                act_tensor, _, rnn_state = policy.actor(obs_tensor, rnn_state, masks)
                act = act_tensor.squeeze(0).detach().cpu().numpy()
                actions[agent_id] = act  #  不再 normalize_action

                ego = env.agents[agent_id]
                radar = RadarModel()  # 可提前放外面，避免重复构造
                ego_pos = np.array(ego.get_position())
                ego_vel = np.array(ego.get_velocity())
                ego_yaw = ego.get_property_value(c.attitude_heading_true_rad)
                ego_pitch = ego.get_property_value(c.attitude_pitch_rad)

                # for enemy in ego.enemies:
                #     tar_pos = np.array(enemy.get_position())
                #     tar_vel = np.array(enemy.get_velocity())
                #     is_locked, info = radar.detect(
                #         ego_pos, ego_vel, ego_yaw, ego_pitch,
                #         tar_pos, tar_vel
                #     )
                # in_doppler_blind = info["in_doppler_blind"]
                # print(f"目标是否在多普勒盲区？{in_doppler_blind}")
                # if step<=300:in_doppler_blind=True
                # if in_doppler_blind:
                #     print("⚠️ 雷达无法锁定：目标速度与我机径向速度太小！")
                # else:
                #     print("✅ 雷达可正常跟踪目标。")
                #
                # #在多普勒盲区不让发射导弹
                # if in_doppler_blind:
                #     # 假设环境里“act[-1] > 阈值”才会真正发射导弹，
                #     # 这里把它改成明显的“不开火”值（-1 或 0 都可以，一般更保险用 -1）
                #     act[-1] = 0
                # actions[agent_id] = act
                #     print(
                #         f"[{agent_id}] -> [{enemy.uid}] Vr = {info['Vr']:.2f} m/s | Doppler盲区: {info['in_doppler_blind']}")

                # print(
                #     f"[{agent_id}] 蓝方 PPO 输出: {act.tolist()}"
                # )
                # pos = agent.get_position()  # 返回 (x, y, z)
                # print(f"[{agent_id}] 位置: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

        # === 打包动作，推进一步仿真 ===
        ordered_actions = [actions[agent_id] for agent_id in env.agents.keys()]
        obs, reward, done, info = env.step(ordered_actions)

        # 每步记录 acmi 状态
        for agent in env.agents.values():
            acmi_line = agent.log()
            acmi_log.append((step, agent.uid, acmi_line, agent.dt, agent.color))

        env.render(mode='txt', filepath="C:/Users/ww/Desktop/simulation_log.acmi")

    # # === 打印结果 ===
    # print(" 仿真推进完成，奖励如下：")
    # for agent_id, rew in zip(env.agents.keys(), reward):
    #     print(f"[{agent_id}] reward: {rew[0]:.4f}")

if __name__ == "__main__":
    main()