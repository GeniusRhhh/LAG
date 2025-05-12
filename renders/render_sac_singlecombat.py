import numpy as np
import torch
import logging
from datetime import datetime
import os

# 假设你有 SingleCombatEnv 环境
from envs.JSBSim.envs import SingleCombatEnv  # 这里使用 SingleCombatEnv 环境

# 你的 SAC Actor
from algorithms.sac.sac_actor import ActorNet

logging.basicConfig(level=logging.INFO)

# def main():
#     # 1) 创建环境 (双智能体)
#     env_name = "1v1/NoWeapon/Selfplay"
#     scenario = "1v1_combat"
#     env = SingleCombatEnv(env_name)  # 使用 1v1 combat 环境
#     env.seed(42)
#
#     # 打印观测空间的形状
#     print("Observation space:", env.observation_space)
#     print("Observation space shape:", env.observation_space.shape)
#
#     # 2) 加载模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 你要先构建 ActorNet 的网络结构参数(和训练时一致)
#     class DummyArgs:
#         hidden_size = '128 128'
#         act_hidden_size = '128 128'
#         activation_id = 1
#         use_feature_normalization = False
#         init_alpha = 0.2
#         gamma = 0.99
#         tau = 0.005
#         gain = 0.01
#
#     args = DummyArgs()
#
#     # 构造 ActorNet
#     obs_space = env.observation_space
#     act_space = env.action_space
#     actor0 = ActorNet(args, obs_space, act_space, device=device)  # agent0
#     actor1 = ActorNet(args, obs_space, act_space, device=device)  # agent1
#
#     # 加载权重
#     ego_run_dir = "../scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/sac/v0201/run57"
#     checkpoint0 = torch.load(ego_run_dir + f"/agent0_sac_10000.pt", map_location=device)
#     checkpoint1 = torch.load(ego_run_dir + f"/agent1_sac_10000.pt", map_location=device)
#     actor0.load_state_dict(checkpoint0["actor"])
#     actor1.load_state_dict(checkpoint1["actor"])
#     actor0.eval()
#     actor1.eval()
#
#     # 3) 开始推理并渲染
#     obs = env.reset()
#
#     # 生成 acmi 文件路径
#     current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
#     acmi_path = f"{env_name}_{scenario}_{current_time}.acmi"
#
#     # 确保目标文件夹存在
#     acmi_dir = os.path.dirname(acmi_path)
#     if not os.path.exists(acmi_dir):
#         os.makedirs(acmi_dir)  # 创建文件夹
#
#     # 开启渲染
#     env.render(mode='txt', filepath=acmi_path)
#
#     done = False
#     episode_reward0 = 0.0
#     episode_reward1 = 0.0
#
#     while not done:
#         # SAC actor 输入 shape=(1, obs_dim)
#         obs0_tensor = torch.as_tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)  # agent0
#         obs1_tensor = torch.as_tensor(obs[1], dtype=torch.float32, device=device).unsqueeze(0)  # agent1
#
#         # forward
#         with torch.no_grad():
#             actions0, _ = actor0(obs0_tensor, deterministic=True)  # agent0
#             actions1, _ = actor1(obs1_tensor, deterministic=True)  # agent1
#             print(f"actions0 shape0: {actions0.shape}")
#             print(f"actions1 shape1: {actions1.shape}")
#
#         actions0_np = actions0.cpu().numpy().squeeze(0)  # => (act_dim,)
#         actions1_np = actions1.cpu().numpy().squeeze(0)  # => (act_dim,)
#
#         print(f"actions0_np shape_np: {actions0_np.shape}")
#         print(f"actions1_np shape_np: {actions1_np.shape}")
#
#         # 合并为 (2, 4) 的数组
#         actions = np.vstack([actions0_np, actions1_np])  # 使用 vstack 而不是 array
#
#         # 打印合并后的 shape，确保它是 (2, 4)
#         print(f"actions shape after stacking: {actions.shape}")
#
#         # 传递合并后的动作给 env.step
#         next_obs, rewards, dones, info = env.step(actions)
#         print(f"render Next observation shape: {next_obs.shape}")
#
#         episode_reward0 += rewards[0].sum()  # agent0
#         episode_reward1 += rewards[1].sum()  # agent1
#
#         # 每步渲染
#         env.render(mode='txt', filepath=acmi_path)
#
#         # 更新观察数据
#         obs = next_obs
#         done = dones.all()
#
#     print(f"Episode finished, total reward agent0={episode_reward0}, total reward agent1={episode_reward1}")
#     print(f"ACMI file saved to: {acmi_path}")
#
# if __name__ == "__main__":
#     main()


"""
1v1 SAC 推理 + .acmi 渲染示例
文件: sac_inference_render_1v1.py
"""
import numpy as np
import torch
import logging
from datetime import datetime
import os
from envs.JSBSim.envs import SingleCombatEnv
from algorithms.sac.sac_actor import ActorNet
from gymnasium.spaces import Box

logging.basicConfig(level=logging.INFO)

def main():
    # 1) 创建 1v1 环境
    env_name = "1v1/NoWeapon/Selfplay"
    scenario = "1v1/NoWeapon/Selfplay"
    env = SingleCombatEnv(scenario)
    env.seed(42)

    # 打印环境空间信息
    print("Observation space:", env.observation_space)
    print("Observation space shape:", env.observation_space.shape)
    print("Action space:", env.action_space)
    print("Action space shape:", env.action_space.shape)

    # 2) 提取单智能体的空间
    single_obs_space = Box(
        low=env.observation_space.low[0],
        high=env.observation_space.high[0],
        dtype=env.observation_space.dtype
    )
    single_act_space = Box(
        low=env.action_space.low[0],
        high=env.action_space.high[0],
        dtype=env.action_space.dtype
    )

    # 3) 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DummyArgs:
        hidden_size = '512 512'
        act_hidden_size = '512 512'
        activation_id = 1
        use_feature_normalization = False
        init_alpha = 0.2
        gamma = 0.99
        tau = 0.005
        gain = 0.01

    args = DummyArgs()

    # 使用单智能体空间构造 ActorNet（模型结构与训练时一致）
    actor0 = ActorNet(args, single_obs_space, single_act_space, device=device)
    actor1 = ActorNet(args, single_obs_space, single_act_space, device=device)

    # 模型路径（根据实际情况修改）
    run_dir0 = "../scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/sac/02152307"
    run_dir1 = "../scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/sac/02152307"
    checkpoint0 = torch.load(run_dir0 + "/sac_ep221.pt", map_location=device)
    checkpoint1 = torch.load(run_dir1 + "/sac_ep1.pt", map_location=device)
    actor0.load_state_dict(checkpoint0["actor"])
    actor1.load_state_dict(checkpoint1["actor"])
    actor0.eval()
    actor1.eval()
    # 4) 开始推理并渲染
    obs = env.reset()  # 返回形状 (2,15)
    obs = np.array(obs, dtype=np.float32)
    logging.info("Initial obs shape: %s", obs.shape)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    acmi_path = f"{env_name}_{scenario}_{current_time}.acmi"

    # 确保目标文件夹存在
    acmi_dir = os.path.dirname(acmi_path)
    if not os.path.exists(acmi_dir):
        os.makedirs(acmi_dir)

    env.render(mode='txt', filepath=acmi_path)

    done = False
    episode_reward0 = 0.0
    episode_reward1 = 0.0

    while not done:
        # 分离单个智能体的观测
        obs0 = obs[0]  # shape (15,)
        obs1 = obs[1]  # shape (15,)

        obs_tensor0 = torch.as_tensor(obs0, dtype=torch.float32, device=device).unsqueeze(0)
        obs_tensor1 = torch.as_tensor(obs1, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action0, _ = actor0(obs_tensor0, deterministic=True)
            action1, _ = actor1(obs_tensor1, deterministic=True)

        action0_np = action0.cpu().numpy().squeeze(0)
        action1_np = action1.cpu().numpy().squeeze(0)

        actions = [action0_np, action1_np]
        next_obs, reward, done, info = env.step(actions)
        reward = np.array(reward)
        print("Reward shape:", reward.shape)  # 调试信息

        # 如果 reward 形状为 (1, 2)，则取第一行
        if reward.shape[0] == 1:
            r = reward[0]
        else:
            r = reward
        episode_reward0 += r[0]
        episode_reward1 += r[1]

        env.render(mode='txt', filepath=acmi_path)
        obs = np.array(next_obs, dtype=np.float32)
        if isinstance(done, (list, np.ndarray)):
            done = np.all(done)

    print(f"Episode finished, total reward: agent0={episode_reward0}, agent1={episode_reward1}")
    print(f"ACMI file saved to: {acmi_path}")

if __name__ == "__main__":
    main()
