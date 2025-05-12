import numpy as np
import torch
import logging
from datetime import datetime

# 假设你有 SingleCombatEnv 环境
from envs.JSBSim.envs import SingleCombatEnv

# 你的 SAC Actor
from algorithms.sac.sac_actor import ActorNet

logging.basicConfig(level=logging.INFO)


def main():
    # 1) 创建 1v1 对战环境
    env_name = "SingleCombat"
    scenario = "1v1_combat"
    env = SingleCombatEnv(scenario)
    env.seed(42)

    # 2) 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建 DummyArgs 类来定义网络结构参数
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

    # 构造 ActorNet
    obs_space = env.observation_space
    act_space = env.action_space
    actor = ActorNet(args, obs_space, act_space, device=device)

    # 加载权重
    ego_run_dir = "../scripts/results/SingleCombat/1v1_combat/sac/v0131/01312018"
    checkpoint = torch.load(ego_run_dir + f"/sac_4920000.pt", map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    # 3) 开始推理并渲染
    obs = env.reset()  # 获取初始状态
    # 生成 acmi 文件路径
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    acmi_path = f"{env_name}_{scenario}_{current_time}.acmi"

    # 开启渲染
    env.render(mode='txt', filepath=acmi_path)

    done = False
    episode_reward = 0.0

    while not done:
        # 获取两个智能体的观察值
        obs_tensor_1 = torch.as_tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)  # 智能体1
        obs_tensor_2 = torch.as_tensor(obs[1], dtype=torch.float32, device=device).unsqueeze(0)  # 智能体2

        # 根据 SAC 推理生成动作
        with torch.no_grad():
            action_1, _ = actor(obs_tensor_1, deterministic=True)  # 对智能体1推理
            action_2, _ = actor(obs_tensor_2, deterministic=True)  # 对智能体2推理

        action_1_np = action_1.cpu().numpy().squeeze(0)  # 获取动作
        action_2_np = action_2.cpu().numpy().squeeze(0)

        # 环境步进
        next_obs, reward, done, info = env.step([action_1_np, action_2_np])  # 双智能体环境步进
        episode_reward += reward.sum()  # 总奖励

        # 每步渲染
        env.render(mode='txt', filepath=acmi_path)

        # 更新观察
        obs = next_obs

    print(f"Episode finished, total reward={episode_reward}")
    print(f"ACMI file saved to: {acmi_path}")


if __name__ == "__main__":
    main()
