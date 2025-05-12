import os
import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.DEBUG)

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cuda'))
        self.use_prior = True

def _t2n(x):
    return x.detach().cpu().numpy()

# 环境和参数设置
num_agents = 2
render = True
ego_policy_index = 150 # 我方模型索引
episode_rewards_ego = 0
episode_rewards_enm = 0
ego_run_dir = "../scripts/results/SingleCombat/1v1/NoWeapon/vsBaseline/ppo/v1/02281007pursue"
experiment_name = ego_run_dir.split('/')[-4]

env_name = "1v1/NoWeapon/vsBaseline"
env = SingleCombatEnv(env_name)
env.seed(0)
args = Args()

# 确保使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建并加载我方 PPO 模型
ego_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
ego_policy.eval()
ego_policy.load_state_dict(torch.load(f"{ego_run_dir}/actor_{ego_policy_index}.pt", map_location=device))

print("Start render")
obs = env.reset()

# 临时文件路径
# temp_filepath = f'1v1_vsBaselineManeuver_{ego_policy_index}_0_0.00_0.00.txt.acmi'
temp_filepath = f'1v1_vsBaselinePursue_{ego_policy_index}_0_0.00_0.00.txt.acmi'

if render:
    env.render(mode='txt', filepath=temp_filepath)

# 初始化 RNN 状态
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))

# 分割观察值（我方和敌方）
enm_obs = obs[num_agents // 2:, :]
ego_obs = obs[:num_agents // 2, :]

while True:
    # 我方动作生成（使用 PPO 模型）
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)

    # 敌方动作由环境自动生成（依赖 use_baseline=True 和 baseline_type='maneuver'）
    actions = np.concatenate((ego_actions, np.zeros((1, 4))), axis=0)  # 占位符，环境会覆盖敌方动作

    # 执行一步并获取反馈
    obs, rewards, dones, infos = env.step(actions)

    print(f"Actions: {actions}, Rewards: {rewards}")

    # 分割奖励
    rewards_ego = rewards[:num_agents // 2, ...]
    rewards_enm = rewards[num_agents // 2:, ...]

    # 累积奖励
    episode_rewards_ego += rewards_ego
    episode_rewards_enm += rewards_enm

    # 渲染
    if render:
        env.render(mode='txt', filepath=temp_filepath)

    # 检查是否结束
    if dones.all():
        print(infos)
        break

    # 更新观察值
    enm_obs = obs[num_agents // 2:, ...]
    ego_obs = obs[:num_agents // 2, ...]

# 输出最终奖励
reward_diff = episode_rewards_ego.sum() - episode_rewards_enm.sum()
print(f"Final episode reward of ego agent: {episode_rewards_ego.sum():.2f}")
print(f"Final episode reward of enemy agent: {episode_rewards_enm.sum():.2f}")
print(f"Reward difference (Ego - Enemy): {reward_diff:.2f}")

# 更新文件名并重命名
# final_filepath = f'1v1_vsBaselineManeuver_{ego_policy_index}_0_{episode_rewards_ego.sum():.2f}_{episode_rewards_enm.sum():.2f}.txt.acmi'
final_filepath = f'1v1_vsBaselinePursue_{ego_policy_index}_0_{episode_rewards_ego.sum():.2f}_{episode_rewards_enm.sum():.2f}.txt.acmi'
os.rename(temp_filepath, final_filepath)