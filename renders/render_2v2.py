import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import time
import logging
import sys
import os
from contextlib import redirect_stdout


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
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True


def _t2n(x):
    return x.detach().cpu().numpy()


# 参数设置
num_agents = 4
render = True
ego_policy_index = 50
enm_policy_index =49
episode_rewards = 0

ego_run_dir = "../scripts/results/MultipleCombat/2v2/ShootMissile/HierarchySelfplay/mappo/v1/run181"
enm_run_dir = "../scripts/results/MultipleCombat/2v2/ShootMissile/HierarchySelfplay/mappo/v1/run181"

# 提取run编号
run_number = ego_run_dir.split('/')[-1]  # 获取 "run173"

# 创建新的文件命名格式：runxxx_xx_xx
file_prefix = f"{run_number}_{ego_policy_index:02d}_{enm_policy_index:02d}"

# 设置日志文件
log_filename = f"{file_prefix}.txt"


# 配置logging，让DEBUG日志同时输出到控制台和文件
def setup_logging(log_filename):
    # 获取根logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 清除已有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建formatter
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# 设置日志系统
logger = setup_logging(log_filename)

experiment_name = ego_run_dir.split('/')[-4]
env = MultipleCombatEnv("2v2/ShootMissile/HierarchySelfplay")
env.seed(0)

args = Args()
ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))

ego_policy.eval()
enm_policy.eval()

ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_episode_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_episode_{enm_policy_index}.pt"))

print("Start render")
print(f"文件命名格式: {file_prefix}")
print(f"我方模型序号: {ego_policy_index}")
print(f"敌方模型序号: {enm_policy_index}")
print(f"训练文件编号: {run_number}")

obs, _ = env.reset()

if render:
    env.render(mode='txt', filepath=f'{file_prefix}.txt.acmi')

ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))

enm_obs = obs[num_agents // 2:, :]
ego_obs = obs[:num_agents // 2, :]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)

while True:
    start = time.time()
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    end = time.time()
    # print(f"NN forward time: {end-start}")

    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)

    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)

    actions = np.concatenate((ego_actions, enm_actions), axis=0)

    # Observe reward and next obs
    start = time.time()
    obs, _, rewards, dones, infos = env.step(actions)
    end = time.time()
    # print(f"Env step time: {end-start}")

    rewards = rewards[:num_agents // 2, ...]
    episode_rewards += rewards

    if render:
        env.render(mode='txt', filepath=f'{file_prefix}.txt.acmi')

    if dones.all():
        print("游戏结束信息:")
        print(infos)
        break

    enm_obs = obs[num_agents // 2:, ...]
    ego_obs = obs[:num_agents // 2, ...]

print("最终奖励:")
print(episode_rewards)

print(f"渲染完成！生成文件:")
print(f"- ACMI文件: {file_prefix}.txt.acmi")
print(f"- 日志文件: {log_filename}")