import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
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
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True

def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 2
render = True
ego_policy_index = 1040
enm_policy_index = 0
episode_rewards = 0
ego_run_dir = "../scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/03031629"
enm_run_dir = "../scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/03031629"
experiment_name = ego_run_dir.split('/')[-4]
# 打印动作空间和观察空间

env = SingleCombatEnv("1v1/NoWeapon/HierarchySelfplay")
env.seed(0)
args = Args()
logging.info(f"Action space: {env.action_space}")
logging.info(f"Observation space: {env.observation_space}")
ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))

print("Start render")
obs = env.reset()

# 打印观察空间和初始形状
logging.info(f"Observation space: {env.observation_space}")
logging.info(f"Initial obs shape: {obs.shape}")

# 临时强制设置初始高度
for agent_id, sim in env.agents.items():
    sim.set_property_value(c.position_h_sl_ft, 20000.0)
    logging.info(f"Set {agent_id} initial altitude: {sim.get_property_value(c.position_h_sl_ft)} ft")
obs = env.reset()  # 再次重置以应用高度

if render:
    env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')

# 打印初始高度（动态调整索引）
ego_obs = obs[:num_agents // 2, :]
enm_obs = obs[num_agents // 2:, :]
obs_dim = ego_obs.shape[1]
logging.info(f"Ego obs shape: {ego_obs.shape}, Enm obs shape: {enm_obs.shape}")

# 假设高度索引（根据 obs_dim 动态调整）
ego_altitude_idx = 3 if obs_dim > 3 else (obs_dim - 1)  # 默认取 obs[3] 或最后一个
enm_altitude_idx = 15 if obs_dim > 15 else (obs_dim - 1)  # 默认取 obs[15] 或最后一个
ego_altitude_m = ego_obs[0, ego_altitude_idx] * 1000 + 6000  # 假设单位 1000m，中心化 6000m
ego_altitude_ft = ego_altitude_m * 3.28084
enm_altitude_m = enm_obs[0, enm_altitude_idx] * 1000 + 6000
enm_altitude_ft = enm_altitude_m * 3.28084
logging.info(f"Initial Ego Altitude: {ego_altitude_ft:.2f} ft (idx={ego_altitude_idx})")
logging.info(f"Initial Enemy Altitude: {enm_altitude_ft:.2f} ft (idx={enm_altitude_idx})")
logging.info(f"Raw Ego obs: {ego_obs[0]}, Raw Enm obs: {enm_obs[0]}")

ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)

while True:
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)
    actions = np.concatenate((ego_actions, enm_actions), axis=0)
    # Obser reward and next obs
    obs, rewards, dones, infos = env.step(actions)
    rewards = rewards[:num_agents // 2, ...]
    episode_rewards += rewards
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
    if dones.all():
        print(infos)
        break
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step:{env.current_step}, bloods:{bloods}")

    # 打印当前高度
    ego_obs = obs[:num_agents // 2, :]
    enm_obs = obs[num_agents // 2:, :]
    ego_altitude_m = ego_obs[0, ego_altitude_idx] * 1000 + 6000
    ego_altitude_ft = ego_altitude_m * 3.28084
    enm_altitude_m = enm_obs[0, enm_altitude_idx] * 1000 + 6000
    enm_altitude_ft = enm_altitude_m * 3.28084
    logging.info(f"Step {env.current_step} - Ego Altitude: {ego_altitude_ft:.2f} ft")
    logging.info(f"Step {env.current_step} - Enemy Altitude: {enm_altitude_ft:.2f} ft")

print(episode_rewards)