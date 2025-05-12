import os
import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv
from envs.JSBSim.tasks.singlecombat_task import PursueAgent, StraightFlyAgent
from envs.JSBSim.core.catalog import JsbsimCatalog as Catalog
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class Args:
    def __init__(self):
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = False
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True
        self.cuda = False
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.alpha_lr = 3e-4
        self.init_alpha = 0.2

def _t2n(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().squeeze()
    elif isinstance(x, (tuple, list)):
        return _t2n(x[0])
    return np.array(x).squeeze()

num_agents = 2
render = True
ego_policy_index = 198
episode_rewards_ego = 0
episode_rewards_enm = 0
ego_run_dir = "../scripts/results/SingleCombat/1v1/NoWeapon/vsBaseline/sac/v1/05072311"
experiment_name = ego_run_dir.split('/')[-4]

env_name = "1v1/NoWeapon/vsBaseline"
env = SingleCombatEnv(env_name)
env.seed(0)
args = Args()

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

from algorithms.sac.sac_actor import ActorNet
ego_policy = ActorNet(args, env.observation_space, env.action_space, device=device)
ego_policy.eval()

ego_policy.act_layer.action_low = ego_policy.act_layer.action_low.to(device)
ego_policy.act_layer.action_high = ego_policy.act_layer.action_high.to(device)

try:
    ego_policy.load_state_dict(torch.load(f"{ego_run_dir}/actor_{ego_policy_index}.pt", map_location=device, weights_only=True))
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

print("Start render")
obs = env.reset()

temp_filepath = f'1v1_vsBaselinePursue0502_{ego_policy_index}_0_0.00_0.00.txt.acmi'
if render:
    env.render(mode='txt', filepath=temp_filepath)

ego_agent_id = env.ego_ids[0]
enm_agent_id = env.enm_ids[0]

action_dim = len(env.task.action_var)
logging.debug(f"task.action_var: {env.task.action_var}, action_dim: {action_dim}")
logging.debug(f"env.task.use_baseline: {env.task.use_baseline}, num_agents: {env.task.num_agents}")

enm_obs = obs[num_agents // 2:, :]
ego_obs = obs[:num_agents // 2, :]
logging.debug(f"Initial obs shape: {obs.shape}, value: {obs}")
logging.debug(f"Initial ego_obs shape: {ego_obs.shape}, value: {ego_obs}")

# 初始化敌方代理
enm_agent = PursueAgent()  # 可替换为 StraightFlyAgent() 测试

step = 0
action_history = []
while True:
    # 转换为张量
    ego_obs_tensor = torch.tensor(ego_obs, dtype=torch.float32, device=device)
    # SAC 动作和分布
    feat = ego_policy.base(ego_obs_tensor)
    dist = ego_policy.act_layer.action_out(feat)
    mean = dist.mean.detach().cpu().numpy().squeeze()
    std = dist.stddev.detach().cpu().numpy().squeeze()
    raw_actions, log_probs, _ = ego_policy(ego_obs_tensor, deterministic=False)
    logging.debug(f"Step {step}: SAC mean: {mean}, std: {std}, log_probs: {log_probs}")
    ego_actions = _t2n(raw_actions)
    ego_actions = np.array(ego_actions, dtype=np.float32).ravel()
    if len(ego_actions) != action_dim:
        ego_actions = np.pad(ego_actions, (0, action_dim - len(ego_actions)), mode='constant')[:action_dim]
    action_history.append(ego_actions.tolist())
    logging.debug(f"Step {step}: ego_actions: {ego_actions}, action_history_len: {len(action_history)}")

    # 敌方动作
    enm_action = enm_agent.get_action(env._jsbsims[enm_agent_id])
    logging.debug(f"Step {step}: enm_action: {enm_action}")

    # 飞机状态
    try:
        ego_alt = env._jsbsims[ego_agent_id].get_property_value(Catalog.position_h_sl_ft)
        enm_alt = env._jsbsims[enm_agent_id].get_property_value(Catalog.position_h_sl_ft)
    except Exception as e:
        logging.error(f"Failed to get altitude: {e}")
        raise
    logging.debug(f"Step {step}: ego_altitude: {ego_alt}, enm_altitude: {enm_alt}")

    actions = ego_actions[np.newaxis, :]
    obs, rewards, dones, infos = env.step(actions)

    print(f"Step {step}: Actions: {actions}, Rewards: {rewards}")

    rewards_ego = rewards[:num_agents // 2, ...]
    rewards_enm = rewards[num_agents // 2:, ...]
    episode_rewards_ego += rewards_ego
    episode_rewards_enm += rewards_enm
    logging.debug(f"Step {step}: rewards_ego: {rewards_ego}, rewards_enm: {rewards_enm}")

    if render:
        env.render(mode='txt', filepath=temp_filepath)

    if dones.all():
        print(infos)
        break

    enm_obs = obs[num_agents // 2:, ...]
    ego_obs = obs[:num_agents // 2, ...]
    logging.debug(f"Step {step}: Updated obs shape: {obs.shape}, value: {obs}")
    logging.debug(f"Step {step}: Updated ego_obs shape: {ego_obs.shape}, value: {ego_obs}")

    step += 1

reward_diff = episode_rewards_ego.sum() - episode_rewards_enm.sum()
print(f"Final episode reward of ego agent: {episode_rewards_ego.sum():.2f}")
print(f"Final episode reward of enemy agent: {episode_rewards_enm.sum():.2f}")
print(f"Reward difference (Ego - Enemy): {reward_diff:.2f}")
print(f"Action history (first 5 steps): {action_history[:5]}")

final_filepath = f'1v1_vsBaselinePursue0502_{ego_policy_index}_0_{episode_rewards_ego.sum():.2f}_{episode_rewards_enm.sum():.2f}.txt.acmi'
os.rename(temp_filepath, final_filepath)