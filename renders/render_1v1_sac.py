import numpy as np
import torch
from envs.JSBSim.envs import SingleControlEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.sac.sac_actor import ActorNet
import logging
import os

logging.basicConfig(level=logging.DEBUG)

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = False
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))

def _t2n(x):
    """Convert torch tensor to numpy array."""
    return x.detach().cpu().numpy()

# Configuration
render = True
policy_index = 10000
run_dir = "../scripts/results/SingleControl/1/heading/sac/v0131/05152329"
experiment_name = run_dir.split('/')[-4]

# Check if file exists
model_path = f"{run_dir}/sac_{policy_index}.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")

# Initialize environment
env = SingleControlEnv("heading")
env.seed(0)
args = Args()

# Initialize SAC policy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = ActorNet(args, env.observation_space, env.action_space, device=device)
policy.to(device)  # Ensure model is on the correct device
policy.eval()

# Load actor state_dict
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
policy.load_state_dict(checkpoint["actor"])

print("Start render")
obs = env.reset()
# Log observation space and shape
logging.info(f"Observation space: {env.observation_space}, Obs shape: {obs.shape}")
# Remove batch dimension if present
if obs.ndim == 2 and obs.shape[0] == 1:
    obs = obs.squeeze(0)  # (1, 12) -> (12,)
# Validate obs shape
expected_shape = env.observation_space.shape
if obs.shape != expected_shape:
    raise ValueError(f"Expected obs shape {expected_shape}, got {obs.shape}")
episode_rewards = 0

if render:
    env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')

while True:
    # Convert obs to tensor and add batch dimension
    obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)  # [12,] -> [1, 12]

    # Get action from SAC policy
    actions, _ = policy(obs_tensor, deterministic=True)  # actions: [1, 4]
    actions = _t2n(actions.detach())  # [1, 4]

    # Clip actions to ensure within bounds
    actions = np.clip(actions, env.action_space.low, env.action_space.high)

    # Log step details
    logging.debug(f"Step {env.current_step}: obs={obs.tolist()}, actions={actions.tolist()}")

    # Step the environment
    obs, rewards, dones, infos = env.step(actions)
    # Remove batch dimension if present
    if obs.ndim == 2 and obs.shape[0] == 1:
        obs = obs.squeeze(0)  # (1, 12) -> (12,)
    episode_rewards += rewards

    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')

    print(f"step:{env.current_step}, reward:{rewards}")

    if dones:
        if isinstance(infos, dict) and 'reward_items' in infos:
            logging.info(f"Reward Breakdown: {infos['reward_items']}")
        print(infos)
        break

print(f"Episode rewards: {episode_rewards}")
env.close()