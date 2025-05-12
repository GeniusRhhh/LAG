import torch
import numpy as np
from algorithms.sac.sac_actor import ActorNet
from envs.JSBSim.envs import SingleCombatEnv
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

env = SingleCombatEnv("1v1/NoWeapon/vsBaseline")
args = Args()
device = torch.device("cpu")
ego_policy = ActorNet(args, env.observation_space, env.action_space, device=device)
ego_policy.eval()
ego_policy.load_state_dict(torch.load("../scripts/results/SingleCombat/1v1/NoWeapon/vsBaseline/sac/v1/05061647/actor_30.pt", map_location=device, weights_only=True))

obs = env.reset()
ego_obs = obs[:1, :]  # (1, 1, 15)
for i in range(10):
    feat = ego_policy.base(ego_obs)
    dist = ego_policy.act_layer.action_out(feat)
    mean = dist.mean.detach().cpu().numpy().squeeze()
    std = dist.stddev.detach().cpu().numpy().squeeze()
    actions, log_probs, _ = ego_policy(ego_obs, deterministic=False)
    logging.debug(f"Step {i}: mean: {mean}, std: {std}, actions: {actions.detach().cpu().numpy().squeeze()}")
    ego_obs = torch.randn(1, 1, 15, dtype=torch.float32, device=device) * 10  # 模拟不同输入