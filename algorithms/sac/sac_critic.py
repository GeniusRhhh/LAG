import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from ..utils.mlp import MLPBase
from ..utils.utils import check, get_shape_from_space

class QNetwork(nn.Module):
    def __init__(self, args, obs_space, act_dim, device=torch.device("cpu")):
        super(QNetwork, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)

        # Get observation dimension
        obs_dim = get_shape_from_space(obs_space)
        if isinstance(obs_dim, (tuple, list)):
            assert len(obs_dim) == 1, f"Unexpected obs_dim shape: {obs_dim}"
            obs_dim = obs_dim[0]
        assert isinstance(obs_dim, int), f"obs_dim should be int, got {type(obs_dim)}: {obs_dim}"

        # Input dimension: obs + act
        input_dim = Box(low=-np.inf, high=np.inf, shape=(obs_dim + act_dim,))

        # MLPBase for feature extraction
        self.base = MLPBase(
            input_dim,
            args.hidden_size,
            args.activation_id,
            args.use_feature_normalization
        )

        # Output layer for Q value
        self.q_out = nn.Linear(self.base.output_size, 1)
        self.to(device)

    def forward(self, obs, act):
        obs = check(obs).to(**self.tpdv)
        act = check(act).to(**self.tpdv)
        x = torch.cat([obs, act], dim=-1)
        features = self.base(x)
        q_values = self.q_out(features)
        return q_values

class SACCritic(nn.Module):
    """
    SAC Critic with two Q networks (Q1, Q2) to reduce overestimation bias
    """
    def __init__(self, args, obs_space, act_dim, device=torch.device("cpu")):
        super(SACCritic, self).__init__()
        self.Q1 = QNetwork(args, obs_space, act_dim, device)
        self.Q2 = QNetwork(args, obs_space, act_dim, device)
        self.to(device)

    def forward(self, obs, act):
        q1 = self.Q1(obs, act)
        q2 = self.Q2(obs, act)
        return q1, q2