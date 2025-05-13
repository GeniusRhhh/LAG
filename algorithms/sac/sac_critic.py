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

        # 解析观测维度
        obs_dim = get_shape_from_space(obs_space)
        if isinstance(obs_dim, (tuple, list)):
            assert len(obs_dim) == 1, f"Unexpected obs_dim shape: {obs_dim}"
            obs_dim = obs_dim[0]
        assert isinstance(obs_dim, int), f"obs_dim should be int but got {type(obs_dim)}: {obs_dim}"

        # 组合输入维度 = obs_dim + act_dim
        input_dim1 = obs_dim + act_dim
        input_dim=Box(low=-np.inf,high=np.inf,shape=(input_dim1,))
        # MLPBase 基础网络
        self.base = MLPBase(
            input_dim,
            args.hidden_size,
            args.activation_id,
            args.use_feature_normalization
        )

        # 输出层 Q(s, a)
        self.q_out = nn.Linear(self.base.output_size, 1)

        self.to(device)

    def forward(self, obs, act):
        obs = check(obs).to(**self.tpdv)
        act = check(act).to(**self.tpdv)
        x = torch.cat([obs, act], dim=-1)
        # TF使用拼接后数据, 确保拼接后数据维度正确
        # logging.info(f"QNetwork Combined INPUT shape: {x.shape}")

        features = self.base(x)

        return self.q_out(features)
class SACCritic(nn.Module):
    """
    SAC CRITIC: 双 Q 网络 + 目标网络
    在实际环境中将网络拆分在 POLICY 模块处重整
    """
    def __init__(self, args, obs_space, act_dim, device=torch.device("cpu")):
        super(SACCritic, self).__init__()
        self.Q1 = QNetwork(args, obs_space, act_dim, device)
        self.Q2 = QNetwork(args, obs_space, act_dim, device)

    def forward(self, obs, act):
        """
        返回 Q1, Q2
        """
        q1 = self.Q1(obs, act)
        q2 = self.Q2(obs, act)

        # 打印 q1 和 q2, 确保数值是否异常
        # logging.info("Q1 value: %s", q1)
        # logging.info("Q2 value: %s", q2)

        return q1, q2