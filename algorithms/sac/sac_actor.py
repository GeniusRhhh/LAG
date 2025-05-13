# algorithms/sac/sac_actor.py
import torch
import torch.nn as nn
from gymnasium import spaces
from algorithms.utils.mlp import MLPLayer
from algorithms.utils.act import ACTLayer

class ActorNet(nn.Module):
    def __init__(self, args, obs_space, act_space, device='cpu'):
        super(ActorNet, self).__init__()
        self.device = device
        # 直接传递 args.hidden_size（字符串），不转换为列表
        self.base = MLPLayer(obs_space.shape[0], args.hidden_size, args.activation_id)
        self.act_layer = ACTLayer(
            act_space,
            self.base.output_size,
            args.act_hidden_size,  # 同样传递字符串
            args.activation_id,
            args.gain,
            device=self.device
        )

    def forward(self, obs, deterministic=False):
        feat = self.base(obs)
        actions, log_probs = self.act_layer(feat, deterministic=deterministic)
        return actions, log_probs

    def get_action(self, obs, deterministic=False):
        if isinstance(obs, (list, tuple)):
            obs = torch.FloatTensor(obs).to(self.device)
        elif not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(self.device)
        actions, log_pi = self.forward(obs, deterministic)
        return actions, log_pi

    def evaluate_actions(self, obs, action):
        feat = self.base(obs)
        log_probs, entropy = self.act_layer.evaluate_actions(feat, action)
        return log_probs, entropy