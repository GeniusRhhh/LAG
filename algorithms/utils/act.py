import gymnasium as gym
import torch
import torch.nn as nn

from .distributions import BetaShootBernoulli, Categorical, DiagGaussian, Bernoulli
from .mlp import MLPLayer

class ACTLayer(nn.Module):
    def __init__(self, act_space, input_dim, hidden_size, activation_id, gain):
        super(ACTLayer, self).__init__()
        self._mlp_actlayer = False
        self._continuous_action = False
        self._multidiscrete_action = False
        self._mixed_action = False
        self._shoot_action = False

        if len(hidden_size) > 0:
            self._mlp_actlayer = True
            self.mlp = MLPLayer(input_dim, hidden_size, activation_id)
            input_dim = self.mlp.output_size

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if isinstance(act_space, gym.spaces.Discrete):
            action_dim = act_space.n
            self.action_out = Categorical(input_dim, action_dim, gain).to(device)
        elif isinstance(act_space, gym.spaces.Box):
            self._continuous_action = True
            action_dim = act_space.shape[0]
            self.action_out = DiagGaussian(input_dim, action_dim, gain).to(device)
            # 初始化 action_high 和 action_low 并移至设备
            self.action_high = torch.FloatTensor(act_space.high).to(device)
            self.action_low = torch.FloatTensor(act_space.low).to(device)
        elif isinstance(act_space, gym.spaces.MultiBinary):
            action_dim = act_space.shape[0]
            self.action_out = Bernoulli(input_dim, action_dim, gain).to(device)
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            self._multidiscrete_action = True
            action_dims = act_space.nvec
            action_outs = []
            for action_dim in action_dims:
                action_outs.append(Categorical(input_dim, action_dim, gain).to(device))
            self.action_outs = nn.ModuleList(action_outs)
        elif isinstance(act_space, gym.spaces.Tuple) and \
                isinstance(act_space[0], gym.spaces.MultiDiscrete) and \
                isinstance(act_space[1], gym.spaces.Discrete):
            self._shoot_action = True
            discrete_dims = act_space[0].nvec
            self._discrete_dim = act_space[0].shape[0]
            self._control_shoot_dim = 2
            self._shoot_dim = 1
            action_outs = []
            for discrete_dim in discrete_dims:
                action_outs.append(Categorical(input_dim, discrete_dim, gain).to(device))
            action_outs.append(BetaShootBernoulli(input_dim, self._control_shoot_dim, gain).to(device))
            self.action_outs = nn.ModuleList(action_outs)
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(act_space)}!")

    def forward(self, x, deterministic=False, **kwargs):
        """
        Compute actions and action logprobs from given input.
        前向传播，生成动作和对数概率。
        Args:
            x (torch.Tensor): input to network.
            deterministic (bool): whether to sample from action distribution or return the mode.

        Returns:
            actions (torch.Tensor): actions to take.
            action_log_probs (torch.Tensor): log probabilities of taken actions.
        """
        device = x.device
        if self._mlp_actlayer:
            x = self.mlp(x)

        if self._multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x.to(device))
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)

        elif self._shoot_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs[:-1]:
                action_dist = action_out(x.to(device))
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            shoot_action_dist = self.action_outs[-1](x.to(device), **kwargs)
            shoot_action = shoot_action_dist.mode() if deterministic else shoot_action_dist.sample()
            actions.append(shoot_action)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)

        else:
            action_dist = self.action_out(x.to(device))
            actions = action_dist.mode() if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action, active_masks=None, **kwargs):
        """
        Compute log probability and entropy of given actions.

        Args:
            x (torch.Tensor): input to network.
            action (torch.Tensor): actions whose entropy and log probability to evaluate.
            active_masks (torch.Tensor): denotes whether an agent is active or dead.

        Returns:
            action_log_probs (torch.Tensor): log probabilities of the input actions.
            dist_entropy (torch.Tensor): action distribution entropy for the given inputs.
        """
        device = x.device
        if self._mlp_actlayer:
            x = self.mlp(x)

        if self._multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_dist = action_out(x.to(device))
                action_log_probs.append(action_dist.log_probs(act.unsqueeze(-1)))
                if active_masks is not None:
                    dist_entropy.append((action_dist.entropy() * active_masks.to(device)) / active_masks.sum())
                else:
                    dist_entropy.append(action_dist.entropy() / action_log_probs[-1].size(0))
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            dist_entropy = torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True)

        elif self._shoot_action:
            dis_action, shoot_action = action.split((self._discrete_dim, self._shoot_dim), dim=-1)
            action_log_probs = []
            dist_entropy = []
            dis_action = torch.transpose(dis_action, 0, 1)
            for action_out, act in zip(self.action_outs[:-1], dis_action):
                action_dist = action_out(x.to(device))
                action_log_probs.append(action_dist.log_probs(act.unsqueeze(-1)))
                if active_masks is not None:
                    dist_entropy.append((action_dist.entropy() * active_masks.to(device)) / active_masks.sum())
                else:
                    dist_entropy.append(action_dist.entropy() / action_log_probs[-1].size(0))

            shoot_action_dist = self.action_outs[-1](x.to(device), **kwargs)
            action_log_probs.append(shoot_action_dist.log_probs(shoot_action))
            if active_masks is not None:
                dist_entropy.append((shoot_action_dist.entropy() * active_masks.to(device)) / active_masks.sum())
            else:
                dist_entropy.append(shoot_action_dist.entropy() / action_log_probs[-1].size(0))

            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            dist_entropy = torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True)

        else:
            action_dist = self.action_out(x.to(device))
            action_log_probs = action_dist.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_dist.entropy() * active_masks.to(device)) / active_masks.sum()
            else:
                dist_entropy = action_dist.entropy() / action_log_probs.size(0)
        return action_log_probs, dist_entropy

    def get_probs(self, x):
        """
        Compute action probabilities from inputs.

        Args:
            x (torch.Tensor): input to network.

        Return:
            action_probs (torch.Tensor):
        """
        device = x.device
        if self._mlp_actlayer:
            x = self.mlp(x)

        if self._multidiscrete_action:
            action_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x.to(device))
                action_prob = action_dist.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, dim=-1)
        elif self._continuous_action or self._shoot_action:
            raise ValueError("Normal distribution has no `probs` attribute!")
        else:
            action_dists = self.action_out(x.to(device))
            action_probs = action_dists.probs
        return action_probs

    @property
    def output_size(self) -> int:
        if self._multidiscrete_action or self._shoot_action:
            return len(self.action_outs)
        else:
            return self.action_out.output_size

# import logging
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from gymnasium import spaces
# from .distributions import BetaShootBernoulli, Categorical, DiagGaussian, Bernoulli
# from .mlp import MLPLayer
#
# class ACTLayer(nn.Module):
#     def __init__(self, act_space, input_dim, hidden_size, activation_id, gain, device='cpu'):
#         super(ACTLayer, self).__init__()
#         self._mlp_actlayer = False
#         self._continuous_action = False
#         self._multidiscrete_action = False
#         self._mixed_action = False
#         self._shoot_action = False
#         self._global_step = 0
#         self.device = device
#
#         # Initialize action bounds for continuous action space
#         if isinstance(act_space, spaces.Box):
#             self.action_low = torch.tensor(act_space.low, dtype=torch.float32, device=device)
#             self.action_high = torch.tensor(act_space.high, dtype=torch.float32, device=device)
#
#         if len(hidden_size) > 0:
#             self._mlp_actlayer = True
#             self.mlp = MLPLayer(input_dim, hidden_size, activation_id)
#             input_dim = self.mlp.output_size
#
#         if isinstance(act_space, spaces.Discrete):
#             action_dim = act_space.n
#             self.action_out = Categorical(input_dim, action_dim, gain)
#         elif isinstance(act_space, spaces.Box):
#             self._continuous_action = True
#             action_dim = act_space.shape[0]
#             self.action_out = DiagGaussian(input_dim, action_dim, gain, device=self.device)
#         elif isinstance(act_space, spaces.MultiBinary):
#             action_dim = act_space.shape[0]
#             self.action_out = Bernoulli(input_dim, action_dim, gain)
#         elif isinstance(act_space, spaces.MultiDiscrete):
#             self._multidiscrete_action = True
#             action_dims = act_space.nvec
#             action_outs = []
#             for action_dim in action_dims:
#                 action_outs.append(Categorical(input_dim, action_dim, gain))
#             self.action_outs = nn.ModuleList(action_outs)
#         elif isinstance(act_space, spaces.Tuple) and \
#                 isinstance(act_space[0], spaces.MultiDiscrete) and \
#                 isinstance(act_space[1], spaces.Discrete):
#             self._shoot_action = True
#             discrete_dims = act_space[0].nvec
#             self._discrete_dim = act_space[0].shape[0]
#             self._control_shoot_dim = 2
#             self._shoot_dim = 1
#             action_outs = []
#             for discrete_dim in discrete_dims:
#                 action_outs.append(Categorical(input_dim, discrete_dim, gain))
#             action_outs.append(BetaShootBernoulli(input_dim, self._control_shoot_dim, gain))
#             self.action_outs = nn.ModuleList(action_outs)
#         else:
#             raise NotImplementedError(f"Unsupported action space type: {type(act_space)}!")
#
#     def forward(self, x, deterministic=False, **kwargs):
#         if self._mlp_actlayer:
#             x = self.mlp(x)
#
#         if self._multidiscrete_action:
#             actions = []
#             action_log_probs = []
#             for action_out in self.action_outs:
#                 action_dist = action_out(x)
#                 action = action_dist.mode() if deterministic else action_dist.sample()
#                 action_log_prob = action_dist.log_probs(action)
#                 actions.append(action)
#                 action_log_probs.append(action_log_prob)
#             actions = torch.cat(actions, dim=-1)
#             action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
#             logging.debug(f"Multidiscrete action log_probs shape: {action_log_probs.shape}")
#
#         elif self._shoot_action:
#             actions = []
#             action_log_probs = []
#             for action_out in self.action_outs[:-1]:
#                 action_dist = action_out(x)
#                 action = action_dist.mode() if deterministic else action_dist.sample()
#                 action_log_prob = action_dist.log_probs(action)
#                 actions.append(action)
#                 action_log_probs.append(action_log_prob)
#             shoot_action_dist = self.action_outs[-1](x, **kwargs)
#             shoot_action = shoot_action_dist.mode() if deterministic else shoot_action_dist.sample()
#             actions.append(shoot_action)
#             actions = torch.cat(actions, dim=-1)
#             action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
#             logging.debug(f"Shoot action log_probs shape: {action_log_probs.shape}")
#
#         elif self._continuous_action:
#             action_dist = self.action_out(x)
#             z = action_dist.rsample()
#             actions = torch.tanh(z)
#             # 修改 log_prob 校正公式，增加数值稳定性
#             log_probs = action_dist.log_probs(z)
#             # 添加小的 epsilon 避免数值溢出
#             correction = 2 * (np.log(2) - z - F.softplus(-2 * z + 1e-6)).sum(dim=-1, keepdim=True)
#             action_log_probs = log_probs - correction
#             # 限制 log_prob 范围
#             action_log_probs = torch.clamp(action_log_probs, -10.0, 10.0)
#             logging.debug(f"Continuous action log_probs shape: {action_log_probs.shape}, values: {action_log_probs.mean().item()}")
#             self._global_step += 1
#             if self._global_step % 20000 == 0:
#                 mean = action_dist.loc.cpu().detach().numpy()
#                 std = action_dist.scale.cpu().detach().numpy()
#                 actions_np = actions.cpu().detach().numpy()
#                 sample_action = actions_np[0].tolist()
#                 logging.info(
#                     f"ACTLayer Step {self._global_step}: "
#                     f"mean={mean[0].tolist()}, std={std[0].tolist()}, "
#                     f"sample_action={sample_action}"
#                 )
#
#             # Clip actions to ensure they are within bounds
#             actions = torch.clamp(actions, self.action_low, self.action_high)
#
#             # Boundary check for logging
#             actions_np = actions.cpu().detach().numpy()
#             control_bounds_violated = np.any(
#                 (actions_np[:, :3] < -1.0) | (actions_np[:, :3] > 1.0), axis=1
#             )
#             thrust_bounds_violated = (
#                 (actions_np[:, 3] < 0.4) | (actions_np[:, 3] > 0.9)
#             )
#             actions_out_of_bounds = np.any(control_bounds_violated | thrust_bounds_violated)
#
#             if actions_out_of_bounds:
#                 logging.warning(
#                     f"Step {self._global_step} Actions out of bounds: "
#                     f"actions={actions_np.tolist()}, "
#                     f"control_violated={control_bounds_violated.tolist()}, "
#                     f"thrust_violated={thrust_bounds_violated.tolist()}"
#                 )
#
#         else:
#             # Default case for Discrete action space
#             action_dist = self.action_out(x)
#             actions = action_dist.mode() if deterministic else action_dist.sample()
#             action_log_probs = action_dist.log_probs(actions)
#             logging.debug(f"Discrete action log_probs shape: {action_log_probs.shape}")
#
#         return actions, action_log_probs
#
#     def evaluate_actions(self, state, action):
#         if self._mlp_actlayer:
#             state = self.mlp(state)
#
#         if self._multidiscrete_action:
#             action_log_probs = []
#             action_entropy = []
#             for i, action_out in enumerate(self.action_outs):
#                 action_dist = action_out(state)
#                 action_log_prob = action_dist.log_probs(action[:, i].unsqueeze(-1))
#                 dist_entropy = action_dist.entropy()
#                 action_log_probs.append(action_log_prob)
#                 action_entropy.append(dist_entropy)
#             action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
#             action_entropy = torch.cat(action_entropy, dim=-1).sum(dim=-1, keepdim=True)
#         elif self._shoot_action:
#             action_log_probs = []
#             action_entropy = []
#             for i, action_out in enumerate(self.action_outs[:-1]):
#                 action_dist = action_out(state)
#                 action_log_prob = action_dist.log_probs(action[:, i].unsqueeze(-1))
#                 dist_entropy = action_dist.entropy()
#                 action_log_probs.append(action_log_prob)
#                 action_entropy.append(dist_entropy)
#             shoot_action_dist = self.action_outs[-1](state)
#             shoot_action_log_prob = shoot_action_dist.log_probs(action[:, -self._control_shoot_dim:])
#             shoot_action_entropy = shoot_action_dist.entropy()
#             action_log_probs.append(shoot_action_log_prob)
#             action_entropy.append(shoot_action_entropy)
#             action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
#             action_entropy = torch.cat(action_entropy, dim=-1).sum(dim=-1, keepdim=True)
#         else:
#             action_dist = self.action_out(state)
#             action_log_probs = action_dist.log_probs(action)
#             action_entropy = action_dist.entropy()
#
#         return action_log_probs, action_entropy
#
#     def get_probs(self, state):
#         if self._mlp_actlayer:
#             state = self.mlp(state)
#
#         if self._multidiscrete_action:
#             action_probs = []
#             for action_out in self.action_outs:
#                 action_dist = action_out(state)
#                 action_probs.append(action_dist.probs)
#             action_probs = torch.cat(action_probs, dim=-1)
#         elif self._shoot_action:
#             action_probs = []
#             for action_out in self.action_outs[:-1]:
#                 action_dist = action_out(state)
#                 action_probs.append(action_dist.probs)
#             shoot_action_dist = self.action_outs[-1](state)
#             action_probs.append(shoot_action_dist.probs)
#             action_probs = torch.cat(action_probs, dim=-1)
#         else:
#             action_dist = self.action_out(state)
#             action_probs = action_dist.probs
#
#         return action_probs
#
#     @property
#     def output_size(self) -> int:
#         if self._multidiscrete_action or self._shoot_action:
#             return len(self.action_outs)
#         else:
#             return self.action_out.output_size

