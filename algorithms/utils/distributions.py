# import torch
# import torch.nn as nn
#
# from .utils import init  # 导入初始化函数，用于对网络参数进行初始化
#
# """
# 自定义标准 PyTorch 分布类，并适配为强化学习动作分布的接口。
# 这些分布类会处理采样、对数概率计算、熵计算，以及与动作空间相关的操作。
# """
#
# # 定义自定义分布类，并修改其标准接口
# # =======================================================
#
# # Categorical 分布 (分类分布)
# class FixedCategorical(torch.distributions.Categorical):
#     """
#     自定义的分类分布，用于强化学习中离散动作空间的处理。
#     基于 torch.distributions.Categorical 修改，增加了采样维度调整、对数概率计算等。
#     """
#     def sample(self):
#         # 采样动作，并增加一个额外的维度以方便后续操作
#         return super().sample().unsqueeze(-1)
#
#     def log_probs(self, actions):
#         """
#         计算给定动作的对数概率，处理维度问题以确保兼容。
#         actions: 输入的动作张量。
#         """
#         return (
#             super()
#             .log_prob(actions.squeeze(-1))  # 去掉多余维度后计算对数概率
#             .view(actions.squeeze(-1).unsqueeze(-1).size())  # 恢复形状
#             .sum(-1, keepdim=True)  # 在最后一维上求和并保留维度
#         )
#
#     def mode(self):
#         """
#         获取概率最大的类别作为确定性动作。
#         """
#         return self.probs.argmax(dim=-1, keepdim=True)  # 返回概率最大的索引
#
#     def entropy(self):
#         """
#         计算分布的熵，衡量不确定性。
#         """
#         return super().entropy().unsqueeze(-1)  # 增加一个维度以统一格式
#
#
# # Normal 分布 (正态分布)
# class FixedNormal(torch.distributions.Normal):
#     """
#     自定义的正态分布，用于强化学习中连续动作空间的处理。
#     基于 torch.distributions.Normal 修改，增加了采样和对数概率计算功能。
#     """
#     def log_probs(self, actions):
#         # 计算每个维度的对数概率，并在最后一维上求和返回总对数概率
#         return super().log_prob(actions).sum(-1, keepdim=True)
#
#     def entropy(self):
#         # 计算分布的熵，并在最后一维上求和
#         return super().entropy().sum(-1, keepdim=True)
#
#     def mode(self):
#         # 返回均值作为确定性动作
#         return self.mean
#
#
# # Bernoulli 分布 (二值分布)
# class FixedBernoulli(torch.distributions.Bernoulli):
#     """
#     自定义的伯努利分布，用于强化学习中二值动作空间的处理。
#     基于 torch.distributions.Bernoulli 修改，增加了采样和对数概率计算功能。
#     """
#     def log_probs(self, actions):
#         # 计算给定动作的对数概率，并在最后一维上求和
#         return super().log_prob(actions).sum(-1, keepdim=True)
#
#     def entropy(self):
#         # 计算分布的熵，并在最后一维上求和
#         return super().entropy().sum(-1, keepdim=True)
#
#     def mode(self):
#         # 如果概率大于 0.5，返回 1；否则返回 0，表示动作的二值选择
#         return torch.gt(self.probs, 0.5).float()
#
#
# # 分类动作模块
# class Categorical(nn.Module):
#     """
#     分类动作模块，用于离散动作空间。
#     输入特征向量，输出分类分布，用于生成离散动作。
#     """
#     def __init__(self, num_inputs, num_outputs, gain=0.01):
#         """
#         初始化分类动作模块。
#         num_inputs: 输入特征维度。
#         num_outputs: 动作类别数量。
#         gain: 初始化增益。
#         """
#         super(Categorical, self).__init__()
#
#         def init_(m):
#             # 初始化网络参数，使用正交初始化和常量偏置
#             return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
#
#         # 定义线性层，用于生成分类分布的 logits
#         self.logits_net = init_(nn.Linear(num_inputs, num_outputs))
#
#     def forward(self, x):
#         """
#         前向传播。
#         x: 输入特征张量。
#         返回一个自定义的 FixedCategorical 分布。
#         """
#         x = self.logits_net(x)  # 计算 logits
#         return FixedCategorical(logits=x)  # 返回分类分布
#
#     @property
#     def output_size(self) -> int:
#         # 输出维度为 1，表示选择一个类别
#         return 1
#
#
# #对角高斯分布模块
# class DiagGaussian(nn.Module):
#     """
#     对角高斯分布模块，用于连续动作空间。
#     输入特征向量，输出正态分布，用于生成连续动作。
#     """
#     def __init__(self, num_inputs, num_outputs, gain=0.01):
#         """
#         初始化高斯分布模块。
#         num_inputs: 输入特征维度。
#         num_outputs: 动作维度（连续动作变量数量）。
#         gain: 初始化增益。
#         """
#         super(DiagGaussian, self).__init__()
#
#         def init_(m):
#             # 初始化网络参数
#             return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
#
#         # 定义线性层，用于生成均值
#         self.mu_net = init_(nn.Linear(num_inputs, num_outputs))
#         # 定义标准差的对数值，作为可训练参数
#         self.log_std = nn.Parameter(torch.zeros(num_outputs))
#         self._num_outputs = num_outputs  # 动作维度
#
#     def forward(self, x):
#         """
#         前向传播。
#         x: 输入特征张量。
#         返回一个自定义的 FixedNormal 分布。
#         """
#         action_mean = self.mu_net(x)  # 生成均值
#         return FixedNormal(action_mean, self.log_std.exp())  # 返回正态分布
#
#     @property
#     def output_size(self) -> int:
#         # 输出维度为动作的数量
#         return self._num_outputs
# # class DiagGaussian(nn.Module):
# #     def __init__(self, num_inputs, num_outputs, gain=0.01):
# #         super(DiagGaussian, self).__init__()
# #
# #         # 初始化均值网络
# #         self.mu_net = nn.Linear(num_inputs, num_outputs)
# #         nn.init.uniform_(self.mu_net.weight, -gain, gain)
# #         nn.init.uniform_(self.mu_net.bias, -gain, gain)
# #
# #         # 初始化log_std
# #         # self.log_std = nn.Parameter(torch.zeros(num_outputs))
# #         self.log_std = nn.Parameter(torch.full((num_outputs,), -0.5))  # 初始化为 -0.5
# #         self._num_outputs = num_outputs  # 动作维度
# #
# #     def forward(self, x):
# #         action_mean = self.mu_net(x)
# #         return FixedNormal(action_mean, self.log_std.exp())
# #
# #     @property
# #     def output_size(self) -> int:
# #         return self._num_outputs
# #
#
# # 射击动作模块
# class BetaShootBernoulli(nn.Module):
#     """
#     射击动作模块，用于混合动作空间。
#     将输入特征映射到射击概率，基于 Beta 分布计算射击的概率。
#     """
#     def __init__(self, num_inputs, num_outputs, gain=0.01):
#         super(BetaShootBernoulli, self).__init__()
#
#         def init_(m):
#             # 初始化网络参数
#             return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
#
#         self.net = init_(nn.Linear(num_inputs, num_outputs))  # 定义线性层
#         self._num_outputs = num_outputs  # 动作维度
#         self.constraint = nn.Softplus()  # 用 Softplus 函数约束输出
#
#     def forward(self, x, **kwargs):
#         """
#         前向传播。
#         x: 输入特征张量。
#         alpha0 和 beta0: 先验参数，用于控制射击的概率。
#         返回自定义的 FixedBernoulli 分布。
#         """
#         x = self.net(x)  # 通过线性层计算
#         x = self.constraint(x)  # 确保输出大于 0
#         x = 100 - self.constraint(100 - x)  # 限制最大值为 100
#         alpha = 1 + x[:, 0].unsqueeze(-1)  # 计算 alpha 参数
#         beta = 1 + x[:, 1].unsqueeze(-1)  # 计算 beta 参数
#         alpha_0 = kwargs['alpha0']  # 先验 alpha
#         beta_0 = kwargs['beta0']  # 先验 beta
#         p = (alpha + alpha_0) / (alpha + alpha_0 + beta + beta_0)  # 计算射击概率
#         return FixedBernoulli(p)  # 返回射击分布
#
#     @property
#     def output_size(self) -> int:
#         # 输出维度为动作的数量
#         return self._num_outputs
#
#
# # 伯努利分布模块
# class Bernoulli(nn.Module):
#     """
#     伯努利分布模块，用于二值动作空间。
#     输入特征向量，输出二值分布。
#     """
#     def __init__(self, num_inputs, num_outputs, gain=0.01):
#         super(Bernoulli, self).__init__()
#
#         def init_(m):
#             # 初始化网络参数
#             return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
#
#         self.logits_net = init_(nn.Linear(num_inputs, num_outputs))  # 定义线性层
#         self._num_outputs = num_outputs  # 动作维度
#
#     def forward(self, x):
#         """
#         前向传播。
#         x: 输入特征张量。
#         返回一个自定义的 FixedBernoulli 分布。
#         """
#         x = self.logits_net(x)  # 计算 logits
#         return FixedBernoulli(logits=x)  # 返回二值分布
#
#     @property
#     def output_size(self) -> int:
#         # 输出维度为动作的数量
#         return self._num_outputs


import torch
import torch.nn as nn
import numpy as np
import logging

from .utils import init


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        log_probs = super().log_prob(actions.squeeze(-1)).view(actions.squeeze(-1).unsqueeze(-1).size()).sum(-1, keepdim=True)
        return torch.clamp(log_probs, -1000, 0)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().unsqueeze(-1)

class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        log_probs = super().log_prob(actions).sum(-1, keepdim=True)
        return torch.clamp(log_probs, -1000, 0)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    def mode(self):
        return self.mean

class FixedSquashedNormal(torch.distributions.Normal):
    """
    Squashed Normal distribution for SAC, applies tanh and scales to action bounds.
    """
    def __init__(self, loc, scale, low, high, epsilon=1e-6):
        if torch.isnan(loc).any() or torch.isinf(loc).any():
            logging.warning(f"Invalid loc in FixedSquashedNormal: {loc}")
            loc = torch.clamp(loc, -10, 10)
        if torch.isnan(scale).any() or torch.isinf(scale).any() or (scale <= 0).any():
            logging.warning(f"Invalid scale in FixedSquashedNormal: {scale}")
            scale = torch.clamp(scale, 1e-9, 7.4)
        super().__init__(loc, scale)
        self.low = torch.as_tensor(low, dtype=torch.float32, device=loc.device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=loc.device)
        self.epsilon = epsilon

    def sample(self):
        raw_samples = super().sample()
        squashed = torch.tanh(raw_samples)
        scaled = self.low + (self.high - self.low) * (squashed + 1) / 2
        return scaled

    def rsample(self):
        raw_samples = super().rsample()
        squashed = torch.tanh(raw_samples)
        scaled = self.low + (self.high - self.low) * (squashed + 1) / 2
        return scaled

    def log_probs(self, actions):
        squashed = 2 * (actions - self.low) / (self.high - self.low) - 1
        squashed = torch.clamp(squashed, -1 + self.epsilon, 1 - self.epsilon)
        raw_actions = torch.atanh(squashed)
        if torch.isnan(raw_actions).any() or torch.isinf(raw_actions).any():
            logging.warning(f"Invalid raw_actions in log_probs: {raw_actions}")
            raw_actions = torch.clamp(raw_actions, -10, 10)
        log_prob = super().log_prob(raw_actions)
        log_prob -= torch.log(1 - squashed.pow(2) + self.epsilon).sum(-1, keepdim=True)
        return log_prob.sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    def mode(self):
        raw_mean = self.mean
        squashed = torch.tanh(raw_mean)
        scaled = self.low + (self.high - self.low) * (squashed + 1) / 2
        return scaled


class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(Categorical, self).__init__()
        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
        self.logits_net = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.logits_net(x)
        return FixedCategorical(logits=x)

    @property
    def output_size(self) -> int:
        return 1


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01, low=None, high=None):
        super(DiagGaussian, self).__init__()
        def init_(m):
            return init(m, nn.init.uniform_, lambda x: nn.init.constant_(x, 0), gain)
        self.mu_net = init_(nn.Linear(num_inputs, num_outputs))
        # self.log_std = nn.Parameter(torch.full((num_outputs,), -1.0))
        self.log_std = nn.Parameter(torch.full((num_outputs,), -0.5))  # 初始标准差约为 0.6
        log_std = torch.clamp(self.log_std, -5.0, 0.0)  # 标准差范围 [0.0067, 1.0]
        self._num_outputs = num_outputs
        self.low = low if low is not None else np.array([-1.0] * num_outputs, dtype=np.float32)
        self.high = high if high is not None else np.array([1.0] * num_outputs, dtype=np.float32)

    def forward(self, x):
        # action_mean = self.mu_net(x)
        action_mean = torch.clamp(self.mu_net(x), -10.0, 10.0)
        log_std = torch.clamp(self.log_std, -10, 0)  # 限制 std 在 [0.000045, 1.0]
        std = log_std.exp()
        logging.debug(f"DiagGaussian: mean={action_mean.detach().cpu().numpy().squeeze()}, std={std.detach().cpu().numpy().squeeze()}")
        return FixedSquashedNormal(action_mean, std, self.low, self.high)

    @property
    def output_size(self) -> int:
        return self._num_outputs


class BetaShootBernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(BetaShootBernoulli, self).__init__()
        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
        self.net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs
        self.constraint = nn.Softplus()

    def forward(self, x, **kwargs):
        x = self.net(x)
        x = self.constraint(x)
        x = 100 - self.constraint(100 - x)
        alpha = 1 + x[:, 0].unsqueeze(-1)
        beta = 1 + x[:, 1].unsqueeze(-1)
        alpha_0 = kwargs['alpha0']
        beta_0 = kwargs['beta0']
        p = (alpha + alpha_0) / (alpha + alpha_0 + beta + beta_0)
        return FixedBernoulli(p)

    @property
    def output_size(self) -> int:
        return self._num_outputs


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(Bernoulli, self).__init__()
        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
        self.logits_net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs

    def forward(self, x):
        x = self.logits_net(x)
        return FixedBernoulli(logits=x)

    @property
    def output_size(self) -> int:
        return self._num_outputs