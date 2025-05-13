# import gymnasium
# import torch
# import torch.nn as nn
# from .flatten import build_flattener
#
#
# import torch
# import torch.nn as nn
# from .flatten import build_flattener
#
#
# class MLPLayer(nn.Module):
#     def __init__(self, input_dim, hidden_size, activation_id):
#         super(MLPLayer, self).__init__()
#         self._size = [input_dim] + list(map(int, hidden_size.split(' ')))
#         self._hidden_layers = len(self._size) - 1
#         active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
#
#         fc_h = []
#         for j in range(len(self._size) - 1):
#             fc_h += [
#                 nn.Linear(self._size[j], self._size[j + 1]), active_func, nn.LayerNorm(self._size[j + 1])
#             ]
#         self.fc = nn.Sequential(*fc_h)
#
#     def forward(self, x: torch.Tensor):
#         x = self.fc(x)
#         return x
#
#     @property
#     def output_size(self) -> int:
#         return self._size[-1]
#
#
# # Feature extraction module
# class MLPBase(nn.Module):
#     def __init__(self, obs_space, hidden_size, activation_id, use_feature_normalization):
#         super(MLPBase, self).__init__()
#         self._hidden_size = hidden_size
#         self._activation_id = activation_id
#         self._use_feature_normalization = use_feature_normalization
#
#         self.obs_flattener = build_flattener(obs_space)
#         input_dim = self.obs_flattener.size
#         if self._use_feature_normalization:
#             self.feature_norm = nn.LayerNorm(input_dim)
#         self.mlp = MLPLayer(input_dim, self._hidden_size, self._activation_id)
#
#     def forward(self, x: torch.Tensor):
#         if self._use_feature_normalization:
#             x = self.feature_norm(x)
#         x = self.mlp(x)
#         return x
#
#     @property
#     def output_size(self) -> int:
#         return self.mlp.output_size
import logging

import gymnasium
import torch
import torch.nn as nn
from .flatten import build_flattener

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, activation_id):
        super(MLPLayer, self).__init__()

        # 新层尺寸列表 ("256 256" -> [256, 256])
        self._size = [input_dim] + list(map(int, hidden_size.split(" ")))
        self.hidden_layers = len(self._size) - 1  # 隐藏层数量
        # 激活函数选择
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]

        fc_h = []
        linear_layers=[]
        # 用于定义新的一层 linear 层
        for j in range(len(self._size) - 1):
            linear_layer = nn.Linear(self._size[j], self._size[j + 1])
            fc_h.append(linear_layer)
            linear_layers.append(linear_layer)
            # 添加激活函数
            fc_h.append(active_func)
            # 添加新的一层 LayerNorm
            fc_h.append(nn.LayerNorm(self._size[j + 1]))

        self.fc = nn.Sequential(*fc_h)

        # 权重初始化 - 仅 Linear
        self.last_linear = linear_layers[-1] if linear_layers else None

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.last_linear:  # 最后一层 Linear
                    nn.init.orthogonal_(m.weight, gain=1.0)  # 小增益
                else:  # 其他层
                    nn.init.orthogonal_(m.weight, gain=1.0)  # 标准增益
                    nn.init.constant_(m.bias, 0.0)  # 偏置初始化为0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @property
    def output_size(self) -> int:
        return self._size[-1]
# Feature extraction module
class MLPBase(nn.Module):
    def __init__(self, obs_space, hidden_size, activation_id, use_feature_normalization):
        super(MLPBase, self).__init__()
        self._hidden_size = hidden_size
        self._activation_id = activation_id
        self._use_feature_normalization = use_feature_normalization

        # self.obs_flattener = build_flattener(obs_space)
        if isinstance(obs_space, gymnasium.spaces.Space):
            self.obs_flattener = build_flattener(obs_space)
            input_dim = self.obs_flattener.size
        elif isinstance(obs_space, int):
            self.obs_flattener = None
            input_dim = obs_space  # 直接使用 int
        else:
            raise ValueError(f"Unsupported obs_space type: {type(obs_space)}")

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)

        self.mlp = MLPLayer(input_dim, self._hidden_size, self._activation_id)

    def forward(self, x: torch.Tensor):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x

    @property
    def output_size(self) -> int:
        return self.mlp.output_size


# import torch
# import torch.nn as nn
# from .flatten import build_flattener
#
#
# class MLPLayer(nn.Module):
#     def __init__(self, input_dim, hidden_size, activation_id):
#         super(MLPLayer, self).__init__()
#         self._size = [input_dim] + list(map(int, hidden_size.split(' ')))
#         self._hidden_layers = len(self._size) - 1
#         active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
#
#         fc_h = []
#         for j in range(len(self._size) - 1):
#             fc_h += [
#                 nn.Linear(self._size[j], self._size[j + 1]), active_func, nn.LayerNorm(self._size[j + 1])
#             ]
#         self.fc = nn.Sequential(*fc_h)
#
#     def forward(self, x: torch.Tensor):
#         x = self.fc(x)
#         return x
#
#     @property
#     def output_size(self) -> int:
#         return self._size[-1]
#
#
# # Feature extraction module
# class MLPBase(nn.Module):
#     def __init__(self, obs_space, hidden_size, activation_id, use_feature_normalization):
#         super(MLPBase, self).__init__()
#         self._hidden_size = hidden_size
#         self._activation_id = activation_id
#         self._use_feature_normalization = use_feature_normalization
#
#         self.obs_flattener = build_flattener(obs_space)
#         input_dim = self.obs_flattener.size
#         if self._use_feature_normalization:
#             self.feature_norm = nn.LayerNorm(input_dim)
#         self.mlp = MLPLayer(input_dim, self._hidden_size, self._activation_id)
#
#     def forward(self, x: torch.Tensor):
#         if self._use_feature_normalization:
#             x = self.feature_norm(x)
#         x = self.mlp(x)
#         return x
#
#     @property
#     def output_size(self) -> int:
#         return self.mlp.output_size