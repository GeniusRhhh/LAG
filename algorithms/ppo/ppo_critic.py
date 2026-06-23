import torch
import torch.nn as nn

from ..utils.mlp import MLPBase, MLPLayer  # 从utils模块导入MLP基础类和MLP层类
from ..utils.gru import GRULayer  # 导入GRU层的实现
from ..utils.utils import check  # 导入检查工具函数，用于数据格式和类型的校验

class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        # 网络配置
        self.hidden_size = args.hidden_size  # 隐藏层大小
        self.act_hidden_size = args.act_hidden_size  # 行动隐藏层大小
        self.activation_id = args.activation_id  # 激活函数类型
        self.use_feature_normalization = args.use_feature_normalization  # 是否使用特征归一化
        self.use_recurrent_policy = args.use_recurrent_policy  # 是否使用循环网络策略
        self.recurrent_hidden_size = args.recurrent_hidden_size  # 循环网络隐藏层大小
        self.recurrent_hidden_layers = args.recurrent_hidden_layers  # 循环网络层数
        self.tpdv = dict(dtype=torch.float32, device=device)  # 设备和数据类型配置
        # (1) 特征提取模块
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        # (2) rnn模块
        input_size = self.base.output_size  # 输入大小等于基础网络的输出大小
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size  # 更新输入大小为RNN层的输出大小
        # (3) 价值输出模块
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)  # 如果动作隐藏层非空，增加一个MLP层
        self.value_out = nn.Linear(input_size, 1)  # 线性层输出价值预测

        self.to(device)  # 将网络部署到指定设备

    def forward(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)  # 校验并转移观测到指定设备
        rnn_states = check(rnn_states).to(**self.tpdv)  # 校验并转移RNN状态
        masks = check(masks).to(**self.tpdv)  # 校验并转移掩码

        critic_features = self.base(obs)  # 通过基础网络提取特征

        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)  # 通过RNN处理特征和状态

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)  # 通过MLP层进一步处理特征

        values = self.value_out(critic_features)  # 计算价值预测

        return values, rnn_states  # 返回价值和更新后的RNN状态
