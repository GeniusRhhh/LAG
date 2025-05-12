import torch
import torch.nn as nn

from ..utils.mlp import MLPBase  # 导入多层感知机基类
from ..utils.gru import GRULayer  # 导入GRU层的定义
from ..utils.act import ACTLayer  # 导入动作转换层的定义
from ..utils.utils import check  # 导入检查函数，用于数据验证

class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()  # 调用父类构造函数
        # 网络配置
        self.gain = args.gain  # 设置初始化增益
        self.hidden_size = args.hidden_size  # 隐藏层大小
        self.act_hidden_size = args.act_hidden_size  # 动作层的隐藏层大小
        self.activation_id = args.activation_id  # 激活函数标识
        self.use_feature_normalization = args.use_feature_normalization  # 是否使用特征归一化
        self.use_recurrent_policy = args.use_recurrent_policy  # 是否使用循环策略
        self.recurrent_hidden_size = args.recurrent_hidden_size  # 循环层的隐藏层大小
        self.recurrent_hidden_layers = args.recurrent_hidden_layers  # 循环层的层数
        self.tpdv = dict(dtype=torch.float32, device=device)  # 定义张量属性（数据类型和设备）
        self.use_prior = args.use_prior  # 是否使用先验知识

        # (1) 特征提取模块
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        # (2) RNN模块
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) 动作模块
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)

        self.to(device)  # 将网络的所有模块移到指定设备

    def forward(self, obs, rnn_states, masks, deterministic=False):
        obs = check(obs).to(**self.tpdv)  # 检查并转换观测到指定设备和数据类型
        rnn_states = check(rnn_states).to(**self.tpdv)  # 检查并转换RNN状态
        masks = check(masks).to(**self.tpdv)  # 检查并转换掩码
        if self.use_prior:
            # 控制射击导弹的先验知识
            attack_angle = torch.rad2deg(obs[:, 11])  # 角度单位转换
            distance = obs[:, 13] * 10000  # 距离单位转换
            alpha0 = torch.full(size=(obs.shape[0],1), fill_value=3).to(**self.tpdv)  # 初始alpha值
            beta0 = torch.full(size=(obs.shape[0],1), fill_value=10).to(**self.tpdv)  # 初始beta值
            alpha0[distance<=12000] = 6  # 根据距离调整alpha
            alpha0[distance<=8000] = 10  # 根据更近的距离调整alpha
            beta0[attack_angle<=45] = 6  # 根据攻击角度调整beta
            beta0[attack_angle<=22.5] = 3  # 根据更小的攻击角度调整beta

        actor_features = self.base(obs)  # 通过基础网络提取特征

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)  # 通过RNN处理特征和状态

        if self.use_prior:
            actions, action_log_probs = self.act(actor_features, deterministic, alpha0=alpha0, beta0=beta0)  # 使用先验信息生成动作和对数概率
        else:
            actions, action_log_probs = self.act(actor_features, deterministic)  # 生成动作和对数概率

        return actions, action_log_probs, rnn_states  # 返回动作、动作对数概率和RNN状态

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = check(obs).to(**self.tpdv)  # 检查并转换观测
        rnn_states = check(rnn_states).to(**self.tpdv)  # 检查并转换RNN状态
        action = check(action).to(**self.tpdv)  # 检查并转换动作
        masks = check(masks).to(**self.tpdv)  # 检查并转换掩码
        if self.use_prior:
            # 控制射击导弹的先验知识
            attack_angle = torch.rad2deg(obs[:, 11])  # 角度单位转换
            distance = obs[:, 13] * 10000  # 距离单位转换
            alpha0 = torch.full(size=(obs.shape[0], 1), fill_value=3).to(**self.tpdv)  # 初始alpha值
            beta0 = torch.full(size=(obs.shape[0], 1), fill_value=10).to(**self.tpdv)  # 初始beta值

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)  # 检查并转换活动掩码

        actor_features = self.base(obs)  # 通过基础网络提取特征

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)  # 通过RNN处理特征和状态

        if self.use_prior:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks, alpha0=alpha0, beta0=beta0)  # 使用先验信息评估动作和计算分布熵
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)  # 评估动作和计算分布熵

        return action_log_probs, dist_entropy  # 返回动作对数概率和分布熵


#obs (观测)
# obs （观测数据）：这代表从环境中获取的观测或状态信息，是智能体当前环境状态的数值表示。这些观测数据是智能体决策过程的基础，它们提供了智能体需要的环境信息，如位置、速度、角度等，智能体根据这些信息决定其行动。
# rnn_states (RNN状态)
# rnn_states：这是递归神经网络（Recurrent Neural Network, RNN）的状态。如果策略模型使用了循环网络结构（如GRU或LSTM），这些状态包含了过去行动的历史信息，有助于模型处理序列依赖问题。在模型中维护这些状态可以帮助智能体考虑到之前的行动和观测，对于处理具有时间依赖性的任务尤其重要。
# masks (掩码)
# masks：掩码用于指示哪些序列是有效的，通常用在循环神经网络中处理变长序列时。在强化学习中，当一个环节结束并开始新的环节时，掩码被设置为0，以重置RNN的状态，否则为1。这确保了RNN在新的序列开始时不会携带过去环节的状态信息，帮助模型正确处理每个独立的决策序列。
# active_masks 在神经网络模型中，尤其是在处理序列数据或在某些环境中（例如强化学习或处理具有可变长度的输入的任务）的应用中，非常有用。它的主要作用是指示哪些元素或部分数据是有效的，从而允许模型在进行计算时忽略某些无效或不相关的部分。
