import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#check() 函数用于统一输入数据格式，将 NumPy 数组转换为 PyTorch 张量，确保模型在前向推理过程中数据类型一致，防止类型不匹配错误，是深度学习推理阶段常用的预处理工具函数之一。
def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size) -> None:
        super().__init__()
        # 构建每层的维度列表，例如 input_dim=12, hidden_size='128 128' => [12, 128, 128]
        self._size = [input_dim] + list(map(int, hidden_size.split(' ')))
        self._hidden_layers = len(self._size) - 1 # 隐藏层数量（不含输入层）
        active_func = nn.ReLU() # 激活函数选择 ReLU
        fc_h = [] # 用于收集所有网络层模块

        # 遍历构建每一层：Linear → ReLU → LayerNorm
        for j in range(len(self._size) - 1):
            fc_h += [
                nn.Linear(self._size[j], self._size[j + 1]),  # 全连接层
                active_func,  # 激活函数
                nn.LayerNorm(self._size[j + 1]) # 层归一化，稳定训练
            ]
        # 使用 nn.Sequential 将所有层级打包成一个前馈神经网络
        self.fc = nn.Sequential(*fc_h)

    def forward(self, x):
        # 前向传播，将输入依次通过多层感知机处理
        x = self.fc(x)
        return x

class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()  # 调用父类 nn.Module 的初始化函数
        # 初始化一个多层感知机（MLP），用于特征提取
        # input_dim：输入特征维度（如观测向量的长度）
        # hidden_size：字符串形式，例如 '128 128'，定义每层隐藏单元数
        self.mlp = MLPLayer(input_dim, hidden_size)

    def forward(self, x):
        # 将输入 x 通过 MLP 网络进行前向传播
        # 输入 x 维度通常为 (batch_size, input_dim)
        x = self.mlp(x)
        return x  # 返回经过多层全连接+激活+归一化后的输出特征（通常维度为 [batch_size, 最后一层hidden_size]）


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()  # 初始化父类 nn.Module
        # 创建 GRU 层（门控循环单元）
        self.gru = nn.GRU(input_size=input_size,     # 输入特征维度（例如从 MLP 提取的向量维度）
                          hidden_size=hidden_size,   # GRU 隐状态的维度（内部状态记忆容量）
                          num_layers=num_layers)     # 堆叠的 GRU 层数
        # NOTE: self.gru(x, hxs) needs x=[T, N, input_size] and hxs=[L, N, hidden_size]
        # 添加层归一化，用于提升训练稳定性和收敛速度
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: F.Tensor, hxs: F.Tensor):
        # x=[N, input_size], hxs=[N, L, hidden_size]
        # 输入形状调整：GRU 需要 [T, N, D]，而我们输入是 [N, D]
        x, hxs = self.gru(x.unsqueeze(0), hxs.transpose(0, 1).contiguous())
        # 去掉时间维度，得到 [N, D]
        x = x.squeeze(0)            # [1, N, input_size] => [N, input_size]
        # 恢复隐藏状态形状
        hxs = hxs.transpose(0, 1)   # [L, N, hidden_size] => [N, L, hidden_size]
        x = self.norm(x)    # 输出再做一次归一化
        return x, hxs

class Categorical(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Categorical, self).__init__()
        # 定义一个全连接层，将输入维度映射到动作维度（输出为 logits，即动作的未归一化评分）
        self.logits_net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 通过线性层生成 logits，表示每个离散动作的打分（未归一化）,logits 是“未归一化的分类评分向量”
        logits = self.logits_net(x)
        # 构造一个离散分布 Categorical，并获取对应的概率（softmax后）
        # 然后取最大概率对应的动作索引（即贪婪选择策略）
        return torch.distributions.Categorical(logits=logits).probs.argmax(dim=-1, keepdim=True)

class ACTLayer(nn.Module):
    def __init__(self, input_dim, action_dims, use_mlp_actlayer=False):
        super(ACTLayer, self).__init__()
        # 是否在输出动作前添加额外的MLP层增强表达能力
        self._mlp_actlayer = use_mlp_actlayer
        # 如果启用 MLP，则定义一个结构为 128→128→128(输入维度为 128，一共两层隐藏层，每层输出维度都是 128) 的多层感知机
        if self._mlp_actlayer:
            self.mlp = MLPLayer(128, '128 128')  # 输入输出都是128维
        # 构建多个动作输出头（多动作头结构）
        # 每一个动作维度都用一个Categorical分类器独立建模
        action_outs = []
        for action_dim in action_dims:
            # 对每一个动作维度（如41个副翼档位）构建一个分类器
            # input_dim一般为上层网络输出维度，例如GRU输出的128
            action_outs.append(Categorical(input_dim, action_dim))
            # 使用ModuleList包装多个动作输出头，确保在nn.Module中注册子模块
        self.action_outs = nn.ModuleList(action_outs)

    def forward(self, x):
        # 如果启用了额外的MLP增强层，则先对输入特征进行进一步抽象处理
        if self._mlp_actlayer:
            x = self.mlp(x)   # 输出仍为 128维，用于增强特征表达
        actions = []   # 存储每个动作维度的输出（即分类器输出的动作索引）
        # 遍历每个动作维度对应的分类器（多动作头结构）
        for action_out in self.action_outs:
            action = action_out(x)
            actions.append(action)  # 收集到总动作列表中
        # 将多个动作维度的输出拼接为一个完整的动作向量
        # 如：4个动作维度 -> [batch_size, 4]
        actions = torch.cat(actions, dim=-1)
        # 返回组合后的离散动作索引向量
        return actions


class BaselineActor(nn.Module):
    def __init__(self, input_dim=12, use_mlp_actlayer=False) -> None:
        super().__init__()
        # 设置张量默认数据类型与设备（此处默认为 CPU）
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        # 特征提取器：2层 MLP，将 12维观测 → 128维特征向量
        self.base = MLPBase(input_dim, '128 128')
        # 时序建模器：单层 GRU，输入维度128 → 输出128，记忆能力增强
        self.rnn = GRULayer(128, 128, 1)
        # 多动作输出头：将128维特征映射为4个离散动作（41, 41, 41, 30）
        self.act = ACTLayer(128, [41, 41, 41, 30], use_mlp_actlayer)
        # 将整个模型注册到CPU上（也可以改为 GPU）
        self.to(torch.device('cpu'))

    def check(self, input):
        # 用于将 numpy 数据转换为 torch Tensor，并保证兼容性
        output = torch.from_numpy(input) if type(input) == np.ndarray else input
        return output

    def forward(self, obs, rnn_states):
        # 转换输入为 float32 tensor 并移动到指定设备
        x = check(obs).to(**self.tpdv)  # 当前输入观测
        h_s = check(rnn_states).to(**self.tpdv)    # 上一时刻 GRU 的隐状态
        # 特征提取：MLP
        x = self.base(x)  # 输出
        # 时序建模：GRU
        x, h_s = self.rnn(x, h_s)  # 输出新特征 x, 和更新后的状态 h_s
        # 动作生成：多动作输出（4个动作组成的离散向量）
        actions = self.act(x)
        return actions, h_s
