import torch  # 导入PyTorch库，用于构建和训练神经网络
import torch.nn as nn  # 从torch库中导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口
import numpy as np  # 导入NumPy库，用于处理数组和矩阵运算

# 定义一个函数，用于将numpy数组转换为torch张量，或直接验证输入是否已经是张量
def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

# 定义一个多层感知机（MLP）层类，继承自nn.Module
class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self._size = [input_dim] + list(map(int, hidden_size.split(' ')))# 解析隐藏层尺寸字符串并构建大小列表
        self._hidden_layers = len(self._size) - 1# 计算隐藏层的数量
        active_func = nn.ReLU()# 激活函数使用ReLU

        # 构建全连接层和层归一化，按顺序添加到模块列表
        fc_h = []
        for j in range(len(self._size) - 1):
            fc_h += [
                nn.Linear(self._size[j], self._size[j + 1]),  # 全连接层
                active_func,  # 激活函数
                nn.LayerNorm(self._size[j + 1])  # 层归一化
            ]
        self.fc = nn.Sequential(*fc_h) # 将所有层组合成一个顺序模型

    def forward(self, x):
        x = self.fc(x)  # 通过定义的全连接层传递输入x
        return x

# 定义基础的MLP类，用于创建MLPLayer实例并进行前向传递
class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.mlp = MLPLayer(input_dim, hidden_size)  # 创建MLPLayer实例

    def forward(self, x):
        x = self.mlp(x)  # 将输入x传递给MLPLayer实例
        return x

# 定义GRU层类，用于处理序列数据
class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)  # 创建GRU模块
        self.norm = nn.LayerNorm(hidden_size)  # 创建层归一化模块

    def forward(self, x: F.Tensor, hxs: F.Tensor):
        x, hxs = self.gru(x.unsqueeze(0), hxs.transpose(0, 1).contiguous())  # GRU前向传播
        x = x.squeeze(0)  # 压缩多余的维度
        hxs = hxs.transpose(0, 1)  # 调换维度以符合其他操作的需求
        x = self.norm(x)  # 对输出x应用层归一化
        return x, hxs

# 定义分类模块，用于输出动作概率
class Categorical(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.logits_net = nn.Linear(input_dim, output_dim)  # 创建线性层，输出logits

    def forward(self, x):
        logits = self.logits_net(x)  # 计算logits
        return torch.distributions.Categorical(logits=logits).probs.argmax(dim=-1, keepdim=True)  # 返回概率最大的动作

# 定义动作层类，可以选择使用额外的MLP层处理动作输出
class ACTLayer(nn.Module):
    def __init__(self, input_dim, action_dims, use_mlp_actlayer=False):
        super().__init__()
        self._mlp_actlayer = use_mlp_actlayer  # 是否使用额外MLP层
        if self._mlp_actlayer:
            self.mlp = MLPLayer(128, '128 128')  # 创建MLPLayer实例
        action_outs = []
        for action_dim in action_dims:
            action_outs.append(Categorical(input_dim, action_dim))  # 为每个动作维度创建分类模块。该模块用于生成每个动作的概率分布，并从中选择最可能的动作。
        self.action_outs = nn.ModuleList(action_outs)  # 将所有动作输出模块存储为模块列表

    def forward(self, x):
        if self._mlp_actlayer:
            x = self.mlp(x)  # 如果启用，先通过MLP层处理x
        actions = []
        for action_out in self.action_outs:
            action = action_out(x)  # 为每个动作维度计算输出
            actions.append(action)
        actions = torch.cat(actions, dim=-1)  # 将所有动作输出拼接成一个张量
        return actions

# 定义基线演员模型，包括MLP基础、GRU序列处理和动作输出层
class BaselineActor(nn.Module):
    def __init__(self, input_dim=12, use_mlp_actlayer=False):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))  # 定义默认的张量属性
        self.base = MLPBase(input_dim, '128 128')  # 创建MLP基础实例
        self.rnn = GRULayer(128, 128, 1)
        self.act = ACTLayer(128, [41, 41, 41, 30], use_mlp_actlayer)  # 创建动作层实例
        self.to(torch.device('cpu'))

    def check(self, input):
        output = torch.from_numpy(input) if type(input) == np.ndarray else input
        return output

    def forward(self, obs, rnn_states):
        x = check(obs).to(**self.tpdv)  # 检查并转换obs为适合的张量
        h_s = check(rnn_states).to(**self.tpdv)  # 检查并转换rnn_states
        x = self.base(x)  # 通过MLP基础处理x
        x, h_s = self.rnn(x, h_s)  # 通过GRU处理x和隐藏状态
        actions = self.act(x)  # 生成动作
        return actions, h_s  # 返回动作和新的隐藏状态
