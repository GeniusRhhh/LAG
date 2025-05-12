import torch
import torch.nn as nn
import numpy as np
from baseline_actor import BaselineActor  # 假设这是你提供的第二个代码文件
from utils.utils import get_root_dir  # 假设这是你的项目中的工具函数

# 1. 加载模型
model_path = get_root_dir() + '/model/baseline_model.pt'
try:
    # 实例化 BaselineActor
    actor = BaselineActor(input_dim=12, use_mlp_actlayer=False)
    # 加载权重
    actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    actor.eval()  # 设置为评估模式
    print("成功加载模型权重！")
except Exception as e:
    print(f"加载模型权重失败：{e}")
    exit()

# 2. 打印模型结构
print("\n=== 模型结构 ===")
print(actor)

# 3. 打印模型参数数量
total_params = sum(p.numel() for p in actor.parameters())
print(f"\n=== 模型参数总数 ===")
print(f"总参数数量：{total_params}")

# 4. 打印每一层的参数
print("\n=== 每一层的参数 ===")
for name, param in actor.named_parameters():
    print(f"层：{name}, 参数形状：{param.shape}, 参数数量：{param.numel()}")

# 5. 测试输入输出维度
# 构造一个模拟输入（参考 get_observation 的格式）
obs = np.zeros(12)  # 12维观察值
obs = np.expand_dims(obs, axis=0)  # 增加批次维度，形状为 [1, 12]
rnn_states = np.zeros((1, 1, 128))  # GRU隐藏状态，形状为 [1, 1, 128]

# 转换为张量
obs = torch.from_numpy(obs).float()
rnn_states = torch.from_numpy(rnn_states).float()

try:
    with torch.no_grad():
        actions, new_rnn_states = actor(obs, rnn_states)
    print("\n=== 输入输出维度 ===")
    print(f"输入观察值维度：{obs.shape}")
    print(f"输入GRU隐藏状态维度：{rnn_states.shape}")
    print(f"输出动作维度：{actions.shape}")
    print(f"输出GRU隐藏状态维度：{new_rnn_states.shape}")
    print(f"输出动作示例：{actions}")
except Exception as e:
    print(f"推理失败：{e}")

# 6. 打印部分权重（可选，查看权重大小分布）
print("\n=== 部分权重信息（示例） ===")
for name, param in list(actor.named_parameters())[:5]:  # 只打印前5层权重
    print(f"层：{name}, 权重均值：{param.mean().item():.4f}, 权重标准差：{param.std().item():.4f}")