# # # import numpy as np
# # # import random
# # #
# # # n_states=5
# # # actions=[-1,1]
# # # Q=np.zeros((n_states,len(actions)))
# # # alpha=0.1
# # # gamma=0.9
# # # epsilon=0.1
# # # episodes=1000
# # #
# # # def step(state,action):
# # #     next_state=state+action
# # #     if next_state <0 or next_state >=n_states:
# # #         return state,0
# # #     reward = 1 if next_state ==4 else 0
# # #     return next_state , reward
# # #
# # # def choose_action(state):
# # #     if random.random()<epsilon:
# # #         return random.choice([0,1])
# # #     else:
# # #         return np.argmax(Q[state])
# # # print("Train")
# # # for _ in range(episodes):
# # #     state=random.randint(0,3)
# # #     while state!=4:
# # #         action_idx=choose_action(state)
# # #         action=actions[action_idx]
# # #         next_state,reward=step(state,action)
# # #         Q[state,action_idx]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state,action_idx])
# # #         state=next_state
# # #
# # # print("Learned Q-table:")
# # # print(Q)
# # #
# # # state=0
# # # print("\nTest:")
# # # while state!=4:
# # #     action_idx=np.argmax(Q[state])
# # #     action=actions[action_idx]
# # #     next_state,reward=step(state,action)
# # #     print(f"State:{state},Action:{action},Reward:{reward},NextState:{next_state}")
# # #     state=next_state
# # #
# # import torch
# # import numpy as np
# # import gym
# # import random
# #
# # print("=" * 60)
# # print("PyTorch语法详细解释")
# # print("=" * 60)
# #
# # # ===== 1. 张量 (Tensor) 和转换 =====
# # print("\n1. 张量转换 - 为什么要转换？")
# # print("-" * 40)
# #
# # # 原始数据类型
# # python_list = [1.0, 2.0, 3.0, 4.0]
# # numpy_array = np.array([1.0, 2.0, 3.0, 4.0])
# #
# # print(f"Python列表: {python_list}, 类型: {type(python_list)}")
# # print(f"Numpy数组: {numpy_array}, 类型: {type(numpy_array)}")
# #
# # # 转换为张量
# # tensor_from_list = torch.FloatTensor(python_list)
# # tensor_from_numpy = torch.FloatTensor(numpy_array)
# #
# # print(f"从列表转换的张量: {tensor_from_list}, 类型: {type(tensor_from_list)}")
# # print(f"从numpy转换的张量: {tensor_from_numpy}, 类型: {type(tensor_from_numpy)}")
# #
# # print("\n为什么要转换为张量？")
# # print("• 神经网络只能处理张量，不能处理普通数组")
# # print("• 张量支持自动求导（计算梯度）")
# # print("• 张量可以在GPU上加速计算")
# # print("• 张量有丰富的数学运算方法")
# #
# # # ===== 2. unsqueeze() - 增加维度 =====
# # print("\n\n2. unsqueeze() - 增加维度")
# # print("-" * 40)
# #
# # # 一维张量
# # tensor_1d = torch.FloatTensor([1, 2, 3, 4])
# # print(f"原始1D张量: {tensor_1d}")
# # print(f"原始形状: {tensor_1d.shape}")
# #
# # # 在不同位置增加维度
# # tensor_2d_0 = tensor_1d.unsqueeze(0)  # 在第0维增加
# # tensor_2d_1 = tensor_1d.unsqueeze(1)  # 在第1维增加
# #
# # print(f"unsqueeze(0)后: {tensor_2d_0}")
# # print(f"形状变化: {tensor_1d.shape} -> {tensor_2d_0.shape}")
# # print(f"unsqueeze(1)后: {tensor_2d_1}")
# # print(f"形状变化: {tensor_1d.shape} -> {tensor_2d_1.shape}")
# #
# # print("\n为什么需要unsqueeze(0)？")
# # print("• 神经网络期望输入是批次数据: (batch_size, features)")
# # print("• 单个样本形状: (4,)")
# # print("• 添加批次维度后: (1, 4) - 表示1个样本，4个特征")
# #
# # # 实际示例
# # env = gym.make('CartPole-v1')
# # state = env.reset()
# # print(f"\n环境状态: {state}, 形状: {state.shape}")
# #
# # # 错误的做法（会报错）
# # try:
# #     # 假设有一个简单的神经网络
# #     linear = torch.nn.Linear(4, 2)
# #     # result = linear(torch.FloatTensor(state))  # 这会报错！
# #     print("直接输入会报错，因为缺少批次维度")
# # except:
# #     print("维度不匹配错误！")
# #
# # # 正确的做法
# # state_tensor = torch.FloatTensor(state).unsqueeze(0)
# # print(f"正确处理后: {state_tensor.shape}")
# # linear = torch.nn.Linear(4, 2)
# # result = linear(state_tensor)
# # print(f"神经网络输出: {result.shape}")
# #
# # # ===== 3. .sample() 方法 =====
# # print("\n\n3. .sample() 方法")
# # print("-" * 40)
# #
# # # OpenAI Gym的sample()
# # print("OpenAI Gym环境的.sample():")
# # env = gym.make('CartPole-v1')
# # random_action = env.action_space.sample()
# # print(f"动作空间: {env.action_space}")
# # print(f"随机动作: {random_action}")
# # print("这是gym环境自带的方法，用于随机选择动作")
# #
# # # Python random模块的sample()
# # print("\nPython random模块的sample():")
# # data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# # sampled = random.sample(data, 3)  # 从data中随机选择3个元素
# # print(f"原始数据: {data}")
# # print(f"随机采样3个: {sampled}")
# # print("这是Python标准库的方法，用于随机采样")
# #
# # # ===== 4. .item() 方法 =====
# # print("\n\n4. .item() 方法 - 张量转数值")
# # print("-" * 40)
# #
# # # 创建不同的张量
# # scalar_tensor = torch.tensor(42.0)
# # vector_tensor = torch.tensor([1, 2, 3])
# # matrix_tensor = torch.tensor([[1, 2], [3, 4]])
# #
# # print(f"标量张量: {scalar_tensor}, 类型: {type(scalar_tensor)}")
# # print(f"使用.item(): {scalar_tensor.item()}, 类型: {type(scalar_tensor.item())}")
# #
# # print(f"\n向量张量: {vector_tensor}")
# # try:
# #     # vector_tensor.item()  # 这会报错！
# #     print("向量张量不能使用.item()，因为包含多个元素")
# # except:
# #     print("多元素张量使用.item()会报错")
# #
# # # 实际使用场景
# # q_values = torch.tensor([0.3, 0.7])  # 两个动作的Q值
# # best_action_tensor = q_values.argmax()  # 返回张量
# # best_action_value = q_values.argmax().item()  # 转为Python数值
# #
# # print(f"\nQ值: {q_values}")
# # print(f"最佳动作(张量): {best_action_tensor}, 类型: {type(best_action_tensor)}")
# # print(f"最佳动作(数值): {best_action_value}, 类型: {type(best_action_value)}")
# # print("gym.step()需要Python整数，所以要用.item()转换")
# #
# # # ===== 5. 不计算梯度 torch.no_grad() =====
# # print("\n\n5. 不计算梯度 torch.no_grad()")
# # print("-" * 40)
# #
# # # 创建需要梯度的张量
# # x = torch.tensor([2.0], requires_grad=True)
# # print(f"输入张量: {x}, requires_grad: {x.requires_grad}")
# #
# # # 正常计算（会计算梯度）
# # print("\n正常计算（计算梯度）:")
# # y1 = x ** 2
# # print(f"y = x^2 = {y1}")
# # print(f"y.requires_grad: {y1.requires_grad}")
# # print("梯度图被构建，消耗内存和计算资源")
# #
# # # 使用no_grad()（不计算梯度）
# # print("\n使用no_grad()（不计算梯度）:")
# # with torch.no_grad():
# #     y2 = x ** 2
# #     print(f"y = x^2 = {y2}")
# #     print(f"y.requires_grad: {y2.requires_grad}")
# #     print("没有构建梯度图，节省内存和计算")
# #
# # print("\n在DQN中的应用:")
# # print("• 选择动作时不需要梯度，只是前向传播")
# # print("• 使用no_grad()可以：")
# # print("  - 节省内存（不存储梯度信息）")
# # print("  - 加快计算速度")
# # print("  - 避免意外的梯度计算")
# #
# #
# # # 实际示例
# # class SimpleNet(torch.nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.linear = torch.nn.Linear(4, 2)
# #
# #     def forward(self, x):
# #         return self.linear(x)
# #
# #
# # net = SimpleNet()
# # state = torch.FloatTensor([1, 2, 3, 4]).unsqueeze(0)
# #
# # print(f"\n示例网络输入: {state}")
# #
# # # 训练时（需要梯度）
# # output_train = net(state)
# # print(f"训练时输出: {output_train}")
# # print(f"需要梯度: {output_train.requires_grad}")
# #
# # # 推理时（不需要梯度）
# # with torch.no_grad():
# #     output_inference = net(state)
# #     action = output_inference.argmax().item()
# #
# # print(f"推理时输出: {output_inference}")
# # print(f"需要梯度: {output_inference.requires_grad}")
# # print(f"选择的动作: {action}")
# #
# # print("\n" + "=" * 60)
# # print("总结:")
# # print("• 张量转换：为了神经网络处理")
# # print("• unsqueeze(0)：添加批次维度")
# # print("• .sample()：随机采样方法")
# # print("• .item()：张量转Python数值")
# # print("• no_grad()：不计算梯度，节省资源")
# # print("=" * 60)
#
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
#
# print("=" * 60)
# print("张量类型和梯度概念详解")
# print("=" * 60)
#
# # ===== 1. FloatTensor vs tensor 区别 =====
# print("\n1. FloatTensor vs tensor 的区别")
# print("-" * 40)
#
# # 创建不同类型的张量
# data = [1, 2, 3, 4]
#
# # 方法1：FloatTensor（旧式API）
# tensor1 = torch.FloatTensor(data)
# print(f"FloatTensor: {tensor1}")
# print(f"数据类型: {tensor1.dtype}")
# print(f"设备: {tensor1.device}")
#
# # 方法2：tensor（新式API，推荐）
# tensor2 = torch.tensor(data, dtype=torch.float32)
# print(f"tensor: {tensor2}")
# print(f"数据类型: {tensor2.dtype}")
# print(f"设备: {tensor2.device}")
#
# # 方法3：tensor（自动推断类型）
# tensor3 = torch.tensor(data)
# print(f"tensor(自动): {tensor3}")
# print(f"数据类型: {tensor3.dtype}")
#
# print("\n区别总结:")
# print("• FloatTensor: 旧式API，固定float32类型")
# print("• tensor: 新式API，推荐使用，可指定类型")
# print("• tensor更灵活，支持更多数据类型和选项")
#
# # 不同数据类型示例
# print("\n不同数据类型示例:")
# int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
# float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
# double_tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
#
# print(f"int32: {int_tensor.dtype}")
# print(f"float32: {float_tensor.dtype}")
# print(f"float64: {double_tensor.dtype}")
#
# # ===== 2. 梯度是什么？为什么需要？ =====
# print("\n\n2. 梯度是什么？")
# print("-" * 40)
#
# print("梯度 = 函数的变化率 = 斜率")
# print("告诉我们：参数朝哪个方向变化，函数值变化最快")
#
# # 简单例子：一元函数 y = x²
# print("\n简单例子：y = x²")
# x_values = torch.linspace(-3, 3, 100)
# y_values = x_values ** 2
#
# print("函数 y = x²:")
# print("• 在 x = -2 处，斜率为负（向右下降）")
# print("• 在 x = 0 处，斜率为 0（最低点）")
# print("• 在 x = 2 处，斜率为正（向右上升）")
#
# # 计算梯度示例
# x = torch.tensor(2.0, requires_grad=True)  # 需要计算梯度
# y = x ** 2  # y = x²
#
# print(f"\n当 x = {x.item():.1f} 时:")
# print(f"y = x² = {y.item():.1f}")
#
# # 计算梯度
# y.backward()  # 自动计算梯度
# print(f"梯度 dy/dx = {x.grad.item():.1f}")
# print("数学上：dy/dx = 2x = 2×2 = 4 ✓")
#
# # ===== 3. 神经网络中的梯度 =====
# print("\n\n3. 神经网络中的梯度")
# print("-" * 40)
#
#
# # 简单神经网络示例
# class SimpleNet(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight = torch.nn.Parameter(torch.tensor(0.5))  # 可学习参数
#         self.bias = torch.nn.Parameter(torch.tensor(0.1))
#
#     def forward(self, x):
#         return self.weight * x + self.bias
#
#
# # 创建网络和数据
# net = SimpleNet()
# x = torch.tensor(2.0)
# target = torch.tensor(3.0)  # 期望输出
#
# print("简单神经网络: y = weight × x + bias")
# print(f"初始参数: weight = {net.weight.item():.2f}, bias = {net.bias.item():.2f}")
#
# # 前向传播
# output = net(x)
# print(f"输入 x = {x.item()}")
# print(f"网络输出 = {net.weight.item():.2f} × {x.item()} + {net.bias.item():.2f} = {output.item():.2f}")
# print(f"期望输出 = {target.item()}")
#
# # 计算损失
# loss = (output - target) ** 2
# print(f"损失 = (输出 - 目标)² = ({output.item():.2f} - {target.item()})² = {loss.item():.2f}")
#
# # 反向传播计算梯度
# loss.backward()
# print(f"\n梯度:")
# print(f"weight的梯度 = {net.weight.grad.item():.2f}")
# print(f"bias的梯度 = {net.bias.grad.item():.2f}")
#
# print("\n梯度的含义:")
# print("• weight梯度 > 0: 增加weight会增加损失")
# print("• bias梯度 > 0: 增加bias会增加损失")
# print("• 所以我们要减少这些参数（梯度下降）")
#
# # ===== 4. 梯度下降优化 =====
# print("\n\n4. 梯度下降优化过程")
# print("-" * 40)
#
# # 重新创建网络
# net = SimpleNet()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # 学习率0.1
#
# print("训练过程:")
# for epoch in range(5):
#     # 前向传播
#     output = net(x)
#     loss = (output - target) ** 2
#
#     # 清零梯度
#     optimizer.zero_grad()
#
#     # 反向传播
#     loss.backward()
#
#     # 更新参数
#     optimizer.step()
#
#     print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}, "
#           f"weight = {net.weight.item():.3f}, "
#           f"bias = {net.bias.item():.3f}")
#
# print(f"\n最终输出: {net(x).item():.3f} (目标: {target.item()})")
#
# # ===== 5. 为什么要关闭梯度？ =====
# print("\n\n5. 为什么要关闭梯度？")
# print("-" * 40)
#
# # 创建大一点的网络
# big_net = torch.nn.Sequential(
#     torch.nn.Linear(100, 50),
#     torch.nn.ReLU(),
#     torch.nn.Linear(50, 1)
# )
#
# input_data = torch.randn(1, 100)
#
# print("计算内存使用（模拟）:")
#
# # 正常计算（存储梯度）
# print("\n1. 正常计算（requires_grad=True）:")
# output1 = big_net(input_data)
# print(f"输出: {output1.item():.3f}")
# print(f"是否需要梯度: {output1.requires_grad}")
# print("内存占用: 高（存储所有中间计算图）")
#
# # 关闭梯度计算
# print("\n2. 关闭梯度（torch.no_grad()）:")
# with torch.no_grad():
#     output2 = big_net(input_data)
#     print(f"输出: {output2.item():.3f}")
#     print(f"是否需要梯度: {output2.requires_grad}")
#     print("内存占用: 低（不存储计算图）")
#
# print("\n什么时候关闭梯度？")
# print("• 测试/推理时：只要结果，不需要训练")
# print("• DQN选择动作时：只要Q值，不需要学习")
# print("• 数据预处理时：不涉及模型参数更新")
# print("• 计算统计信息时：如准确率、损失监控等")
#
# # ===== 6. DQN中的实际应用 =====
# print("\n\n6. DQN中的实际应用")
# print("-" * 40)
#
#
# class MiniDQN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = torch.nn.Linear(4, 2)  # 4个状态 -> 2个动作
#
#     def forward(self, x):
#         return self.fc(x)
#
#
# dqn = MiniDQN()
# state = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # 批次大小为1
#
# print("DQN中的两种使用场景:")
#
# # 场景1：训练时（需要梯度）
# print("\n场景1：训练时")
# q_values_train = dqn(state)
# print(f"Q值: {q_values_train}")
# print(f"需要梯度: {q_values_train.requires_grad}")
# print("用途: 计算损失，更新网络参数")
#
# # 场景2：选择动作时（不需要梯度）
# print("\n场景2：选择动作时")
# with torch.no_grad():
#     q_values_action = dqn(state)
#     action = q_values_action.argmax().item()
#
# print(f"Q值: {q_values_action}")
# print(f"需要梯度: {q_values_action.requires_grad}")
# print(f"选择动作: {action}")
# print("用途: 快速选择最佳动作，节省内存")
#
# print("\n" + "=" * 60)
# print("总结:")
# print("• FloatTensor: 旧API，固定float32")
# print("• tensor: 新API，更灵活（推荐）")
# print("• 梯度: 告诉我们参数如何调整以减少误差")
# print("• no_grad(): 推理时使用，节省内存和计算")
# print("=" * 60)
import pandas
