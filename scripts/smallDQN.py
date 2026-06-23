import gym
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque


# 定义 DQN 神经网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        # 初始化父类 nn.Module
        super(DQN, self).__init__()
        # 定义三层全连接网络：输入状态维度 -> 64 -> 64 -> 动作维度
        self.fc1 = nn.Linear(state_dim, 64)  # 第一层：状态到隐藏层
        self.fc2 = nn.Linear(64, 64)  # 第二层：隐藏层到隐藏层
        self.fc3 = nn.Linear(64, action_dim)  # 第三层：隐藏层到动作 Q 值

    def forward(self, x):
        # 前向传播
        x = torch.relu(self.fc1(x))  # 第一层激活
        x = torch.relu(self.fc2(x))  # 第二层激活
        return self.fc3(x)  # 输出每个动作的 Q 值


# DQN 算法
def train_dqn(env, episodes=500, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]  # 状态向量维度（如 CartPole 的 4）
    action_dim = env.action_space.n  # 动作数量（如左、右为 2）

    # 创建经验回放缓冲区，存储 (s, a, r, s', done) 元组
    memory = deque(maxlen=10000)
    batch_size = 64  # 每次训练的批量大小

    # 初始化主网络和目标网络
    policy_net = DQN(state_dim, action_dim)  # 主网络，实时更新
    target_net = DQN(state_dim, action_dim)  # 目标网络，定期同步
    target_net.load_state_dict(policy_net.state_dict())  # 复制主网络参数
    target_net.eval()  # 目标网络仅用于推理

    # 定义优化器，使用 Adam 优化主网络参数
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)

    # 训练循环
    for episode in range(episodes):
        # 重置环境，获取初始状态
        state = env.reset()  # 返回初始状态，例如 [0.02, 0.01, -0.03, 0.04]
        state = torch.FloatTensor(state).unsqueeze(0)  # 转换为张量，形状 (1, 4)
        total_reward = 0  # 记录回合总奖励

        while True:
            # ε-贪心策略选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()  # 随机选择动作（探索）
            else:
                with torch.no_grad():  # 无需计算梯度，节省内存
                    action = policy_net(state).argmax().item()  # 选择 Q 值最大的动作

            # 执行动作，获取下一状态、奖励和终止标志
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)  # 转换为张量
            # 存储经验到缓冲区
            memory.append((state, action, reward, next_state, done))
            state = next_state  # 更新状态
            total_reward += reward  # 累加奖励

            # 经验回放：当缓冲区足够大时进行训练
            if len(memory) >= batch_size:
                # 随机采样一批经验
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # 将批量数据转换为张量
                states = torch.cat(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.cat(next_states)
                dones = torch.FloatTensor(dones)

                # 计算当前 Q 值：Q(s, a)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                # 计算目标 Q 值：r + γ * max_a' Q(s', a'; θ^-)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + gamma * next_q_values * (1 - dones)

                # 计算均方误差损失
                loss = nn.MSELoss()(q_values, targets)

                # 反向传播优化主网络
                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数

            if done:
                break  # 回合结束

        # 每 10 回合同步目标网络
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 衰减探索率 ε
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 打印回合信息
        print(f"Episode {episode}, Total Reward: {total_reward}")

    return policy_net


# 主程序
# if __name__ == "__main__":
#     # 创建 CartPole 环境
#     env = gym.make('CartPole-v1')
#     # 运行 DQN 算法
#     policy_net = train_dqn(env)

X=np.full(shape=(8,1),fill_value=1)
print("x:",X)

Y=np.concatenate([X,np.full(shape=(8,1),fill_value=1)],axis=1)
print("y:",Y)
