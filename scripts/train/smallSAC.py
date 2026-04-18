import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt  # 用于绘制奖励曲线

# Actor 网络定义：生成动作的网络
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        """
        初始化 Actor 网络
        :param state_dim: 状态维度
        :param action_dim: 动作维度
        :param max_action: 动作的最大值
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # 第一个全连接层
        self.fc2 = nn.Linear(256, 256)  # 第二个全连接层
        self.mean = nn.Linear(256, action_dim)  # 输出动作均值
        self.log_std = nn.Linear(256, action_dim)  # 输出动作的对数标准差
        self.max_action = max_action  # 最大动作值，用于限制动作范围

    def forward(self, state):
        """
        前向传播
        :param state: 当前状态
        :return: 动作均值和对数标准差
        """
        x = torch.relu(self.fc1(state))  # 激活函数 ReLU
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x)) * self.max_action  # 使用 tanh 将动作范围限制到 [-max_action, max_action]
        log_std = torch.clamp(self.log_std(x), min=-10, max=2)  # 限制 log_std 的范围
        return mean, log_std

    def sample(self, state):
        """
        从策略分布中采样动作
        :param state: 当前状态
        :return: 采样的动作和对应的 log 概率
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)  # 将 log 标准差转为标准差
        std = torch.clamp(std, min=1e-6)  # 防止标准差为 0
        normal = torch.distributions.Normal(mean, std)  # 正态分布
        action = normal.rsample()  # 使用重参数化技巧采样动作
        action = torch.clamp(action, -self.max_action, self.max_action)  # 限制动作范围
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)  # 计算动作的 log 概率
        return action, log_prob

# Critic 网络定义：计算 Q 值的网络
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        初始化 Critic 网络
        :param state_dim: 状态维度
        :param action_dim: 动作维度
        """
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)  # 输入状态和动作
        self.fc2 = nn.Linear(256, 256)
        self.q_value = nn.Linear(256, 1)  # 输出 Q 值

    def forward(self, state, action):
        """
        前向传播
        :param state: 当前状态
        :param action: 当前动作
        :return: Q 值
        """
        x = torch.cat([state, action], dim=1)  # 拼接状态和动作作为输入
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q_value(x)

# SAC 算法定义
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        """
        初始化 SAC 智能体
        :param state_dim: 状态维度
        :param action_dim: 动作维度
        :param max_action: 动作的最大值
        """
        self.actor = ActorNetwork(state_dim, action_dim, max_action).cuda()  # 策略网络
        self.critic1 = CriticNetwork(state_dim, action_dim).cuda()  # 第一个 Q 网络
        self.critic2 = CriticNetwork(state_dim, action_dim).cuda()  # 第二个 Q 网络
        self.target_critic1 = CriticNetwork(state_dim, action_dim).cuda()  # 第一个目标 Q 网络
        self.target_critic2 = CriticNetwork(state_dim, action_dim).cuda()  # 第二个目标 Q 网络

        # 将目标 Q 网络的参数设置为与 Q 网络相同
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005  # 软更新系数
        self.alpha = 0.2  # 熵正则项系数

    def select_action(self, state):
        """
        选择动作
        :param state: 当前状态
        :return: 动作
        """
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        """
        训练 SAC 模型
        :param replay_buffer: 回放缓冲区
        :param batch_size: 批量大小
        """
        # 从回放缓冲区中采样
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).cuda()
        action = torch.FloatTensor(action).cuda()
        reward = torch.FloatTensor(reward).unsqueeze(1).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        done = torch.FloatTensor(done).unsqueeze(1).cuda()

        # 计算目标 Q 值
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # 更新 Critic 网络
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新 Actor 网络
        new_action, log_prob = self.actor.sample(state)
        q1_value = self.critic1(state, new_action)
        actor_loss = (self.alpha * log_prob - q1_value).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标 Q 网络
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# 回放缓冲区
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        data = [self.storage[i] for i in ind]
        return map(np.stack, zip(*data))

# 训练函数
def train():
    """
    训练 SAC 模型并绘制奖励曲线和动态可视化
    """
    env = gym.make("Pendulum-v1")  # 创建环境
    state_dim = env.observation_space.shape[0]  # 状态维度
    action_dim = env.action_space.shape[0]  # 动作维度
    max_action = float(env.action_space.high[0])  # 最大动作值

    agent = SACAgent(state_dim, action_dim, max_action)  # 初始化 SAC 智能体
    replay_buffer = ReplayBuffer()  # 初始化回放缓冲区
    episodes = 200  # 训练回合数
    batch_size = 256  # 批量大小

    rewards = []  # 存储每回合奖励

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0  # 累计奖励
        for _ in range(1000):
            env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))  # 存储经验
            state = next_state  # 更新状态
            ep_reward += reward

            if len(replay_buffer.storage) > batch_size:
                agent.train(replay_buffer, batch_size)

            if done:
                break
        rewards.append(ep_reward)  # 记录奖励
        print(f"Episode {ep + 1}, Reward: {ep_reward}")  # 打印奖励

    # 绘制奖励曲线
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SAC Training Rewards")
    plt.show()

if __name__ == "__main__":
    train()
