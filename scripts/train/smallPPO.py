import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

    def evaluate(self, states, actions):
        probs = self.forward(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# PPO 算法主体
class PPO:
    def __init__(self, state_dim, action_dim, batch_size=64, update_epochs=4):
        self.actor = Actor(state_dim, action_dim).cuda()
        self.critic = Critic(state_dim).cuda()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.batch_size = batch_size
        self.update_epochs = update_epochs

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns_and_advantages(self, rewards, values, done):
        returns = []
        advantages = []
        discounted_return = 0
        for step in reversed(range(len(rewards))):
            if done[step]:
                discounted_return = 0
            discounted_return = rewards[step] + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        returns = torch.tensor(returns).cuda()
        values = torch.tensor(values).cuda()
        advantages = returns - values
        return returns, advantages

    def update(self, memory):
        states = torch.FloatTensor(memory['states']).cuda()
        actions = torch.LongTensor(memory['actions']).cuda()
        log_probs = torch.FloatTensor(memory['log_probs']).cuda()
        returns = torch.FloatTensor(memory['returns']).cuda()
        advantages = torch.FloatTensor(memory['advantages']).cuda()

        for _ in range(self.update_epochs):
            for i in range(0, len(states), self.batch_size):
                state_batch = states[i:i + self.batch_size]
                action_batch = actions[i:i + self.batch_size]
                old_log_probs = log_probs[i:i + self.batch_size]
                return_batch = returns[i:i + self.batch_size]
                advantage_batch = advantages[i:i + self.batch_size]

                new_log_probs, entropy = self.actor.evaluate(state_batch, action_batch)
                critic_value = self.critic(state_batch).squeeze()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surrogate1 = ratio * advantage_batch
                surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage_batch
                actor_loss = -torch.min(surrogate1, surrogate2).mean() - 0.01 * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                critic_loss = nn.MSELoss()(critic_value, return_batch)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

# 主函数
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, action_dim)
    max_episodes = 500
    max_steps = 200
    rewards_history = []

    for episode in range(max_episodes):
        state = env.reset()
        memory = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'values': [], 'dones': []}
        total_reward = 0

        for _ in range(max_steps):
            value = agent.critic(torch.FloatTensor(state).unsqueeze(0).cuda()).item()
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            memory['states'].append(state)
            memory['actions'].append(action)
            memory['rewards'].append(reward)
            memory['log_probs'].append(log_prob.item())
            memory['values'].append(value)
            memory['dones'].append(done)

            state = next_state
            total_reward += reward

            if done:
                break

        returns, advantages = agent.compute_returns_and_advantages(
            memory['rewards'], memory['values'], memory['dones']
        )
        memory['returns'] = returns
        memory['advantages'] = advantages

        agent.update(memory)
        rewards_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Rewards")
    plt.show()
