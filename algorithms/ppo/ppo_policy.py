import logging

import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic

class PPOPolicy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args  # 参数配置
        self.device = device  # 运行设备配置

        # 优化器配置
        self.lr = args.lr  # 学习率
        self.obs_space = obs_space  # 观测空间
        self.act_space = act_space  # 动作空间

        # 创建Actor和Critic对象
        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.obs_space, self.device)

        # 配置优化器，同时优化actor和critic的参数
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        计算动作、价值和日志概率。
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks):
        """
        获取当前观测的价值评估。
        Returns:
            values
        """
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        """
        评估给定动作的价值、动作日志概率和分布熵。用于训练损失计算。
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        仅计算动作和更新RNN状态，用于推断。
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        """
        准备训练，将Actor和Critic设为训练模式。
        """
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        """
        准备推断，将Actor和Critic设为评估模式。
        """
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        """
        创建这个策略的一个副本。
        """
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
