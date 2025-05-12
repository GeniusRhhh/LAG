import torch
import torch.nn as nn
from typing import Union, List
from .ppo_policy import PPOPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm

class PPOTrainer():
    def __init__(self, args, device=torch.device("cpu")):
        """
        初始化PPO训练器。
        :param args: 配置参数。
        :param device: 计算设备，默认为CPU。
        """
        self.device = device  # 设置计算设备
        self.tpdv = dict(dtype=torch.float32, device=device)  # 数据类型和设备配置
        # PPO算法配置参数
        self.ppo_epoch = args.ppo_epoch  # PPO训练的周期数
        self.clip_param = args.clip_param  # PPO的裁剪参数
        self.use_clipped_value_loss = args.use_clipped_value_loss  # 是否使用裁剪的价值损失
        self.num_mini_batch = args.num_mini_batch  # 每次更新的最小批次数
        self.value_loss_coef = args.value_loss_coef  # 价值损失的系数
        self.entropy_coef = args.entropy_coef  # 熵系数
        self.use_max_grad_norm = args.use_max_grad_norm  # 是否使用最大梯度规范化
        self.max_grad_norm = args.max_grad_norm  # 最大梯度范数
        self.use_recurrent_policy = args.use_recurrent_policy  # 是否使用递归策略
        self.data_chunk_length = args.data_chunk_length  # rnn数据块长度

    def ppo_update(self, policy: PPOPolicy, sample):
        """
        执行一次PPO更新。
        :param policy: PPO策略对象。
        :param sample: 采样的数据。
        :returns: 各种损失和梯度信息。
        """
        # 解包采样数据
        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample
        # 转换数据类型和设备
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # 在所有步骤上执行单次前向传递
        values, action_log_probs, dist_entropy = policy.evaluate_actions(
            obs_batch, rnn_states_actor_batch, rnn_states_critic_batch, actions_batch, masks_batch
        )

        # 计算损失
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
        policy_loss = -policy_loss.mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()#计算策略的熵损失，熵越大表示策略越随机。

        # 总损失
        loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef

        # 优化损失函数
        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, policy: PPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        """
        训练策略。
        :param policy: PPO策略对象。
        :param buffer: 数据缓冲区。
        :returns: 训练信息字典。
        """
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                raise NotImplementedError

            for sample in data_generator:
                policy_loss, value_loss, policy_entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
