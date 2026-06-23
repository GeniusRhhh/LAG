# import logging
# import torch
# import torch.nn.functional as F
# import wandb
#
# class SACTrainer:
#     """
#     SAC Trainer for LAG platform
#     """
#     def __init__(self, args, policy, device=torch.device("cpu")):
#         self.args = args
#         self.policy = policy
#         self.device = device
#         self.max_grad_norm = 1.0  # Increased from 0.5
#         self.use_recurrent_policy = args.use_recurrent_policy
#
#     def update(self, batch):
#         self.policy.prep_training()
#
#         obs = torch.from_numpy(batch["obs"]).float().to(self.device)
#         act = torch.from_numpy(batch["act"]).float().to(self.device)
#         rew = torch.from_numpy(batch["rew"]).float().to(self.device)
#         next_obs = torch.from_numpy(batch["next_obs"]).float().to(self.device)
#         done = torch.from_numpy(batch["done"]).float().to(self.device)
#
#         B, N, A = obs.shape[:3]
#         obs_flat = obs.reshape(B * N * A, -1)
#         act_flat = act.reshape(B * N * A, -1)
#         next_obs_flat = next_obs.reshape(B * N * A, -1)
#         rew_flat = rew.reshape(B * N * A, 1)
#         done_flat = done.reshape(B * N * A, 1)
#
#         with torch.no_grad():
#             next_a, next_logp, _ = self.policy.actor(next_obs_flat, deterministic=False)
#             q1_next, q2_next = self.policy.critic_target(next_obs_flat, next_a)
#             q_next = torch.min(q1_next, q2_next) - self.policy.alpha * next_logp
#             q_target = rew_flat + self.policy.gamma * (1 - done_flat) * q_next
#             q_target = torch.clamp(q_target, -100, 100)  # Clip Q values
#
#         q1, q2 = self.policy.critic(obs_flat, act_flat)
#         critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
#         self.policy.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
#         self.policy.critic_optimizer.step()
#
#         curr_a, curr_logp, _ = self.policy.actor(obs_flat, deterministic=False)
#         q1_pi, q2_pi = self.policy.critic(obs_flat, curr_a)
#         q_pi = torch.min(q1_pi, q2_pi)
#         curr_logp = torch.clamp(curr_logp, -20.0, 0.0)
#         actor_loss = (self.policy.alpha * curr_logp - q_pi).mean()
#         self.policy.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
#         self.policy.actor_optimizer.step()
#
#         alpha_loss = (self.policy.alpha * (-curr_logp + self.policy.target_entropy).detach()).mean()
#         self.policy.alpha_optimizer.zero_grad()
#         alpha_loss.backward()
#         self.policy.alpha_optimizer.step()
#
#         self.policy.soft_update()
#
#         metrics = {
#             "critic_loss": critic_loss.item(),
#             "actor_loss": actor_loss.item(),
#             "alpha_loss": alpha_loss.item(),
#             "alpha": self.policy.alpha.item(),
#             "q_mean": q_pi.mean().item(),
#             "q_std": q_pi.std().item(),
#             "reward_mean": rew_flat.mean().item(),
#             "reward_std": rew_flat.std().item()
#         }
#         logging.debug(
#             f"Training metrics: critic_loss={metrics['critic_loss']:.4f}, actor_loss={metrics['actor_loss']:.4f}, alpha={metrics['alpha']:.4f}, q_mean={metrics['q_mean']:.4f}, reward_mean={metrics['reward_mean']:.4f}")
#         return metrics

import logging
import torch
import torch.nn.functional as F
import wandb
import numpy as np

class SACTrainer:
    def __init__(self, policy):
        self.policy = policy

    def update(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, total_steps=0):
        self.policy.prep_training()

        obs_batch = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.policy.device)
        act_batch = torch.as_tensor(act_batch, dtype=torch.float32, device=self.policy.device)
        rew_batch = torch.as_tensor(rew_batch, dtype=torch.float32, device=self.policy.device)
        next_obs_batch = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.policy.device)
        done_batch = torch.as_tensor(done_batch, dtype=torch.float32, device=self.policy.device)

        # Flatten B*N env
        B, N, obs_dim = obs_batch.shape
        act_dim = act_batch.shape[-1]
        obs_batch = obs_batch.reshape(B * N, obs_dim)
        act_batch = act_batch.reshape(B * N, act_dim)
        next_obs_batch = next_obs_batch.reshape(B * N, obs_dim)
        rew_batch = rew_batch.reshape(B * N, 1)
        done_batch = done_batch.reshape(B * N, 1)

        # 计算 Q_target
        with torch.no_grad():
            next_a, next_logp = self.policy.actor(next_obs_batch, deterministic=False)
            q1_next, q2_next = self.policy.critic_target(next_obs_batch, next_a)
            q_next = torch.min(q1_next, q2_next) - self.policy.alpha * next_logp
            q_target = rew_batch + self.policy.gamma * (1 - done_batch) * q_next
            q_target = torch.clamp(q_target, -100, 100)

        # 优化 Critic
        q1, q2 = self.policy.critic(obs_batch, act_batch)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=1.0)
        self.policy.critic_optimizer.step()

        # 优化 Actor
        curr_a, curr_logp = self.policy.actor(obs_batch, deterministic=False)
        q1_pi, q2_pi = self.policy.critic(obs_batch, curr_a)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.policy.alpha * curr_logp - q_pi).mean()
        self.policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=1.0)
        self.policy.actor_optimizer.step()

        # 优化 alpha，添加最小值限制
        alpha_loss = (self.policy.alpha * (-curr_logp - self.policy.target_entropy).detach()).mean()
        self.policy.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.policy.alpha_optimizer.step()
        # 限制 alpha 不低于 0.01
        with torch.no_grad():
            self.policy.log_alpha.copy_(torch.clamp(self.policy.log_alpha, min=np.log(0.01)))

        # 软更新
        self.policy.soft_update()

        # 日志记录动作分布
        action_mean = curr_a.mean(dim=0).cpu().detach().numpy()
        action_std = curr_a.std(dim=0).cpu().detach().numpy()

        train_metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "q_mean": q_pi.mean().item(),
            "alpha": self.policy.alpha.item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item()
        }

        if total_steps % 500 == 0 and total_steps > 0:
            logging.info(
                f"Step {total_steps} Train Metrics - "
                f"critic_loss={train_metrics['critic_loss']:.4f}, "
                f"actor_loss={train_metrics['actor_loss']:.4f}, "
                f"q1_mean={train_metrics['q1_mean']:.4f}, q2_mean={train_metrics['q2_mean']:.4f}, "
                f"q_mean={train_metrics['q_mean']:.4f}, alpha={train_metrics['alpha']:.4f}, "
                f"actor_grad_norm={train_metrics['actor_grad_norm']:.4f}, "
                f"critic_grad_norm={train_metrics['critic_grad_norm']:.4f}"
            )
            logging.info(
                f"Step {total_steps} Action Distribution: "
                f"mean={action_mean.tolist()}, std={action_std.tolist()}"
            )
            if hasattr(self.policy, 'use_wandb') and self.policy.use_wandb:
                wandb.log({
                    "step": total_steps,
                    "critic_loss": train_metrics['critic_loss'],
                    "actor_loss": train_metrics['actor_loss'],
                    "q1_mean": train_metrics['q1_mean'],
                    "q2_mean": train_metrics['q2_mean'],
                    "q_mean": train_metrics['q_mean'],
                    "alpha": train_metrics['alpha'],
                    "actor_grad_norm": train_metrics['actor_grad_norm'],
                    "critic_grad_norm": train_metrics['critic_grad_norm'],
                    "action_mean": action_mean.tolist(),
                    "action_std": action_std.tolist()
                })

        return train_metrics