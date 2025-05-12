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

class SACTrainer:
    """
    SAC Trainer for LAG platform
    """
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.args = args
        self.policy = policy
        self.device = device
        self.max_grad_norm = 10.0  # Increased from 1.0
        self.use_recurrent_policy = args.use_recurrent_policy

    def update(self, batch):
        self.policy.prep_training()

        obs = torch.from_numpy(batch["obs"]).float().to(self.device)
        act = torch.from_numpy(batch["act"]).float().to(self.device)
        rew = torch.from_numpy(batch["rew"]).float().to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).float().to(self.device)
        done = torch.from_numpy(batch["done"]).float().to(self.device)

        B, N, A = obs.shape[:3]
        obs_flat = obs.reshape(B * N * A, -1)
        act_flat = act.reshape(B * N * A, -1)
        next_obs_flat = next_obs.reshape(B * N * A, -1)
        rew_flat = rew.reshape(B * N * A, 1)
        done_flat = done.reshape(B * N * A, 1)

        # Prepare data dictionary for SACPolicy.update
        data = {
            "obs": obs_flat,
            "actions": act_flat,
            "rewards": rew_flat,
            "next_obs": next_obs_flat,
            "dones": done_flat
        }

        # Call SACPolicy.update to perform the training step
        actor_loss, critic_loss, alpha_loss = self.policy.update(data)

        # Compute additional metrics
        with torch.no_grad():
            q1, q2 = self.policy.critic(obs_flat, act_flat)
            q_pi = torch.min(q1, q2)

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.policy.alpha.item(),
            "q_mean": q_pi.mean().item(),
            "q_std": q_pi.std().item(),
            "reward_mean": rew_flat.mean().item(),
            "reward_std": rew_flat.std().item()
        }
        logging.debug(
            f"Training metrics: critic_loss={metrics['critic_loss']:.4f}, actor_loss={metrics['actor_loss']:.4f}, "
            f"alpha={metrics['alpha']:.4f}, q_mean={metrics['q_mean']:.4f}, reward_mean={metrics['reward_mean']:.4f}")
        return metrics