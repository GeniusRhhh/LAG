import logging
import torch
import torch.nn as nn
from algorithms.sac.sac_actor import ActorNet
from algorithms.sac.sac_critic import SACCritic
from ..utils.utils import check

class SACPolicy:
    def __init__(self, args, obs_space, act_space):
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
        self.gamma = args.gamma
        self.tau = args.tau

        # Actor
        act_dim = act_space.shape[0]
        self.actor = ActorNet(args, obs_space, act_space, device=self.device).to(self.device)

        # Critic + Target Critic
        self.critic = SACCritic(args, obs_space, act_dim, device=self.device).to(self.device)
        self.critic_target = SACCritic(args, obs_space, act_dim, device=self.device).to(self.device)

        # Alpha
        self.log_alpha = torch.tensor([float(args.init_alpha)], device=self.device).log().requires_grad_()
        self.target_entropy = -act_dim
        logging.info(f"SACPolicy 初始化: target_entropy={self.target_entropy}, init_alpha={args.init_alpha}")

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)

    @property
    def alpha(self):
        """Return the exponential of log_alpha as the SAC alpha parameter."""
        return self.log_alpha.exp()

    def prep_training(self):
        """Set models to training mode."""
        self.actor.train()
        self.critic.train()
        self.critic_target.train()

    def get_action(self, obs, deterministic=False):
        obs = check(obs).to(self.device)
        action, log_pi = self.actor.get_action(obs, deterministic=deterministic)
        return action, log_pi

    def soft_update(self):
        """Soft update the target critic network."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'])