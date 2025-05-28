import logging
import torch
import torch.nn as nn
from algorithms.sac.sac_actor import ActorNet
from algorithms.sac.sac_critic import SACCritic
from ..utils.utils import check

class SACPolicy(nn.Module):
    def __init__(self, args, obs_space, act_space):
        super(SACPolicy, self).__init__()  # 继承 nn.Module
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
        self.gamma = args.gamma
        self.tau = getattr(args, 'tau', 0.005)
        self.smooth_factor = 0.01

        act_dim = act_space.shape[0]
        self.actor = ActorNet(args, obs_space, act_space, device=self.device).to(self.device)
        self.prev_action = None

        self.critic = SACCritic(args, obs_space, act_dim, device=self.device).to(self.device)
        self.critic_target = SACCritic(args, obs_space, act_dim, device=self.device).to(self.device)

        self.log_alpha = torch.tensor([float(args.init_alpha)], device=self.device).log().requires_grad_()
        self.target_entropy = -1.0
        logging.info(f"SACPolicy 初始化: target_entropy={self.target_entropy}, init_alpha={args.init_alpha}")

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def prep_training(self):
        self.actor.train()
        self.critic.train()
        self.critic_target.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

    def get_action(self, obs, deterministic=False):
        obs = check(obs).to(self.device)
        action, log_pi = self.actor.get_action(obs, deterministic=deterministic)
        if self.prev_action is not None:
            action_diff = action - self.prev_action
            smooth_loss = self.smooth_factor * torch.mean(action_diff ** 2)
            action = action - smooth_loss.detach()
        action = torch.clamp(action, torch.tensor([-1.0, -1.0, -1.0, 0.4], device=self.device),
                            torch.tensor([1.0, 1.0, 1.0, 0.9], device=self.device))
        self.prev_action = action.detach().clone()
        action = action.detach().cpu().numpy()
        log_pi = log_pi.detach().cpu().numpy() if log_pi is not None else None
        return action, log_pi

    def soft_update(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def state_dict(self):
        """自定义 state_dict，包含所有组件状态"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'prev_action': self.prev_action
        }

    def load_state_dict(self, state_dict):
        """自定义 load_state_dict，加载所有组件状态"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.log_alpha.data.copy_(state_dict['log_alpha'])
        self.prev_action = state_dict.get('prev_action', None)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)