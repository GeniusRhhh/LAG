import logging
import torch
import torch.nn as nn
from algorithms.sac.sac_actor import ActorNet
from algorithms.sac.sac_critic import SACCritic
from ..utils.utils import check

class SACPolicy(nn.Module):
    """
    SAC Policy integrating Actor, Critic, Target Critic, and alpha
    """
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        super(SACPolicy, self).__init__()
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.num_agents = num_agents
        self.n_rollout_threads = getattr(args, 'n_rollout_threads', 1)
        logging.info(f"SACPolicy initialized: use_recurrent_policy={self.use_recurrent_policy}, num_agents={self.num_agents}, n_rollout_threads={self.n_rollout_threads}")

        act_dim = act_space.shape[0]
        self.actor = ActorNet(args, obs_space, act_space, self.device)
        self.critic = SACCritic(args, obs_space, act_dim, self.device)
        self.critic_target = SACCritic(args, obs_space, act_dim, self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor([float(args.init_alpha)], device=self.device).log().requires_grad_()
        self.target_entropy = args.target_entropy if hasattr(args, "target_entropy") else -act_dim

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, weight_decay=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=1e-4)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.to(self.device)

    @property
    def alpha(self):
        return torch.clamp(self.log_alpha.exp(), min=0.01, max=10.0)

    def soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_actions(self, obs, rnn_states_actor=None, masks=None, deterministic=False):
        obs = check(obs).to(device=self.device)
        batch_shape = obs.shape[:-1]
        obs_flat = obs.reshape(-1, obs.shape[-1])

        if torch.isnan(obs_flat).any() or torch.isinf(obs_flat).any():
            logging.warning(f"Invalid obs detected: {obs_flat}")
            obs_flat = torch.clamp(obs_flat, -10, 10)

        if self.use_recurrent_policy:
            if rnn_states_actor is None:
                rnn_states_actor = torch.zeros(
                    (self.recurrent_hidden_layers, obs_flat.size(0), self.recurrent_hidden_size),
                    dtype=torch.float32, device=self.device
                )
            else:
                rnn_states_actor = check(rnn_states_actor).to(device=self.device)
                if rnn_states_actor.dim() == 4:
                    rnn_states_actor = rnn_states_actor.view(-1, *rnn_states_actor.shape[2:])
                if rnn_states_actor.dim() == 3 and rnn_states_actor.size(0) != self.recurrent_hidden_layers:
                    rnn_states_actor = rnn_states_actor.transpose(0, 1)
            if masks is None:
                masks = torch.ones((obs_flat.size(0), 1), dtype=torch.float32, device=self.device)
            else:
                masks = check(masks).to(device=self.device)
                if masks.dim() == 3:
                    masks = masks.view(-1, 1)
            rnn_states_flat = rnn_states_actor
            masks_flat = masks
        else:
            rnn_states_flat = None
            masks_flat = None
            n_threads = batch_shape[0]
            rnn_states_actor = torch.zeros(
                (n_threads, self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size),
                dtype=torch.float32, device=self.device
            )

        self.actor.eval()
        with torch.no_grad():
            actions, log_probs, new_rnn_states = self.actor(obs_flat, rnn_states_flat, masks_flat,
                                                            deterministic=deterministic)

        actions = actions.reshape(*batch_shape, -1)
        if log_probs is not None:
            # log_probs 已经是 [batch_size, 1]，直接重塑
            log_probs = log_probs.reshape(*batch_shape, 1)
        else:
            log_probs = torch.zeros((*batch_shape, 1), device=self.device)

        if self.use_recurrent_policy and new_rnn_states is not None:
            new_rnn_states = new_rnn_states.transpose(0, 1).reshape(
                batch_shape[0], self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size
            )
        else:
            new_rnn_states = rnn_states_actor

        if hasattr(self.actor.act_layer, 'action_low'):
            low, high = self.actor.act_layer.action_low, self.actor.act_layer.action_high
            actions = torch.clamp(actions, low, high)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), new_rnn_states.cpu().numpy()
    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def update(self, data):
        obs = check(data['obs']).to(self.device)
        actions = check(data['actions']).to(self.device)
        rewards = check(data['rewards']).to(self.device)
        next_obs = check(data['next_obs']).to(self.device)
        dones = check(data['dones']).to(self.device)

        for name, tensor in [('obs', obs), ('actions', actions), ('rewards', rewards), ('next_obs', next_obs), ('dones', dones)]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logging.warning(f"Invalid {name} detected: {tensor}")
                tensor = torch.clamp(tensor, -10, 10)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor(next_obs, deterministic=False)
            next_log_probs = torch.clamp(next_log_probs, -1000, 0)
            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
            target_q = torch.clamp(target_q, -100, 100)  # Clip Q values

        q1, q2 = self.critic(obs, actions)
        critic_loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actions, log_probs, _ = self.actor(obs, deterministic=False)
        log_probs = torch.clamp(log_probs, -1000, 0)
        q1_pi, q2_pi = self.critic(obs, actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_probs - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update()

        return actor_loss.item(), critic_loss.item(), alpha_loss.item()

    def save(self, path):
        data = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "alpha_opt": self.alpha_optimizer.state_dict(),
        }
        torch.save(data, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.critic_target.load_state_dict(data["critic_target"])
        self.log_alpha.data = torch.tensor(data["log_alpha"], device=self.device, requires_grad=True)
        self.actor_optimizer.load_state_dict(data["actor_opt"])
        self.critic_optimizer.load_state_dict(data["critic_opt"])
        self.alpha_optimizer.load_state_dict(data["alpha_opt"])