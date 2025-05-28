import logging
import time
import numpy as np
import torch

from algorithms.utils.buffer import SharedReplayBuffer
from .base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class ShareJSBSimRunner(Runner):
    def load(self):
        self.obs_space = self.envs.observation_space  # (33,)
        self.share_obs_space = self.envs.share_observation_space  # (132,)
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay

        from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
        from algorithms.mappo.ppo_policy import PPOPolicy as Policy
        self.policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        self.buffer = SharedReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.share_obs_space, self.act_space)

    def run(self):
        self.warmup()
        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads

        for episode in range(episodes):
            obs, share_obs = self.envs.reset()
            for step in range(self.buffer_size):
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)
                obs, share_obs, rewards, dones, infos = self.envs.step(actions)
                data = (obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic)
                self.insert(data)
            logging.info("Before calling ShareJSBSimRunner.compute")
            # 显式调用 ShareJSBSimRunner.compute，并添加调试信息
            logging.info("Calling ShareJSBSimRunner.compute")
            self.compute()  # 确保调用的是 ShareJSBSimRunner.compute
            logging.info("After calling ShareJSBSimRunner.compute")
            train_infos = self.train()
            self.total_num_steps += self.buffer_size * self.n_rollout_threads

            if episode % self.log_interval == 0:
                end = time.time()
                logging.info(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.all_args.scenario_name, self.algorithm_name, self.experiment_name,
                            episode, episodes, self.total_num_steps, self.num_env_steps,
                            int(self.total_num_steps / (end - start))))
                train_infos["average_episode_rewards"] = np.mean([rew.mean() for rew in self.buffer.rewards])
                logging.info(f"Average episode rewards: {train_infos['average_episode_rewards']:.4f}")

            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(self.total_num_steps)

            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

    def warmup(self):
        obs, share_obs = self.envs.reset()
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        # 获取 obs 和 share_obs，并确保形状正确
        obs = self.buffer.obs[step]  # 形状 (1, 4, 33)
        share_obs = self.buffer.share_obs[step]  # 形状 (1, 4, 132)
        masks = self.buffer.masks[step].squeeze(-1)  # 形状 (1, 4)

        # 重塑为 (n_rollout_threads * num_agents, obs_dim)
        obs = obs.reshape(-1, obs.shape[-1])  # (4, 33)
        share_obs = share_obs.reshape(-1, share_obs.shape[-1])  # (4, 132)
        masks = masks.reshape(-1, 1)  # (4, 1)

        # 获取 RNN 状态并重塑
        rnn_states_actor = self.buffer.rnn_states_actor[step].reshape(-1, self.all_args.recurrent_hidden_layers,
                                                                      self.all_args.recurrent_hidden_size)  # (4, 1, 64)
        rnn_states_critic = self.buffer.rnn_states_critic[step].reshape(-1, self.all_args.recurrent_hidden_layers,
                                                                        self.all_args.recurrent_hidden_size)  # (4, 1, 64)

        # 转换为张量并移至设备
        obs = torch.FloatTensor(obs).to(self.device)
        share_obs = torch.FloatTensor(share_obs).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        rnn_states_actor = torch.FloatTensor(rnn_states_actor).to(self.device)
        rnn_states_critic = torch.FloatTensor(rnn_states_critic).to(self.device)

        # 获取动作和值
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.policy.get_actions(
            share_obs, obs, rnn_states_actor, rnn_states_critic, masks
        )

        # 转换为 numpy 数组并拆分
        values = np.array(np.split(_t2n(values), self.n_rollout_threads, axis=0))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads, axis=0))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads, axis=0))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads, axis=0))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads, axis=0))

        # 调整形状为 (n_rollout_threads, num_agents, recurrent_hidden_layers, recurrent_hidden_size)
        rnn_states_actor = rnn_states_actor.reshape(self.n_rollout_threads, self.num_agents,
                                                    self.all_args.recurrent_hidden_layers,
                                                    self.all_args.recurrent_hidden_size)
        rnn_states_critic = rnn_states_critic.reshape(self.n_rollout_threads, self.num_agents,
                                                      self.all_args.recurrent_hidden_layers,
                                                      self.all_args.recurrent_hidden_size)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data):
        obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
        dones_env = np.all(dones.squeeze(-1), axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env] = np.zeros(((dones_env).sum(), self.num_agents, 1), dtype=np.float32)
        # print(f"Insert obs shape: {obs.shape}, share_obs shape: {share_obs.shape}")
        self.buffer.insert(obs, share_obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        cent_obs = np.concatenate(self.buffer.share_obs[-1])  # 形状 (4, 132)
        print(f"cent_obs shape in compute: {cent_obs.shape}")
        rnn_states_critic = self.buffer.rnn_states_critic[-1].reshape(
            -1, self.buffer.rnn_states_critic.shape[-2], self.buffer.rnn_states_critic.shape[-1]
        )  # 形状 (4, 1, 64)
        masks = self.buffer.masks[-1].reshape(-1, 1)  # 形状 (4, 1)

        next_values = self.policy.get_values(cent_obs, rnn_states_critic, masks)
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')
        logging.info(f"Saved model for episode {episode} to {self.save_dir}")