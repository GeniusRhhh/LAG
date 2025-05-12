# import time
# import torch
# import logging
# import numpy as np
# import os
# from typing import List
# from .base_runner import Runner, ReplayBuffer
#
#
# def _t2n(x):
#     return x.detach().cpu().numpy()
#
#
# class JSBSimRunner(Runner):
#
#     def load(self):
#         self.obs_space = self.envs.observation_space
#         self.act_space = self.envs.action_space
#         self.num_agents = self.envs.num_agents
#         self.use_selfplay = self.all_args.use_selfplay
#         self.render_mode = None
#
#         # policy & algorithm
#         if self.algorithm_name == "ppo":
#             from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
#             from algorithms.ppo.ppo_policy import PPOPolicy as Policy
#         else:
#             raise NotImplementedError
#         self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
#         self.trainer = Trainer(self.all_args, device=self.device)
#
#         # buffer
#         self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)
#
#         if self.model_dir is not None:
#             logging.info(f"Model directory set to {self.model_dir}")
#             self.save_dir = self.model_dir  # 设置保存目录
#             if not os.path.exists(self.save_dir):
#                 os.makedirs(self.save_dir)
#
#     def run(self):
#         # 检查是否存在保存的状态，如果有则恢复
#         # self.model_dir = "../scripts/results/SingleCombat/1v1/NoWeapon/vsBaseline/ppo/v1/run9"
#         if self.model_dir is not None and os.path.exists(self.model_dir):
#             episodes = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(self.model_dir)
#                         if f.startswith('actor_') and f.endswith('.pt') and f != 'actor_latest.pt']
#             if episodes:
#                 latest_episode = max(episodes)
#                 self.restore(latest_episode)
#                 logging.info(f"Restored from episode {latest_episode}, start_episode={self.start_episode}, "
#                              f"total_num_steps={self.total_num_steps}, save_dir set to {self.save_dir}")
#             else:
#                 logging.info(f"No episode checkpoints found in {self.model_dir}, starting from scratch")
#                 self.start_episode = 0
#                 self.total_num_steps = 0
#                 self.save_dir = self.model_dir
#                 if not os.path.exists(self.save_dir):
#                     os.makedirs(self.save_dir)
#         else:
#             logging.info(f"Model directory {self.model_dir} not found or not set, starting from scratch")
#             self.start_episode = 0
#             self.total_num_steps = 0
#             self.save_dir = self.run_dir  # 默认保存目录
#             if not os.path.exists(self.save_dir):
#                 os.makedirs(self.save_dir)
#             logging.info(f"Default save directory set to {self.save_dir}")
#
#         self.warmup()
#
#         start = time.time()
#         episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads
#         for episode in range(self.start_episode, episodes):
#             heading_turns_list = []
#             for step in range(self.buffer_size):
#                 # 采样动作
#                 values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)
#
#                 # 观察奖励和下一状态
#                 obs, rewards, dones, infos = self.envs.step(actions)
#
#                 # 额外记录的信息
#                 for info in infos:
#                     if 'heading_turn_counts' in info:
#                         heading_turns_list.append(info['heading_turn_counts'])
#
#                 data = obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic
#
#                 # 将数据插入缓冲区
#                 self.insert(data)
#
#             # 计算回报并更新网络
#             self.compute()
#             train_infos = self.train()
#
#             # 后处理
#             self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads
#
#             # 记录信息
#             if episode % self.log_interval == 0:
#                 end = time.time()
#                 logging.info(
#                     "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
#                     .format(self.all_args.scenario_name,
#                             self.algorithm_name,
#                             self.experiment_name,
#                             episode,
#                             episodes,
#                             self.total_num_steps,
#                             self.num_env_steps,
#                             int(self.total_num_steps / (end - start))))
#                 train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
#                 logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
#                 if len(heading_turns_list):
#                     train_infos["average_heading_turns"] = np.mean(heading_turns_list)
#                     logging.info("average heading turns is {}".format(train_infos["average_heading_turns"]))
#                 self.log_info(train_infos, self.total_num_steps)
#
#             # 评估
#             if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
#                 self.eval(self.total_num_steps)
#
#             # 保存模型
#             if (episode % self.save_interval == 0) or (episode == episodes - 1):
#                 self.save(episode)
#
#     def warmup(self):
#         # reset env
#         obs = self.envs.reset()
#         self.buffer.step = 0
#         self.buffer.obs[0] = obs.copy()
#
#     @torch.no_grad()
#     def collect(self, step):
#         self.policy.prep_rollout()
#         values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
#             = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
#                                       np.concatenate(self.buffer.rnn_states_actor[step]),
#                                       np.concatenate(self.buffer.rnn_states_critic[step]),
#                                       np.concatenate(self.buffer.masks[step]))
#         # split parallel data [N*M, shape] => [N, M, shape]
#         values = np.array(np.split(_t2n(values), self.n_rollout_threads))
#         actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
#         action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
#         rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
#         rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
#         return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
#
#     def insert(self, data: List[np.ndarray]):
#         obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
#
#         dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
#
#         rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]),
#                                                        dtype=np.float32)
#         rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]),
#                                                         dtype=np.float32)
#
#         masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
#         masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
#
#         self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)
#
#     @torch.no_grad()
#     def eval(self, total_num_steps):
#         logging.info("\nStart evaluation...")
#         total_episodes, eval_episode_rewards = 0, []
#         eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]),
#                                            dtype=np.float32)
#
#         eval_obs = self.eval_envs.reset()
#         eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
#         eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]),
#                                    dtype=np.float32)
#
#         self.timestamp = 0  # use for tacview real time render
#
#         while total_episodes < self.eval_episodes:
#             self.policy.prep_rollout()
#             eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
#                                                             np.concatenate(eval_rnn_states),
#                                                             np.concatenate(eval_masks), deterministic=True)
#             eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
#             eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
#
#             # Obser reward and next obs
#             eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
#
#             # real render with tacview
#             if self.render_mode == "real_time" and self.tacview:
#                 render_data = [f"#{self.timestamp:.2f}\n"]
#                 for sim in self.eval_envs.envs[0]._jsbsims.values():
#                     log_msg = sim.log()
#                     if log_msg is not None:
#                         render_data.append(log_msg + "\n")
#
#                 render_data_str = "".join(render_data)
#                 try:
#                     self.tacview.send_data_to_client(render_data_str)
#                 except Exception as e:
#                     logging.error(f"Tacview rendering error: {e}")
#
#             self.timestamp += 0.2  # step 0.2s
#             eval_cumulative_rewards += eval_rewards
#             eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
#             total_episodes += np.sum(eval_dones_env)
#             if np.any(eval_dones_env == True):
#                 rewards_to_add = eval_cumulative_rewards[eval_dones_env == True].copy()
#                 logging.debug(f"Adding rewards shape: {rewards_to_add.shape}")
#                 eval_episode_rewards.append(rewards_to_add)
#                 eval_cumulative_rewards[eval_dones_env == True] = 0
#
#             eval_masks = np.ones_like(eval_masks, dtype=np.float32)
#             eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]),
#                                                           dtype=np.float32)
#             eval_rnn_states[eval_dones_env == True] = np.zeros(
#                 ((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)
#
#         eval_infos = {}
#         if len(eval_episode_rewards) == 0:
#             logging.warning("No evaluation episodes completed, using default reward 0.0")
#             mean_reward = 0.0
#         else:
#             try:
#                 logging.debug(f"Eval episode rewards shapes: {[r.shape for r in eval_episode_rewards]}")
#                 flat_rewards = []
#                 for rewards in eval_episode_rewards:
#                     if rewards.size > 0:
#                         if rewards.ndim > 1:
#                             agent_rewards = rewards.mean(axis=tuple(range(1, rewards.ndim)))
#                         else:
#                             agent_rewards = rewards
#                         flat_rewards.extend(agent_rewards)
#                 mean_reward = float(np.mean(flat_rewards)) if len(flat_rewards) > 0 else 0.0
#                 if np.isnan(mean_reward):
#                     logging.warning("Computed mean reward is NaN, using default 0.0")
#                     mean_reward = 0.0
#             except Exception as e:
#                 logging.error(f"Error computing eval rewards: {e}, using default 0.0")
#                 mean_reward = 0.0
#
#         eval_infos['eval_average_episode_rewards'] = mean_reward
#         logging.debug(f"Final eval reward for wandb: {mean_reward}, type: {type(mean_reward)}")
#         logging.info(" eval average episode rewards: " + str(mean_reward))
#         self.log_info(eval_infos, total_num_steps)
#         logging.info("...End evaluation")
#
#     @torch.no_grad()
#     def render(self):
#         logging.info(f"\nStart render, render mode is {self.render_mode} ... ...")
#         render_episode_rewards = 0
#         render_obs = self.envs.reset()
#         render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
#         render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
#         self.envs.render(mode=self.render_mode, filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi',
#                          tacview=self.tacview)
#         while True:
#             self.policy.prep_rollout()
#             render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
#                                                                 np.concatenate(render_rnn_states),
#                                                                 np.concatenate(render_masks),
#                                                                 deterministic=True)
#             render_actions = np.expand_dims(_t2n(render_actions), axis=0)
#             render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
#
#             render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
#             if self.use_selfplay:
#                 render_rewards = render_rewards[:, :self.num_agents // 2, ...]
#             render_episode_rewards += render_rewards
#             self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
#             if render_dones.all():
#                 break
#         render_infos = {}
#         render_infos['render_episode_reward'] = render_episode_rewards
#         logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))
#
#     def save(self, episode):
#         # 保存 actor 和 critic
#         policy_actor_state_dict = self.policy.actor.state_dict()
#         torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
#         policy_critic_state_dict = self.policy.critic.state_dict()
#         torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
#         torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')
#
#         # 保存优化器状态
#         if hasattr(self.policy, 'optimizer') and self.policy.optimizer is not None:
#             torch.save(self.policy.optimizer.state_dict(), str(self.save_dir) + '/optimizer_latest.pt')
#         else:
#             logging.warning("Optimizer not available, skipping save.")
#
#         # 保存回放缓冲区状态
#         torch.save(self.buffer, str(self.save_dir) + '/buffer_latest.pt')
#
#         # 保存训练状态
#         training_state = {
#             'episode': episode,
#             'total_num_steps': self.total_num_steps,
#         }
#         torch.save(training_state, str(self.save_dir) + '/training_state_latest.pt')
#
#     def restore(self, episode=None):
#         if episode is None:
#             actor_path = str(self.model_dir) + '/actor_latest.pt'
#             critic_path = str(self.model_dir) + '/critic_latest.pt'
#             optimizer_path = str(self.model_dir) + '/optimizer_latest.pt'
#             buffer_path = str(self.model_dir) + '/buffer_latest.pt'
#             training_state_path = str(self.model_dir) + '/training_state_latest.pt'
#         else:
#             actor_path = str(self.model_dir) + f'/actor_{episode}.pt'
#             critic_path = str(self.model_dir) + '/critic_latest.pt'
#             optimizer_path = str(self.model_dir) + '/optimizer_latest.pt'
#             buffer_path = str(self.model_dir) + '/buffer_latest.pt'
#             training_state_path = str(self.model_dir) + '/training_state_latest.pt'
#
#         # 恢复 actor
#         if os.path.exists(actor_path):
#             self.policy.actor.load_state_dict(torch.load(actor_path, weights_only=True))
#             logging.info(f"Loaded actor from {actor_path}")
#         else:
#             logging.error(f"Actor file {actor_path} not found!")
#             raise FileNotFoundError(f"Actor file {actor_path} not found!")
#
#         # 恢复 critic
#         if os.path.exists(critic_path):
#             self.policy.critic.load_state_dict(torch.load(critic_path, weights_only=True))
#             logging.info(f"Loaded critic from {critic_path}")
#         else:
#             logging.error(f"Critic file {critic_path} not found!")
#             raise FileNotFoundError(f"Critic file {critic_path} not found!")
#
#         # 恢复优化器
#         if hasattr(self.policy, 'optimizer') and os.path.exists(optimizer_path):
#             self.policy.optimizer.load_state_dict(torch.load(optimizer_path, weights_only=False))
#             logging.info(f"Loaded optimizer from {optimizer_path}")
#         else:
#             logging.warning(f"Optimizer file {optimizer_path} not found, skipping optimizer restore.")
#
#         # 恢复缓冲区
#         if os.path.exists(buffer_path):
#             self.buffer = torch.load(buffer_path, weights_only=False)
#             logging.info(f"Loaded buffer from {buffer_path}")
#         else:
#             logging.warning(f"Buffer file {buffer_path} not found, initializing new buffer.")
#             self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)
#
#         # 恢复训练状态
#         if os.path.exists(training_state_path):
#             training_state = torch.load(training_state_path, weights_only=False)
#             self.start_episode = training_state['episode'] + 1  # 从下一轮开始
#             self.total_num_steps = training_state['total_num_steps']
#             logging.info(f"Loaded training state: episode={training_state['episode']}, total_num_steps={self.total_num_steps}")
#         else:
#             logging.warning(f"Training state file {training_state_path} not found, starting from scratch.")
#             self.start_episode = 0
#             self.total_num_steps = 0



import time
import torch
import logging
import numpy as np
import os
from typing import List, Union
from collections import deque
from .base_runner import Runner
from algorithms.utils.buffer import SACReplayBuffer, ReplayBuffer

def _t2n(x):
    """Convert torch tensor to numpy array."""
    return x.detach().cpu().numpy()

class JSBSimRunner(Runner):
    """
    JSBSim环境运行器，适配SAC算法，支持连续动作空间。
    """

    def load(self):
        """加载环境、策略、训练器和缓冲区。"""
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay
        self.render_mode = None

        logging.info(f"Num agents: {self.num_agents}")
        logging.info(f"n_rollout_threads: {self.all_args.n_rollout_threads}")
        logging.info(f"Buffer params: buffer_size={self.all_args.buffer_size}, n_rollout_threads={self.all_args.n_rollout_threads}, num_agents={self.num_agents}")

        # policy & algorithm
        if self.algorithm_name == "sac":
            from algorithms.sac.sac_trainer import SACTrainer as Trainer
            from algorithms.sac.sac_policy import SACPolicy as Policy
            from algorithms.utils.buffer import SACReplayBuffer
        elif self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
            from algorithms.utils.buffer import ReplayBuffer
        else:
            raise NotImplementedError(f"Algorithm {self.algorithm_name} not supported.")

        self.policy = Policy(self.all_args, self.obs_space, self.act_space, self.num_agents, device=self.device)
        self.trainer = Trainer(self.all_args, policy=self.policy, device=self.device)

        # buffer
        if self.algorithm_name == "sac":
            self.buffer = SACReplayBuffer(
                self.all_args,
                self.obs_space,
                self.act_space,
                self.all_args.n_rollout_threads,
                self.num_agents,
                capacity=self.all_args.buffer_size
            )
        else:
            self.buffer = ReplayBuffer(
                self.all_args,
                self.num_agents,
                self.obs_space,
                self.act_space
            )

        logging.info(f"Buffer: {type(self.buffer)}")

        if self.model_dir is not None:
            logging.info(f"Model directory set to {self.model_dir}")
            self.save_dir = self.model_dir
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def run(self):
        """运行训练循环，适配SAC算法。"""
        if self.model_dir is not None and os.path.exists(self.model_dir):
            files = os.listdir(self.model_dir)
            episodes = []
            for f in files:
                if f.startswith('actor_') and f.endswith('.pt') and f != 'actor_latest.pt':
                    try:
                        episode_str = f.split('_')[-1].split('.')[0]
                        episode = int(episode_str)
                        episodes.append(episode)
                    except ValueError:
                        logging.warning(f"Skipping invalid episode file: {f}")
                        continue

            if episodes:
                latest_episode = max(episodes)
                self.restore(latest_episode)
                logging.info(f"Restored from episode {latest_episode}, start_episode={self.start_episode}, "
                             f"total_num_steps={self.total_num_steps}, save_dir set to {self.save_dir}")
            else:
                logging.info(f"No valid episode checkpoints found in {self.model_dir}, starting from scratch")
                self.start_episode = 0
                self.total_num_steps = 0
                self.save_dir = self.model_dir
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
        else:
            logging.info(f"Model directory {self.model_dir} not found or not set, starting from scratch")
            self.start_episode = 0
            self.total_num_steps = 0
            self.save_dir = self.run_dir
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                logging.info(f"Default save directory set to {self.save_dir}")

        self.warmup()

        start = time.time()
        episodes = self.num_env_steps // self.all_args.buffer_size // self.n_rollout_threads
        for episode in range(self.start_episode, episodes):
            heading_turns_list = []
            train_infos = {}
            step = 0  # 回合内步数重置
            while step < self.all_args.buffer_size:
                try:
                    # 采样动作
                    actions, rnn_states_actor = self.collect(step)

                    expected_action_shape = (self.n_rollout_threads, self.num_agents, self.act_space.shape[0])
                    if actions.shape != expected_action_shape:
                        logging.error(
                            f"run: invalid actions shape, got {actions.shape}, expected {expected_action_shape}")
                        raise ValueError(
                            f"Invalid actions shape: got {actions.shape}, expected {expected_action_shape}")

                    # 观察奖励和下一状态
                    obs, rewards, dones, infos = self.envs.step(actions)
                    if step % 100 == 0:  # 每 100 步打印一次
                        logging.info(f"Obs type: {type(obs)}, Obs content: {obs}")  # 调试信息

                    # 确保状态归一化
                    if isinstance(obs, dict):
                        for key in obs:
                            obs[key] = np.clip(obs[key], -10.0, 10.0)
                            if not (-10.0 <= obs[key]).all() or not (obs[key] <= 10.0).all():
                                logging.warning(f"Observation out of bounds after clipping for key {key}: {obs[key]}")
                                dones = np.ones(dones.shape, dtype=bool) if dones.ndim > 0 else np.ones_like(dones,
                                                                                                             dtype=bool)
                                break
                    else:
                        obs = np.clip(obs, -10.0, 10.0)
                        if not (-10.0 <= obs).all() or not (obs <= 10.0).all():
                            logging.warning(f"Observation out of bounds after clipping: {obs}")
                            dones = np.ones(dones.shape, dtype=bool) if dones.ndim > 0 else np.ones_like(dones,
                                                                                                         dtype=bool)

                    next_obs = obs
                    self.current_obs = obs

                    # 记录信息
                    for info in infos:
                        if 'heading_turn_counts' in info:
                            heading_turns_list.append(info['heading_turn_counts'])
                        logging.debug(f"Env info: {info}")  # 记录完整的 infos

                    data = (obs, actions, rewards, next_obs, dones, rnn_states_actor)
                    self.insert(data)
                    self.total_num_steps += self.n_rollout_threads
                    step += 1

                    # SAC 更新
                    if self.algorithm_name == "sac" and self.total_num_steps >= self.all_args.warmup_steps:
                        if self.buffer.size >= self.all_args.batch_size:
                            batch = self.buffer.sample_batch(batch_size=self.all_args.batch_size)
                            train_infos = self.trainer.update(batch) or {}
                            if step % 100 == 0:  # 每 100 步打印一次关键信息
                                self.log_critical_info(step, obs, actions, rewards, train_infos)
                        else:
                            logging.warning(
                                f"run: episode={episode}, step={step}, buffer_size={self.buffer.size}, batch_size={self.all_args.batch_size}, skipping update")
                    # 添加终止条件
                    if step >= 10000:
                        dones = np.ones(dones.shape, dtype=bool) if dones.ndim > 0 else np.ones_like(dones, dtype=bool)
                        logging.info(f"Episode terminated due to max steps at step {step}")
                        break

                    if dones.any():
                        logging.info(f"Episode terminated due to done signal at step {step}")
                        break

                except Exception as e:
                    logging.error(f"run: error at episode={episode}, step={step}, error={str(e)}")
                    raise

            # 后处理
            total_steps_per_episode = self.all_args.buffer_size * self.n_rollout_threads

            if episode % self.log_interval == 0:
                end = time.time()
                logging.info(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                self.total_num_steps,
                                self.num_env_steps,
                                int(self.total_num_steps / (end - start))))
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rew_buf[-self.all_args.buffer_size:])
                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                if len(heading_turns_list):
                    train_infos["average_heading_turns"] = np.mean(heading_turns_list)
                    logging.info("average heading turns is {}".format(train_infos["average_heading_turns"]))
                self.log_info(train_infos, self.total_num_steps)

            if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
                self.eval(self.total_num_steps)

            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)
    def warmup(self):
        """初始化环境和缓冲区。"""
        obs = self.envs.reset()
        self.current_obs = obs  # 初始化当前观察
        if self.algorithm_name != "sac":
            self.buffer.step = 0
            self.buffer.obs[0] = obs.copy()
        else:
            self.buffer.clear()  # 清空 SACReplayBuffer

    def insert(self, data: List[np.ndarray]):
        """将数据插入缓冲区。"""
        obs, actions, rewards, next_obs, dones, rnn_states_actor = data

        # 调试日志
        # logging.info(f"Inserting data - Obs shape: {obs.shape}, Actions shape: {actions.shape}, "
        #              f"Rewards shape: {rewards.shape}, Next Obs shape: {next_obs.shape}, Dones shape: {dones.shape}")

        # 确保 rewards 形状正确
        expected_rewards_shape = (self.n_rollout_threads, self.num_agents, 1)
        if rewards.shape != expected_rewards_shape:
            logging.warning(f"Unexpected rewards shape: got {rewards.shape}, expected {expected_rewards_shape}")
            if rewards.ndim == 2:  # (n_rollout_threads, num_agents)
                rewards = rewards.reshape(self.n_rollout_threads, self.num_agents, 1)
            elif rewards.ndim == 1:  # (n_rollout_threads,)
                rewards = rewards.reshape(self.n_rollout_threads, 1, 1)
            else:
                raise ValueError(f"Cannot reshape rewards from {rewards.shape} to {expected_rewards_shape}")

        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)

        if self.algorithm_name == "sac":
            self.buffer.insert(obs, actions, rewards, next_obs, dones)
        else:
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            masks[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )
            self.buffer.insert(obs, actions, rewards, masks, rnn_states_actor)

    @torch.no_grad()
    def collect(self, step):
        """从策略中采样动作，适配SAC的连续动作。"""
        self.policy.prep_rollout()
        if self.algorithm_name == "sac":
            if not hasattr(self, 'current_obs'):
                self.current_obs = self.envs.reset()
            obs = self.current_obs
            rnn_states = np.zeros(
                (self.n_rollout_threads, self.num_agents, self.all_args.recurrent_hidden_layers,
                 self.all_args.recurrent_hidden_size),
                dtype=np.float32
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            actions, log_probs, rnn_states_actor = self.policy.get_actions(
                obs, rnn_states_actor=rnn_states, masks=masks, deterministic=False
            )
            expected_shape = (self.n_rollout_threads, self.num_agents, self.all_args.recurrent_hidden_layers,
                              self.all_args.recurrent_hidden_size)
            if rnn_states_actor.shape != expected_shape:
                logging.error(
                    f"collect: invalid rnn_states_actor shape, got {rnn_states_actor.shape}, expected {expected_shape}")
                rnn_states_actor = rnn_states_actor.reshape(expected_shape)
        else:
            obs = self.buffer.obs[step]
            rnn_states = self.buffer.rnn_states_actor[step]
            masks = self.buffer.masks[step]
            actions, rnn_states_actor = self.policy.act(
                np.concatenate(obs),
                np.concatenate(rnn_states),
                np.concatenate(masks),
                deterministic=False
            )
            actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
            rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        return actions, rnn_states_actor

    def eval(self, total_num_steps):
        """评估策略性能。"""
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.all_args.recurrent_hidden_layers,
             self.all_args.recurrent_hidden_size),
            dtype=np.float32
        )

        self.timestamp = 0

        while total_episodes < self.eval_episodes:
            self.policy.prep_rollout()
            if self.algorithm_name == "sac":
                eval_actions, _, eval_rnn_states = self.policy.get_actions(
                    eval_obs,
                    rnn_states_actor=eval_rnn_states,
                    masks=eval_masks,
                    deterministic=True
                )
                eval_actions = eval_actions.reshape(self.n_eval_rollout_threads, self.num_agents, -1)
                expected_rnn_shape = (
                    self.n_eval_rollout_threads, self.num_agents, self.all_args.recurrent_hidden_layers,
                    self.all_args.recurrent_hidden_size
                )
                if eval_rnn_states.shape != expected_rnn_shape:
                    logging.error(
                        f"eval: invalid eval_rnn_states shape, got {eval_rnn_states.shape}, expected {expected_rnn_shape}")
                    eval_rnn_states = eval_rnn_states.reshape(expected_rnn_shape)
            else:
                eval_actions, eval_rnn_states = self.policy.act(
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    deterministic=True
                )
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            if self.render_mode == "real_time" and self.tacview:
                render_data = [f"#{self.timestamp:.2f}\n"]
                for sim in self.eval_envs.envs[0]._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")
                render_data_str = "".join(render_data)
                try:
                    self.tacview.send_data_to_client(render_data_str)
                except Exception as e:
                    logging.error(f"Tacview rendering error: {e}")

            self.timestamp += 0.2
            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            if np.any(eval_dones_env):
                rewards_to_add = eval_cumulative_rewards[eval_dones_env].copy()
                eval_episode_rewards.append(rewards_to_add)
                eval_cumulative_rewards[eval_dones_env] = 0
                eval_masks[eval_dones_env] = np.zeros(
                    (np.sum(eval_dones_env), self.num_agents, 1), dtype=np.float32
                )
                eval_rnn_states[eval_dones_env] = np.zeros(
                    (np.sum(eval_dones_env), self.num_agents, self.all_args.recurrent_hidden_layers,
                     self.all_args.recurrent_hidden_size),
                    dtype=np.float32
                )

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)

        eval_infos = {}
        if len(eval_episode_rewards) == 0:
            logging.warning("No evaluation episodes completed, using default reward 0.0")
            mean_reward = 0.0
        else:
            try:
                flat_rewards = []
                for rewards in eval_episode_rewards:
                    if rewards.size > 0:
                        if rewards.ndim > 1:
                            agent_rewards = rewards.mean(axis=tuple(range(1, rewards.ndim)))
                        else:
                            agent_rewards = rewards
                        flat_rewards.extend(agent_rewards)
                mean_reward = float(np.mean(flat_rewards)) if len(flat_rewards) > 0 else 0.0
                if np.isnan(mean_reward):
                    logging.warning("Computed mean reward is NaN, using default 0.0")
                    mean_reward = 0.0
            except Exception as e:
                logging.error(f"Error computing eval rewards: {e}, using default 0.0")
                mean_reward = 0.0

        eval_infos['eval_average_episode_rewards'] = mean_reward
        logging.debug(f"Final eval reward for wandb: {mean_reward}, type: {type(mean_reward)}")
        logging.info(" eval average episode rewards: " + str(mean_reward))
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

    def render(self):
        """渲染环境，适配连续动作。"""
        logging.info(f"\nStart render, render mode is {self.render_mode} ... ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_masks = np.ones((self.num_agents, 1), dtype=np.float32)
        render_rnn_states = np.zeros(
            (self.num_agents, self.all_args.recurrent_hidden_layers, self.all_args.recurrent_hidden_size),
            dtype=np.float32
        )
        self.envs.render(mode=self.render_mode, filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi',
                         tacview=self.tacview)
        while True:
            self.policy.prep_rollout()
            if self.algorithm_name == "sac":
                render_actions, _, render_rnn_states = self.policy.get_actions(
                    render_obs.reshape(-1, render_obs.shape[-1]),
                    rnn_states_actor=render_rnn_states,
                    masks=render_masks,
                    deterministic=True
                )
                render_actions = render_actions.reshape(1, self.num_agents, -1)
                render_rnn_states = render_rnn_states.reshape(
                    self.num_agents, self.all_args.recurrent_hidden_layers, self.all_args.recurrent_hidden_size
                )
            else:
                render_actions, render_rnn_states = self.policy.act(
                    np.concatenate(render_obs),
                    np.concatenate(render_rnn_states),
                    np.concatenate(render_masks),
                    deterministic=True
                )
                render_actions = np.expand_dims(_t2n(render_actions), axis=0)
                render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            if self.use_selfplay:
                render_rewards = render_rewards[:, :self.num_agents // 2]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        """保存模型、优化器和缓冲区状态。"""
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')

        if self.algorithm_name == "sac":
            torch.save(self.policy.critic.state_dict(), str(self.save_dir) + '/critic_latest.pt')
            torch.save(self.policy.critic.state_dict(), str(self.save_dir) + f'/critic_{episode}.pt')
            torch.save(self.policy.critic_target.state_dict(), str(self.save_dir) + '/critic_target_latest.pt')
            torch.save(self.policy.critic_target.state_dict(), str(self.save_dir) + f'/critic_target_{episode}.pt')

            # 保存优化器
            if hasattr(self.policy, 'actor_optimizer') and self.policy.actor_optimizer is not None:
                torch.save(self.policy.actor_optimizer.state_dict(), str(self.save_dir) + '/actor_optimizer_latest.pt')
                torch.save(self.policy.actor_optimizer.state_dict(),
                           str(self.save_dir) + f'/actor_optimizer_{episode}.pt')
            else:
                logging.warning("Actor optimizer not available, skipping save.")

            if hasattr(self.policy, 'critic_optimizer') and self.policy.critic_optimizer is not None:
                torch.save(self.policy.critic_optimizer.state_dict(),
                           str(self.save_dir) + '/critic_optimizer_latest.pt')
                torch.save(self.policy.critic_optimizer.state_dict(),
                           str(self.save_dir) + f'/critic_optimizer_{episode}.pt')
            else:
                logging.warning("Critic optimizer not available, skipping save.")

            if hasattr(self.policy, 'alpha_optimizer') and self.policy.alpha_optimizer is not None:
                torch.save(self.policy.alpha_optimizer.state_dict(), str(self.save_dir) + '/alpha_optimizer_latest.pt')
                torch.save(self.policy.alpha_optimizer.state_dict(),
                           str(self.save_dir) + f'/alpha_optimizer_{episode}.pt')
            else:
                logging.warning("Alpha optimizer not available, skipping save.")
        else:
            policy_critic_state_dict = self.policy.critic.state_dict()
            torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
            torch.save(policy_critic_state_dict, str(self.save_dir) + f'/critic_{episode}.pt')

        torch.save(self.buffer, str(self.save_dir) + '/buffer_latest.pt')
        torch.save(self.buffer, str(self.save_dir) + f'/buffer_{episode}.pt')

        training_state = {
            'episode': episode,
            'total_num_steps': self.total_num_steps,
        }
        torch.save(training_state, str(self.save_dir) + '/training_state_latest.pt')
        torch.save(training_state, str(self.save_dir) + f'/training_state_{episode}.pt')

    def restore(self, episode=None):
        """恢复模型、优化器和缓冲区状态。"""
        if episode is None:
            actor_path = str(self.model_dir) + '/actor_latest.pt'
            critic_path = str(self.model_dir) + '/critic_latest.pt'
            critic_target_path = str(self.model_dir) + '/critic_target_latest.pt'
            optimizer_path = str(self.model_dir) + '/optimizer_latest.pt'
            buffer_path = str(self.model_dir) + '/buffer_latest.pt'
            training_state_path = str(self.model_dir) + '/training_state_latest.pt'
        else:
            actor_path = str(self.model_dir) + f'/actor_{episode}.pt'
            critic_path = str(self.model_dir) + f'/critic_{episode}.pt'
            critic_target_path = str(self.model_dir) + f'/critic_target_{episode}.pt'
            optimizer_path = str(self.model_dir) + f'/optimizer_{episode}.pt'
            buffer_path = str(self.model_dir) + f'/buffer_{episode}.pt'
            training_state_path = str(self.model_dir) + f'/training_state_{episode}.pt'

        if os.path.exists(actor_path):
            self.policy.actor.load_state_dict(torch.load(actor_path, weights_only=True))
            logging.info(f"Loaded actor from {actor_path}")
        else:
            logging.error(f"Actor file {actor_path} not found!")
            raise FileNotFoundError(f"Actor file {actor_path} not found!")

        if self.algorithm_name == "sac":
            if os.path.exists(critic_path):
                self.policy.critic.load_state_dict(torch.load(critic_path, weights_only=True))
                logging.info(f"Loaded critic from {critic_path}")
            else:
                logging.error(f"Critic file {critic_path} not found!")
                raise FileNotFoundError(f"Critic file {critic_path} not found!")
            if os.path.exists(critic_target_path):
                self.policy.critic_target.load_state_dict(torch.load(critic_target_path, weights_only=True))
                logging.info(f"Loaded critic_target from {critic_target_path}")
            else:
                logging.error(f"Critic_target file {critic_target_path} not found!")
                raise FileNotFoundError(f"Critic_target file {critic_target_path} not found!")
        else:
            if os.path.exists(critic_path):
                self.policy.critic.load_state_dict(torch.load(critic_path, weights_only=True))
                logging.info(f"Loaded critic from {critic_path}")
            else:
                logging.error(f"Critic file {critic_path} not found!")
                raise FileNotFoundError(f"Critic file {critic_path} not found!")

        if hasattr(self.policy, 'optimizer') and os.path.exists(optimizer_path):
            self.policy.optimizer.load_state_dict(torch.load(optimizer_path, weights_only=False))
            logging.info(f"Loaded optimizer from {optimizer_path}")
        else:
            logging.warning(f"Optimizer file {optimizer_path} not found, skipping optimizer restore.")

        if os.path.exists(buffer_path):
            self.buffer = torch.load(buffer_path, weights_only=False)
            logging.info(f"Loaded buffer from {buffer_path}")
        else:
            logging.warning(f"Buffer file {buffer_path} not found, initializing new buffer.")
            if self.algorithm_name == "sac":
                self.buffer = SACReplayBuffer(
                    self.all_args,
                    self.obs_space,
                    self.act_space,
                    self.all_args.n_rollout_threads,
                    self.num_agents,
                    capacity=self.all_args.buffer_size
                )
            else:
                self.buffer = ReplayBuffer(
                    self.all_args,
                    self.num_agents,
                    self.obs_space,
                    self.act_space
                )

        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)
            self.start_episode = training_state['episode'] + 1
            self.total_num_steps = training_state['total_num_steps']
            logging.info(f"Loaded training state: episode={training_state['episode']}, total_num_steps={self.total_num_steps}")
        else:
            logging.warning(f"Training state file {training_state_path} not found, starting from scratch.")
            self.start_episode = 0
            self.total_num_steps = 0

    # 添加的方法：记录关键信息
    def log_critical_info(self, step, obs, actions, rewards, train_infos):
        logging.info(f"反归一化Step {step}:")
        # 1. 观察值检查
        obs_sample = obs[0, 0]  # 取第一个线程、第一个智能体的观察
        logging.info(f"  Obs - delta_altitude={obs_sample[0]*1000:.2f}m, altitude={obs_sample[3]*5000:.2f}m, "
                     f"delta_heading={obs_sample[1]*180/np.pi:.2f}°")
        if np.any(np.isnan(obs_sample)) or np.any(np.isinf(obs_sample)):
            logging.warning(f"  Invalid obs detected: {obs_sample}")

        # 2. 动作检查
        action_sample = actions[0, 0]
        logging.info(f"  Action - aileron={action_sample[0]:.4f}, elevator={action_sample[1]:.4f}, "
                     f"rudder={action_sample[2]:.4f}, throttle={action_sample[3]:.4f}")
        if np.any(action_sample < self.act_space.low) or np.any(action_sample > self.act_space.high):
            logging.warning(f"  Action out of bounds: {action_sample}")

        # 3. 奖励检查
        reward_sample = rewards[0, 0, 0]
        logging.info(f"  Reward - value={reward_sample:.4f}")
        if reward_sample < -1.0:  # 假设 -1.0 是异常阈值
            logging.warning(f"  Reward unusually low: {reward_sample}")

        # 4. 训练指标检查
        if train_infos:
            logging.info(f"  Train Metrics - critic_loss={train_infos['critic_loss']:.4f}, "
                         f"actor_loss={train_infos['actor_loss']:.4f}, q_mean={train_infos['q_mean']:.4f}, "
                         f"alpha={train_infos['alpha']:.4f}")
            if train_infos['critic_loss'] > 100.0:  # 假设 100.0 是异常阈值
                logging.warning(f"  High critic_loss detected: {train_infos['critic_loss']}")