# import os
# import torch
# import logging
# import numpy as np
# from typing import List
# from .base_runner import Runner, ReplayBuffer
# from .jsbsim_runner import JSBSimRunner
#
#
# def _t2n(x):
#     return x.detach().cpu().numpy()
#
#
# class SelfplayJSBSimRunner(JSBSimRunner):
#
#     def load(self):
#         self.use_selfplay = self.all_args.use_selfplay
#         assert self.use_selfplay == True, "Only selfplay can use SelfplayRunner"
#         self.obs_space = self.envs.observation_space
#         self.act_space = self.envs.action_space
#         self.num_agents = self.envs.num_agents
#         self.num_opponents = self.all_args.n_choose_opponents
#         assert self.eval_episodes >= self.num_opponents, \
#             f"Number of evaluation episodes:{self.eval_episodes} should be greater than number of opponents:{self.num_opponents}"
#         self.init_elo = self.all_args.init_elo
#         self.latest_elo = self.init_elo
#         self.render_mode = None
#         self.elo_k_factor = 32
#         self.min_pool_size = 1562
#         self.max_pool_size = 1562
#         self.score_threshold = 0.1
#
#         if self.algorithm_name == "ppo":
#             from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
#             from algorithms.ppo.ppo_policy import PPOPolicy as Policy
#         else:
#             raise NotImplementedError
#         self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
#         self.trainer = Trainer(self.all_args, device=self.device)
#
#         self.buffer = ReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.act_space)
#
#         from algorithms.utils.selfplay import get_algorithm
#         self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)
#
#         assert self.num_opponents <= self.n_rollout_threads, \
#             "Number of different opponents({}) must less than or equal to number of training threads({})!" \
#                 .format(self.num_opponents, self.n_rollout_threads)
#
#         self.policy_pool = {}
#         if self.model_dir is not None and os.path.exists(self.model_dir):
#             self.restore()
#             for file in os.listdir(self.model_dir):
#                 if file.startswith('actor_') and file.endswith('.pt') and file != 'actor_latest.pt':
#                     episode = file.replace('actor_', '').replace('.pt', '')
#                     self.policy_pool[episode] = self.init_elo
#         default_path = str(self.save_dir) + '/actor_0.pt'
#         if not os.path.exists(default_path):
#             torch.save(self.policy.actor.state_dict(), default_path)
#         self.policy_pool['0'] = self.init_elo
#         #logging.info(f"Initialized policy pool with default policy '0' at Elo {self.init_elo}")
#
#         self.opponent_policy = [
#             Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
#             for _ in range(self.num_opponents)]
#         self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
#         self.opponent_obs = np.zeros_like(self.buffer.obs[0])
#         self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
#         self.opponent_masks = np.ones_like(self.buffer.masks[0])
#
#         if self.use_eval:
#             self.eval_opponent_policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
#
#         logging.info("\n Load selfplay opponents: Algo {}, num_opponents {}, policy_pool size {}.\n"
#                      .format(self.all_args.selfplay_algorithm, self.num_opponents, len(self.policy_pool)))
#
#     def warmup(self):
#         obs = self.envs.reset()
#         self.opponent_obs = obs[:, self.num_agents // 2:, ...]
#         obs = obs[:, :self.num_agents // 2, ...]
#         self.buffer.step = 0
#         self.buffer.obs[0] = obs.copy()
#
#     @torch.no_grad()
#     def collect(self, step):
#         try:
#             self.policy.prep_rollout()
#             values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
#                 = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
#                                           np.concatenate(self.buffer.rnn_states_actor[step]),
#                                           np.concatenate(self.buffer.rnn_states_critic[step]),
#                                           np.concatenate(self.buffer.masks[step]))
#             values = np.array(np.split(_t2n(values), self.n_rollout_threads))
#             actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
#             action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
#             rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
#             rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
#
#             opponent_actions = np.zeros_like(actions)
#             for policy_idx, policy in enumerate(self.opponent_policy):
#                 env_idx = self.opponent_env_split[policy_idx]
#                 opponent_action, opponent_rnn_states \
#                     = policy.act(np.concatenate(self.opponent_obs[env_idx]),
#                                  np.concatenate(self.opponent_rnn_states[env_idx]),
#                                  np.concatenate(self.opponent_masks[env_idx]))
#                 opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
#                 self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
#             actions = np.concatenate((actions, opponent_actions), axis=1)
#
#             return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
#         except Exception as e:
#             logging.error(f"Error in collect: {str(e)}")
#             raise
#
#     def insert(self, data: List[np.ndarray]):
#         obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
#         dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
#         rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
#         rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)
#         masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
#         masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
#         self.opponent_obs = obs[:, self.num_agents // 2:, ...]
#         self.opponent_masks = masks[:, self.num_agents // 2:, ...]
#         self.opponent_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
#         obs = obs[:, :self.num_agents // 2, ...]
#         actions = actions[:, :self.num_agents // 2, ...]
#         rewards = rewards[:, :self.num_agents // 2, ...]
#         masks = masks[:, :self.num_agents // 2, ...]
#         self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)
#
#     @torch.no_grad()
#     def eval(self, total_num_steps):
#         logging.info("\nStart evaluation...")
#         self.policy.prep_rollout()
#         total_episodes = 0
#         episode_rewards, opponent_episode_rewards = [], []
#         cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)
#         opponent_cumulative_rewards = np.zeros_like(cumulative_rewards)
#
#         #logging.info(f"Current policy_pool before opponent selection: {self.policy_pool}")
#         eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in range(self.num_opponents)]
#         eval_each_episodes = self.eval_episodes // self.num_opponents
#         logging.info(f" Choose opponents {eval_choose_opponents} for evaluation")
#
#         for opponent in eval_choose_opponents:
#             if opponent not in self.policy_pool:
#                 self.policy_pool[opponent] = self.init_elo
#                 #logging.warning(f"Opponent {opponent} not in policy_pool; added with default Elo {self.init_elo}")
#
#         eval_cur_opponent_idx = 0
#         self.timestamp = 0
#         while total_episodes < self.eval_episodes:
#             if total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
#                 policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
#                 try:
#                     self.eval_opponent_policy.actor.load_state_dict(
#                         torch.load(str(self.save_dir) + f'/actor_{policy_idx}.pt', weights_only=True))
#                 except FileNotFoundError:
#                     #logging.warning(f"Policy {policy_idx} not found; using latest policy as fallback.")
#                     self.eval_opponent_policy.actor.load_state_dict(
#                         torch.load(str(self.save_dir) + '/actor_latest.pt', weights_only=True))
#                 self.eval_opponent_policy.prep_rollout()
#                 eval_cur_opponent_idx += 1
#                 logging.info(f" Load opponent {policy_idx} for evaluation ({total_episodes}/{self.eval_episodes})")
#                 policy_path = str(self.save_dir) + f'/actor_{policy_idx}.pt'
#                 #logging.info(f"Eval Loading model from: {policy_path}")
#
#                 obs = self.eval_envs.reset()
#                 masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
#                 rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
#                 opponent_obs = obs[:, self.num_agents // 2:, ...]
#                 obs = obs[:, :self.num_agents // 2:, ...]
#                 opponent_masks = np.ones_like(masks, dtype=np.float32)
#                 opponent_rnn_states = np.zeros_like(rnn_states, dtype=np.float32)
#
#             actions, rnn_states = self.policy.act(np.concatenate(obs),
#                                                   np.concatenate(rnn_states),
#                                                   np.concatenate(masks), deterministic=True)
#             actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
#             rnn_states = np.array(np.split(_t2n(rnn_states), self.n_eval_rollout_threads))
#
#             opponent_actions, opponent_rnn_states \
#                 = self.eval_opponent_policy.act(np.concatenate(opponent_obs),
#                                                 np.concatenate(opponent_rnn_states),
#                                                 np.concatenate(opponent_masks), deterministic=True)
#             opponent_rnn_states = np.array(np.split(_t2n(opponent_rnn_states), self.n_eval_rollout_threads))
#             opponent_actions = np.array(np.split(_t2n(opponent_actions), self.n_eval_rollout_threads))
#             actions = np.concatenate((actions, opponent_actions), axis=1)
#
#             obs, eval_rewards, dones, eval_infos = self.eval_envs.step(actions)
#             dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
#             if np.any(dones_env):
#                 total_episodes += np.sum(dones_env)
#
#             opponent_obs = obs[:, self.num_agents // 2:, ...]
#             obs = obs[:, :self.num_agents // 2, ...]
#             masks[dones_env == True] = np.zeros(((dones_env == True).sum(), *masks.shape[1:]), dtype=np.float32)
#             rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states.shape[1:]), dtype=np.float32)
#             opponent_masks[dones_env == True] = np.zeros(((dones_env == True).sum(), *opponent_masks.shape[1:]), dtype=np.float32)
#             opponent_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *opponent_rnn_states.shape[1:]), dtype=np.float32)
#
#             opponent_rewards = eval_rewards[:, self.num_agents // 2:, ...]
#             opponent_cumulative_rewards += opponent_rewards
#             opponent_episode_rewards.append(opponent_cumulative_rewards[dones_env == True])
#             opponent_cumulative_rewards[dones_env == True] = 0
#
#             eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]
#             cumulative_rewards += eval_rewards
#             episode_rewards.append(cumulative_rewards[dones_env == True])
#             cumulative_rewards[dones_env == True] = 0
#
#             if self.render_mode == "real_time":
#                 render_data = [f"#{self.timestamp:.2f}\n"]
#                 for sim in self.eval_envs.envs[0]._jsbsims.values():
#                     log_msg = sim.log()
#                     if log_msg is not None:
#                         render_data.append(log_msg + "\n")
#                 render_data_str = "".join(render_data)
#                 self.tacview.send_data_to_client(render_data_str)
#             self.timestamp += 0.2
#
#         episode_rewards = np.concatenate(episode_rewards) if episode_rewards else np.array([0])
#         episode_rewards = episode_rewards.squeeze(-1).mean(axis=-1)
#         eval_average_episode_rewards = np.array(np.split(episode_rewards, self.num_opponents)).mean(axis=-1)
#
#         opponent_episode_rewards = np.concatenate(opponent_episode_rewards) if opponent_episode_rewards else np.array([0])
#         opponent_episode_rewards = opponent_episode_rewards.squeeze(-1).mean(axis=-1)
#         opponent_average_episode_rewards = np.array(np.split(opponent_episode_rewards, self.num_opponents)).mean(axis=-1)
#
#         ego_elo = np.array([self.latest_elo for _ in range(self.n_eval_rollout_threads)])
#         opponent_elo = np.array([self.policy_pool[key] for key in eval_choose_opponents])
#         expected_score = 1 / (1 + 10 ** ((opponent_elo - ego_elo) / 400))
#
#         all_rewards = np.concatenate([episode_rewards, opponent_episode_rewards])
#         dynamic_threshold = self._calculate_dynamic_threshold(all_rewards)
#
#         diff = opponent_average_episode_rewards - eval_average_episode_rewards
#         actual_score = np.zeros_like(expected_score)
#         win_mask = diff < -dynamic_threshold
#         lose_mask = diff > dynamic_threshold
#         actual_score[win_mask] = 1
#         actual_score[lose_mask] = 0
#         actual_score[~(win_mask | lose_mask)] = 0.5
#
#         elo_gain = self.elo_k_factor * (actual_score - expected_score)
#         update_opponent_elo = opponent_elo - elo_gain
#         for i, key in enumerate(eval_choose_opponents):
#             self.policy_pool[key] = update_opponent_elo[i]
#         ego_elo = ego_elo + elo_gain
#         self.latest_elo = ego_elo.mean()
#
#         # # 修复：手动同步策略池
#         # for ep in range(total_num_steps // 8000 + 1):  # 假设每 8000 步保存一次
#         #     policy_path = str(self.save_dir) + f'/actor_{ep}.pt'
#         #     if os.path.exists(policy_path) and str(ep) not in self.policy_pool:
#         #         self.policy_pool[str(ep)] = self.latest_elo if str(ep) == str(total_num_steps // 8000) else self.init_elo
#         #         logging.info(f"Added policy {ep} to pool with Elo {self.policy_pool[str(ep)]}")
#         #
#         # logging.info(f"Policy pool after eval: {self.policy_pool}")
#         eval_infos = {}
#         eval_infos['eval_average_episode_rewards'] = eval_average_episode_rewards.mean()
#         eval_infos['latest_elo'] = self.latest_elo
#         logging.info(" eval average episode rewards: " + str(eval_infos['eval_average_episode_rewards']))
#         logging.info(" latest elo score: " + str(self.latest_elo))
#         self.log_info(eval_infos, total_num_steps)
#         logging.info("...End evaluation")
#         self.reset_opponent()
#
#
#
#     def save(self, episode):
#         #logging.info(f"save Saving policy for episode: {episode}")
#         policy_actor_state_dict = self.policy.actor.state_dict()
#         policy_critic_state_dict = self.policy.critic.state_dict()
#         torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
#         torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
#         policy_path = str(self.save_dir) + f'/actor_{episode}.pt'
#         try:
#             torch.save(policy_actor_state_dict, policy_path)
#             if os.path.exists(policy_path):
#                 self.policy_pool[str(episode)] = self.latest_elo  # 直接添加到池中
#                 # logging.info(f"Successfully saved policy {episode} to {policy_path} and added to pool with Elo {self.latest_elo}")
#             else:
#                 logging.error(f"Failed to save policy {episode}: file not found at {policy_path}")
#         except Exception as e:
#             logging.error(f"Error saving policy {episode}: {str(e)}")
#
#     def _calculate_dynamic_threshold(self, rewards):
#         std = np.std(rewards)
#         threshold = std * self.score_threshold
#         return max(min(threshold, 10.0), 1.0)
#
#     def _load_opponent_policy(self, policy, policy_idx):
#         #logging.info(f"_load_opponent_policy Loading opponent policy {policy_idx}...")
#         policy_path = str(self.save_dir) + f'/actor_{policy_idx}.pt'
#         #logging.info(f"_load_opponent_policy Loading model from: {policy_path}")
#         try:
#             policy.actor.load_state_dict(torch.load(policy_path, weights_only=True))
#             #logging.info(f"Successfully loaded policy {policy_idx}")
#         except FileNotFoundError as e:
#             #logging.error(f"File not found: {policy_path}")
#             raise e
#         policy.prep_rollout()
#
#     def _update_policy_pool(self, episode):
#         if not isinstance(episode, int) or episode < 0:
#             logging.error(f"Invalid episode value: {episode}. Expected a non-negative integer.")
#             return
#
#         policy_path = str(self.save_dir) + f'/actor_{episode}.pt'
#         if os.path.exists(policy_path) and str(episode) not in self.policy_pool:
#             self.policy_pool[str(episode)] = self.latest_elo
#             #logging.info(f"_update_policy_pool Added policy {episode} to the pool with Elo {self.latest_elo}")
#
#         invalid_keys = [k for k in self.policy_pool if k != '0' and not os.path.exists(str(self.save_dir) + f'/actor_{k}.pt')]
#         for k in invalid_keys:
#             del self.policy_pool[k]
#             #logging.info(f"Removed invalid policy {k} from pool")
#
#     def reset_opponent(self):
#         choose_opponents = []
#         available_pool = [k for k in self.policy_pool.keys() if os.path.exists(str(self.save_dir) + f'/actor_{k}.pt')]
#
#         if not available_pool:
#             #logging.warning("No valid policies found in the pool! Using latest policy as fallback.")
#             available_pool = ['latest']
#
#         #logging.info(f"Available pool for opponent selection: {available_pool}")
#         for _ in range(self.num_opponents):
#             opponent = self.selfplay_algo.choose(self.policy_pool)
#             while opponent in choose_opponents and len(choose_opponents) < len(available_pool):
#                 opponent = self.selfplay_algo.choose(self.policy_pool)
#             choose_opponents.append(opponent)
#
#         for i, policy in enumerate(self.opponent_policy):
#             if i < len(choose_opponents):
#                 choose_idx = choose_opponents[i]
#                 logging.info(f"Loading opponent {choose_idx} for training...")
#                 self._load_opponent_policy(policy, choose_idx)
#             else:
#                 continue
#
#         logging.info(f"Choose opponents {choose_opponents} for training")
#         self.buffer.clear()
#         self.opponent_obs = np.zeros_like(self.opponent_obs)
#         self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
#         self.opponent_masks = np.ones_like(self.opponent_masks)
#         obs = self.envs.reset()
#         if self.num_opponents > 0:
#             self.opponent_obs = obs[:, self.num_agents // 2:, ...]
#             obs = obs[:, :self.num_agents // 2, ...]
#         self.buffer.obs[0] = obs.copy()
#
#     @torch.no_grad()
#     def render(self):
#         idx = self.all_args.render_index
#         opponent_idx = self.all_args.render_opponent_index
#         dir_list = str(self.run_dir).split('/')
#         file_path = '/'.join(dir_list[:dir_list.index('results') + 1])
#         self.policy.actor.load_state_dict(torch.load(str(self.model_dir) + f'/actor_{idx}.pt'))
#         self.eval_opponent_policy.actor.load_state_dict(
#             torch.load(str(self.model_dir) + f'/actor_{opponent_idx}.pt', weights_only=True))
#         self.eval_opponent_policy.prep_rollout()
#         logging.info("\nStart render ...")
#         render_episode_rewards = 0
#         render_obs = self.envs.reset()
#         self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
#         render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
#         render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
#         render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
#         render_obs = render_obs[:, :self.num_agents // 2, ...]
#         render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
#         render_opponent_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
#         while True:
#             self.policy.prep_rollout()
#             render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
#                                                                 np.concatenate(render_rnn_states),
#                                                                 np.concatenate(render_masks),
#                                         0                        deterministic=True)
#             render_actions = np.expand_dims(_t2n(render_actions), axis=0)
#             render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
#             render_opponent_actions, render_opponent_rnn_states \
#                 = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs),
#                                                 np.concatenate(render_opponent_rnn_states),
#                                                 np.concatenate(render_opponent_masks),
#                                                 deterministic=True)
#             render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
#             render_opponent_rnn_states = np.expand_dims(_t2n(render_opponent_rnn_states), axis=0)
#             render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
#             render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
#             render_rewards = render_rewards[:, :self.num_agents // 2, ...]
#             render_episode_rewards += render_rewards
#             self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
#             if render_dones.all():
#                 break
#             render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
#             render_obs = render_obs[:, :self.num_agents // 2, ...]
#         print(render_episode_rewards)

import time
import torch
import logging
import numpy as np
import os
import copy
from typing import List, Tuple, Dict
from .base_runner import Runner, ReplayBuffer
from algorithms.utils.selfplay import get_algorithm, PSRO

def _t2n(x):
    return x.detach().cpu().numpy()

class SelfplayJSBSimRunner(Runner):
    def load(self):
        self.use_selfplay = self.all_args.use_selfplay
        assert self.use_selfplay, "Only selfplay can use SelfplayRunner"
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.num_opponents = self.all_args.n_choose_opponents
        assert self.eval_episodes >= self.num_opponents, \
            f"Number of evaluation episodes ({self.eval_episodes}) should be >= number of opponents ({self.num_opponents})"
        self.init_elo = self.all_args.init_elo
        self.latest_elo = self.init_elo
        self.render_mode = None
        self.global_threshold = 10.0  # 初始动态阈值
        self.last_evaluated_opponents = []  # 最近评估的对手
        self.eval_counts = {}  # 每个对手的评估次数
        # self.eval_cache = {}  # 缓存历史策略对的评估结果
        self.opponent_select_counts = {}  # 记录对手选择频率

        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError

        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        self.buffer = ReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.act_space)
        self.policy_pool = {}

        if self.all_args.selfplay_algorithm == 'psro':
            self.selfplay_algo = PSRO({})
        else:
            self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)

        assert self.num_opponents <= self.n_rollout_threads, \
            f"Number of opponents ({self.num_opponents}) must be <= training threads ({self.n_rollout_threads})!"

        self.opponent_policy = [
            Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
            for _ in range(self.num_opponents)]
        self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
        self.opponent_obs = np.zeros_like(self.buffer.obs[0])
        self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
        self.opponent_masks = np.ones_like(self.buffer.masks[0])

        if self.use_eval:
            self.eval_opponent_policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)

        if self.model_dir is not None:
            self.restore()
            # 检查初始对手是否为预训练模型
            if '0' in self.policy_pool and not os.path.exists(str(self.model_dir) + '/actor_0.pt'):
                logging.warning("Initial opponent '0' is untrained, which may lead to poor early training data. "
                               "Consider using a pre-trained model or increasing warmup episodes.")

    def select_opponents(self, all_opponents: List[str], max_eval_opponents: int = 5) -> List[str]:
        num_opponents = min(len(all_opponents), max_eval_opponents)
        if not all_opponents:
            return []

        scores = {}
        payoff_matrix = self.selfplay_algo.payoff_matrix
        for opp in all_opponents:
            idx = all_opponents.index(opp)
            win_rate = np.mean(payoff_matrix[:, idx])
            deviation = abs(win_rate - 0.5)
            latest_elo = self.latest_elo
            elo_diff = abs(self.policy_pool.get(opp, self.init_elo) - latest_elo) / 400
            elo_score = 1 / (1 + 10 ** elo_diff)
            unevaluated_bonus = 0.1 if opp not in self.last_evaluated_opponents else 0
            random_term = 0.3 * np.random.rand()  # 增加随机项权重以探索高挑战性旧策略
            score = deviation + 0.5 * elo_score + unevaluated_bonus + random_term
            scores[opp] = score

        logging.info(f"Opponent scores: {scores}")
        indices = np.argsort(list(scores.values()))[-num_opponents:]
        selected = [all_opponents[i] for i in indices]

        self.last_evaluated_opponents = selected
        for opp in selected:
            self.eval_counts[opp] = self.eval_counts.get(opp, 0) + 1
            self.opponent_select_counts[opp] = self.opponent_select_counts.get(opp, 0) + 1
        logging.info(f"Opponent selection counts: {self.opponent_select_counts}")

        return selected

    def prune_policy_pool(self, threshold: float = 0.2) -> None:
        keys = list(self.policy_pool.keys())
        n = len(keys)
        to_remove = []
        payoff_matrix = self.selfplay_algo.payoff_matrix
        win_rates = []
        for i in range(n):
            avg_win_rate = np.mean(payoff_matrix[i, :])
            win_rates.append((keys[i], avg_win_rate))
            if avg_win_rate < threshold:
                to_remove.append(keys[i])

        logging.info(f"Policy win rates: {win_rates}")
        for key in to_remove:
            del self.policy_pool[key]

        new_n = len(self.policy_pool)
        new_matrix = np.full((new_n, new_n), 0.5)
        new_keys = list(self.policy_pool.keys())
        for i, ki in enumerate(new_keys):
            for j, kj in enumerate(new_keys):
                old_i = keys.index(ki) if ki in keys else -1
                old_j = keys.index(kj) if kj in keys else -1
                if old_i >= 0 and old_j >= 0:
                    new_matrix[i, j] = payoff_matrix[old_i, old_j]

        self.selfplay_algo.payoff_matrix = new_matrix
        self.selfplay_algo.policy_pool = copy.deepcopy(self.policy_pool)
        logging.info(f"Pruned {len(to_remove)} strategies: {to_remove}")

    def run_evaluation(self, ego_id: str, opponent_id: str, eval_episodes: int) -> Tuple[np.ndarray, np.ndarray]:
        if ego_id == 'latest':
            ego_policy = self.policy
        else:
            ego_policy = self.eval_opponent_policy
            actor_path = str(self.save_dir) + f'/actor_{ego_id}.pt'
            if os.path.exists(actor_path):
                ego_policy.actor.load_state_dict(torch.load(actor_path, weights_only=True))
            else:
                logging.warning(f"Ego actor {actor_path} not found, using current policy.")
                ego_policy.actor.load_state_dict(self.policy.actor.state_dict())

        actor_path = str(self.save_dir) + f'/actor_{opponent_id}.pt'
        if os.path.exists(actor_path):
            self.eval_opponent_policy.actor.load_state_dict(torch.load(actor_path, weights_only=True))
        else:
            logging.warning(f"Opponent actor {actor_path} not found, using current policy.")
            self.eval_opponent_policy.actor.load_state_dict(self.policy.actor.state_dict())

        ego_policy.prep_rollout()
        self.eval_opponent_policy.prep_rollout()

        obs = self.eval_envs.reset()
        masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        opponent_obs = obs[:, self.num_agents // 2:, ...]
        obs = obs[:, :self.num_agents // 2:, ...]
        opponent_masks = np.ones_like(masks, dtype=np.float32)
        opponent_rnn_states = np.zeros_like(rnn_states, dtype=np.float32)
        cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)
        opponent_cumulative_rewards = np.zeros_like(cumulative_rewards)
        episode_rewards = []
        opponent_episode_rewards = []
        episodes_completed = 0

        while episodes_completed < eval_episodes:
            actions, rnn_states = ego_policy.act(
                np.concatenate(obs), np.concatenate(rnn_states), np.concatenate(masks), deterministic=False
            )
            actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_eval_rollout_threads))

            opponent_actions, opponent_rnn_states = self.eval_opponent_policy.act(
                np.concatenate(opponent_obs), np.concatenate(opponent_rnn_states), np.concatenate(opponent_masks),
                deterministic=False
            )
            opponent_rnn_states = np.array(np.split(_t2n(opponent_rnn_states), self.n_eval_rollout_threads))
            opponent_actions = np.array(np.split(_t2n(opponent_actions), self.n_eval_rollout_threads))
            actions = np.concatenate((actions, opponent_actions), axis=1)

            obs, rewards, dones, infos = self.eval_envs.step(actions)
            dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
            episodes_completed += np.sum(dones_env)

            opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2:, ...]
            masks[dones_env] = np.zeros(((dones_env).sum(), *masks.shape[1:]), dtype=np.float32)
            rnn_states[dones_env] = np.zeros(((dones_env).sum(), *rnn_states.shape[1:]), dtype=np.float32)
            opponent_masks[dones_env] = np.zeros(((dones_env).sum(), *opponent_masks.shape[1:]), dtype=np.float32)
            opponent_rnn_states[dones_env] = np.zeros(((dones_env).sum(), *opponent_rnn_states.shape[1:]),
                                                      dtype=np.float32)

            eval_rewards = rewards[:, :self.num_agents // 2:, ...]
            opponent_rewards = rewards[:, self.num_agents // 2:, ...]
            cumulative_rewards += eval_rewards
            opponent_cumulative_rewards += opponent_rewards

            if dones_env.any():
                episode_rewards.append(cumulative_rewards[dones_env].copy())
                opponent_episode_rewards.append(opponent_cumulative_rewards[dones_env].copy())
                cumulative_rewards[dones_env] = 0
                opponent_cumulative_rewards[dones_env] = 0

        ego_rewards = np.concatenate(episode_rewards).squeeze(-1) if episode_rewards else np.array([0])
        opp_rewards = np.concatenate(opponent_episode_rewards).squeeze(-1) if opponent_episode_rewards else np.array([0])
        diff = np.mean(ego_rewards) - np.mean(opp_rewards)
        # 调整胜率计算以生成平滑胜率
        c = max(self.global_threshold * 2, np.abs(diff) / 2)
        result = np.clip((diff + c) / (2 * c), 0, 1)
        logging.info(f"Evaluated {ego_id} vs {opponent_id}: diff={diff:.2f}, c={c:.2f}, result={result:.2f}, "
                    f"ego_rewards: mean={np.mean(ego_rewards):.2f}, std={np.std(ego_rewards):.2f}, "
                    f"min={np.min(ego_rewards):.2f}, max={np.max(ego_rewards):.2f}")
        return ego_rewards, opp_rewards

    def run(self):
        self.model_dir = "../scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/run92"
        if self.model_dir is not None and os.path.exists(self.model_dir):
            episodes = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(self.model_dir)
                        if f.startswith('actor_') and f.endswith('.pt') and f != 'actor_latest.pt']
            if episodes:
                latest_episode = max(episodes)
                self.restore(latest_episode)
                self.save_dir = self.model_dir
                logging.info(f"Restored from episode {latest_episode}, start_episode={self.start_episode}, "
                            f"total_num_steps={self.total_num_steps}, save_dir set to {self.save_dir}")
                if not self.policy_pool:
                    logging.warning("Policy pool is empty after restore, initializing with default.")
                    self.policy_pool = {'0': self.init_elo}
                    self.selfplay_algo.policy_pool = copy.deepcopy(self.policy_pool)
            else:
                self.start_episode = 0
                self.total_num_steps = 0
                self.save_dir = self.model_dir
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
        else:
            logging.info("Model directory not set or not found, starting from scratch with default save directory.")
            self.start_episode = 0
            self.total_num_steps = 0
            self.save_dir = self.run_dir
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.policy_pool = {'0': self.init_elo}
            self.selfplay_algo.policy_pool = copy.deepcopy(self.policy_pool)
            self.eval(0)

        self.warmup()

        start = time.time()
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads
        for episode in range(self.start_episode, episodes):
            heading_turns_list = []
            for step in range(self.buffer_size):
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions)

                for info in infos:
                    if 'heading_turn_counts' in info:
                        heading_turns_list.append(info['heading_turn_counts'])

                data = obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic
                self.insert(data)

            self.compute()
            train_infos = self.train()
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

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
                train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                # 添加训练损失日志
                if "policy_loss" in train_infos:
                    logging.info(f"Policy loss: {train_infos['policy_loss']:.4f}")
                if "value_loss" in train_infos:
                    logging.info(f"Value loss: {train_infos['value_loss']:.4f}")
                logging.info(f"Average episode rewards: {train_infos['average_episode_rewards']:.4f}")
                self.log_info(train_infos, self.total_num_steps)

            if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
                self.eval(self.total_num_steps)

            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)
                self.reset_opponent()

            if episode % 10 == 0 and episode != 0:
                self.prune_policy_pool(threshold=0.2)

    def warmup(self):
        obs = self.envs.reset()
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        obs = obs[:, :self.num_agents // 2:, ...]
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        opponent_actions = np.zeros_like(actions)
        for policy_idx, policy in enumerate(self.opponent_policy):
            env_idx = self.opponent_env_split[policy_idx]
            opponent_action, opponent_rnn_states \
                = policy.act(np.concatenate(self.opponent_obs[env_idx]),
                             np.concatenate(self.opponent_rnn_states[env_idx]),
                             np.concatenate(self.opponent_masks[env_idx]))
            opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
            self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
        actions = np.concatenate((actions, opponent_actions), axis=1)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
        rnn_states_actor[dones_env] = np.zeros(((dones_env).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env] = np.zeros(((dones_env).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env] = np.zeros(((dones_env).sum(), self.num_agents, 1), dtype=np.float32)

        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        self.opponent_masks = masks[:, self.num_agents // 2:, ...]
        self.opponent_rnn_states[dones_env] = np.zeros(((dones_env).sum(), *rnn_states_actor.shape[1:]),
                                                       dtype=np.float32)
        obs = obs[:, :self.num_agents // 2:, ...]
        actions = actions[:, :self.num_agents // 2:, ...]
        rewards = rewards[:, :self.num_agents // 2:, ...]
        masks = masks[:, :self.num_agents // 2:, ...]
        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    def reset_opponent(self):
        choose_opponents = []
        for policy in self.opponent_policy:
            choose_idx = self.selfplay_algo.choose(self.policy_pool)
            choose_opponents.append(choose_idx)
            actor_path = str(self.save_dir) + f'/actor_{choose_idx}.pt'
            if not os.path.exists(actor_path):
                logging.info(f"Using current policy as opponent {choose_idx} since file {actor_path} does not exist.")
                policy.actor.load_state_dict(self.policy.actor.state_dict())
            else:
                policy.actor.load_state_dict(torch.load(actor_path, weights_only=True))
            policy.prep_rollout()
        logging.info(f"Choose opponents {choose_opponents} for training, saving to {self.save_dir}")
        self.buffer.clear()
        self.opponent_obs = np.zeros_like(self.opponent_obs)
        self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
        self.opponent_masks = np.ones_like(self.opponent_masks)
        obs = self.envs.reset()
        if self.num_opponents > 0:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2:, ...]
        self.buffer.obs[0] = obs.copy()

    def eval(self, total_num_steps):
        self.policy.prep_rollout()
        all_opponents = list(self.policy_pool.keys())
        if not all_opponents:
            logging.warning("Policy pool is empty; initializing with '0'.")
            all_opponents = ['0']
            self.policy_pool['0'] = self.init_elo
            self.selfplay_algo.policy_pool = copy.deepcopy(self.policy_pool)

        eval_results = {}
        eval_data = []
        eval_choose_opponents = []

        episode = total_num_steps // (self.buffer_size * self.n_rollout_threads)
        logging.info(f"Evaluating at total_num_steps={total_num_steps}, episode={episode}")

        max_eval_opponents = 5
        select_opponents = self.select_opponents(all_opponents, max_eval_opponents)
        if not select_opponents and episode == 0:
            select_opponents = all_opponents  # 确保初始评估包含 '0'

        if episode % 50 == 0 and episode > 0:  # 全评估每 50 个 episode
            logging.info(f"Performing full evaluation at episode={episode}")
            # 评估 latest 对所有对手
            eval_choose_opponents = all_opponents
            eval_each_episodes = max(1, self.eval_episodes // len(eval_choose_opponents))
            for opponent_idx in eval_choose_opponents:
                ego_rewards, opp_rewards = self.run_evaluation('latest', opponent_idx, eval_each_episodes)
                diff = np.mean(ego_rewards) - np.mean(opp_rewards)
                if abs(diff) > 10 * self.global_threshold:  # 过滤异常 diff
                    logging.warning(f"Filtered outlier diff={diff:.6f} for pair (latest, {opponent_idx})")
                    diff = 0.0
                eval_results[opponent_idx] = [diff]
                eval_data.append((ego_rewards, opp_rewards))
                result = np.clip((diff + self.global_threshold * 2) / (2 * self.global_threshold * 2), 0, 1)
                logging.info(f"Evaluated latest vs {opponent_idx}: diff={diff:.6f}, c={self.global_threshold * 2:.2f}, "
                             f"result={result:.6f}, ego_rewards: mean={np.mean(ego_rewards):.2f}, std={np.std(ego_rewards):.2f}")

            # 评估对手之间的对战（可选，保持 payoff_matrix 完整性）
            batch_size = 20
            pairs = [(opp1, opp2) for i, opp1 in enumerate(all_opponents) for opp2 in all_opponents[i:] if opp1 != opp2]
            eval_pairs = pairs[:batch_size] if len(pairs) > batch_size else pairs
            for opp1, opp2 in eval_pairs:
                ego_rewards, opp_rewards = self.run_evaluation(opp1, opp2, eval_each_episodes)
                diff = np.mean(ego_rewards) - np.mean(opp_rewards)
                if abs(diff) > 10 * self.global_threshold:
                    logging.warning(f"Filtered outlier diff={diff:.6f} for pair ({opp1}, {opp2})")
                    diff = 0.0
                eval_results[(opp1, opp2)] = [diff]
                eval_results[(opp2, opp1)] = [-diff]  # 对称性
                eval_data.append((ego_rewards, opp_rewards))
                result = np.clip((diff + self.global_threshold * 2) / (2 * self.global_threshold * 2), 0, 1)
                logging.info(f"Evaluated {opp1} vs {opp2}: diff={diff:.6f}, c={self.global_threshold * 2:.2f}, "
                             f"result={result:.6f}, ego_rewards: mean={np.mean(ego_rewards):.2f}, std={np.std(ego_rewards):.2f}")

        else:
            logging.info(f"Performing partial evaluation at episode={episode}: {select_opponents}")
            eval_choose_opponents = select_opponents
            eval_each_episodes = max(1, self.eval_episodes // len(eval_choose_opponents))
            for opponent_idx in eval_choose_opponents:
                ego_rewards, opp_rewards = self.run_evaluation('latest', opponent_idx, eval_each_episodes)
                diff = np.mean(ego_rewards) - np.mean(opp_rewards)
                if abs(diff) > 10 * self.global_threshold:  # 过滤异常 diff
                    logging.warning(f"Filtered outlier diff={diff:.6f} for pair (latest, {opponent_idx})")
                    diff = 0.0
                eval_results[opponent_idx] = [diff]
                eval_data.append((ego_rewards, opp_rewards))
                result = np.clip((diff + self.global_threshold * 2) / (2 * self.global_threshold * 2), 0, 1)
                logging.info(f"Evaluated latest vs {opponent_idx}: diff={diff:.6f}, c={self.global_threshold * 2:.2f}, "
                             f"result={result:.6f}, ego_rewards: mean={np.mean(ego_rewards):.2f}, std={np.std(ego_rewards):.2f}")

        # 计算 diffs
        diffs = []
        for ego_rewards, opp_rewards in eval_data:
            diff = np.mean(ego_rewards) - np.mean(opp_rewards)
            if abs(diff) > 10 * self.global_threshold:
                diff = 0.0
            diffs.append(diff)

        if len(diffs) > 1:
            current_threshold = np.clip(1.5 * np.std(diffs), 1, 50)
            alpha = 0.5
            self.global_threshold = alpha * current_threshold + (1 - alpha) * self.global_threshold
        logging.info(f"diffs: mean={np.mean(diffs):.2f}, std={np.std(diffs):.2f}, count={len(diffs)}, "
                     f"threshold={self.global_threshold:.2f}")

        logging.info(f"Evaluated opponents: {eval_choose_opponents}")
        unevaluated = [opp for opp in all_opponents if opp not in eval_choose_opponents]
        logging.info(f"Unevaluated opponents: {unevaluated}")
        if eval_data:
            ego_rewards_list = [d[0] for d in eval_data]
            opp_rewards_list = [d[1] for d in eval_data]
            ego_rewards_all = np.concatenate(ego_rewards_list)
            opp_rewards_all = np.concatenate(opp_rewards_list)
            logging.info(f"ego_rewards mean={np.mean(ego_rewards_all):.2f}, std={np.std(ego_rewards_all):.2f}")
            logging.info(f"opp_rewards mean={np.mean(opp_rewards_all):.2f}, std={np.std(opp_rewards_all):.2f}")

        if isinstance(self.selfplay_algo, PSRO):
            policy_pool_copy = copy.deepcopy(self.policy_pool)
            self.selfplay_algo.update(policy_pool_copy, eval_results)
            self.selfplay_algo.policy_pool = copy.deepcopy(self.policy_pool)

        # ELO 更新
        if eval_choose_opponents:
            ego_elo = np.array([self.latest_elo] * len(eval_choose_opponents))
            opponent_elo = np.array([self.policy_pool.get(key, self.init_elo) for key in eval_choose_opponents])
            expected_score = 1 / (1 + 10 ** ((opponent_elo - ego_elo) / 400))
            actual_score = np.array([
                np.clip((eval_results.get(key, [0])[0] + self.global_threshold * 2) / (2 * self.global_threshold * 2),
                        0, 1)
                for key in eval_choose_opponents
            ])
            K = np.clip(32 * (1 + np.tanh(np.abs(diffs[:len(eval_choose_opponents)]) / (self.global_threshold + 1e-5))),
                        16, 64)
            elo_gain = K * (actual_score - expected_score)
            logging.info(f"ELO updates: opponent_elo={opponent_elo}, expected_score={expected_score}, "
                         f"actual_score={actual_score}, K={K}, elo_gain={elo_gain}")
            for i, key in enumerate(eval_choose_opponents):
                self.policy_pool[key] = float(opponent_elo[i] + elo_gain[i])
            self.latest_elo = float((ego_elo + elo_gain).mean())

        eval_infos = {
            'eval_average_episode_rewards': np.mean(
                [np.mean(ego_rewards) for ego_rewards, _ in eval_data]
            ) if eval_data else 0.0,
            'latest_elo': self.latest_elo
        }
        self.log_info(eval_infos, self.total_num_steps)
        self.reset_opponent()
    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
        torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')
        self.policy_pool[str(episode)] = self.latest_elo
        self.selfplay_algo.policy_pool = copy.deepcopy(self.policy_pool)

        if hasattr(self.policy, 'optimizer') and self.policy.optimizer is not None:
            torch.save(self.policy.optimizer.state_dict(), str(self.save_dir) + '/optimizer_latest.pt')
        else:
            logging.warning("Optimizer not available, skipping save.")
        torch.save(self.buffer, str(self.save_dir) + '/buffer_latest.pt')
        training_state = {
            'episode': episode,
            'total_num_steps': self.total_num_steps,
            'latest_elo': self.latest_elo,
            'policy_pool': self.policy_pool
        }
        torch.save(training_state, str(self.save_dir) + '/training_state_latest.pt')
        if isinstance(self.selfplay_algo, PSRO):
            np.save(str(self.save_dir) + '/payoff_matrix.npy', self.selfplay_algo.payoff_matrix)

    def restore(self, episode=None):
        if episode is None:
            actor_path = str(self.model_dir) + '/actor_latest.pt'
            critic_path = str(self.model_dir) + '/critic_latest.pt'
            optimizer_path = str(self.model_dir) + '/optimizer_latest.pt'
            buffer_path = str(self.model_dir) + '/buffer_latest.pt'
            training_state_path = str(self.model_dir) + '/training_state_latest.pt'
            payoff_matrix_path = str(self.model_dir) + '/payoff_matrix.npy'
        else:
            actor_path = str(self.model_dir) + f'/actor_{episode}.pt'
            critic_path = str(self.model_dir) + '/critic_latest.pt'
            optimizer_path = str(self.model_dir) + '/optimizer_latest.pt'
            buffer_path = str(self.model_dir) + '/buffer_latest.pt'
            training_state_path = str(self.model_dir) + '/training_state_latest.pt'
            payoff_matrix_path = str(self.model_dir) + '/payoff_matrix.npy'

        if not os.path.exists(actor_path):
            logging.error(f"Actor file {actor_path} not found!")
            raise FileNotFoundError(f"Actor file {actor_path} not found!")
        self.policy.actor.load_state_dict(torch.load(actor_path, weights_only=True))

        if not os.path.exists(critic_path):
            logging.error(f"Critic file {critic_path} not found!")
            raise FileNotFoundError(f"Critic file {critic_path} not found!")
        self.policy.critic.load_state_dict(torch.load(critic_path, weights_only=True))

        if hasattr(self.trainer, 'optimizer') and os.path.exists(optimizer_path):
            self.policy.optimizer.load_state_dict(torch.load(optimizer_path, weights_only=False))
        else:
            logging.warning(f"Optimizer file {optimizer_path} not found, skipping optimizer restore.")

        if os.path.exists(buffer_path):
            self.buffer = torch.load(buffer_path, weights_only=False)
        else:
            logging.warning(f"Buffer file {buffer_path} not found, initializing new buffer.")
            self.buffer = ReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.act_space)

        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)
            self.start_episode = training_state['episode'] + 1
            self.total_num_steps = training_state['total_num_steps']
            self.latest_elo = training_state['latest_elo']
            self.policy_pool = training_state.get('policy_pool', {})
            logging.info(
                f"Loaded training state: episode={training_state['episode']}, total_num_steps={self.total_num_steps}, "
                f"latest_elo={self.latest_elo}, policy_pool={self.policy_pool}")
        else:
            logging.warning(f"Training state file {training_state_path} not found, starting from scratch.")
            self.start_episode = 0
            self.total_num_steps = 0
            self.latest_elo = self.init_elo
            self.policy_pool = {'0': self.init_elo}

        if isinstance(self.selfplay_algo, PSRO):
            if os.path.exists(payoff_matrix_path):
                self.selfplay_algo.payoff_matrix = np.load(payoff_matrix_path)
                if self.selfplay_algo.payoff_matrix.shape[0] != len(self.policy_pool):
                    logging.warning(
                        f"Payoff matrix size {self.selfplay_algo.payoff_matrix.shape} does not match policy_pool size {len(self.policy_pool)}, resizing.")
                    n = len(self.policy_pool)
                    new_matrix = np.full((n, n), 0.5)
                    old_n = min(self.selfplay_algo.payoff_matrix.shape[0], n)
                    new_matrix[:old_n, :old_n] = self.selfplay_algo.payoff_matrix[:old_n, :old_n]
                    self.selfplay_algo.payoff_matrix = new_matrix
            else:
                n = len(self.policy_pool)
                self.selfplay_algo.payoff_matrix = np.full((n, n), 0.5)
                logging.warning(f"Payoff matrix file {payoff_matrix_path} not found, initialized to {n}x{n} with 0.5.")
            self.selfplay_algo.policy_pool = copy.deepcopy(self.policy_pool)

    @torch.no_grad()
    def render(self):
        idx = self.all_args.render_index
        opponent_idx = self.all_args.render_opponent_index
        dir_list = str(self.run_dir).split('/')
        file_path = '/'.join(dir_list[:dir_list.index('results') + 1])
        self.policy.actor.load_state_dict(torch.load(str(self.model_dir) + f'/actor_{idx}.pt', weights_only=True))
        self.policy.prep_rollout()
        self.eval_opponent_policy.actor.load_state_dict(
            torch.load(str(self.model_dir) + f'/actor_{opponent_idx}.pt', weights_only=True))
        self.eval_opponent_policy.prep_rollout()
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
        render_obs = render_obs[:, :self.num_agents // 2:, ...]
        render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
        render_opponent_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            render_opponent_actions, render_opponent_rnn_states \
                = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs),
                                                np.concatenate(render_opponent_rnn_states),
                                                np.concatenate(render_opponent_masks),
                                                deterministic=True)
            render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
            render_opponent_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            render_rewards = render_rewards[:, :self.num_agents // 2:, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
            render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
            render_obs = render_obs[:, :self.num_agents // 2:, ...]
        print(render_episode_rewards)