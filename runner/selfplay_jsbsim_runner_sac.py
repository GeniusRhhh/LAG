import torch
import logging
import numpy as np
import os
from typing import List

from algorithms.utils.buffer import ReplayBuffer
from algorithms.utils.selfplay import get_algorithm
from .base_runner import Runner
from .jsbsim_runner import JSBSimRunner
from .base_runner import Runner
from .jsbsim_runner import JSBSimRunner

def _t2n(x):
    return x.detach().cpu().numpy()

class SelfplayJSBSimRunner(JSBSimRunner):
    def load(self):
        self.use_selfplay = self.all_args.use_selfplay
        assert self.use_selfplay, "Only selfplay can use SelfplayRunner"
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.num_opponents = self.all_args.n_choose_opponents
        assert self.eval_episodes >= self.num_opponents, \
            f"Number of evaluation episodes:{self.eval_episodes} should be greater than number of opponents:{self.num_opponents}"
        self.init_elo = self.all_args.init_elo
        self.latest_elo = self.init_elo

        # 当前policy和trainer在train_jsbsim.py中已根据algorithm_name赋值
        # 若为SAC, policy和trainer为SACPolicy和SACTrainer对象
        # 若仍为PPO，需要在train_jsbsim.py中切换为SAC。

        # 对SAC需要off-policy replay buffer
        obs_dim = self.obs_space.shape[0]
        act_dim = self.act_space.shape[0]
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, self.all_args.buffer_size)

        self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)
        assert self.num_opponents <= self.n_rollout_threads, \
            f"Number of opponents({self.num_opponents}) must be <= number of training threads({self.n_rollout_threads})!"

        # 对手策略同主策略算法类型，这里以SAC为例
        from algorithms.sac.sac_policy import SACPolicy
        self.opponent_policy = [
            SACPolicy(self.all_args, self.obs_space, self.act_space, device=self.device)
            for _ in range(self.num_opponents)
        ]
        self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))

        self.opponent_obs = None
        self.opponent_rnn_states = None
        self.opponent_masks = None

        if self.use_eval:
            self.eval_opponent_policy = SACPolicy(self.all_args, self.obs_space, self.act_space, device=self.device)

        logging.info("\n Load selfplay opponents: Algo {}, num_opponents {}.\n"
                     .format(self.all_args.selfplay_algorithm, self.num_opponents))

        if self.model_dir is not None:
            self.restore()

        # SAC相关参数（需在config或命令行中定义这些参数）
        self.start_steps = getattr(self.all_args, 'start_steps', 10000)
        self.update_interval = getattr(self.all_args, 'update_interval', 50)
        self.update_times = getattr(self.all_args, 'update_times', 1)
        self.batch_size = getattr(self.all_args, 'batch_size', 256)
        self.total_steps = 0

    def run(self):
        # 使用SAC进行训练：off-policy
        # 在start_steps前仅收集数据不更新，之后每隔update_interval步更新update_times次
        for episode in range(self.all_args.num_episodes):
            obs = self.envs.reset()
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            ego_obs = obs[:, :self.num_agents // 2, ...]
            done = [False]*self.n_rollout_threads

            while not all(done):
                # 主策略选择动作
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.policy.get_actions(
                    np.concatenate(ego_obs),
                    np.zeros((self.n_rollout_threads,1,self.all_args.recurrent_hidden_size), dtype=np.float32),
                    np.zeros((self.n_rollout_threads,1,self.all_args.recurrent_hidden_size), dtype=np.float32),
                    np.ones((self.n_rollout_threads,self.num_agents//2,1), dtype=np.float32)
                )
                actions = np.array(np.split(actions, self.n_rollout_threads))

                # 对手策略选择动作
                opponent_actions = np.zeros_like(actions)
                for policy_idx, opp_pi in enumerate(self.opponent_policy):
                    env_idx = self.opponent_env_split[policy_idx]
                    opp_act, _ = opp_pi.act(
                        np.concatenate(self.opponent_obs[env_idx]),
                        np.zeros((len(env_idx),1,self.all_args.recurrent_hidden_size), dtype=np.float32),
                        np.ones((len(env_idx),self.num_agents//2,1), dtype=np.float32),
                        deterministic=False
                    )
                    opp_act = np.array(np.split(opp_act, len(env_idx)))
                    opponent_actions[env_idx] = opp_act

                joint_actions = np.concatenate((actions, opponent_actions), axis=1)
                next_obs, rewards, dones, infos = self.envs.step(joint_actions)
                self.total_steps += 1

                next_opponent_obs = next_obs[:, self.num_agents // 2:, ...]
                next_ego_obs = next_obs[:, :self.num_agents // 2, ...]

                # 将transition存入replay buffer（对ego侧智能体）
                ego_obs_flat = np.squeeze(ego_obs, axis=1)
                next_ego_obs_flat = np.squeeze(next_ego_obs, axis=1)
                rewards_flat = np.squeeze(rewards[:, :self.num_agents//2, ...], axis=1)
                done_flags = np.all(dones, axis=-1)

                for i in range(self.n_rollout_threads):
                    self.replay_buffer.store(ego_obs_flat[i], actions[i][0], rewards_flat[i], next_ego_obs_flat[i], done_flags[i])

                ego_obs = next_ego_obs
                self.opponent_obs = next_opponent_obs
                done = done_flags.tolist()

                # SAC更新逻辑
                if (self.total_steps > self.start_steps and
                    self.total_steps % self.update_interval == 0 and
                    self.replay_buffer.size > self.batch_size):
                    for _ in range(self.update_times):
                        train_info = self.trainer.train(self.replay_buffer)
                        # 可在此处记录train_info到日志或wandb

            # 每个episode结束后评估并更新对手策略池
            if (episode+1) % self.all_args.eval_interval == 0:
                self.eval(episode)
                self.after_update(episode)

    def after_update(self, episode):
        # 保存当前策略
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, os.path.join(self.save_dir, f'actor_{episode}.pt'))
        # 更新策略池ELO分数
        self.policy_pool[str(episode)] = self.latest_elo

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        self.policy.prep_rollout()
        total_episodes = 0
        episode_rewards = []
        opponent_episode_rewards = []

        cumulative_rewards = np.zeros((self.n_eval_rollout_threads,1), dtype=np.float32)
        opponent_cumulative_rewards = np.zeros_like(cumulative_rewards)

        # 根据PSRO或FSP等算法选择对手
        eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in range(self.num_opponents)]
        eval_each_episodes = self.eval_episodes // self.num_opponents
        logging.info(f" Choose opponents {eval_choose_opponents} for evaluation")

        eval_cur_opponent_idx = 0
        while total_episodes < self.eval_episodes:
            if total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
                policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
                self.eval_opponent_policy.actor.load_state_dict(
                    torch.load(os.path.join(self.save_dir,f'actor_{policy_idx}.pt'), map_location=self.device)
                )
                self.eval_opponent_policy.prep_rollout()
                eval_cur_opponent_idx += 1
                logging.info(f" Load opponent {policy_idx} for evaluation ({total_episodes}/{self.eval_episodes})")

                obs = self.eval_envs.reset()
                masks = np.ones((self.n_eval_rollout_threads,1,1), dtype=np.float32)
                rnn_states = np.zeros((self.n_eval_rollout_threads,1,self.all_args.recurrent_hidden_size), dtype=np.float32)

                opponent_obs = obs[:, self.num_agents // 2:, ...]
                obs = obs[:, :self.num_agents // 2, ...]
                opponent_masks = np.ones_like(masks, dtype=np.float32)
                opponent_rnn_states = np.zeros_like(rnn_states, dtype=np.float32)

            actions, rnn_states = self.policy.act(np.concatenate(obs),
                                                  np.concatenate(rnn_states),
                                                  np.concatenate(masks), deterministic=True)
            actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_eval_rollout_threads))

            opponent_actions, opponent_rnn_states = self.eval_opponent_policy.act(
                np.concatenate(opponent_obs),
                np.concatenate(opponent_rnn_states),
                np.concatenate(opponent_masks),
                deterministic=True
            )
            opponent_rnn_states = np.array(np.split(_t2n(opponent_rnn_states), self.n_eval_rollout_threads))
            opponent_actions = np.array(np.split(_t2n(opponent_actions), self.n_eval_rollout_threads))
            joint_actions = np.concatenate((actions, opponent_actions), axis=1)

            obs, eval_rewards, dones, eval_infos = self.eval_envs.step(joint_actions)
            dones_env = np.all(dones.squeeze(-1), axis=-1)
            total_episodes += np.sum(dones_env)

            opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            masks[dones_env == True] = 0.0
            rnn_states[dones_env == True] = 0.0
            opponent_masks[dones_env == True] = 0.0
            opponent_rnn_states[dones_env == True] = 0.0

            opponent_rewards = eval_rewards[:, self.num_agents // 2:, ...]
            opponent_cumulative_rewards += opponent_rewards
            opponent_episode_rewards.append(opponent_cumulative_rewards[dones_env==True])
            opponent_cumulative_rewards[dones_env==True] = 0

            eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]
            cumulative_rewards += eval_rewards
            episode_rewards.append(cumulative_rewards[dones_env==True])
            cumulative_rewards[dones_env==True] = 0

        episode_rewards = np.concatenate(episode_rewards)
        episode_rewards = episode_rewards.squeeze(-1).mean(axis=-1)
        eval_average_episode_rewards = np.array(np.split(episode_rewards, self.num_opponents)).mean(axis=-1)

        opponent_episode_rewards = np.concatenate(opponent_episode_rewards)
        opponent_episode_rewards = opponent_episode_rewards.squeeze(-1).mean(axis=-1)
        opponent_average_episode_rewards = np.array(np.split(opponent_episode_rewards, self.num_opponents)).mean(axis=-1)

        # 更新ELO
        ego_elo = np.array([self.latest_elo for _ in range(self.n_eval_rollout_threads)])
        opponent_elo = np.array([self.policy_pool[key] for key in eval_choose_opponents])
        expected_score = 1/(1+10**((ego_elo - opponent_elo)/400))

        actual_score = np.zeros_like(expected_score)
        diff = opponent_average_episode_rewards - eval_average_episode_rewards
        actual_score[diff>100] = 1
        actual_score[np.abs(diff)<100] = 0.5
        actual_score[diff<-100] = 0

        elo_gain = 32*(actual_score - expected_score)
        update_opponent_elo = opponent_elo + elo_gain
        for i, key in enumerate(eval_choose_opponents):
            self.policy_pool[key] = update_opponent_elo[i]
        ego_elo = ego_elo - elo_gain
        self.latest_elo = ego_elo.mean()

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = eval_average_episode_rewards.mean()
        eval_infos['latest_elo'] = self.latest_elo
        logging.info(" eval average episode rewards: " + str(eval_infos['eval_average_episode_rewards']))
        logging.info(" latest elo score: " + str(self.latest_elo))
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

        self.reset_opponent()

    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, os.path.join(self.save_dir, 'actor_latest.pt'))
        policy_critic1_state_dict = self.policy.critic1.state_dict()
        policy_critic2_state_dict = self.policy.critic2.state_dict()
        torch.save(policy_critic1_state_dict, os.path.join(self.save_dir, 'critic1_latest.pt'))
        torch.save(policy_critic2_state_dict, os.path.join(self.save_dir, 'critic2_latest.pt'))
        torch.save(policy_actor_state_dict, os.path.join(self.save_dir,f'actor_{episode}.pt'))
        torch.save(self.policy_pool, os.path.join(self.save_dir,f'policy_pool_{episode}.pt'))
        self.policy_pool[str(episode)] = self.latest_elo

    def reset_opponent(self):
        choose_opponents = []
        for policy in self.opponent_policy:
            choose_idx = self.selfplay_algo.choose(self.policy_pool)
            choose_opponents.append(choose_idx)
            policy.actor.load_state_dict(
                torch.load(os.path.join(self.save_dir,f'actor_{choose_idx}.pt'), map_location=self.device)
            )
            policy.prep_rollout()
        logging.info(f" Choose opponents {choose_opponents} for training")

        # 重置replay buffer，开始新的对抗
        self.replay_buffer = ReplayBuffer(self.obs_space.shape[0], self.act_space.shape[0], self.all_args.buffer_size)
        obs = self.envs.reset()
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]

    @torch.no_grad()
    def render(self):
        idx = self.all_args.render_index
        opponent_idx = self.all_args.render_opponent_index
        dir_list = str(self.run_dir).split('/')
        file_path = '/'.join(dir_list[:dir_list.index('results') + 1])
        self.policy.actor.load_state_dict(torch.load(os.path.join(self.model_dir,f'actor_{idx}.pt'), map_location=self.device))
        self.policy.prep_rollout()
        self.eval_opponent_policy.actor.load_state_dict(
            torch.load(os.path.join(self.model_dir,f'actor_{opponent_idx}.pt'), map_location=self.device)
        )
        self.eval_opponent_policy.prep_rollout()
        logging.info("\nStart render ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
        render_masks = np.ones((1,1,1), dtype=np.float32)
        render_rnn_states = np.zeros((1,1,self.all_args.recurrent_hidden_size), dtype=np.float32)
        render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
        render_obs = render_obs[:, :self.num_agents // 2, ...]
        render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
        render_opponent_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(
                np.concatenate(render_obs),
                np.concatenate(render_rnn_states),
                np.concatenate(render_masks),
                deterministic=True
            )
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            render_opponent_actions, render_opponent_rnn_states = self.eval_opponent_policy.act(
                np.concatenate(render_opponent_obs),
                np.concatenate(render_opponent_rnn_states),
                np.concatenate(render_opponent_masks),
                deterministic=True
            )
            render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
            render_opponent_rnn_states = np.expand_dims(_t2n(render_opponent_rnn_states), axis=0)
            render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)

            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
            logging.debug(
                f"Render step: Aircraft states and actions logged to {file_path}/{self.experiment_name}.txt.acmi")
            render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
            render_obs = render_obs[:, :self.num_agents // 2, ...]
        print("Render complete with rewards:", render_episode_rewards)
