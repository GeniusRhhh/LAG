import logging
import time
import numpy as np
import torch
from algorithms.utils.buffer import SharedReplayBuffer
from .base_runner import Runner

def _t2n(x):
    """将张量转换为 numpy 数组。

    Args:
        x: 输入张量。

    Returns:
        np.ndarray: 转换后的数组。
    """
    return x.detach().cpu().numpy()

class ShareJSBSimRunner(Runner):
    """多线程 JSBSim 运行器，支持自博弈。"""

    def load(self):
        """加载环境、策略和缓冲区。"""
        self.obs_space = self.envs.observation_space
        self.share_obs_space = self.envs.share_observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay
        self.use_rule_opponent = getattr(self.all_args, 'use_rule_opponent', True)
        if self.algorithm_name == "mappo":
            from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.mappo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError("Only MAPPO algorithm is supported")
        self.policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)
        self.buffer = SharedReplayBuffer(self.all_args, self.num_agents // 2 if self.use_selfplay else self.num_agents,
                                        self.obs_space, self.share_obs_space, self.act_space)
        if self.use_selfplay:
            from algorithms.utils.selfplay import get_algorithm
            self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)
            self.n_choose_opponents = max(getattr(self.all_args, 'n_choose_opponents', 1), 1)
            if self.n_choose_opponents > self.n_rollout_threads:
                logging.warning(f"Number of opponent choices {self.n_choose_opponents} > number of threads {self.n_rollout_threads}, set to {self.n_rollout_threads}")
                self.n_choose_opponents = self.n_rollout_threads
            self.policy_pool = {'latest': self.all_args.init_elo}
            self.opponent_policy = [
                Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
                for _ in range(self.n_choose_opponents)]
            self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
            self.opponent_obs = np.zeros_like(self.buffer.obs[0])
            self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
            self.opponent_masks = np.ones_like(self.buffer.masks[0])
            if self.use_eval:
                self.eval_opponent_policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
            logging.info(f"Loaded self-play opponents: algorithm={self.all_args.selfplay_algorithm}, opponent count={self.n_choose_opponents}, rule-based opponent={self.use_rule_opponent}")
        else:
            self.opponent_policy = []
            self.opponent_env_split = []
        if self.model_dir is not None:
            self.restore()

    # 文件：share_jsbsim_runner.py
    def run(self):
        self.warmup()
        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads
        win_rates = []
        for episode in range(episodes):
            episode_rewards = []
            episode_actions = []
            episode_phases = []
            for step in range(self.buffer_size):
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)
                obs, share_obs, rewards, dones, infos = self.envs.step(actions)
                episode_rewards.append(rewards[0, :self.num_agents // 2])
                episode_actions.append(actions[0, :self.num_agents // 2])
                phase_list = []
                reward_comps = {}
                infos_dict = infos[0] if isinstance(infos, np.ndarray) else infos
                logging.debug(f"Step {step} infos_dict: {infos_dict}")
                agent_names = [f"A0{i + 1}00" for i in range(self.num_agents // 2)]
                for idx, agent_name in enumerate(agent_names):
                    phase_list.append(infos_dict.get(agent_name, {}).get("current_phase", "unknown"))
                    reward_comps[agent_name] = infos_dict.get(agent_name, {}).get("reward_details", {})
                    if not reward_comps[agent_name]:
                        logging.warning(f"No reward details for {agent_name} at step {step}, episode {episode}")
                episode_phases.append(phase_list)
                data = obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic
                self.insert(data)
            self.compute()
            train_infos = self.train()
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads
            if "win" in infos_dict:
                win_rates.append(infos_dict["win"])
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)
            if episode % self.log_interval == 0:
                end = time.time()
                avg_reward = np.mean([r.mean() for r in episode_rewards])
                win_rate = np.mean(win_rates[-self.log_interval:]) if win_rates else 0
                actions_array = np.array(episode_actions)
                template_ids = actions_array[:, :, 0]
                shoot_flags = actions_array[:, :, 1]
                template_dist = {i: np.sum(template_ids == i) / template_ids.size for i in range(15)}
                shoot_ratio = np.mean(shoot_flags)
                phases_array = np.array(episode_phases)
                phase_counts = {p: np.sum(phases_array == p) / phases_array.size for p in np.unique(phases_array)}
                for idx, agent_name in enumerate(agent_names):
                    state = obs[0, idx]
                    action = actions[0, idx]
                    phase = infos_dict.get(agent_name, {}).get("current_phase", "unknown")
                    reward = float(episode_rewards[step][idx].item()) if isinstance(
                        episode_rewards[step][idx], np.ndarray) else float(episode_rewards[step][idx])
                    altitude = float(state[0].item()) * 5000
                    velocity = float(state[5].item()) * 340
                    distance_idx = 14 + 4
                    distance = float(state[distance_idx].item()) * 10000 if len(state) > distance_idx else float('inf')
                    action_scalar = action.tolist() if isinstance(action, np.ndarray) and action.size > 1 else float(
                        action.item())
                    logging.debug(f"Agent {agent_name} state: {state}, reward: {reward}")
                    logging.info(
                        f"Agent {agent_name} - Episode {episode}: Altitude={altitude:.1f}m, Velocity={velocity:.1f}m/s, Distance={distance:.1f}m, Action={action_scalar}, Phase={phase}, Reward={reward:.2f}")
                logging.info(
                    f"Scenario: {self.all_args.scenario_name} ... FPS: {int(self.total_num_steps / (end - start))}")
                logging.info(f"Action Distribution (template_id): {template_dist}")
                logging.info(f"Shoot Action Ratio: {shoot_ratio:.3f}")
                logging.info(f"Phase Distribution: {phase_counts}")
                logging.info(f"Reward Components: {reward_comps}")
                train_infos["average_episode_rewards"] = avg_reward
                train_infos["win_rate"] = win_rate
                logging.info(f"Average episode reward: {avg_reward}, Win rate: {win_rate}")
                self.log_info(train_infos, self.total_num_steps)
                if win_rate > 0.7 and self.use_rule_opponent:
                    logging.info("Switching to self-play mode due to high win rate")
                    self.use_rule_opponent = False
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.total_num_steps)
    def warmup(self):
        """预热环境，初始化缓冲区。"""
        obs, share_obs = self.envs.reset()
        if self.use_selfplay:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """收集一步数据。

        Args:
            step: 当前步骤索引。

        Returns:
            tuple: 值、动作、概率、RNN 状态等。
        """
        self.policy.prep_rollout()
        obs = np.concatenate(self.buffer.obs[step])
        rnn_states_actor = np.concatenate(self.buffer.rnn_states_actor[step])
        if np.isnan(obs).any() or np.isnan(rnn_states_actor).any():
            logging.error(f"NaN detected in obs or rnn_states at step {step}: obs={obs}, rnn_states={rnn_states_actor}")
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            obs,
            rnn_states_actor,
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]))
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        if self.use_selfplay:
            opponent_actions = np.zeros_like(actions)
            if self.use_rule_opponent:
                opponent_actions = actions  # 蓝方动作由环境生成
            else:
                for policy_idx, policy in enumerate(self.opponent_policy):
                    env_idx = self.opponent_env_split[policy_idx]
                    opponent_action, opponent_rnn_states = policy.act(
                        np.concatenate(self.opponent_obs[env_idx]),
                        np.concatenate(self.opponent_rnn_states[env_idx]),
                        np.concatenate(self.opponent_masks[env_idx]))
                    opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
                    self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
            actions = np.concatenate((actions, opponent_actions), axis=1)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    @torch.no_grad()
    def compute(self):
        """计算回报值。"""
        self.policy.prep_rollout()
        next_values = self.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def insert(self, data):
        """插入数据到缓冲区。

        Args:
            data: 包含观测、动作、奖励等的数据元组。
        """
        obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
        dones = dones.squeeze(axis=-1)
        dones_env = np.all(dones, axis=-1)
        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        if self.use_selfplay:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            self.opponent_masks = masks[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
            actions = actions[:, :self.num_agents // 2, ...]
            rewards = rewards[:, :self.num_agents // 2, ...]
            masks = masks[:, :self.num_agents // 2, ...]
            active_masks = active_masks[:, :self.num_agents // 2, ...]
        self.buffer.insert(obs, share_obs, actions, rewards, masks, action_log_probs, values,
                           rnn_states_actor, rnn_states_critic, active_masks=active_masks)

    def save(self, episode):
        """保存当前策略模型，包含回合编号，并支持自博弈模式。

        Args:
            episode (int): 当前回合编号。
        """
        torch.save(self.policy.actor.state_dict(), str(self.save_dir) + "/actor_latest.pt")
        torch.save(self.policy.critic.state_dict(), str(self.save_dir) + "/critic_latest.pt")
        torch.save(self.policy.actor.state_dict(), str(self.save_dir) + f"/actor_episode_{episode}.pt")
        torch.save(self.policy.critic.state_dict(), str(self.save_dir) + f"/critic_episode_{episode}.pt")
        if self.use_selfplay and not self.use_rule_opponent:
            policy_id = f"episode_{episode}"
            self.policy_pool[policy_id] = self.all_args.init_elo
            torch.save(self.policy.state_dict(), str(self.save_dir) + f"/policy_{policy_id}.pt")
            for idx, policy in enumerate(self.opponent_policy):
                available_policies = list(self.policy_pool.keys())
                selected_policy = np.random.choice(available_policies)
                policy.load_state_dict(torch.load(str(self.save_dir) + f"/policy_{selected_policy}.pt"))
        logging.info(f"Saved model: {self.save_dir}/actor_episode_{episode}.pt, {self.save_dir}/critic_episode_{episode}.pt")

    @torch.no_grad()
    def eval(self, total_num_steps):
        """评估当前策略。

        Args:
            total_num_steps: 当前总步数。
        """
        logging.info("Starting evaluation...")
        logging.info(f"n_eval_rollout_threads: {self.n_eval_rollout_threads}, num_agents: {self.num_agents}")
        total_episodes, eval_episode_rewards = 0, []
        # Initialize cumulative rewards for all agents
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_obs, eval_share_obs = self.eval_envs.reset()
        logging.info(f"eval_obs shape: {eval_obs.shape}, eval_share_obs shape: {eval_share_obs.shape}")
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, *self.buffer.rnn_states_actor.shape[3:]), dtype=np.float32)
        logging.info(f"eval_masks shape: {eval_masks.shape}, eval_rnn_states shape: {eval_rnn_states.shape}")

        if self.use_selfplay and not self.use_rule_opponent:
            eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in
                                     range(self.n_choose_opponents)]
            assert self.eval_episodes >= self.n_choose_opponents, \
                f"Evaluation episodes {self.eval_episodes} should be greater than opponent count {self.n_choose_opponents}"
            eval_each_episodes = self.eval_episodes // self.n_choose_opponents
            eval_cur_opponent_idx = 0
            logging.info(f"Selected evaluation opponents: {eval_choose_opponents}")

        while total_episodes < self.eval_episodes:
            if self.use_selfplay and not self.use_rule_opponent and total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
                policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
                self.eval_opponent_policy.actor.load_state_dict(
                    torch.load(str(self.save_dir) + f'/actor_{policy_idx}.pt', weights_only=True))
                self.eval_opponent_policy.prep_rollout()
                eval_cur_opponent_idx += 1
                logging.info(
                    f"Loaded opponent policy {policy_idx} for evaluation ({total_episodes + 1}/{self.eval_episodes})")
                eval_obs, eval_share_obs = self.eval_envs.reset()
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_rnn_states = np.zeros(
                    (self.n_eval_rollout_threads, self.num_agents, *self.buffer.rnn_states_actor.shape[3:]),
                    dtype=np.float32)
                eval_opponent_obs = eval_obs[:, self.num_agents // 2:, ...]
                eval_obs = eval_obs[:, :self.num_agents // 2, ...]
                eval_opponent_masks = np.ones((self.n_eval_rollout_threads, self.num_agents // 2, 1), dtype=np.float32)
                eval_opponent_rnn_states = np.zeros(
                    (self.n_eval_rollout_threads, self.num_agents // 2, *self.buffer.rnn_states_actor.shape[3:]),
                    dtype=np.float32)
                logging.info(
                    f"eval_obs shape after split: {eval_obs.shape}, eval_opponent_obs shape: {eval_opponent_obs.shape}")
                logging.info(
                    f"eval_masks shape: {eval_masks.shape}, eval_opponent_masks shape: {eval_opponent_masks.shape}")

            self.policy.prep_rollout()
            eval_actions, eval_rnn_states_ego = self.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states[:, :self.num_agents // 2, ...]),  # Only ego agents
                np.concatenate(eval_masks),  # All agents for GRU compatibility
                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states_ego = np.array(np.split(_t2n(eval_rnn_states_ego), self.n_eval_rollout_threads))

            if self.use_selfplay and not self.use_rule_opponent:
                eval_opponent_actions, eval_opponent_rnn_states = self.eval_opponent_policy.act(
                    np.concatenate(eval_opponent_obs),
                    np.concatenate(eval_opponent_rnn_states),
                    np.concatenate(eval_opponent_masks),
                    deterministic=True)
                eval_opponent_actions = np.array(np.split(_t2n(eval_opponent_actions), self.n_eval_rollout_threads))
                eval_opponent_rnn_states = np.array(
                    np.split(_t2n(eval_opponent_rnn_states), self.n_rollout_threads))
                eval_actions = np.concatenate((eval_actions, eval_opponent_actions), axis=1)
                # Update eval_rnn_states: ego agents
                eval_rnn_states[:, :self.num_agents // 2, ...] = eval_rnn_states_ego
            else:
                # Update eval_rnn_states: all agents
                eval_rnn_states = eval_rnn_states_ego

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            if self.use_selfplay:
                eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]

            eval_cumulative_rewards[:, :self.num_agents // 2, :] += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            if eval_dones_env.any():
                eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env])
                eval_cumulative_rewards[eval_dones_env] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            if eval_dones_env.any():
                eval_masks[eval_dones_env] = np.zeros((eval_dones_env.sum(), self.num_agents, 1), dtype=np.float32)
                eval_rnn_states[eval_dones_env, :self.num_agents // 2, ...] = np.zeros(
                    (eval_dones_env.sum(), self.num_agents // 2, *self.buffer.rnn_states_actor.shape[3:]),
                    dtype=np.float32)
                if self.use_selfplay and not self.use_rule_opponent:
                    eval_opponent_masks[eval_dones_env] = np.zeros(
                        (eval_dones_env.sum(), self.num_agents // 2, 1), dtype=np.float32)
                    eval_opponent_rnn_states[eval_dones_env] = np.zeros(
                        (eval_dones_env.sum(), self.num_agents // 2, *self.buffer.rnn_states_actor.shape[3:]),
                        dtype=np.float32)

            if self.use_selfplay and not self.use_rule_opponent:
                eval_opponent_obs = eval_obs[:, self.num_agents // 2:, ...]
                eval_obs = eval_obs[:, :self.num_agents // 2, ...]

        eval_infos = {}
        if eval_episode_rewards:
            eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean()
        else:
            eval_infos['eval_average_episode_rewards'] = 0.0
        logging.info(f"Evaluation average episode reward: {eval_infos['eval_average_episode_rewards']}")
        self.log_info(eval_infos, total_num_steps)

        if self.use_selfplay:
            self.reset_opponent()
        logging.info("Evaluation completed")

    def reset_opponent(self):
        """重置对手策略和缓冲区。"""
        choose_opponents = []
        for policy in self.opponent_policy:
            choose_idx = self.selfplay_algo.choose(self.policy_pool)
            choose_opponents.append(choose_idx)
            policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{choose_idx}.pt'))
            policy.prep_rollout()
        logging.info(f"Selected training opponents: {choose_opponents}")

        self.buffer.clear()
        self.opponent_obs = np.zeros_like(self.opponent_obs)
        self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
        self.opponent_masks = np.ones_like(self.opponent_masks)

        obs, share_obs = self.envs.reset()
        if self.all_args.n_choose_opponents > 0:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()