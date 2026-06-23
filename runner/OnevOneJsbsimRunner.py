import time
import torch
import logging
import numpy as np
import os
from typing import List
from .base_runner import Runner
from algorithms.sac.sac_policy import SACPolicy
from algorithms.sac.sac_replay_buffer import SACReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()

class SACJSBSimRunner(Runner):
    def load(self):
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.render_mode = None
        # self.n_rollout_threads = getattr(self.all_args, 'n_rollout_threads', 4)
        logging.info(f"加载环境：n_rollout_threads={self.n_rollout_threads}, num_agents={self.num_agents}, "
                     f"obs_space={self.obs_space}, act_space={self.act_space}")
        from algorithms.sac.sac_policy import SACPolicy as Policy
        from algorithms.sac.sac_trainer import SACTrainer as Trainer
        self.policy = Policy(self.all_args, self.obs_space, self.act_space)
        self.trainer = Trainer(self.all_args)
        self.buffer = SACReplayBuffer(self.obs_space, self.act_space, self.n_rollout_threads,
                                      capacity=self.all_args.buffer_size)

        if self.model_dir is not None:
            logging.info(f"模型目录设置为 {self.model_dir}")
            self.save_dir = self.model_dir
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def run(self):
        if self.model_dir is not None and os.path.exists(self.model_dir):
            episodes = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(self.model_dir)
                        if f.startswith('actor_') and f.endswith('.pt') and f != 'actor_latest.pt']
            if episodes:
                latest_episode = max(episodes)
                self.restore(latest_episode)
                logging.info(f"Restored from episode {latest_episode}, start_episode={self.start_episode}, "
                             f"total_num_steps={self.total_num_steps}, save_dir set to {self.save_dir}")
            else:
                logging.info(f"No episode checkpoints found in {self.model_dir}, starting from scratch")
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

        logging.info("Starting warmup...")
        start_time = time.time()
        self.warmup()
        logging.info(f"Warmup completed in {time.time() - start_time:.2f}s")

        start = time.time()
        buffer_size = self.all_args.buffer_size
        episodes = self.num_env_steps // buffer_size // self.n_rollout_threads
        logging.info(
            f"Starting training loop: episodes={episodes}, buffer_size={buffer_size}, batch_size={self.all_args.batch_size}")
        for episode in range(self.start_episode, episodes):
            logging.info(f"Episode {episode}/{episodes} started at {time.strftime('%H:%M:%S')}")
            heading_turns_list = []
            for step in range(buffer_size):
                if step % 500 == 0:
                    logging.info(
                        f"Step {step}/{buffer_size} in episode {episode} started at {time.strftime('%H:%M:%S')}")
                start_step = time.time()
                actions = self.collect(step)
                if step % 500 == 0:
                    logging.info(
                        f"Step {step}: Actions collected in {time.time() - start_step:.2f}s, shape={actions.shape}")

                start_env_step = time.time()
                obs, rewards, dones, infos = self.envs.step(actions)
                if step % 500 == 0:
                    logging.info(
                        f"Step {step}: Environment stepped in {time.time() - start_env_step:.2f}s, obs_shape={obs.shape}")

                for info in infos:
                    if 'heading_turn_counts' in info:
                        heading_turns_list.append(info['heading_turn_counts'])

                start_insert = time.time()
                self.insert(step, obs, actions, rewards, dones)
                if step % 500 == 0:
                    logging.info(f"Step {step}: Data inserted in {time.time() - start_insert:.2f}s")

                # 提前触发训练
                if step == 0:
                    logging.info(
                        f"Early training test at step 0, total_num_steps={self.total_num_steps}, buffer_size={self.buffer.size}")
                    if self.buffer.size >= self.all_args.batch_size:
                        batch = self.buffer.sample_batch(self.all_args.batch_size)
                        train_infos = self.trainer.update(
                            batch["obs"], batch["act"], batch["rew"], batch["next_obs"], batch["done"],
                            self.total_num_steps
                        )
                        logging.info(f"Early training completed: {train_infos}")
                    else:
                        logging.info("Buffer size too small for early training, skipping...")

            train_infos = {}
            warmup_steps = getattr(self.all_args, 'warmup_steps', 1000)
            if self.total_num_steps > warmup_steps:
                logging.info(
                    f"Training policy at total_num_steps={self.total_num_steps}, buffer_size={self.buffer.size}")
                if self.buffer.size >= self.all_args.batch_size:
                    batch = self.buffer.sample_batch(self.all_args.batch_size)
                    train_infos = self.trainer.update(
                        batch["obs"], batch["act"], batch["rew"], batch["next_obs"], batch["done"], self.total_num_steps
                    )
                    logging.info(f"Training completed: {train_infos}")
                else:
                    logging.info("Buffer size too small for training, skipping...")

            self.total_num_steps = (episode + 1) * buffer_size * self.n_rollout_threads

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
                if train_infos:
                    train_infos["average_episode_rewards"] = np.mean([self.buffer.rew_buf[:self.buffer.size].sum(
                        axis=(1, 2)) / (self.buffer.done_buf[:self.buffer.size] == False).sum()])
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
        logging.info("Inside warmup: resetting environment...")
        start_time = time.time()
        obs = self.envs.reset()
        expected_shape = (self.n_rollout_threads, self.num_agents, 15)
        if obs.shape != expected_shape:
            logging.error(f"Reset obs shape mismatch: expected {expected_shape}, got {obs.shape}")
            raise ValueError(f"Reset obs shape mismatch: expected {expected_shape}, got {obs.shape}")
        obs = obs.squeeze(1)
        logging.info(f"Environment reset in {time.time() - start_time:.2f}s: obs_shape={obs.shape}")
        self.buffer.store(obs, np.zeros((self.n_rollout_threads, self.act_space.shape[0])),
                          np.zeros((self.n_rollout_threads, 1)), obs, np.zeros((self.n_rollout_threads, 1)))
        logging.info(f"Warmup buffer stored in {time.time() - start_time:.2f}s")

    def collect(self, step):
        self.policy.prep_rollout()
        if step % 500 == 0:  # 添加频率控制
            logging.info(f"Collecting actions for step {step} at {time.strftime('%H:%M:%S')}")
        start_time = time.time()
        obs = self.envs.reset() if step == 0 else \
              self.envs.step(np.zeros((self.n_rollout_threads, self.num_agents, self.act_space.shape[0])))[0]
        expected_shape = (self.n_rollout_threads, self.num_agents, 15)
        if obs.shape != expected_shape:
            logging.error(f"Collect obs shape mismatch: expected {expected_shape}, got {obs.shape}")
            raise ValueError(f"Collect obs shape mismatch: expected {expected_shape}, got {obs.shape}")
        obs = obs.squeeze(1)
        if step % 500 == 0:
            logging.info(f"Observations collected in {time.time() - start_time:.2f}s: shape={obs.shape}")

        start_policy = time.time()
        ego_actions, _ = self.policy.get_action(obs)
        if step % 500 == 0:
            logging.info(f"Policy action generated in {time.time() - start_policy:.2f}s: shape={ego_actions.shape}")

        actions = np.zeros((self.n_rollout_threads, self.num_agents, self.act_space.shape[0]), dtype=np.float32)
        actions[:, 0, :] = ego_actions
        if step % 500 == 0:
            logging.info(f"Actions generated in {time.time() - start_time:.2f}s: shape={actions.shape}")

        return actions

    def insert(self, step, obs, actions, rewards, dones):
        if step % 500 == 0:  # 添加频率控制
            logging.info(f"Inserting data for step {step} at {time.strftime('%H:%M:%S')}")
        if obs.shape[1] == 1:
            obs = obs.squeeze(1)
        if actions.shape[1] == 1:
            actions = actions.squeeze(1)
        if rewards.shape[1] == 1:
            rewards = rewards.squeeze(1)
        if dones.shape[1] == 1:
            dones = dones.squeeze(1)

        next_obs = obs
        self.buffer.store(obs, actions, rewards, next_obs, dones)
        if step % 500 == 0:
            logging.info(f"Data stored in buffer for step {step}")

    # 以下方法保持不变
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, 1), dtype=np.float32)

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, 1), dtype=np.float32)

        while total_episodes < self.eval_episodes:
            actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.act_space.shape[0]),
                               dtype=np.float32)
            ego_obs = eval_obs.squeeze(1)
            ego_actions, _ = self.policy.get_action(ego_obs, deterministic=True)
            actions[:, 0, :] = ego_actions

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(actions)
            eval_rewards = eval_rewards.squeeze(1)
            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(), axis=-1)
            total_episodes += np.sum(eval_dones_env)

            if np.any(eval_dones_env):
                rewards_to_add = eval_cumulative_rewards[eval_dones_env].copy()
                eval_episode_rewards.append(rewards_to_add)
                eval_cumulative_rewards[eval_dones_env] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env] = np.zeros(((eval_dones_env).sum(), 1), dtype=np.float32)

        eval_infos = {}
        if len(eval_episode_rewards) == 0:
            logging.warning("No evaluation episodes completed, using default reward 0.0")
            mean_reward = 0.0
        else:
            flat_rewards = np.concatenate(eval_episode_rewards).flatten()
            mean_reward = float(np.mean(flat_rewards)) if len(flat_rewards) > 0 else 0.0
        eval_infos['eval_average_episode_rewards'] = mean_reward
        logging.info(" eval average episode rewards: " + str(mean_reward))
        self.log_info(eval_infos, total_num_steps)

    def save(self, episode):
        self.policy.save(str(self.save_dir) + '/policy_latest.pt')
        self.policy.save(str(self.save_dir) + f'/policy_{episode}.pt')
        torch.save(self.buffer, str(self.save_dir) + '/buffer_latest.pt')
        training_state = {'episode': episode, 'total_num_steps': self.total_num_steps}
        torch.save(training_state, str(self.save_dir) + '/training_state_latest.pt')

    def restore(self, episode=None):
        if episode is None:
            policy_path = str(self.model_dir) + '/policy_latest.pt'
            buffer_path = str(self.model_dir) + '/buffer_latest.pt'
            training_state_path = str(self.model_dir) + '/training_state_latest.pt'
        else:
            policy_path = str(self.model_dir) + f'/policy_{episode}.pt'
            buffer_path = str(self.model_dir) + '/buffer_latest.pt'
            training_state_path = str(self.model_dir) + '/training_state_latest.pt'

        if os.path.exists(policy_path):
            self.policy.load_state_dict(torch.load(policy_path, weights_only=True))
            logging.info(f"Loaded policy from {policy_path}")
        if os.path.exists(buffer_path):
            self.buffer = torch.load(buffer_path, weights_only=False)
            logging.info(f"Loaded buffer from {buffer_path}")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)
            self.start_episode = training_state['episode'] + 1
            self.total_num_steps = training_state['total_num_steps']
            logging.info(
                f"Loaded training state: episode={training_state['episode']}, total_num_steps={self.total_num_steps}")