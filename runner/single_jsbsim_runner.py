import time
import torch
import logging
import numpy as np
import wandb

from algorithms.sac.sac_policy import SACPolicy
from algorithms.sac.sac_trainer import SACTrainer
from algorithms.sac.sac_replay_buffer import SACReplayBuffer

class SingleJSBSimRunner:
    def __init__(self, config):
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.all_args = config["all_args"]
        self.device = config["device"]
        self.run_dir = config["run_dir"]

        self.num_env_steps = self.all_args.num_env_steps
        # 移除 self.max_episodes，改为估算 episodes 作为参考
        self.estimated_episodes = self.num_env_steps // (self.all_args.batch_size * self.all_args.update_per_step * self.all_args.n_rollout_threads)
        logging.info(f"[Runner] Estimated episodes: {self.estimated_episodes} based on num_env_steps={self.num_env_steps}, batch_size={self.all_args.batch_size}, update_per_step={self.all_args.update_per_step}, n_rollout_threads={self.all_args.n_rollout_threads}")
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.batch_size = getattr(self.all_args, "batch_size", 256)
        self.update_per_step = getattr(self.all_args, "update_per_step", 1)

        self.use_eval = getattr(self.all_args, "use_eval", False)
        self.eval_episodes = getattr(self.all_args, "eval_episodes", 5)
        self.log_interval = getattr(self.all_args, "log_interval", 1000)
        self.eval_interval = getattr(self.all_args, "eval_interval", 5000)
        self.save_interval = getattr(self.all_args, "save_interval", 10000)

        obs_space = self.envs.observation_space
        act_space = self.envs.action_space

        logging.info(f"[Runner] obs_space={obs_space}, act_space={act_space}")
        logging.info(f"Action space: low={act_space.low.tolist()}, high={act_space.high.tolist()}")

        self.policy = SACPolicy(self.all_args, obs_space, act_space)
        self.trainer = SACTrainer(self.policy)

        buffer_capacity = self.all_args.buffer_size
        self.buffer = SACReplayBuffer(
            obs_space=obs_space,
            act_space=act_space,
            n_env=self.n_rollout_threads,
            capacity=buffer_capacity
        )

        self.total_env_steps = 0
        self.episode_count = 0  # 仅用于记录实际 episode 数量
        self.episode_rewards = []
        self.heading_turn_counts = []
        self.step_count = 0
        self.total_rewards = 0

    def run(self):
        obs = self.envs.reset()
        obs = self._fix_obs_shape(obs)
        obs = np.array(obs, dtype=np.float32)

        start_time = time.time()
        while self.total_env_steps < self.num_env_steps:
            obs = self._fix_obs_shape(obs)
            self.step_count += 1

            # Get actions
            actions, _ = self.policy.get_action(obs, deterministic=False)
            actions = actions.cpu().detach().numpy()
            # Clip actions
            actions = np.clip(actions, self.envs.action_space.low, self.envs.action_space.high)
            # Encourage higher throttle
            actions[:, 3] = np.clip(actions[:, 3] + 0.4, 0.4, 0.9)
            # Constrain elevator to reduce negative values
            actions[:, 1] = np.clip(actions[:, 1], -0.5, 0.5)
            # Extra check for throttle
            throttle = actions[:, 3]
            throttle_clipped = np.clip(throttle, 0.4, 1.0)
            if not np.allclose(throttle, throttle_clipped):
                actions[:, 3] = throttle_clipped
                logging.warning(
                    f"Step {self.total_env_steps} Throttle clipped: original={throttle.tolist()}, "
                    f"clipped={throttle_clipped.tolist()}"
                )

            actions_for_env = actions[:, np.newaxis, :]

            # Step environment
            next_obs, rewards, dones, infos = self.envs.step(actions_for_env)
            next_obs = self._fix_obs_shape(next_obs)
            next_obs = np.array(next_obs, dtype=np.float32)
            rewards = np.array(rewards)
            dones = np.array(dones)
            rewards = self._fix_rew_done_shape(rewards)
            dones = self._fix_rew_done_shape(dones)

            # 调试日志
            if self.step_count % 100 == 0:
                logging.info(f"Step {self.total_env_steps}: "
                             f"Rewards={rewards[0, 0]:.4f}, Dones={dones[0, 0]}, Infos={infos}")

            # Store in buffer
            self.buffer.store(obs, actions, rewards, next_obs, dones)
            self.total_rewards += rewards.sum()

            # Detailed log
            if self.step_count % 500 == 0:
                i = 0
                logging.info(
                    f"Step {self.total_env_steps} (Env 0): "
                    f"Reward={rewards[i, 0]:.4f}, Throttle={actions[i, 3]:.4f}, "
                    f"Elevator={actions[i, 1]:.4f}"
                )

            # Handle done signals
            if np.any(dones):
                termination_reasons = []
                for i, done in enumerate(dones.flatten()):
                    if done:
                        info = infos[i]
                        reason = info.get('termination_reason', 'Unknown')
                        termination_reasons.append(f"Env {i}: {reason}")
                        heading_turns = info.get('heading_turn_counts', 0)
                        self.heading_turn_counts.append(heading_turns)
                        logging.debug(f"Env {i} at Step {self.total_env_steps}: heading_turn_counts={heading_turns}")

                # 重置环境
                next_obs = self.envs.reset()
                next_obs = self._fix_obs_shape(next_obs)
                next_obs = np.array(next_obs, dtype=np.float32)
                self.episode_count += 1
                avg_reward = self.total_rewards / self.step_count if self.step_count > 0 else 0
                self.episode_rewards.append(avg_reward)
                avg_turns = np.mean(self.heading_turn_counts[-10:]) if self.heading_turn_counts else 0
                logging.info(
                    f"Episode {self.episode_count}: "
                    f"Avg reward={avg_reward:.4f}, Avg heading turns={avg_turns:.2f}, "
                    f"Reasons={termination_reasons}"
                )
                if self.all_args.use_wandb:
                    wandb.log({
                        "episode": self.episode_count,
                        "avg_episode_reward": avg_reward,
                        "avg_heading_turns": avg_turns
                    })
                self.total_rewards = 0
                self.step_count = 0

            obs = next_obs
            self.total_env_steps += self.n_rollout_threads

            # Train SAC
            if len(self.buffer) > self.batch_size:
                for _ in range(self.update_per_step):
                    batch = self.buffer.sample_batch(self.batch_size)
                    train_metrics = self.trainer.update(
                        batch["obs"], batch["act"], batch["rew"],
                        batch["next_obs"], batch["done"],
                        total_steps=self.total_env_steps
                    )
                    if self.step_count % 500 == 0:
                        logging.info(
                            f"Step {self.total_env_steps} Train: "
                            f"critic_loss={train_metrics['critic_loss']:.4f}, "
                            f"actor_loss={train_metrics['actor_loss']:.4f}, "
                            f"alpha={train_metrics['alpha']:.4f}"
                        )

            # Log summary
            if self.total_env_steps % self.log_interval == 0 and self.total_env_steps > 0:
                cost_time = time.time() - start_time
                fps = int(self.total_env_steps / (cost_time + 1e-6))
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                avg_turns = np.mean(self.heading_turn_counts[-10:]) if self.heading_turn_counts else 0
                logging.info(
                    f"Scenario 1/heading Algo sac Exp v0131 updates {self.episode_count}/{self.estimated_episodes} episodes, "
                    f"total num timesteps {self.total_env_steps}/{int(self.num_env_steps)}, FPS {fps}, "
                    f"average episode rewards is {avg_reward:.4f}, "
                    f"average heading turns is {avg_turns:.2f}"
                )

            if self.use_eval and self.total_env_steps > 0 and (self.total_env_steps % self.eval_interval == 0):
                self.eval()
            if self.total_env_steps > 0 and (self.total_env_steps % self.save_interval == 0):
                self.save()

        logging.info(f"Training done: {self.episode_count} episodes, {self.total_env_steps} steps")

    def _fix_obs_shape(self, obs):
        if isinstance(obs, np.ndarray):
            if obs.ndim == 3 and obs.shape[1] == 1:
                obs = obs.squeeze(1)
        return obs

    def _fix_rew_done_shape(self, arr):
        if arr.ndim >= 2:
            while arr.ndim > 2:
                arr = np.squeeze(arr, axis=-1)
            if arr.shape[-1] != 1:
                raise ValueError(f"Got shape={arr.shape}, expected final dim=1.")
        else:
            arr = arr.reshape(-1, 1)
        return arr

    def eval(self):
        logging.info("[Eval] Starting evaluation")
        if self.eval_envs is None:
            return
        returns = []
        for ep in range(self.eval_episodes):
            obs = self.eval_envs.reset()
            obs = self._fix_obs_shape(obs)
            obs = np.array(obs, dtype=np.float32)
            ep_ret = 0
            while True:
                act, _ = self.policy.get_action(obs, deterministic=True)
                act = act.cpu().detach().numpy()
                act = np.clip(act, self.envs.action_space.low, self.envs.action_space.high)
                act_env = act[:, np.newaxis, :]
                next_obs, rewards, dones, infos = self.eval_envs.step(act_env)
                next_obs = self._fix_obs_shape(next_obs)
                next_obs = np.array(next_obs, dtype=np.float32)
                ep_ret += np.array(rewards).sum()
                obs = next_obs
                if np.array(dones).all():
                    break
            returns.append(ep_ret)
        avg_return = np.mean(returns)
        logging.info(f"[Eval] Avg return={avg_return:.4f}")
        if self.all_args.use_wandb:
            wandb.log({"eval_avg_return": avg_return, "step": self.total_env_steps})

    def save(self):
        path = f"{self.run_dir}/sac_{self.total_env_steps}.pt"
        self.policy.save(path)
        logging.info(f"[Runner] Saved => {path}")

    def restore(self, load_path):
        self.policy.load(load_path)
        logging.info(f"[Runner] Loaded => {load_path}")