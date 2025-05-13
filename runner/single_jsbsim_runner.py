import time
import torch
import logging
import numpy as np

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

        self.max_episodes = getattr(self.all_args, "max_episodes", 1000)  # 动态轮次，默认 1000
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

        logging.info(f"[SingleJSBSimRunner] obs_space={obs_space}, act_space={act_space}")
        logging.info(f"Action space: low={act_space.low.tolist()}, high={act_space.high.tolist()}")

        self.policy = SACPolicy(self.all_args, obs_space, act_space)
        self.trainer = SACTrainer(self.policy)

        buffer_capacity = getattr(self.all_args, "buffer_size", 10000)
        self.buffer = SACReplayBuffer(
            obs_space=obs_space,
            act_space=act_space,
            n_env=self.n_rollout_threads,
            capacity=buffer_capacity
        )

        self.total_env_steps = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.step_count = 0
        self.total_rewards = 0

    def run(self):
        obs = self.envs.reset()
        obs = self._fix_obs_shape(obs)
        obs = np.array(obs, dtype=np.float32)

        start_time = time.time()
        while self.episode_count < self.max_episodes:  # 动态轮次
            obs = self._fix_obs_shape(obs)
            self.step_count += 1

            # Get actions
            actions, _ = self.policy.get_action(obs, deterministic=False)
            # 将 actions 从 GPU 移到 CPU 并转换为 NumPy 数组
            actions = actions.cpu().detach().numpy()
            # Clip actions
            actions = np.clip(actions, self.envs.action_space.low, self.envs.action_space.high)
            # Check for out-of-bounds actions (avoid broadcast error)
            actions_out_of_bounds = (
                np.any(actions[:, :3] < -1.0) or
                np.any(actions[:, :3] > 1.0) or
                np.any(actions[:, 3] < 0.4) or
                np.any(actions[:, 3] > 0.9)
            )
            if actions_out_of_bounds:
                logging.warning(
                    f"Step {self.total_env_steps}: Actions out of bounds: {actions.tolist()}"
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

            # Store in buffer
            self.buffer.store(obs, actions, rewards, next_obs, dones)
            self.total_rewards += rewards.sum()

            # Log state, action, reward every 1000 steps for first environment
            if self.step_count % 1000 == 0:
                i = 0
                delta_altitude_m = obs[i, 0] * 2000
                altitude_m = obs[i, 3] * 10000
                delta_heading_deg = obs[i, 1] * 180
                velocity_u_mh = obs[i, 2] * 340
                logging.info(
                    f"Step {self.total_env_steps} (Env 0):\n"
                    f"  Obs - delta_altitude={delta_altitude_m:.2f}m, altitude={altitude_m:.2f}m, "
                    f"delta_heading={delta_heading_deg:.2f}°, velocity_u={velocity_u_mh:.2f}m/s\n"
                    f"  Action - aileron={actions[i, 0]:.4f}, elevator={actions[i, 1]:.4f}, "
                    f"rudder={actions[i, 2]:.4f}, throttle={actions[i, 3]:.4f}\n"
                    f"  Reward - value={rewards[i, 0]:.4f}"
                )
                if isinstance(infos[i], dict) and 'reward_items' in infos[i]:
                    reward_items = infos[i]['reward_items']
                    logging.info(
                        f"  Reward Breakdown - "
                        f"HeadingReward={reward_items.get('HeadingReward', 0):.4f}, "
                        f"AltitudeReward={reward_items.get('AltitudeReward', 0):.4f}"
                    )

            # Handle done signals
            if np.any(dones):
                next_obs = self.envs.reset()
                next_obs = self._fix_obs_shape(next_obs)
                next_obs = np.array(next_obs, dtype=np.float32)
                self.episode_count += 1
                avg_reward = self.total_rewards / self.step_count if self.step_count > 0 else 0
                self.episode_rewards.append(avg_reward)
                logging.info(
                    f"Episode {self.episode_count} terminated at step {self.total_env_steps}, "
                    f"Dones: {dones.tolist()}, Infos: {infos}"
                )
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
                    if self.step_count % 1000 == 0:
                        logging.info(
                            f"  Train Metrics - critic_loss={train_metrics['critic_loss']:.4f}, "
                            f"actor_loss={train_metrics['actor_loss']:.4f}, "
                            f"q_mean={train_metrics['q_mean']:.4f}, alpha={train_metrics['alpha']:.4f}"
                        )

            # Log summary
            if self.total_env_steps % self.log_interval == 0 and self.total_env_steps > 0:
                cost_time = time.time() - start_time
                fps = int(self.total_env_steps / (cost_time + 1e-6))
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                logging.info(
                    f"\nScenario 1/heading Algo sac Exp v1 updates {self.episode_count}/{self.max_episodes} episodes, "
                    f"total num timesteps {self.total_env_steps}, FPS {fps}\n"
                    f"average episode rewards is {avg_reward:.4f}\n"
                    f"average heading turns is 1.0"
                )

            # Eval
            if self.use_eval and self.total_env_steps > 0 and (self.total_env_steps % self.eval_interval == 0):
                self.eval()

            # Save
            if self.total_env_steps > 0 and (self.total_env_steps % self.save_interval == 0):
                self.save()

        logging.info(f"训练完成，共 {self.episode_count} 回合，{self.total_env_steps} 步")

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

    @torch.no_grad()
    def eval(self):
        logging.info("[Eval] start evaluation")
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
                act = act.cpu().detach().numpy()  # 同样在 eval 中转换
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
        logging.info(f"[Eval] average return={np.mean(returns)}")

    def save(self):
        path = f"{self.run_dir}/sac_{self.total_env_steps}.pt"
        self.policy.save(path)
        logging.info(f"[Runner] saved => {path}")

    def restore(self, load_path):
        self.policy.load(load_path)
        logging.info(f"[Runner] loaded => {load_path}")