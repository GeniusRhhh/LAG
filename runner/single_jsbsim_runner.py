import time
import torch
import logging
import numpy as np

from algorithms.sac.sac_policy import SACPolicy
from algorithms.sac.sac_trainer import SACTrainer
from algorithms.sac.sac_replay_buffer import SACReplayBuffer

class SingleJSBSimRunner:
    """
    单智能体 + 连续动作 SAC Runner, 多并行环境 (n_env).
    兼容旧env的 “(n_env,1,obs_dim)/(n_env,1,act_dim)” 要求.
    """

    def __init__(self, config):
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.all_args = config["all_args"]
        self.device = config["device"]
        self.run_dir = config["run_dir"]

        self.num_env_steps = self.all_args.num_env_steps
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.batch_size = getattr(self.all_args, "batch_size", 128)
        self.update_per_step = getattr(self.all_args, "update_per_step", 1)

        self.use_eval = getattr(self.all_args, "use_eval", False)
        self.eval_episodes = getattr(self.all_args, "eval_episodes", 5)
        self.log_interval = getattr(self.all_args, "log_interval", 1000)
        self.eval_interval = getattr(self.all_args, "eval_interval", 5000)
        self.save_interval = getattr(self.all_args, "save_interval", 10000)

        # 获取env.observation_space/action_space
        obs_space = self.envs.observation_space
        act_space = self.envs.action_space

        logging.info(f"[SingleJSBSimRunner] obs_space={obs_space}, act_space={act_space}")

        # 构建SAC
        self.policy = SACPolicy(self.all_args, obs_space, act_space)
        self.trainer = SACTrainer(self.policy)

        # ReplayBuffer
        buffer_capacity = getattr(self.all_args, "buffer_size", 10**5)
        self.buffer = SACReplayBuffer(
            obs_space=obs_space,
            act_space=act_space,
            n_env=self.n_rollout_threads,
            capacity=buffer_capacity
        )

        self.total_env_steps = 0

    def run(self):
        # 1) reset
        obs = self.envs.reset()
        # 如果env返回 (n_env,1,obs_dim) => squeeze一下
        obs = self._fix_obs_shape(obs)    # shape=(n_env, obs_dim)
        obs = np.array(obs, dtype=np.float32)

        start_time = time.time()
        while self.total_env_steps < self.num_env_steps:
            # --- 先把 obs 再做一次确认 ---
            obs = self._fix_obs_shape(obs)  # 确保 (n_env, obs_dim)

            # 2) 策略 => actions (n_env, act_dim), 用于存入 ReplayBuffer
            actions, _ = self.policy.get_action(obs, deterministic=False)
            # => shape=(n_env, act_dim)

            # 3) 构造给env的动作 => (n_env,1,act_dim)
            actions_for_env = actions[:, np.newaxis, :]

            # 4) 与环境交互
            next_obs, rewards, dones, infos = self.envs.step(actions_for_env)

            # 修正 next_obs shape => (n_env, obs_dim)
            next_obs = self._fix_obs_shape(next_obs)

            # 转成 float32
            next_obs = np.array(next_obs, dtype=np.float32)
            rewards  = np.array(rewards)
            dones    = np.array(dones)

            # 可能出现 (n_env,1,1) => 要先 squeeze -> (n_env,1)
            # 可能出现 (n_env,) => reshape -> (n_env,1)
            rewards = self._fix_rew_done_shape(rewards)
            dones   = self._fix_rew_done_shape(dones)

            # 5) 存入 ReplayBuffer => (n_env, obs_dim)/(n_env, act_dim)/(n_env,1)/(n_env, obs_dim)/(n_env,1)
            self.buffer.store(obs, actions, rewards, next_obs, dones)

            # 6) 如果 done => reset_single
            # for i in range(self.n_rollout_threads):
            #     if dones[i][0]:
            #         obs_i = self.envs.reset_single(i)
            #         # 可能 shape=(1, obs_dim) => squeeze(0)
            #         if isinstance(obs_i, np.ndarray) and obs_i.ndim==2 and obs_i.shape[0]==1:
            #             obs_i = obs_i.squeeze(0)
            #         next_obs[i] = obs_i

            obs = next_obs
            self.total_env_steps += self.n_rollout_threads

            # --- 训练SAC ---
            if len(self.buffer) > self.batch_size:
                for _ in range(self.update_per_step):
                    batch = self.buffer.sample_batch(self.batch_size)
                    self.trainer.update(
                        batch["obs"], batch["act"], batch["rew"],
                        batch["next_obs"], batch["done"]
                    )

            # --- 日志 ---
            if self.total_env_steps % self.log_interval == 0:
                cost_time = time.time() - start_time
                fps = int(self.total_env_steps/(cost_time+1e-6))
                logging.info(f"[Runner] Steps={self.total_env_steps}/{self.num_env_steps}, FPS={fps}, BufferSize={len(self.buffer)}")

            # --- eval ---
            if self.use_eval and self.total_env_steps>0 and (self.total_env_steps%self.eval_interval==0):
                self.eval()

            # --- save ---
            if self.total_env_steps>0 and (self.total_env_steps%self.save_interval==0):
                self.save()

    def _fix_obs_shape(self, obs):
        """
        将 obs 无论是 (n_env,obs_dim) or (n_env,1,obs_dim) => 统一成 (n_env, obs_dim).
        """
        if isinstance(obs, np.ndarray):
            if obs.ndim == 3 and obs.shape[1] == 1:
                # => shape=(n_env,1,obs_dim) => squeeze axis=1 => (n_env,obs_dim)
                obs = obs.squeeze(1)
        return obs

    def _fix_rew_done_shape(self, arr):
        """
        将 rewards / dones 不管是 (n_env,), (n_env,1), (n_env,1,1) => 统一成 (n_env,1).
        """
        if arr.ndim >= 2:
            # 如果 (n_env,1,1) => squeeze到 (n_env,1)
            while arr.ndim > 2:
                arr = np.squeeze(arr, axis=-1)
            # 可能是 (n_env,1) =>OK
            if arr.shape[-1] != 1:
                # 如果出现 (n_env, x)  x>1 => 视情况
                raise ValueError(f"Got shape={arr.shape}, expected final dim=1.")
        else:
            # (n_env,) => reshape => (n_env,1)
            arr = arr.reshape(-1,1)
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
                # 给env => 需要 (n_env,1,act_dim)?
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
