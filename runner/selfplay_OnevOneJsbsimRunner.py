import time
import copy
import logging
import numpy as np
import torch
from typing import Dict
from gymnasium.spaces import Box

from algorithms.sac.sac_policy import SACPolicy
from algorithms.sac.sac_trainer import SACTrainer
from algorithms.sac.sac_replay_buffer import SACReplayBuffer

class selfplayOnevOneJSBSimRunner:
    """支持自对弈机制的1v1对抗训练Runner"""

    def __init__(self, config: Dict):
        # 环境配置
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.all_args = config["all_args"]
        self.device = config["device"]
        self.run_dir = config["run_dir"]

        # 训练参数
        self.num_env_steps = self.all_args.num_env_steps
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.batch_size = getattr(self.all_args, "batch_size", 256)
        self.update_per_step = getattr(self.all_args, "update_per_step", 1)

        # 评估参数
        self.use_eval = getattr(self.all_args, "use_eval", False)
        self.eval_episodes = getattr(self.all_args, "eval_episodes", 5)
        self.log_interval = getattr(self.all_args, "log_interval", 5000)
        self.eval_interval = getattr(self.all_args, "eval_interval", 10000)
        self.save_interval = getattr(self.all_args, "save_interval", 20000)

        # 自对弈参数
        self.policy_pool_size = 5  # 策略池容量
        self.policy_pool: Dict[int, Dict] = {}  # {episode: {"policy": state_dict, "elo": float}}
        self.current_elo = 1200.0  # 当前策略ELO分数
        self.opponent_update_interval = 10000  # 对手更新间隔

        # 环境空间处理
        self._process_spaces()

        # 初始化主策略
        self.agent = SACPolicy(self.all_args, self.obs_space, self.act_space)
        self.trainer = SACTrainer(self.agent)

        # 共享经验回放
        self.buffer = SACReplayBuffer(
            obs_space=self.obs_space,
            act_space=self.act_space,
            n_env=self.n_rollout_threads,
            capacity=int(1e6)
        )

        # 训练状态
        self.total_env_steps = 0
        self.current_episode = 0
        self.start_time = time.time()

    def _process_spaces(self):
        """处理多智能体观测/动作空间"""
        multi_obs_space = self.envs.observation_space
        multi_act_space = self.envs.action_space

        # 提取单个智能体空间
        self.obs_space = self._extract_single_space(multi_obs_space)
        self.act_space = self._extract_single_space(multi_act_space)

        logging.info(f"Processed obs space: {self.obs_space}")
        logging.info(f"Processed act space: {self.act_space}")

    def _extract_single_space(self, multi_space: Box, idx: int = 0) -> Box:
        """从多智能体空间提取单智能体空间"""
        return Box(
            low=multi_space.low[idx],
            high=multi_space.high[idx],
            shape=(multi_space.shape[-1],),
            dtype=multi_space.dtype
        )

    def run(self):
        """主训练循环"""
        obs = self._env_reset()
        while self.total_env_steps < self.num_env_steps:
            # 生成对抗动作
            agent_act, opponent_act = self._generate_actions(obs)

            # 环境交互
            next_obs, rewards, dones = self._env_step(agent_act, opponent_act)

            # 存储经验
            self._store_experience(
                curr_obs=obs[:, 0],  # 选择第一个智能体的观测
                actions=agent_act,
                rewards=rewards[:, 0],  # 选择第一个智能体的奖励
                next_obs=next_obs[:, 0],  # 选择第一个智能体的下一个观测
                dones=dones[:, 0]  # 选择第一个智能体的done标志
            )

            # 策略更新
            if len(self.buffer) > self.batch_size:
                self._update_agent()

            # 对手管理
            if self.total_env_steps % self.opponent_update_interval == 0:
                self._update_opponent_pool()
                self._adjust_elo(rewards[:, 0].mean(), rewards[:, 1].mean())

            # 评估与保存
            if self.total_env_steps % self.eval_interval == 0:
                self.evaluate()
            if self.total_env_steps % self.save_interval == 0:
                self.save()

            obs = next_obs
            self.total_env_steps += self.n_rollout_threads

    def _generate_actions(self, obs: np.ndarray) -> tuple:
        """生成对抗动作"""
        # 主策略动作
        agent_act, _ = self.agent.get_action(obs[:, 0], deterministic=False)

        # 对手策略动作
        opponent = self._select_opponent()
        opponent_act, _ = opponent.get_action(obs[:, 1], deterministic=False)

        return agent_act, opponent_act

    def _select_opponent(self):
        """根据ELO选择对手策略"""
        if not self.policy_pool:
            return self._create_random_opponent()

        # 按ELO加权选择
        episodes, elos = zip(*[(k, v["elo"]) for k, v in self.policy_pool.items()])
        probs = torch.softmax(torch.tensor(elos) / 100, dim=0).numpy()
        chosen_ep = np.random.choice(episodes, p=probs)

        opponent = SACPolicy(self.all_args, self.obs_space, self.act_space)
        opponent.load_state_dict(self.policy_pool[chosen_ep]["policy"])
        return opponent

    def _create_random_opponent(self):
        """创建随机初始化对手"""
        opponent = SACPolicy(self.all_args, self.obs_space, self.act_space)
        for p in opponent.parameters():
            p.data.normal_(0, 0.1)
        return opponent

    def _update_agent(self):
        """更新主策略"""
        for _ in range(self.update_per_step):
            batch = self.buffer.sample_batch(self.batch_size)
            losses = self.trainer.update(
                obs_batch=batch["obs"],  # 对应 update 方法中的 obs_batch
                act_batch=batch["act"],  # 对应 update 方法中的 act_batch
                rew_batch=batch["rew"],  # 对应 update 方法中的 rew_batch
                next_obs_batch=batch["next_obs"],  # 对应 update 方法中的 next_obs_batch
                done_batch=batch["done"]  # 对应 update 方法中的 done_batch
            )

            # 定期记录训练指标
            if self.total_env_steps % self.log_interval == 0:
                logging.info(
                    f"Step {self.total_env_steps} | "
                    f"Critic Loss: {losses['critic_loss']:.3f} | "
                    f"Actor Loss: {losses['actor_loss']:.3f} | "
                    f"Alpha: {losses['alpha']:.2f} | "
                    f"Current ELO: {self.current_elo:.1f}"
                )

    def _update_opponent_pool(self):
        """更新策略池"""
        self.current_episode += 1
        self.policy_pool[self.current_episode] = {
            "policy": copy.deepcopy(self.agent.state_dict()),
            "elo": self.current_elo
        }

        # 保持池大小
        if len(self.policy_pool) > self.policy_pool_size:
            oldest = min(self.policy_pool.keys())
            del self.policy_pool[oldest]

    def _adjust_elo(self, agent_reward: float, opponent_reward: float):
        """动态调整ELO评分"""
        expected = 1 / (1 + 10 ** ((self.current_elo - 1200) / 400))
        actual = 1.0 if agent_reward > opponent_reward else 0.5 if agent_reward == opponent_reward else 0.0
        delta = 32 * (actual - expected)

        self.current_elo += delta
        logging.info(f"ELO Updated: {self.current_elo:.1f} (Δ{delta:+.1f})")

    @torch.no_grad()
    def evaluate(self):
        """评估当前策略"""
        if not self.use_eval or not self.eval_envs:
            return

        total_rewards = []
        for opponent in self.policy_pool.values():
            ep_rewards = []
            for episode_idx in range(self.eval_episodes):  # 使用 episode_idx 替换 ep
                obs = self.eval_envs.reset()
                episode_reward = 0.0
                while True:
                    agent_act, _ = self.agent.get_action(obs[:, 0], deterministic=True)
                    opp_act, _ = SACPolicy.load_from_state(opponent["policy"]).get_action(obs[:, 1], deterministic=True)
                    next_obs, rewards, dones, _ = self.eval_envs.step(np.stack([agent_act, opp_act], axis=1))
                    episode_reward += rewards[:, 0].mean()
                    obs = next_obs
                    if dones.all():
                        break
                ep_rewards.append(episode_reward)
            avg_reward = np.mean(ep_rewards)
            total_rewards.append(avg_reward)
            logging.info(f"VS Episode {episode_idx}: Avg Reward {avg_reward:.1f}")  # 使用 episode_idx 打印日志

        logging.info(f"Evaluation Complete | Mean Reward: {np.mean(total_rewards):.1f}")

    def save(self):
        """保存策略"""
        save_path = f"{self.run_dir}/sac_ep{self.current_episode}.pt"
        torch.save({
            "policy": self.agent.state_dict(),
            "elo": self.current_elo,
            "step": self.total_env_steps
        }, save_path)
        logging.info(f"Model saved to {save_path}")

    def _env_reset(self):
        """环境重置"""
        obs = self.envs.reset()
        return np.asarray(obs, dtype=np.float32)

    def _env_step(self, agent_act: np.ndarray, opponent_act: np.ndarray):
        """环境交互（修复奖励形状）"""
        actions = np.stack([agent_act, opponent_act], axis=1)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # 保持奖励和done标志的二维结构
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 2)  # 强制转换为 (n_env, 2)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1, 2)  # 强制转换为 (n_env, 2)

        return (
            np.asarray(next_obs, dtype=np.float32),  # shape (n_env, 2, obs_dim)
            rewards,  # shape (n_env, 2)
            dones  # shape (n_env, 2)
        )

    def _store_experience(self, curr_obs: np.ndarray, actions: np.ndarray,
                          rewards: np.ndarray, next_obs: np.ndarray, dones: np.ndarray):
        """存储经验（修复形状问题）"""

        # 确保所有输入都是二维数组
        def ensure_2d(arr, name):
            """确保输入的数组是二维的，如果是三维数组并且第二维的大小为1，则压缩它"""
            if arr.ndim == 1:
                return arr.reshape(-1, 1)
            elif arr.ndim == 3:  # 处理 (n_env, 1, dim) 的情况
                if arr.shape[1] == 1:
                    return arr.squeeze(axis=1)  # 只压缩维度为1的轴
                else:
                    return arr  # 如果第二维不为1，保持原始形状
            elif arr.ndim == 2:
                return arr  # 直接返回二维数组
            else:
                raise ValueError(f"Unsupported array dimensions for {name}: {arr.shape}")

        # 转换形状
        curr_obs = ensure_2d(curr_obs[:, 0], "current observations")  # 选择第一个智能体的观测
        actions = ensure_2d(actions[:, 0], "actions")  # 选择第一个智能体的动作
        rewards = ensure_2d(rewards, "rewards")
        next_obs = ensure_2d(next_obs[:, 0], "next observations")  # 选择第一个智能体的下一个观测
        dones = ensure_2d(dones, "dones")

        # 验证最终形状
        assert curr_obs.shape == (self.n_rollout_threads, self.obs_space.shape[0]), \
            f"Obs shape mismatch: {curr_obs.shape} vs expected {(self.n_rollout_threads, self.obs_space.shape[0])}"

        assert actions.shape == (self.n_rollout_threads, self.act_space.shape[0]), \
            f"Action shape mismatch: {actions.shape} vs expected {(self.n_rollout_threads, self.act_space.shape[0])}"

        # 存储经验
        self.buffer.store(
            obs=curr_obs,  # shape (n_env, obs_dim)
            act=actions,  # shape (n_env, act_dim)
            rew=rewards,  # shape (n_env, 1)
            next_obs=next_obs,  # shape (n_env, obs_dim)
            done=dones  # shape (n_env, 1)
        )
