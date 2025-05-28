# # import logging
# #
# # import numpy as np
# #
# # class SACReplayBuffer:
# #     """
# #     SAC 经验回放缓冲区，支持 n_env 个并行环境
# #     每个时间步存储 (n_env, obs_dim), (n_env, act_dim) 数据。
# #     """
# #
# #     def __init__(self, obs_space, act_space, n_env, capacity=10**6):
# #         """
# #         初始化 SACReplayBuffer
# #         :param obs_space: 观察空间 (gym.Space)
# #         :param act_space: 动作空间 (gym.Space)
# #         :param n_env:  并行环境数量 (int)
# #         :param capacity: 经验回放容量 (int)
# #         """
# #
# #         obs_dim = obs_space.shape[0]  # 获取观察空间维度
# #         act_dim = act_space.shape[0]  # 获取动作空间维度
# #         self.n_env = n_env
# #         self.capacity = capacity
# #
# #         # 初始化存储结构，增加 n_env 维度
# #         self.obs_buf = np.zeros((capacity, n_env, obs_dim), dtype=np.float32)
# #         self.next_obs_buf = np.zeros((capacity, n_env, obs_dim), dtype=np.float32)
# #         self.act_buf = np.zeros((capacity, n_env, act_dim), dtype=np.float32)
# #         self.rew_buf = np.zeros((capacity, n_env, 1), dtype=np.float32)
# #         self.done_buf = np.zeros((capacity, n_env, 1), dtype=np.float32)
# #
# #
# #         self.ptr = 0  # 当前存储指针
# #         self.size = 0  # 当前缓冲区存储的样本数
# #
# #
# #     def store(self, obs, act, rew, next_obs, done):
# #         # logging.info(f"replaybuffer存入的 obs 形状: {obs.shape}")  # 应为 (n_env, obs_dim)
# #         # 删除所有 squeeze 操作，保持原始维度
# #         idx = self.ptr
# #         self.obs_buf[idx] = obs     # 直接存储原始形状（如 [n_env, obs_dim]）
# #         self.act_buf[idx] = act
# #         self.rew_buf[idx] = rew
# #         self.next_obs_buf[idx] = next_obs
# #         self.done_buf[idx] = done
# #         self.ptr = (self.ptr + 1) % self.capacity
# #         self.size = min(self.size + 1, self.capacity)
# #
# #     def sample_batch(self, batch_size=256):
# #         idxs = np.random.randint(0, self.size, size=batch_size)
# #         batch = {
# #             "obs": self.obs_buf[idxs],  # 保持 [batch_size, n_env, obs_dim]
# #             "act": self.act_buf[idxs],
# #             "rew": self.rew_buf[idxs],
# #             "next_obs": self.next_obs_buf[idxs],
# #             "done": self.done_buf[idxs]
# #         }
# #         return batch  # 不再自动压缩维度
# #     def clear(self):
# #         """ 清空缓冲区 """
# #         self.ptr = 0
# #         self.size = 0
# #         self.obs_buf.fill(0)
# #         self.act_buf.fill(0)
# #         self.rew_buf.fill(0)
# #         self.next_obs_buf.fill(0)
# #         self.done_buf.fill(0)
# #         print("[INFO] SACReplayBuffer cleared.")
# #
# #     def __len__(self):
# #         """ 返回当前缓冲区大小 """
# #         return self.size
#
# import numpy as np
# from collections import deque
#
# class SACReplayBuffer:
#     """
#     SAC Replay Buffer for LAG platform
#     - Supports n_rollout_threads and num_agents
#     - Includes reward normalization
#     - Supports RNN states and masks
#     """
#     def __init__(self, args, obs_space, act_space, n_rollout_threads, num_agents, capacity=1000000):
#         self.n_rollout_threads = n_rollout_threads
#         self.num_agents = num_agents
#         self.capacity = capacity
#         self.reward_norm = args.reward_norm
#         self.use_recurrent_policy = args.use_recurrent_policy
#         self.recurrent_hidden_size = args.recurrent_hidden_size
#         self.recurrent_hidden_layers = args.recurrent_hidden_layers
#
#         # Reward normalization
#         self.reward_mean = np.zeros(num_agents)
#         self.reward_var = np.ones(num_agents)
#         self.reward_count = np.zeros(num_agents)
#         self.reward_buffers = [deque(maxlen=1000) for _ in range(num_agents)]
#
#         obs_dim = obs_space.shape[0]
#         act_dim = act_space.shape[0]
#
#         # Buffers for multiple agents
#         self.obs_buf = np.zeros((capacity, n_rollout_threads, num_agents, obs_dim), dtype=np.float32)
#         self.next_obs_buf = np.zeros((capacity, n_rollout_threads, num_agents, obs_dim), dtype=np.float32)
#         self.act_buf = np.zeros((capacity, n_rollout_threads, num_agents, act_dim), dtype=np.float32)
#         self.rew_buf = np.zeros((capacity, n_rollout_threads, num_agents, 1), dtype=np.float32)
#         self.done_buf = np.zeros((capacity, n_rollout_threads, num_agents, 1), dtype=np.float32)
#         self.masks_buf = np.ones((capacity, n_rollout_threads, num_agents, 1), dtype=np.float32)
#
#         # RNN states
#         if self.use_recurrent_policy:
#             self.rnn_states_actor_buf = np.zeros(
#                 (capacity, n_rollout_threads, num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size),
#                 dtype=np.float32
#             )
#         else:
#             self.rnn_states_actor_buf = None
#
#         self.ptr = 0
#         self.size = 0
#
#     def store(self, obs, act, rew, next_obs, done, masks, rnn_states_actor=None):
#         """
#         Store experience for multiple agents
#         Input shapes: [n_rollout_threads, num_agents, dim]
#         """
#         idx = self.ptr
#         self.obs_buf[idx] = obs
#         self.act_buf[idx] = act
#
#         # Update reward statistics per agent
#         if self.reward_norm:
#             for a in range(self.num_agents):
#                 rew_a = rew[:, :, a, :].reshape(-1)
#                 self.reward_buffers[a].extend(rew_a)
#                 self.reward_count[a] += len(rew_a)
#                 self.reward_mean[a] = np.mean(self.reward_buffers[a])
#                 self.reward_var[a] = np.var(self.reward_buffers[a]) + 1e-6
#                 rew[:, :, a, :] = (rew[:, :, a, :] - self.reward_mean[a]) / np.sqrt(self.reward_var[a])
#
#         self.rew_buf[idx] = rew
#         self.next_obs_buf[idx] = next_obs
#         self.done_buf[idx] = done
#         self.masks_buf[idx] = masks
#         if self.use_recurrent_policy and rnn_states_actor is not None:
#             self.rnn_states_actor_buf[idx] = rnn_states_actor
#
#         self.ptr = (self.ptr + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)
#
#     def sample_batch(self, batch_size=256):
#         """
#         Sample a batch of experiences
#         Output shapes: [batch_size, n_rollout_threads, num_agents, dim]
#         """
#         idxs = np.random.randint(0, self.size, size=batch_size)
#         batch = {
#             "obs": self.obs_buf[idxs],
#             "act": self.act_buf[idxs],
#             "rew": self.rew_buf[idxs],
#             "next_obs": self.next_obs_buf[idxs],
#             "done": self.done_buf[idxs],
#             "masks": self.masks_buf[idxs]
#         }
#         if self.use_recurrent_policy:
#             batch["rnn_states_actor"] = self.rnn_states_actor_buf[idxs]
#         return batch
#
#     def clear(self):
#         self.ptr = 0
#         self.size = 0
#         self.obs_buf.fill(0)
#         self.act_buf.fill(0)
#         self.rew_buf.fill(0)
#         self.next_obs_buf.fill(0)
#         self.done_buf.fill(0)
#         self.masks_buf.fill(1)
#         if self.use_recurrent_policy:
#             self.rnn_states_actor_buf.fill(0)
#         for buf in self.reward_buffers:
#             buf.clear()
#         self.reward_mean.fill(0)
#         self.reward_var.fill(1)
#         self.reward_count.fill(0)
import logging

import numpy as np

class SACReplayBuffer:
    """
    SAC 经验回放缓冲区(N, 支持 n_env 个并行环境,
    存储每个环境产生的序列 (n_env, obs_dim), (n_env, act_dim), 奖励
    """
    def __init__(self, obs_space, act_space, n_env, capacity=10**6):
        """
        初始化 SACReplayBuffer
        :param obs_space: 观察空间 (gym.Space)
        :param act_space: 动作空间 (gym.Space)
        :param n_env: 并行环境数量 (int)
        :param capacity: 经验回放容量 (int)
        """
        # print(f"[DEBUG] SACReplayBuffer received obs_space={obs_space}, act_space={act_space}, n_env={n_env}")
        obs_dim = obs_space.shape[0]  # 共取观察空间已知维度
        act_dim = act_space.shape[0]  # 共取动作空间已知维度
        self.n_env = n_env
        self.capacity = capacity

        # 动态初始化缓冲池, 增加 n_env 维度
        self.obs_buf = np.zeros((capacity, n_env, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, n_env, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, n_env, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, n_env, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, n_env, 1), dtype=np.float32)
        # print(f"[DEBUG] SACReplayBuffer initialized with obs_dim={obs_dim}, act_dim={act_dim}, n_env={n_env}, ...")

        self.ptr = 0  # 当前存储指针计数
        self.size = 0  # 总经验数据区存储数据计数

    def store(self, obs, act, rew, next_obs, done):
        """
        存储 n_env 个环境的 (s, a, r, s', done) 经验
        :param obs: (n_env, obs_dim) 观察
        :param act: (n_env, act_dim) 动作
        :param rew: (n_env, 1) 奖励
        :param next_obs: (n_env, obs_dim) 下一步状态
        :param done: (n_env, 1) 终止信号
        """
        # 确保 obs, next_obs 符合是 (n_env, obs_dim) 而不是 (n_env, n_agents, obs_dim)
        if obs.ndim == 3 and obs.shape[1] == 1:
            obs = obs.squeeze(1)
            next_obs = next_obs.squeeze(1)

        if act.ndim == 3 and act.shape[1] == 1:
            act = act.squeeze(1)

        idx = self.ptr
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.next_obs_buf[idx] = next_obs
        self.done_buf[idx] = done

        self.ptr = (self.ptr + 1) % self.capacity
        # 每次 store 增加 n_env 个样本
        self.size = min(self.size + self.n_env, self.capacity)

    def sample_batch(self, batch_size=256):
        """
        采样一个批次的数据
        :param batch_size: 采样大小
        :return: 采样
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "obs": self.obs_buf[idxs],  # shape (batch_size, n_env, obs_dim)
            "act": self.act_buf[idxs],  # shape (batch_size, n_env, act_dim)
            "rew": self.rew_buf[idxs],  # shape (batch_size, n_env, 1)
            "next_obs": self.next_obs_buf[idxs],  # shape (batch_size, n_env, obs_dim)
            "done": self.done_buf[idxs]  # shape (batch_size, n_env, 1)
        }

        # 确保 obs 维度 (batch_size, obs_dim) 而不是 (batch_size, n_agents, obs_dim)
        if batch["obs"].shape[1] == 1:
            batch["obs"] = batch["obs"].squeeze(1)  # 共用维去掉 1 维去掉 squeeze
        if batch["next_obs"].shape[1] == 1:
            batch["next_obs"] = batch["next_obs"].squeeze(1)
        elif self.size < batch_size:
            logging.info(f"[WARNING] Replay Buffer insufficient: size={self.size}, requested={batch_size}")
        # logging.info(f"Replay Buffer Size: {len(self.obs_buf)}, Sampling Batch SIZE: {batch_size}")

        return batch

    def clear(self):
        """
        清空缓冲区
        """
        self.ptr = 0
        self.size = 0
        self.obs_buf.fill(0)
        self.act_buf.fill(0)
        self.rew_buf.fill(0)
        self.next_obs_buf.fill(0)
        self.done_buf.fill(0)
        print("[INFO] SACReplayBuffer cleared.")

    def __len__(self):
        """
        返回当前缓冲区大小
        """
        return self.size