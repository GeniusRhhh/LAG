import torch
import logging
import numpy as np
import time



class OnevOneJSBSimRunner:
    """
    1v1 双智能体 SAC Runner
    - 每个 agent 都有自己的 SACPolicy / ReplayBuffer / Trainer
    - 环境 SingleCombatEnv => num_agents=2
    - obs shape=(n_env, 2, obs_dim), rew/done shape=(n_env,2)
    """
    def __init__(self, config):
        self.envs = config["envs"]        # SubprocVecEnv or DummyVecEnv
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
        # 计算每个训练轮次的步数
        self.steps_per_round = self.log_interval  # 每次输出间隔就是每轮的训练步数
        self.total_rounds = self.num_env_steps // self.steps_per_round  # 总轮次

        # 初始化其他参数
        self.current_round = 1  # 当前轮次编号
        self.steps_in_current_round = 0  # 当前轮次的步数

        logging.info(f"Total rounds: {self.total_rounds}, Steps per round: {self.steps_per_round}")

        # 1) 解析环境的 obs_space/act_space => shape=(2,obs_dim)/(2,act_dim)
        multi_obs_space = self.envs.observation_space  # shape=(2,obs_dim)
        multi_act_space = self.envs.action_space       # shape=(2,act_dim)
        # agent0 obs_box, agent1 obs_box
        # simplest approach => same shape => copy
        self.obs_space0 = self._extract_single_agent_box(multi_obs_space, agent_index=0)
        self.obs_space1 = self._extract_single_agent_box(multi_obs_space, agent_index=1)
        self.act_space0 = self._extract_single_agent_box(multi_act_space, agent_index=0)
        self.act_space1 = self._extract_single_agent_box(multi_act_space, agent_index=1)
        # logging.info(f"[OnevOneJSBSimRunner] Agent0: obs={self.obs_space0}, act={self.act_space0}")
        # logging.info(f"[OnevOneJSBSimRunner] Agent1: obs={self.obs_space1}, act={self.act_space1}")

        # 2) 构建 两个 SACPolicy/Trainer/ReplayBuffer
        from algorithms.sac.sac_policy import SACPolicy
        from algorithms.sac.sac_trainer import SACTrainer
        from algorithms.sac.sac_replay_buffer import SACReplayBuffer

        self.agent0 = SACPolicy(self.all_args, self.obs_space0, self.act_space0)
        self.agent1 = SACPolicy(self.all_args, self.obs_space1, self.act_space1)
        self.trainer0 = SACTrainer(self.agent0)
        self.trainer1 = SACTrainer(self.agent1)

        # ReplayBuffer
        buffer_capacity = getattr(self.all_args, "buffer_size", 10**5)
        # Each agent stores its own data
        self.buffer0 = SACReplayBuffer(
            obs_space=self.obs_space0,
            act_space=self.act_space0,
            n_env=self.n_rollout_threads,
            capacity=buffer_capacity
        )
        self.buffer1 = SACReplayBuffer(
            obs_space=self.obs_space1,
            act_space=self.act_space1,
            n_env=self.n_rollout_threads,
            capacity=buffer_capacity
        )

        self.total_env_steps = 0
        self.start_time = time.time()

    def run(self):
        # reset => shape=(n_env,2,obs_dim)
        obs = self.envs.reset()
        obs = np.array(obs, dtype=np.float32)
        if len(obs.shape) == 4:
            obs = obs[:, :, 0, :]  # 转换为 (n_env, 2, obs_dim)
        while self.total_env_steps < self.num_env_steps:
            # 1) split => obs0=(n_env,obs_dim), obs1=(n_env,obs_dim)
            obs0 = obs[:,0,:]
            obs1 = obs[:,1,:]

            # 2) get_action => shape=(n_env,act_dim)
            actions0, _ = self.agent0.get_action(obs0, deterministic=False)
            actions1, _ = self.agent1.get_action(obs1, deterministic=False)

            # 3) combine => (n_env,2,act_dim)
            actions_for_env = np.stack([actions0, actions1], axis=1)

            # 4) env step => next_obs=(n_env,2,obs_dim), rewards=(n_env,2), dones=(n_env,2)
            next_obs, rewards, dones, infos = self.envs.step(actions_for_env)
            rewards = np.squeeze(rewards, axis=(1, 3))  # squeeze 后 rewards.shape 变为 (8, 2)
            # logging.info("After squeeze, rewards shape: %s", rewards.shape)
            dones = np.squeeze(dones, axis=(1, 3))
            # logging.info("After squeeze, dones shape: %s", dones.shape)

            next_obs = np.array(next_obs, dtype=np.float32)
            if len(next_obs.shape) == 4:
                next_obs = next_obs[:, :, 0, :]  # 变为 (n_env, 2, obs_dim)
            # logging.info("Rewards array shape: %s", rewards.shape)
            # logging.info("Rewards content: %s", rewards)

            # reshape => rew0=(n_env,1), done0=(n_env,1)
            rew0 = rewards[:,0].reshape(-1,1)

            rew1 = rewards[:,1].reshape(-1,1)
            done0= dones[:,0].reshape(-1,1)
            done1= dones[:,1].reshape(-1,1)

            # next_obs => shape=(n_env,2,obs_dim)
            next_obs0 = next_obs[:,0,:]
            next_obs1 = next_obs[:,1,:]

            # 5) 存入各自 buffer
            self.buffer0.store(obs0, actions0, rew0, next_obs0, done0)
            self.buffer1.store(obs1, actions1, rew1, next_obs1, done1)

            obs = next_obs
            self.total_env_steps += self.n_rollout_threads
            self.steps_in_current_round += self.n_rollout_threads
            # 每当完成一轮训练，更新轮次信息
            if self.steps_in_current_round >= self.steps_per_round:
                logging.info(f"Round {self.current_round}/{self.total_rounds} completed, steps: {self.total_env_steps}")
                self.steps_in_current_round = 0
                self.current_round += 1

            # 6) 训练
            if len(self.buffer0)>self.batch_size:
                for _ in range(self.update_per_step):
                    batch0 = self.buffer0.sample_batch(self.batch_size)
                    losses2 = self.trainer0.update(batch0["obs"], batch0["act"], batch0["rew"],
                                                  batch0["next_obs"], batch0["done"])
                    critic_loss1 = losses2['critic_loss']
                    actor_loss1 = losses2['actor_loss']
                    alpha_loss1 = losses2['alpha_loss']
                    # 固定轮次打印
                    if self.total_env_steps % 5000 == 0:
                        logging.info(f"[1v1Runner] Steps={self.total_env_steps}, "
                                     f"Round={self.current_round}, "
                                     f"Ego critic loss={critic_loss1}, "
                                     f"Actor loss={actor_loss1}, "
                                     f"Alpha loss={alpha_loss1}")
            if len(self.buffer1)>self.batch_size:
                for _ in range(self.update_per_step):
                    batch1 = self.buffer1.sample_batch(self.batch_size)
                    logging.info(f"Batch shape: {batch1['obs'].shape}")
                    logging.info(f"Batch action shape: {batch1['act'].shape}")
                    losses = self.trainer1.update(batch1["obs"], batch1["act"], batch1["rew"],
                                                  batch1["next_obs"], batch1["done"])
                    critic_loss = losses['critic_loss']
                    actor_loss = losses['actor_loss']
                    alpha_loss = losses['alpha_loss']
                    # 固定轮次打印
                    if self.total_env_steps % 5000 == 0:
                        logging.info(f"[1v1Runner] Steps={self.total_env_steps}, "
                                     f"Round={self.current_round}, "
                                     f"Enemy critic loss={critic_loss}, "
                                     f"Actor loss={actor_loss}, "
                                     f"Alpha loss={alpha_loss}")

                    for name, param in self.agent1.critic.named_parameters():
                        if param.grad is not None:
                            logging.debug(f"Enemy Critic {name} grad norm: {param.grad.norm().item()}")

                    for name, param in self.agent1.actor.named_parameters():
                        if param.grad is not None:
                            logging.debug(f"Enemy Actor {name} grad norm: {param.grad.norm().item()}")

            # logging
            if self.total_env_steps%self.log_interval==0:
                cost_time = time.time()-self.start_time
                fps = int(self.total_env_steps/(cost_time+1e-6))
                logging.info(f"[1v1Runner] Steps={self.total_env_steps}/{self.num_env_steps}, FPS={fps}, "
                             f"Buffer0={len(self.buffer0)}, Buffer1={len(self.buffer1)}")

            # eval
            if self.use_eval and self.total_env_steps>0 and self.total_env_steps%self.eval_interval==0:
                self.eval()

            # save
            if self.total_env_steps>0 and self.total_env_steps%self.save_interval==0:
                self.save()

    @torch.no_grad()
    def eval(self):
        logging.info("[Eval] start evaluation ...")
        if self.eval_envs is None:
            return
        # 仅简化 => 跑 `eval_episodes` 回合
        returns0=[]
        returns1=[]
        for ep in range(self.eval_episodes):
            obs = self.eval_envs.reset()  # shape=(n_eval_env,2,obs_dim)
            ep_ret0=0
            ep_ret1=0
            while True:
                obs0=obs[:,0,:]
                obs1=obs[:,1,:]
                act0,_=self.agent0.get_action(obs0, deterministic=True)
                act1,_=self.agent1.get_action(obs1, deterministic=True)
                acts=np.stack([act0,act1], axis=1)
                next_obs, rews, dons, infos= self.eval_envs.step(acts)
                ep_ret0 += rews[:,0].sum()
                ep_ret1 += rews[:,1].sum()
                obs=next_obs
                if dons.all():
                    break
            returns0.append(ep_ret0)
            returns1.append(ep_ret1)
        logging.info(f"[Eval] agent0 avgRet={np.mean(returns0)}, agent1 avgRet={np.mean(returns1)}")

    def save(self):
        # 分别保存 agent0, agent1
        path0 = f"{self.run_dir}/agent0_sac_{self.current_round}.pt"
        path1 = f"{self.run_dir}/agent1_sac_{self.current_round}.pt"
        self.agent0.save(path0)
        self.agent1.save(path1)
        logging.info(f"[1v1Runner] saved => {path0} and {path1}")

    def restore(self, path0, path1):
        self.agent0.load(path0)
        self.agent1.load(path1)
        logging.info(f"[1v1Runner] loaded => {path0} & {path1}")

    def _extract_single_agent_box(self, multi_box, agent_index=0):
        """
        从多智能体空间提取单个智能体空间
        Args:
            multi_box (gym.Box): 原始多智能体空间，形状为(2, dim)
            agent_index (int): 要提取的智能体索引（0或1）
        Returns:
            Box: 单个智能体的空间定义
        """
        from gymnasium.spaces import Box
        if not isinstance(multi_box, Box):
            raise NotImplementedError("Only support Box shape=(2, dim) for 1v1")
        shape=multi_box.shape  # e.g. (2, 15)
        assert len(shape)==2, f"Expect shape=(2, dim) got {shape}"
        assert shape[0]==2, f"Expect 2 agents, got {shape}"
        # => 取 [agent_index,:]
        low_  = multi_box.low[agent_index,:]
        high_ = multi_box.high[agent_index,:]
        new_box= Box(low=low_, high=high_, dtype=multi_box.dtype)
        return new_box






# import time
# import torch
# import logging
# import numpy as np
#
# from algorithms.sac.sac_policy import SACPolicy
# from algorithms.sac.sac_trainer import SACTrainer
# from algorithms.sac.sac_replay_buffer import SACReplayBuffer
#
# class OnevOneJSBSimRunner:
#     """
#     1v1 双智能体 SAC Runner
#     - 每个 agent 都有自己的 SACPolicy / ReplayBuffer / Trainer
#     - 环境 SingleCombatEnv => num_agents=2
#     - obs shape=(n_env, 2, obs_dim), rew/done shape=(n_env, 2)
#     """
#     def __init__(self, config):
#         self.envs = config["envs"]        # SubprocVecEnv or DummyVecEnv
#         self.eval_envs = config["eval_envs"]
#         self.all_args = config["all_args"]
#         self.device = config["device"]
#         self.run_dir = config["run_dir"]
#
#         self.num_env_steps = self.all_args.num_env_steps
#         self.n_rollout_threads = self.all_args.n_rollout_threads
#         self.batch_size = getattr(self.all_args, "batch_size", 128)
#         self.update_per_step = getattr(self.all_args, "update_per_step", 1)
#
#         self.use_eval = getattr(self.all_args, "use_eval", False)
#         self.eval_episodes = getattr(self.all_args, "eval_episodes", 5)
#         self.log_interval = getattr(self.all_args, "log_interval", 1000)
#         self.eval_interval = getattr(self.all_args, "eval_interval", 5000)
#         self.save_interval = getattr(self.all_args, "save_interval", 10000)
#         # 计算每个训练轮次的步数
#         self.steps_per_round = self.log_interval  # 每次输出间隔就是每轮的训练步数
#         self.total_rounds = self.num_env_steps // self.steps_per_round  # 总轮次
#
#         # 初始化其他参数
#         self.current_round = 1  # 当前轮次编号
#         self.steps_in_current_round = 0  # 当前轮次的步数
#
#         logging.info(f"Total rounds: {self.total_rounds}, Steps per round: {self.steps_per_round}")
#
#         # 固定设置：我方(agent0)采用基线策略，只训练敌方(agent1)
#         self.only_train_enemy = True
#
#         # 定义预训练模型的路径（基线策略模型），如果该路径存在模型，则加载该模型，
#         # 否则 agent0 将退化为随机策略。请将下行路径替换为你预训练模型的实际路径。
#         baseline_model_path = "../scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/sac/02051346/agent0_sac_5960000.pt"
#
#
#         # 1) 解析环境的 obs_space/act_space => shape=(2, obs_dim)/(2, act_dim)
#         multi_obs_space = self.envs.observation_space  # shape=(2, obs_dim)
#         multi_act_space = self.envs.action_space       # shape=(2, act_dim)
#         self.obs_space0 = self._extract_single_agent_box(multi_obs_space, agent_index=0)
#         self.obs_space1 = self._extract_single_agent_box(multi_obs_space, agent_index=1)
#         self.act_space0 = self._extract_single_agent_box(multi_act_space, agent_index=0)
#         self.act_space1 = self._extract_single_agent_box(multi_act_space, agent_index=1)
#
#         # 2) 构建两个 SACPolicy/Trainer/ReplayBuffer
#         self.agent0 = SACPolicy(self.all_args, self.obs_space0, self.act_space0)
#         self.agent1 = SACPolicy(self.all_args, self.obs_space1, self.act_space1)
#         self.trainer0 = SACTrainer(self.agent0)
#         self.trainer1 = SACTrainer(self.agent1)
#         # 在 __init__ 方法中添加
#         if self.only_train_enemy:
#             # 假设 SACPolicy 内部有 self.actor 和 self.critic 是 nn.Module
#             for param in self.agent0.actor.parameters():
#                 param.requires_grad_(False)
#             for param in self.agent0.critic.parameters():
#                 param.requires_grad_(False)
#             self.trainer0 = None  # 禁用 agent0 的训练器
#
#         # 如果 baseline_model_path 存在，则加载预训练的模型作为 agent0 的基线策略
#         # 注意：agent0 的模型加载后不再进行训练更新
#         self.use_loaded_baseline = False
#         try:
#             self.agent0.load(baseline_model_path)
#             self.use_loaded_baseline = True
#             logging.info(f"[OnevOneJSBSimRunner] Loaded baseline model for agent0 from {baseline_model_path}")
#         except Exception as e:
#             logging.info(f"[OnevOneJSBSimRunner] Failed to load baseline model from {baseline_model_path}: {e}")
#             logging.info("[OnevOneJSBSimRunner] agent0 will use random actions as baseline.")
#
#         buffer_capacity = getattr(self.all_args, "buffer_size", 10**5)
#         self.buffer0 = SACReplayBuffer(
#             obs_space=self.obs_space0,
#             act_space=self.act_space0,
#             n_env=self.n_rollout_threads,
#             capacity=buffer_capacity
#         )
#         self.buffer1 = SACReplayBuffer(
#             obs_space=self.obs_space1,
#             act_space=self.act_space1,
#             n_env=self.n_rollout_threads,
#             capacity=buffer_capacity
#         )
#
#         self.total_env_steps = 0
#         self.start_time = time.time()
#
#     def run(self):
#         # reset => shape=(n_env, 2, obs_dim)
#         obs = self.envs.reset()
#         obs = np.array(obs, dtype=np.float32)
#         if len(obs.shape) == 4:
#             obs = obs[:, :, 0, :]  # 转换为 (n_env, 2, obs_dim)
#         while self.total_env_steps < self.num_env_steps:
#             # 1) 分离 obs: obs0=(n_env, obs_dim), obs1=(n_env, obs_dim)
#             obs0 = obs[:, 0, :]
#             obs1 = obs[:, 1, :]
#
#             # 2) 动作选择：对于 agent0，如果 only_train_enemy 为 True，
#             #   则采用基线策略：
#             #       如果加载了预训练模型，则使用其 deterministic 模式；
#             #       否则，使用随机动作。
#             if self.only_train_enemy:
#                 if self.use_loaded_baseline:
#                     actions0, _ = self.agent0.get_action(obs0, deterministic=True)
#                 else:
#                     actions0 = np.array([self.act_space0.sample() for _ in range(self.n_rollout_threads)])
#             else:
#                 actions0, _ = self.agent0.get_action(obs0, deterministic=False)
#             actions1, _ = self.agent1.get_action(obs1, deterministic=False)
#
#             # 3) 合并动作 => (n_env, 2, act_dim)
#             actions_for_env = np.stack([actions0, actions1], axis=1)
#
#             # 4) 环境 step，获得 next_obs, rewards, dones, infos
#             next_obs, rewards, dones, infos = self.envs.step(actions_for_env)
#             rewards = np.squeeze(rewards, axis=(1, 3))  # rewards 形状变为 (n_env, 2)
#             dones = np.squeeze(dones, axis=(1, 3))
#
#             next_obs = np.array(next_obs, dtype=np.float32)
#             if len(next_obs.shape) == 4:
#                 next_obs = next_obs[:, :, 0, :]  # 转换为 (n_env, 2, obs_dim)
#
#             # 重塑 rewards 和 dones 为 (n_env, 1)
#             rew0 = rewards[:, 0].reshape(-1, 1)
#             rew1 = rewards[:, 1].reshape(-1, 1)
#             done0 = dones[:, 0].reshape(-1, 1)
#             done1 = dones[:, 1].reshape(-1, 1)
#
#             next_obs0 = next_obs[:, 0, :]
#             next_obs1 = next_obs[:, 1, :]
#
#             # 5) 存入各自的 ReplayBuffer
#             self.buffer0.store(obs0, actions0, rew0, next_obs0, done0)
#             self.buffer1.store(obs1, actions1, rew1, next_obs1, done1)
#
#             obs = next_obs
#             self.total_env_steps += self.n_rollout_threads
#             self.steps_in_current_round += self.n_rollout_threads
#             # 每当完成一轮训练，更新轮次信息
#             if self.steps_in_current_round >= self.steps_per_round:
#                 logging.info(f"Round {self.current_round}/{self.total_rounds} completed, steps: {self.total_env_steps}")
#                 self.steps_in_current_round = 0
#                 self.current_round += 1
#
#             # 6) 训练更新：只更新 agent1（敌方）的参数，agent0 保持固定
#             if len(self.buffer1) > self.batch_size:
#                 for _ in range(self.update_per_step):
#                     batch1 = self.buffer1.sample_batch(self.batch_size)
#                     losses = self.trainer1.update(batch1["obs"], batch1["act"], batch1["rew"],
#                                                   batch1["next_obs"], batch1["done"])
#
#                     critic_loss = losses['critic_loss']
#                     actor_loss = losses['actor_loss']
#                     alpha_loss = losses['alpha_loss']
#
#                     # 固定轮次打印
#                     if self.total_env_steps % 5000 == 0:
#                         logging.info(f"[1v1Runner] Steps={self.total_env_steps}, "
#                                      f"Round={self.current_round}, "
#                                      f"Enemy critic loss={critic_loss}, "
#                                      f"Actor loss={actor_loss}, "
#                                      f"Alpha loss={alpha_loss}")
#
#                     for name, param in self.agent1.critic.named_parameters():
#                         if param.grad is not None:
#                             logging.debug(f"Enemy Critic {name} grad norm: {param.grad.norm().item()}")
#
#                     for name, param in self.agent1.actor.named_parameters():
#                         if param.grad is not None:
#                             logging.debug(f"Enemy Actor {name} grad norm: {param.grad.norm().item()}")
#             # logging
#             if self.total_env_steps % self.log_interval == 0:
#                 cost_time = time.time() - self.start_time
#                 fps = int(self.total_env_steps / (cost_time + 1e-6))
#                 logging.info(f"[1v1Runner] Steps={self.total_env_steps}/{self.num_env_steps}, FPS={fps}, "
#                              f"Buffer0={len(self.buffer0)}, Buffer1={len(self.buffer1)}")
#             # eval
#             if self.use_eval and self.total_env_steps > 0 and self.total_env_steps % self.eval_interval == 0:
#                 self.eval()
#
#             # save
#             if self.total_env_steps > 0 and self.total_env_steps % self.save_interval == 0:
#                 self.save()
#
#     @torch.no_grad()
#     def eval(self):
#         logging.info("[Eval] start evaluation ...")
#         if self.eval_envs is None:
#             return
#         returns0 = []
#         returns1 = []
#         for ep in range(self.eval_episodes):
#             obs = self.eval_envs.reset()  # shape=(n_eval_env, 2, obs_dim)
#             ep_ret0 = 0
#             ep_ret1 = 0
#             while True:
#                 obs0 = obs[:, 0, :]
#                 obs1 = obs[:, 1, :]
#                 # 在评估时，对 agent0 同样采用基线策略：
#                 if self.only_train_enemy:
#                     if self.use_loaded_baseline:
#                         act0, _ = self.agent0.get_action(obs0, deterministic=True)
#                     else:
#                         act0 = np.array([self.act_space0.sample() for _ in range(obs0.shape[0])])
#                 else:
#                     act0, _ = self.agent0.get_action(obs0, deterministic=True)
#                 act1, _ = self.agent1.get_action(obs1, deterministic=True)
#                 acts = np.stack([act0, act1], axis=1)
#                 next_obs, rews, dons, infos = self.eval_envs.step(acts)
#                 ep_ret0 += rews[:, 0].sum()
#                 ep_ret1 += rews[:, 1].sum()
#                 obs = next_obs
#                 if dons.all():
#                     break
#             returns0.append(ep_ret0)
#             returns1.append(ep_ret1)
#         logging.info(f"[Eval] agent0 avgRet={np.mean(returns0)}, agent1 avgRet={np.mean(returns1)}")
#
#     def save(self):
#         # 在每轮训练时保存模型，使用轮次命名
#         if not self.only_train_enemy:
#             path0 = f"{self.run_dir}/agent0_sac_{self.current_round}.pt"
#             self.agent0.save(path0)
#             logging.info(f"[1v1Runner] saved agent0 => {path0}")
#
#         path1 = f"{self.run_dir}/agent1_sac_{self.current_round}.pt"
#         self.agent1.save(path1)
#         logging.info(f"[1v1Runner] saved agent1 => {path1}")
#
#     def restore(self, path0, path1):
#         self.agent0.load(path0)
#         self.agent1.load(path1)
#         logging.info(f"[1v1Runner] loaded => {path0} & {path1}")
#
#     def _extract_single_agent_box(self, multi_box, agent_index=0):
#         """
#         从多智能体空间提取单个智能体空间
#         Args:
#             multi_box (gym.Box): 原始多智能体空间，形状为(2, dim)
#             agent_index (int): 要提取的智能体索引（0或1）
#         Returns:
#             Box: 单个智能体的空间定义
#         """
#         from gymnasium.spaces import Box
#         if not isinstance(multi_box, Box):
#             raise NotImplementedError("Only support Box shape=(2, dim) for 1v1")
#         shape = multi_box.shape  # 例如 (2, 15)
#         assert len(shape) == 2, f"Expect shape=(2, dim) got {shape}"
#         assert shape[0] == 2, f"Expect 2 agents, got {shape}"
#         low_ = multi_box.low[agent_index, :]
#         high_ = multi_box.high[agent_index, :]
#         new_box = Box(low=low_, high=high_, dtype=multi_box.dtype)
#         return new_box
