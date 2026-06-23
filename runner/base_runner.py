import logging
import os
import sys
import wandb
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.utils.buffer import ReplayBuffer
from algorithms.sac.sac_replay_buffer import SACReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.buffer_size = self.all_args.buffer_size
        self.use_wandb = self.all_args.use_wandb

        # interval
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.eval_episodes = self.all_args.eval_episodes

        # dir
        self.model_dir = self.all_args.model_dir
        self.run_dir = config["run_dir"]
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.save_dir = str(self.run_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.load()

    def load(self):
        # algorithm
        if self.algorithm_name == "ppo":
            from ..algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from ..algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args,
                             self.envs.observation_space,
                             self.envs.action_space,
                             device=self.device)
        self.trainer = Trainer(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args,
                                   self.envs.observation_space,
                                   self.envs.action_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def rollout(self):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        logging.debug("Computing advantages and returns...")
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def train(self):
        self.policy.prep_training()
        train_infos = self.trainer.train(self.policy, self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        policy_actor = self.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_latest.pt")
        policy_critic = self.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_latest.pt")

    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_latest.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_latest.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)
    # def restore(self, episode=None):
    #     if episode is None:
    #         # 加载最新的保存状态
    #         actor_path = str(self.model_dir) + '/actor_latest.pt'
    #         critic_path = str(self.model_dir) + '/critic_latest.pt'
    #         optimizer_path = str(self.model_dir) + '/optimizer_latest.pt'
    #         buffer_path = str(self.model_dir) + '/buffer_latest.pt'
    #         training_state_path = str(self.model_dir) + '/training_state_latest.pt'
    #     else:
    #         # 加载特定 episode 的状态（仅 actor 和 critic 支持特定 episode）
    #         actor_path = str(self.model_dir) + f'/actor_{episode}.pt'
    #         critic_path = str(self.model_dir) + f'/critic_latest.pt'
    #         optimizer_path = str(self.model_dir) + '/optimizer_latest.pt'
    #         buffer_path = str(self.model_dir) + '/buffer_latest.pt'
    #         training_state_path = str(self.model_dir) + '/training_state_latest.pt'
    #     # 加载支付矩阵（仅 PSRO）
    #     if isinstance(self.selfplay_algo, PSRO) and os.path.exists(str(self.model_dir) + '/payoff_matrix.npy'):
    #         self.selfplay_algo.payoff_matrix = np.load(str(self.model_dir) + '/payoff_matrix.npy')
    #     else:
    #         # 如果没有保存的支付矩阵，调整大小与 policy_pool 匹配
    #         if isinstance(self.selfplay_algo, PSRO):
    #             n = len(self.policy_pool)
    #             self.selfplay_algo.payoff_matrix = np.zeros((n, n))
    #     # 加载模型参数
    #     self.policy.actor.load_state_dict(torch.load(actor_path))
    #     self.policy.critic.load_state_dict(torch.load(critic_path))
    #
    #     # 加载优化器状态（总是加载最新版本）
    #     if hasattr(self.trainer, 'optimizer') and os.path.exists(optimizer_path):
    #         self.trainer.optimizer.load_state_dict(torch.load(optimizer_path))
    #
    #     # 加载回放缓冲区（总是加载最新版本）
    #     if os.path.exists(buffer_path):
    #         self.buffer = torch.load(buffer_path)
    #
    #     # 加载训练进度（总是加载最新版本）
    #     if os.path.exists(training_state_path):
    #         training_state = torch.load(training_state_path)
    #         self.start_episode = training_state['episode'] + 1  # 从下一个 episode 恢复
    #         self.total_num_steps = training_state['total_num_steps']
    #     else:
    #         self.start_episode = 0
    #         self.total_num_steps = 0

    def log_info(self, infos, total_num_steps):
        if self.use_wandb:
            for k, v in infos.items():
                # 添加类型检查和NaN检查
                if v is not None and (not isinstance(v, np.ndarray) or not np.isnan(v).any()):
                    # 确保是标量值
                    if isinstance(v, np.ndarray):
                        if v.size == 1:
                            v = float(v.item())  # 转换为Python标量
                        else:
                            v = float(np.mean(v))  # 取平均值
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    logging.warning(f"Skipping logging {k} due to None or NaN value: {v}")
        else:
            pass