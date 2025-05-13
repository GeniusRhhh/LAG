#!/usr/bin/env python
import sys
import os
import traceback
import wandb
import socket
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle

# 加入项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from config import get_config
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
# 如果你还有 share_vecenv, 这里就先不用; 我们单智能体即可

def make_train_env(all_args):
    """
    创建训练用的并行环境: SingleControlEnv -> HeadingTask(连续版)
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Unsupported env_name: " + all_args.env_name)
                raise NotImplementedError
            # 设置随机种子
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    # 单智能体 => 就用 DummyVecEnv/SubprocVecEnv
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """
    若需要评估环境 => 结构类似 train_env
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Unsupported env_name: " + all_args.env_name)
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--scenario-name', type=str, default='heading',
                       help="Which scenario to run on, e.g. 'heading'")
    # 还可以添加其他SAC需要的参数,例如 actor_lr, critic_lr等
    group.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient for SAC (default: 0.005)")
    group.add_argument("--init-alpha", type=float, default=0.2,
                        help="Initial temperature coefficient for entropy regularization in SAC")
    group.add_argument('--actor-lr', type=float, default=3e-4, help="Actor learning rate")
    group.add_argument('--critic-lr', type=float, default=3e-4, help="Critic learning rate")
    group.add_argument('--alpha-lr', type=float, default=3e-4, help="Alpha learning rate")
    group.add_argument('--batch-size', type=int, default=256, help="SAC batch size")
    group.add_argument('--update-per-step', type=int, default=1, help="How many gradient updates per env step")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # 1) 设置随机种子
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # 2) 选择设备
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 3) 结果存放目录
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 4) wandb
    if all_args.use_wandb:
        import wandb
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         notes=socket.gethostname(),
                         name=f"{all_args.experiment_name}_seed{all_args.seed}",
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        # 如果不使用wandb, 就在本地写log
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir()
                             if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # 设置进程标题
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # 5) 创建并行环境
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    # 6) Runner配置
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir
    }

    # 7) 引入单智能体 SAC Runner
    from runner.single_jsbsim_runner import SingleJSBSimRunner
    runner = SingleJSBSimRunner(config)
    # from runner.OnevOneJsbsimRunner import OnevOneJSBSimRunner
    # runner = OnevOneJSBSimRunner(config)
    # from runner.selfplay_OnevOneJsbsimRunner import selfplayOnevOneJSBSimRunner
    # runner= selfplayOnevOneJSBSimRunner(config)

    # 8) 开始训练
    try:
        runner.run()
    except BaseException:
        traceback.print_exc()
    finally:
        envs.close()
        if all_args.use_wandb:
            run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
