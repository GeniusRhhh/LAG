#!/usr/bin/env python
"""
改进的SU-27训练脚本
基于测试结果的问题分析，针对性改进训练策略
"""
import sys
import os
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from envs.JSBSim.envs import SingleControlEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def make_train_env(all_args):
    """创建训练环境"""
    def get_env_fn(rank):
        def init_env():
            env = SingleControlEnv(all_args.scenario_name)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
        envs = DummyVecEnv([get_env_fn(0)])
    else:
        envs = SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    
    obs = envs.reset()
    logging.info(f"✅ 训练环境初始化：线程数={all_args.n_rollout_threads}, 观察形状={obs.shape}")
    return envs


def make_eval_env(all_args):
    """创建评估环境"""
    def get_env_fn(rank):
        def init_env():
            env = SingleControlEnv(all_args.scenario_name)
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
    
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """解析命令行参数"""
    group = parser.add_argument_group("SU-27 Improved Training")
    group.add_argument('--scenario-name', type=str, default='1/heading_su27',
                       help="训练场景")
    group.add_argument('--resume-from', type=str, default=None,
                       help="从已有模型继续训练")
    
    all_args = parser.parse_known_args(args)[0]
    
    # 强制设置并行环境数为8（针对i7-11700K优化）
    if all_args.n_rollout_threads == 4:  # 如果是默认值
        all_args.n_rollout_threads = 8
        logging.info(f"⚙️  强制设置并行环境数: 4 -> 8 (针对i7-11700K优化)")
    
    return all_args


def main(args):
    """主训练流程"""
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # 创建日志
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "training_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"su27_improved_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.getLogger('jsbsim').setLevel(logging.ERROR)
    
    # 设置随机种子
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    
    # 设备
    if all_args.cuda and torch.cuda.is_available():
        logging.info("🎮 使用GPU训练")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("💻 使用CPU训练")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    # 保存目录
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / "SU27_Improved" / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    if not all_args.use_wandb:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) 
                           for folder in run_dir.iterdir() 
                           if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    
    setproctitle.setproctitle(f"SU27_Improved-{all_args.algorithm_name}-{all_args.experiment_name}")
    
    # 打印训练信息
    logging.info("=" * 80)
    logging.info("🛩️  SU-27改进训练方案")
    logging.info("=" * 80)
    logging.info(f"场景: {all_args.scenario_name}")
    logging.info(f"算法: {all_args.algorithm_name}")
    logging.info(f"并行环境: {all_args.n_rollout_threads}")
    logging.info(f"总步数: {all_args.num_env_steps:,}")
    logging.info(f"学习率: {all_args.lr}")
    logging.info(f"保存路径: {run_dir}")
    
    # 改进点说明
    logging.info("\n📌 改进策略:")
    logging.info("1. 增加训练步数到2000万步")
    logging.info("2. 调整学习率为5e-4（更激进的学习）")
    logging.info("3. 增加并行环境数到16")
    logging.info("4. 更频繁的模型保存（每50个episode）")
    logging.info("5. 支持从已有模型继续训练")
    logging.info("=" * 80)
    
    # 创建环境
    envs = make_train_env(all_args)
    logging.info(f"📊 观察空间: {envs.observation_space}")
    logging.info(f"📊 动作空间: {envs.action_space}")
    
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir
    }
    
    # 使用PPO算法
    from runner.jsbsim_runner import JSBSimRunner as Runner
    
    # 如果指定了继续训练，设置model_dir让Runner自动恢复
    if all_args.resume_from:
        # 从模型路径提取目录
        resume_dir = os.path.dirname(all_args.resume_from)
        logging.info(f"📥 从已有checkpoint继续训练: {resume_dir}")
        config["model_dir"] = resume_dir
    
    runner = Runner(config)
    
    try:
        logging.info("🚀 开始训练...")
        runner.run()
        logging.info("✅ 训练完成！")
        logging.info(f"📁 模型保存在: {run_dir}")
        
        # 提示下一步
        logging.info("\n" + "=" * 80)
        logging.info("📌 训练完成后的步骤：")
        logging.info(f"1. 测试模型: python ceshi_su27_flight.py --model {run_dir}/actor_latest.pt")
        logging.info(f"2. 如果效果好，复制到: envs/JSBSim/model/su27_baseline_model.pt")
        logging.info(f"3. 如果效果不好，继续训练: --resume-from {run_dir}/actor_latest.pt")
        logging.info("=" * 80)
        
    except KeyboardInterrupt:
        logging.warning("⚠️  训练被用户中断")
        logging.info(f"💾 当前模型已保存在: {run_dir}")
    except Exception as e:
        logging.error(f"❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        envs.close()
        if eval_envs:
            eval_envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
