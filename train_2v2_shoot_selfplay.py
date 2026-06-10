#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 第四章入口：2v2 ShootMissile（HierarchySelfplay）训练封装（MAPPO）

import argparse, os, subprocess, sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda", type=int, default=0, help="1=使用GPU, 0=CPU")
    p.add_argument("--num-env-steps", type=float, default=1e6, help="烟雾测试建议 1e5~1e6；正式复现可调大")
    p.add_argument("--n-rollout-threads", type=int, default=8)
    p.add_argument("--experiment-name", type=str, default="ch04_popt_2v2_shoot")
    return p.parse_args()

def main():
    a = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cmd = [
        sys.executable, os.path.join(repo_root, "scripts", "train", "train_jsbsim.py"),
        "--env-name", "MultipleCombat",
        "--algorithm-name", "mappo",
        "--scenario-name", "2v2/ShootMissile/HierarchySelfplay",
        "--experiment-name", a.experiment_name,
        "--seed", str(a.seed),
        "--n-training-threads", "1",
        "--n-rollout-threads", str(a.n_rollout_threads),
        "--num-env-steps", str(a.num_env_steps),
        "--log-interval", "1",
        "--save-interval", "1",
        "--num-mini-batch", "5",
        "--buffer-size", "3000",
        "--lr", "3e-4",
        "--gamma", "0.99",
        "--ppo-epoch", "4",
        "--clip-params", "0.2",
        "--max-grad-norm", "2",
        "--entropy-coef", "1e-3",
        "--hidden-size", "128 128",
        "--act-hidden-size", "128 128",
        "--recurrent-hidden-size", "128",
        "--recurrent-hidden-layers", "1",
        "--data-chunk-length", "8",
        "--use-selfplay",
        "--selfplay-algorithm", "fsp",
        "--n-choose-opponents", "1",
        "--use-eval",
        "--n-eval-rollout-threads", "1",
        "--eval-interval", "1",
        "--eval-episodes", "1",
        "--user-name", "thesis"
    ]
    if a.cuda == 1:
        cmd.insert(2, "--cuda")
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
