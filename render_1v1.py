#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 第四章入口：1v1 渲染（生成 .acmi）

import argparse, os, subprocess, sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True, help="训练输出 run* 目录路径")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", type=int, default=0, help="1=使用GPU, 0=CPU")
    p.add_argument("--episode-length", type=int, default=1200)
    p.add_argument("--experiment-name", type=str, default="ch04_popt_1v1_shoot")
    return p.parse_args()

def main():
    a = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cmd = [
        sys.executable, os.path.join(repo_root, "scripts", "render", "render_jsbsim.py"),
        "--env-name", "SingleCombat",
        "--algorithm-name", "ppo",
        "--scenario-name", "1v1/ShootMissile/HierarchySelfplay",
        "--experiment-name", a.experiment_name,
        "--seed", str(a.seed),
        "--episode-length", str(a.episode_length),
        "--n-training-threads", "1",
        "--n-rollout-threads", "1",
        "--model-dir", a.model_dir,
        "--user-name", "thesis",
        "--use-selfplay",
    ]
    if a.cuda == 1:
        cmd.insert(2, "--cuda")
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
