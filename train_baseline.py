#!/usr/bin/env python
# -*- coding: utf-8 -*-

#第五章入口：关闭 CC-IVB/CC-SVB 与 CCC 协同奖励塑形（对照组）
- USE_CC_SVB=0
- USE_CCC_RSHAPE=0
- EXP_NAME=ch05_baseline

import os, subprocess, sys
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["USE_CC_SVB"] = "0"
    env["USE_CCC_RSHAPE"] = "0"
    env["EXP_NAME"] = env.get("EXP_NAME", "ch05_baseline")

    cmd = [sys.executable, str(repo_root / "scripts" / "train" / "train_hybrid_red.py")]

    log_dir = repo_root / "scripts" / "results" / "MultipleCombat" / "2v2" / "ShootMissile" / "HybridRL" / env["EXP_NAME"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    print("[cmd]", " ".join(cmd))
    print("[log]", log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line)
            print(line, end="")
        p.wait()

if __name__ == "__main__":
    main()
