#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 第五章结果复核：对比两份训练日志（粗粒度）
# - 提取每次出现的 [CCC] rate
# - 提取 avg_ep_ret_team(...) 的数值

import argparse, re
from pathlib import Path

RE_CCC = re.compile(r"\[CCC\].*rate=([0-9.]+)")
RE_RET = re.compile(r"avg_ep_ret_team\(last\d+\)=([\-0-9.]+)")

def parse_one(path: Path):
    ccc = []
    ret = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_CCC.search(line)
        if m: ccc.append(float(m.group(1)))
        m2 = RE_RET.search(line)
        if m2: ret.append(float(m2.group(1)))
    return {
        "ccc_rate_last": ccc[-1] if ccc else None,
        "ccc_rate_mean": sum(ccc)/len(ccc) if ccc else None,
        "ret_last": ret[-1] if ret else None,
        "ret_mean": sum(ret)/len(ret) if ret else None,
        "ccc_points": len(ccc),
        "ret_points": len(ret),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-a", required=True, help="日志A路径（例如 协同训练）")
    ap.add_argument("--log-b", required=True, help="日志B路径（例如 对照组）")
    args = ap.parse_args()

    a = parse_one(Path(args.log_a))
    b = parse_one(Path(args.log_b))

    print("=== Summary A ===")
    for k,v in a.items(): print(f"{k}: {v}")
    print("\n=== Summary B ===")
    for k,v in b.items(): print(f"{k}: {v}")

if __name__ == "__main__":
    main()
