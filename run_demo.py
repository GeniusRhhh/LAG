#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os
import numpy as np

from envs.JSBSim.envs import MultipleCombatEnv
from envs.JSBSim.core.radar import RadarModel
from Tactical_Rule_Template.ruleset import tactical_templates_v2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=str,
                   default="envs/JSBSim/configs/2v2/ShootMissile/HierarchySelfplay.yaml",
                   help="场景yaml路径（相对仓库根目录）")
    p.add_argument("--steps", type=int, default=800, help="推进步数（烟雾测试建议 500~1000）")
    p.add_argument("--out", type=str, default="outputs/ch03/demo.acmi", help="TacView .acmi 输出路径")
    p.add_argument("--render-every", type=int, default=1, help="每隔多少步写一次 acmi（1=每步）")
    p.add_argument("--verbose", action="store_true", help="打印更多模板调度信息")
    return p.parse_args()

def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    scenario_path = os.path.join(repo_root, args.scenario)
    out_path = os.path.join(repo_root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = MultipleCombatEnv(scenario_path)
    radar = RadarModel()

    env.reset()
    control_ids = (env.ego_ids + env.enm_ids)[:env.num_agents]
    ctx_A = {"GR": {}}

    for t in range(args.steps):
        actions = {}
        for aid in control_ids:
            if str(aid).startswith("A"):  # 红方
                if t < 250:
                    tpl = "PINCER"
                elif t < 450:
                    tpl = "T2_LAD"
                else:
                    tpl = "T3_DEF"
                actions[aid] = tactical_templates_v2._dispatch_template(tpl, env, aid, radar, ctx=ctx_A)
                if args.verbose and (t % 50 == 0):
                    print(f"[t={t}] {aid}: tpl={tpl} act={actions[aid]}")
            else:
                actions[aid] = [1, 2, 1, 0]

        ordered = [actions[aid] for aid in control_ids]
        ret = env.step(ordered)

        if args.render_every > 0 and (t % args.render_every == 0):
            env.render(mode="txt", filepath=out_path)

        if len(ret) == 5:
            _, _, _, dones, _ = ret
        else:
            _, _, dones, _ = ret
        done_all = bool(dones.get("__all__", False)) if isinstance(dones, dict) else bool(np.all(dones))
        if done_all:
            break

    print(f"[OK] wrote acmi -> {args.out}")

if __name__ == "__main__":
    main()
