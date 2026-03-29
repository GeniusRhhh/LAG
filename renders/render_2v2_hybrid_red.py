# -*- coding: utf-8 -*-
import os, sys, torch, numpy as np
from pathlib import Path
from config import get_config
from algorithms.ppo.ppo_policy import PPOPolicy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.radar import RadarModel
from algorithms.hybrid.ppo_policy_hybrid import PPOPolicyHybrid
from Tactical_Rule_Template.ruleset.rl_template_driver import (
    list_templates, max_param_dim, param_mask_by_id, step_red_agent_pure_rl, reset_agent_state
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # ===== 指向训练好的 run 目录 =====
    run_dir = Path(ROOT) / "scripts/results/MultipleCombat/2v2/ShootMissile/HybridRL/hybrid_red_exp/run133"
    actor_path = run_dir / "actor_latest.pt"
    acmi_path = run_dir / "render_run.txt.acmi"  # 保存到同一目录
    # ===== 环境 & 雷达 =====
    env = MultipleCombatEnv(os.path.join(ROOT, "envs/JSBSim/configs/2v2/ShootMissile/HierarchySelfplay.yaml"))
    radar = RadarModel()
    env.reset()
    env.render(mode='txt', filepath=str(acmi_path))  # reset 后先写一帧（和独立脚本一致）
    # ===== 构建策略并加载权重 =====
    boot_obs = env.get_obs()
    any_id = (env.ego_ids + env.enm_ids)[0]
    obs_dim = len(boot_obs[any_id])
    tpl_num, z_dim = len(list_templates()), max_param_dim()

    pi = PPOPolicyHybrid(obs_dim=obs_dim, tpl_num=tpl_num, z_dim=z_dim, device=DEVICE)
    pi.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE, weights_only=False))
    pi.actor.eval();pi.critic.eval()
    # —— 新增：蓝方 PPO，与训练脚本一致 —— #
    parser = get_config()
    args = parser.parse_args([])  # 用默认/占位超参即可构建同款网络
    blue_pi = PPOPolicy(args=args, obs_space=env.observation_space, act_space=env.action_space, device=DEVICE)
    blue_pi.actor.to(DEVICE).eval()
    blue_rnn = {}  # {bid: rnn_state}

    red_ids, blue_ids = env.ego_ids, env.enm_ids
    for aid in red_ids: reset_agent_state(aid)

    obs_dict = env.get_obs()
    ep_ret = 0.0
    while True:
        actions = {}
        # 红方：按策略选模板+参数，再桥接成高层动作
        for aid in red_ids:
            ob = np.asarray(obs_dict[aid], dtype=np.float32)
            with torch.no_grad():
                tpl_id, z_vec, _, _ = pi.act(ob)
            mask_vec = param_mask_by_id(tpl_id, z_dim)
            z_vec = z_vec * mask_vec
            actions[aid] = step_red_agent_pure_rl(env, radar, aid, tpl_id, z_vec)

        # 蓝方：先用零动作(占位)。如果你想和训练一致，这里也可以加载蓝方 PPO 的 actor。
        for bid in blue_ids:
            ob = torch.as_tensor(obs_dict[bid], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            st = blue_rnn.get(bid, torch.zeros(1, 1, 128, device=DEVICE))
            masks = torch.ones(1, 1, device=DEVICE)
            with torch.no_grad():
                act_tensor, _, st2 = blue_pi.actor(ob, st, masks)
            blue_rnn[bid] = st2
            actions[bid] = act_tensor.squeeze(0).detach().cpu().numpy()

        ordered = [actions[a] for a in (env.ego_ids + env.enm_ids)]
        ret = env.step(ordered)
        if len(ret) == 5:
            _, _, rewards, dones, info = ret
        else:
            _, rewards, dones, info = ret

        # ===== 团队回报（兼容 dict / list / ndarray）=====
        order = env.ego_ids + env.enm_ids
        if isinstance(rewards, dict):
            team_step_rew = sum(float(np.asarray(rewards.get(a, 0.0)).reshape(-1)[0]) for a in red_ids)
        else:
            r_arr = np.asarray(rewards, dtype=float)
            idxs = [order.index(a) for a in red_ids]
            team_step_rew = float(r_arr[idxs].sum())
        ep_ret += team_step_rew

        # 写入 ACMI（每步）
        env.render(mode='txt', filepath=str(acmi_path))
        obs_dict = env.get_obs()
        # 兼容两种 done 表示
        if isinstance(dones, dict):
            done_all = bool(dones.get("__all__", False))
        else:
            done_all = bool(np.asarray(dones).all())
        if done_all:
            print(f"[Episode Return] team={ep_ret:.2f}")
            break

if __name__ == "__main__":
    main()
