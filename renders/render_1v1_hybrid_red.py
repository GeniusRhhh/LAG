# -*- coding: utf-8 -*-
"""
1v1 POP-T Hybrid 推理脚本（写入 Tacview ACMI）

红方：PPOPolicyHybrid（模板选择 + 连续参数）
蓝方：已有 PPOPolicy（baseline 对手）
环境：SingleCombatEnv，配置文件：
       envs/JSBSim/configs/1v1/ShootMissile/HierarchySelfplay.yaml
"""
import os, sys, torch, numpy as np
from pathlib import Path
from config import get_config
from algorithms.ppo.ppo_policy import PPOPolicy

# 工程根目录
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.core.radar import RadarModel
from algorithms.hybrid.ppo_policy_hybrid import PPOPolicyHybrid
from Tactical_Rule_Template.ruleset.rl_template_driver import (
    list_templates, max_param_dim, param_mask_by_id,
    step_red_agent_pure_rl, reset_agent_state,
    idx2name,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # ===== 1) 指向训练好的 1v1 run 目录 =====
    # ⚠ 这里的 run 号记得改成你想看的那个，比如 run8 / run9
    run_dir = Path(ROOT) / "scripts/results/SingleCombat/1v1/ShootMissile/HybridRL/hybrid_1v1_exp/run100"
    actor_path = run_dir / "actor_latest.pt"
    acmi_path = run_dir / "render_1v1_hybrid.acmi"  # ACMI 输出文件

    # ===== 2) 环境 & 雷达 =====
    env_cfg = os.path.join(
        ROOT, "envs/JSBSim/configs/1v1/ShootMissile/HierarchySelfplay.yaml"
    )
    env = SingleCombatEnv(env_cfg)
    radar = RadarModel()

    env.reset()
    # reset 后先写一帧（和你以前的脚本一致）
    env.render(mode="txt", filepath=str(acmi_path))

    # ===== 3) 模板映射：必须和 1v1 训练脚本完全一致 =====
    ID2NAME = idx2name()
    DUAL_TEMPLATES = {"GRINDER", "PINCER"}  # 1v1 中禁用的双机模板

    # 仅保留“非双机模板”的全局模板 ID（注意要排序，保持索引稳定）
    SINGLE_TPL_IDS = [i for i, name in ID2NAME.items() if name not in DUAL_TEMPLATES]
    SINGLE_TPL_IDS = sorted(SINGLE_TPL_IDS)

    boot_obs = env.get_obs()
    any_id = (env.ego_ids + env.enm_ids)[0]
    obs_dim = len(boot_obs[any_id])
    tpl_num = len(SINGLE_TPL_IDS)
    z_dim = max_param_dim()

    # ===== 4) 构建 Hybrid 策略并加载 1v1 训练得到的权重 =====
    pi = PPOPolicyHybrid(
        obs_dim=obs_dim,
        tpl_num=tpl_num,
        z_dim=z_dim,
        device=DEVICE,
    )
    pi.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE, weights_only=False))
    pi.actor.to(DEVICE).eval()
    pi.critic.to(DEVICE).eval()

    # ===== 5) 蓝方 PPO baseline（和训练脚本保持一致） =====
    parser = get_config()
    args = parser.parse_args([])  # 用默认/占位超参即可构建同款网络
    blue_pi = PPOPolicy(
        args=args,
        obs_space=env.observation_space,
        act_space=env.action_space,
        device=DEVICE,
    )
    blue_pi.actor.to(DEVICE).eval()
    blue_rnn = {}  # {bid: rnn_state}

    red_ids, blue_ids = env.ego_ids, env.enm_ids
    assert len(red_ids) == 1 and len(blue_ids) == 1, "1v1 场景应为 1 红 + 1 蓝"

    # 重置红方模板内部状态机
    for aid in red_ids:
        reset_agent_state(aid)

    obs_dict = env.get_obs()
    ep_ret = 0.0

    while True:
        actions = {}

        # ===== 红方：模板选择 + 连续参数（和 1v1 训练脚本完全同构）=====
        for aid in red_ids:
            ob = np.asarray(obs_dict[aid], dtype=np.float32)
            with torch.no_grad():
                # 这里返回的是“本地索引 tpl_local”，范围是 [0, tpl_num-1]
                tpl_local, z_vec, _, _ = pi.act(ob)
            # 映射为全局模板 ID
            tpl_id = SINGLE_TPL_IDS[tpl_local]
            mask_vec = param_mask_by_id(tpl_id, z_dim)
            z_vec = z_vec * mask_vec
            # 用全局 tpl_id 调用模板执行器
            actions[aid] = step_red_agent_pure_rl(env, radar, aid, tpl_id, z_vec)

        # ===== 蓝方：PPO baseline 对手 =====
        for bid in blue_ids:
            ob = torch.as_tensor(
                obs_dict[bid], dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)
            st = blue_rnn.get(bid, torch.zeros(1, 1, 128, device=DEVICE))
            masks = torch.ones(1, 1, device=DEVICE)
            with torch.no_grad():
                act_tensor, _, st2 = blue_pi.actor(ob, st, masks)
            blue_rnn[bid] = st2
            actions[bid] = act_tensor.squeeze(0).detach().cpu().numpy()

        # ===== 推进环境，动作顺序按 ego_ids + enm_ids =====
        ordered = [actions[a] for a in (env.ego_ids + env.enm_ids)]
        ret = env.step(ordered)

        if len(ret) == 5:
            _, _, rewards, dones, info = ret
        else:
            _, rewards, dones, info = ret

        # ===== 累计红方回报（单机，简单取 A0100 的 reward）=====
        if isinstance(rewards, dict):
            r_step = float(np.asarray(rewards.get(red_ids[0], 0.0)).reshape(-1)[0])
        else:
            r_arr = np.asarray(rewards, dtype=float).reshape(-1)
            # SingleCombat 下，一般顺序就是 [A0100, B0100]
            r_step = float(r_arr[0])
        ep_ret += r_step

        # 写一帧 ACMI
        env.render(mode="txt", filepath=str(acmi_path))

        # 更新观测
        obs_dict = env.get_obs()

        # ===== 终止判定 =====
        if isinstance(dones, dict):
            done_all = bool(dones.get("__all__", False))
        else:
            done_all = bool(np.asarray(dones).all())

        if done_all:
            print(f"[Episode Return] red={ep_ret:.2f}")
            break

if __name__ == "__main__":
    main()