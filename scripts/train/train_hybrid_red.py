# -*- coding: utf-8 -*-
# scripts/train/train_hybrid_red.py
import os, sys, numpy as np, torch
from collections import deque
from pathlib import Path
import argparse
from Tactical_Rule_Template.ruleset.rl_template_driver import idx2name, name2idx, _SCHED
from Tactical_Rule_Template.ruleset.rl_template_driver import reset_team_ctx
from algorithms.hybrid.ppo_policy_hybrid import USE_CC_SVB #是否使用CC-SVB算法
from envs.JSBSim.reward_functions.ccc_reward import apply_ccc_reward
from envs.JSBSim.reward_functions.ccc_reward import get_ccc_stats

USE_CCC_RSHAPE = True # 是否启用协同奖励塑形

# ========= 工程根路径（一定要在用 ROOT 之前定义）=========
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
ROOT = PROJECT_ROOT

# ==== 输出与目录管理（只保留这一套）====
EXP_NAME  = "hybrid_red_exp"   # 可改/可做成命令行参数
SAVE_EVERY = 5                # 每多少个 Iter 存一次 ckpt
BASE_OUT = os.path.join(
    ROOT, "scripts", "results", "MultipleCombat", "2v2", "ShootMissile", "HybridRL", EXP_NAME
)

def _next_run_dir(base_dir: str) -> Path:
    os.makedirs(base_dir, exist_ok=True)
    names = [d for d in os.listdir(base_dir) if d.startswith("run")]
    ids = []
    for n in names:
        try:
            ids.append(int(n[3:]))  # 提取数字部分
        except:
            pass
    nxt = (max(ids) + 1) if ids else 1  # 如果存在文件夹，则递增
    run_dir = Path(base_dir) / f"run{nxt}"  # 使用 Path 拼接路径
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# ========= 路径 =========
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# 可避免相对导入问题
if os.path.join(ROOT, "Tactical_Rule_Template") not in sys.path:
    sys.path.append(os.path.join(ROOT, "Tactical_Rule_Template"))

# ========= 环境 / 雷达 =========
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.radar import RadarModel

# ========= 红方（Hybrid）=========
from algorithms.hybrid.ppo_policy_hybrid import PPOPolicyHybrid
from algorithms.hybrid.rollout_buffer_hybrid import HybridRolloutBuffer
from Tactical_Rule_Template.ruleset.rl_template_driver import (
    list_templates, max_param_dim, param_mask_by_id,
    step_red_agent_pure_rl, reset_agent_state,
)

# ========= 蓝方（独立 PPO，对齐 test_2.py 的用法）=========
from config import get_config
from algorithms.ppo.ppo_policy import PPOPolicy  # 你工程里的 PPOPolicy（非 MAPPO）

# ----------------- 超参 -----------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA      = 0.99
LAMBDA     = 0.95
LR         = 3e-4
CLIP_EPS   = 0.2
ENT_COEF   = 0.01
VF_COEF    = 0.5
MAX_ITERS  = 2000
ROLLOUT_H  = 10000          # 每局最大步数上限（用不上），每局仿真步数大于它时才会被截断
BATCH_SIZE = 4096        # 每次更新目标采样量（样本=时间步×红方机数）
MINI_BSZ   = 512
EPOCHS     = 4
GRAD_CLIP  = 0.5
PRINT_EVERY= 10

# ----------------- 小工具 -----------------
def compute_gae(rews, dones, vals, gamma=GAMMA, lam=LAMBDA):
    """
    GAE（按扁平时间序列计算）：vals 需要比 rews 多 1 个 bootstrap 值
    rews: [T], dones: [T], vals: [T+1]
    """
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rews[t] + gamma * vals[t+1] * nonterminal - vals[t]
        gae   = delta + gamma * lam * nonterminal * gae
        adv[t]= gae
    ret = adv + vals[:-1]
    return adv, ret

def _scalar(x):
    a = np.asarray(x)
    return float(a.reshape(-1)[0]) if a.size else 0.0

def pick_red_ids(env):
    if hasattr(env, "ego_ids") and env.ego_ids:
        return list(env.ego_ids)
    return [aid for aid in env.agents.keys() if str(aid).startswith("A")]

def pick_blue_ids(env):
    if hasattr(env, "enm_ids") and env.enm_ids:
        return list(env.enm_ids)
    return [aid for aid in env.agents.keys() if str(aid).startswith("B")]

def get_boot_obs_dim(env):
    """
    与 test_2.py 一致：reset -> get_obs -> 任选一架的观测长度作为 obs_dim
    """
    env.reset()
    boot_obs = env.get_obs()  # dict: {agent_id: obs_vec}
    any_id = (env.ego_ids + env.enm_ids)[0]
    return len(boot_obs[any_id])

# ----------------- 主程序 -----------------
def main():
    def save_ckpt(it):
        run_dir = _next_run_dir(BASE_OUT)  # 调用 _next_run_dir 创建新文件夹
        torch.save(red_pi.actor.state_dict(), str(run_dir / "actor_latest.pt"))
        torch.save(red_pi.critic.state_dict(), str(run_dir / "critic_latest.pt"))
        print(f"[save to] {run_dir}")  # 输出保存路径

    # 1) 环境 / 雷达（2v2）
    env_cfg = os.path.join(ROOT, "envs/JSBSim/configs/2v2/ShootMissile/HierarchySelfplay.yaml")
    env  = MultipleCombatEnv(env_cfg)
    radar= RadarModel()

    # 2) 参与控制的 ID 顺序（与 test_2.py 的思想一致）
    control_ids = (env.ego_ids + env.enm_ids)[:env.num_agents]
    print("[control_ids used by task]:", control_ids)

    # 3) 观测维度（与 test_2.py 一致的获取方式）
    obs_dim = get_boot_obs_dim(env)

    # 4) 红方 Hybrid 策略（模板头 + 连续参数头）
    tpl_num  = len(list_templates())
    z_dim    = max_param_dim()
    red_pi   = PPOPolicyHybrid(obs_dim=obs_dim, tpl_num=tpl_num, z_dim=z_dim, device=DEVICE)
    red_opt  = torch.optim.Adam(list(red_pi.actor.parameters()) + list(red_pi.critic.parameters()), lr=LR)

    # 5) 蓝方（独立 PPO，对齐 test_2.py）
    parser = get_config()
    args   = parser.parse_args([])  # 用默认/占位超参就能构建出和工程一致的策略网络
    blue_pi= PPOPolicy(args=args, obs_space=env.observation_space, act_space=env.action_space, device=DEVICE)
    blue_pi.actor.to(DEVICE).eval()
    blue_rnn = {}  # {bid: rnn_state}

    red_ids  = pick_red_ids(env)
    blue_ids = pick_blue_ids(env)
    assert len(red_ids)>0 and len(blue_ids)>0, "请确认 env.ego_ids / env.enm_ids 是否正确初始化"

    ENV_STEPS_TOTAL = 0  # 环境实际推进了多少步（一次 env.step 就 +1）
    global_episodes = 0  # 已完成多少个 episode（回合）
    score_meter = deque(maxlen=20)  # 最近N局的总回报均值，用来观察学习是否抬头
    # —— 每个红方的回合回报滑窗（最近20回合）
    agent_meters = {aid: deque(maxlen=20) for aid in red_ids}

    DUAL_SET = {"GRINDER", "PINCER"}
    ID2NAME = idx2name()
    NAME2ID = name2idx()

    # ====== 训练迭代 ======
    for it in range(1, MAX_ITERS+1):
        # 每次迭代收集 ~BATCH_SIZE 样本（按红方样本数计）
        buf = HybridRolloutBuffer(device=DEVICE)
        steps_collected = 0
        ep_return_sum = 0.0
        ep_steps = 0

        # 开新局
        env.reset()
        reset_team_ctx()  # <—— 清掉 A/B 队的共享状态机
        # 清红方模板上下文（护栏的驻留/冷却不能跨局）
        for aid in red_ids:
            reset_agent_state(aid)
        obs_dict = env.get_obs()
        # —— 本回合每机回报累加器（团队你已用 ep_return_sum 累加了）
        ep_return_agent = {aid: 0.0 for aid in red_ids}

        while steps_collected < BATCH_SIZE:
            # 一个 episode 内最多 ROLLOUT_H 步（稳妥起见）
            for t in range(ROLLOUT_H):
                actions = {}
                #—— 红方：Hybrid 采样 + 掩码 + 桥接调用执行模板 —— #
                start = len(buf.rew)  # 记录本拍红方写入的起点，用于稳妥回填
                for aid in red_ids:
                    obs_np = np.asarray(obs_dict[aid], dtype=np.float32)
                    with torch.no_grad():
                        tpl_id, z_vec, logp, v = red_pi.act(obs_np)   # 你的 Hybrid actor：返回模板ID、连续参数、logp、V
                    mask_vec = param_mask_by_id(tpl_id, z_dim)
                    z_vec = z_vec * mask_vec  # 显式屏蔽无效维

                    # 通过桥接层，把 (模板ID, z) → YAML 覆盖 → 执行模板 → 高层动作 [alt, heading, vel, shoot]
                    actions[aid] = step_red_agent_pure_rl(env, radar, aid, tpl_id, z_vec)

                    # 占位写入（reward/done 稍后回填）
                    buf.add_step(obs_np, tpl_id, z_vec, mask_vec, logp, v, rew=0.0, done=0.0)

                # # ------- 先收集红方两机的提议 -------
                # st, obs_cache = {}, {}
                # start = len(buf.rew)  # 记住红方写入起点（后面回填 reward 用）
                # for aid in red_ids:
                #     if aid not in _SCHED:  # 确保初始化
                #         _SCHED[aid] = {
                #             "tpl_id": None,  # 默认模板ID
                #             "tpl": None,  # 默认模板名称
                #             "steps_on_tpl": 0,  # 当前模板上驻留的步数
                #             "cooldown": 0,  # 切换后的冷却时间
                #             "proposed_hist": [],  # 提议模板的历史记录，用于迟滞一致性
                #         }
                #
                #     obs_np = np.asarray(obs_dict[aid], dtype=np.float32)
                #     obs_cache[aid] = obs_np
                #     with torch.no_grad():
                #         tpl_id, z_vec, logp, v = red_pi.act(obs_np)
                #     # 这里的st现在从_SCHED获取当前执行的模板
                #     st[aid] = {
                #         "tpl_id": _SCHED[aid]["tpl_id"],  # 当前执行的模板ID
                #         "tpl_name": _SCHED[aid]["tpl"],  # 当前执行的模板名称
                #         "z": z_vec,  # 参数向量（这里是提议的z_vec，但你也可以更新为实际执行模板的参数）
                #         "logp": float(logp),  # 提议的logp
                #         "v": float(v),  # 提议的v
                #     }
                #
                # # ------- 队内仲裁：按你的3条规则 -------
                # alive_red = [aid for aid in red_ids if getattr(env.agents[aid], "is_alive", True)]
                # #alive_red = red_ids
                # forced_tpl_name = None
                # if len(alive_red) >= 2:
                #     a0, a1 = alive_red[0], alive_red[1]
                #     n0, n1 = st[a0]["tpl_name"], st[a1]["tpl_name"]
                #     d0, d1 = (n0 in DUAL_SET), (n1 in DUAL_SET)
                #     if d0 ^ d1:
                #         forced_tpl_name = n0 if d0 else n1
                #     elif d0 and d1 and n0 != n1:
                #         forced_tpl_name = n0 if st[a0]["logp"] >= st[a1]["logp"] else n1
                #     # else: 两机都不是双机 -> 不强制
                #
                # # ------- 【仅第一回合】打印每步模板选择 -------
                # if global_episodes == 0:  # 第一回合尚未结束时 global_episodes==0
                #     if ep_steps == 0:
                #         print("\n[Episode 1 trace] step, agent, proposed -> executed  (forced if any)")
                #     for aid in red_ids:
                #         proposed = st[aid]["tpl_name"]
                #         executed = get_current_tpl(aid) or "None"
                #         forced_s = " (forced)" if (forced_tpl_name is not None and executed == forced_tpl_name) else ""
                #         print(f"[t={ep_steps:04d}] {aid}: {proposed} -> {executed}{forced_s} {forced_tpl_name}")
                #
                # # ------- 下发动作（必要时强制） -------
                # for aid in red_ids:
                #     tpl_id_final = st[aid]["tpl_id"]
                #     z_vec = st[aid]["z"]
                #     if forced_tpl_name is not None:
                #         tpl_id_final = NAME2ID[forced_tpl_name]
                #     # 用“最终tpl_id”生成掩码并重置 z
                #     mask_vec = param_mask_by_id(tpl_id_final, z_dim)
                #     z_final = z_vec * mask_vec
                #     # 调用执行器
                #     actions[aid] = step_red_agent_pure_rl(
                #         env, radar, aid, tpl_id_final, z_final,
                #         forced_tpl_name=forced_tpl_name if forced_tpl_name else None
                #     )
                #     # 关键：用“最终动作”重算 logp 与 V，再入 buffer
                #     with torch.no_grad():
                #         obs_t = torch.as_tensor(obs_cache[aid][None], dtype=torch.float32, device=DEVICE)
                #         tpl_t = torch.as_tensor([tpl_id_final], dtype=torch.long, device=DEVICE)
                #         z_t = torch.as_tensor(z_final[None], dtype=torch.float32, device=DEVICE)
                #         logp_new, ent_dummy, v_pred = red_pi.evaluate_actions(obs_t, tpl_t, z_t)
                #     buf.add_step(obs_cache[aid], tpl_id_final, z_final, mask_vec,
                #                  float(logp_new.squeeze().cpu().item()),
                #                  float(v_pred.squeeze().cpu().item()),
                #                  rew=0.0, done=0.0)

                # —— 蓝方：独立 PPO（对齐 test_2.py） —— #
                for bid in blue_ids:
                    ob = torch.as_tensor(obs_dict[bid], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    st = blue_rnn.get(bid, torch.zeros(1, 1, 128, device=DEVICE))
                    masks = torch.ones(1, 1, device=DEVICE)
                    with torch.no_grad():
                        act_tensor, _, st2 = blue_pi.actor(ob, st, masks)
                    blue_rnn[bid] = st2
                    actions[bid] = act_tensor.squeeze(0).detach().cpu().numpy()

                # —— 打包顺序与 control_ids 一致（关键！） —— #
                ordered_actions = [actions[aid] for aid in control_ids]

                # 推进环境（对齐 test_2.py 的返回判定）
                ret = env.step(ordered_actions)
                ENV_STEPS_TOTAL += 1  # 每推进一步环境就+1
                if len(ret) == 5:
                    _, _, rewards, dones, info = ret
                else:
                    _, rewards, dones, info = ret

                # ===== 在这里加入协同奖励形状（方案 B：只在训练脚本里改红方奖励）=====
                if USE_CCC_RSHAPE:
                    # 假设 rewards 是“按 agent_id 的 dict”，与 altitude_reward 那类奖励保持一致风格
                    rewards = apply_ccc_reward(
                        env=env,  # 按函数定义传 env，就算内部暂时没用也要传
                        rew_dict=rewards,  # 原来的 reward 字典
                        red_ids=red_ids,  # ['A0100', 'A0200']
                    )

                # 关键：统一从环境取“按 agent_id 索引”的观测
                obs_dict = env.get_obs()

                # —— 回填红方 reward/done —— #
                # 你的环境通常返回 dict：rewards[aid], dones[aid] 或 "__all__"
                for j, aid in enumerate(red_ids):
                    r = _scalar(rewards.get(aid, 0.0)) if isinstance(rewards, dict) else _scalar(rewards[j])

                    if isinstance(dones, dict):
                        d = 1.0 if bool(dones.get(aid, False)) else 0.0
                    elif isinstance(dones, (list, tuple, np.ndarray)):
                        d = 1.0 if bool(np.asarray(dones)[j]) else 0.0
                    else:
                        d = 0.0

                    buf.rew[start + j] = r
                    buf.done[start + j] = d

                    # —— 累加团队与每机回报 —— #
                    ep_return_sum += r
                    ep_return_agent[aid] += r

                steps_collected += len(red_ids)
                ep_steps += 1

                # 收尾判定（对齐 test_2.py）
                done_all = False
                if isinstance(dones, dict):
                    done_all = bool(dones.get("__all__", False))
                elif isinstance(dones, (list, tuple, np.ndarray)):
                    done_all = bool(np.all(dones))
                else:
                    done_all = bool(dones)

                # if done_all or steps_collected >= BATCH_SIZE:
                #     # —— 写入本回合统计 —— #
                #     score_meter.append(ep_return_sum)  # 团队回报（红方两机求和）
                #     for aid in red_ids:
                #         agent_meters[aid].append(ep_return_agent[aid])  # 每机回报
                #     global_episodes += 1
                #
                #     # —— 重置回合累加器 —— #
                #     ep_return_sum = 0.0
                #     ep_return_agent = {aid: 0.0 for aid in red_ids}
                #
                #     # —— 开新局（保持你原有流程） —— #
                #     blue_rnn.clear()
                #     env.reset()
                #     for aid in red_ids:
                #         reset_agent_state(aid)
                #     obs_dict = env.get_obs()
                #     break
                # —— 判定：自然结束 vs. 批次到界 —— #
                if done_all:
                    # —— 只在自然结束时：记回合、清回合累计、reset 环境/护栏 —— #
                    score_meter.append(ep_return_sum)  # 团队回报
                    for aid in red_ids:
                        agent_meters[aid].append(ep_return_agent[aid])  # 每机回报
                    global_episodes += 1
                    print(f"End of episode {global_episodes}, total reward: {ep_return_sum}")
                    # 清回合累计
                    ep_return_sum = 0.0
                    ep_return_agent = {aid: 0.0 for aid in red_ids}
                    # 开新局（自然结束才 reset）
                    blue_rnn.clear()
                    env.reset()
                    reset_team_ctx()
                    for aid in red_ids:
                        reset_agent_state(aid)
                    obs_dict = env.get_obs()
                    break  # 退出当前 inner for，到 while 外面；若样本已够会去更新

                elif steps_collected >= BATCH_SIZE:
                    # —— 仅批次到界：不计回合、不清累计、不 reset；直接去做更新 —— #
                    break

        # ===== Bootstrap 值（对齐到最后一个样本的 obs）=====
        last_obs = buf.obs[-1]
        with torch.no_grad():
            v_last = float(red_pi.critic(torch.as_tensor(last_obs[None], dtype=torch.float32, device=DEVICE)).item())

        # ===== 张量化 & GAE =====
        obs_t, tpl_t, z_t, m_t, logp_old_t, v_t, rew_t, done_t = buf.as_tensors()
        vals_np = np.concatenate([v_t.cpu().numpy(), np.array([v_last], dtype=np.float32)], axis=0)
        rews_np = rew_t.cpu().numpy()
        done_np = done_t.cpu().numpy()
        adv_np, ret_np = compute_gae(rews_np, done_np, vals_np)
        adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=DEVICE)
        ret_t = torch.as_tensor(ret_np, dtype=torch.float32, device=DEVICE)
        # === 这里插入 CC-SVB 开关（仅替换 advantage），否则走原逻辑 ===
        if USE_CC_SVB:
            with torch.no_grad():
                # 取整批观测的团队价值和个体基线头（需在 ppo_policy_hybrid.py 里实现 value_heads）
                v_team_all, b_heads_all = red_pi.value_heads(obs_t.cpu().numpy())  # v:(N,), b:(N,2)
                v_team_all = v_team_all.to(DEVICE)
                b_heads_all = b_heads_all.to(DEVICE)
            N = obs_t.shape[0]
            # 如果样本严格按 A0100/A0200 交替写入，可用模2槽位；否则用你自己记录的 slot_id 来索引
            slots = torch.arange(N, device=DEVICE) % 2  # 0/1
            b_indiv = b_heads_all[torch.arange(N, device=DEVICE), slots]  # (N,)
            # 用个体基线替换优势
            adv_t = (ret_t - b_indiv).detach()

        # 优势标准化
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        # ===== PPO 更新（红方 Hybrid）=====
        N = obs_t.shape[0]
        # a) 累加器（本次迭代的平均损失与指标）
        pg_acc = 0.0;vf_acc = 0.0;ent_acc = 0.0;kl_acc = 0.0;cf_acc = 0.0;mb_count = 0
        for _ in range(EPOCHS):
            idx = np.random.permutation(N)
            for s in range(0, N, MINI_BSZ):
                b = idx[s:s + MINI_BSZ]
                # 新策略下 logp/熵/值
                logp_new, ent, v_pred = red_pi.evaluate_actions(
                    obs_t[b], tpl_t[b], z_t[b]
                )
                ratio = torch.exp(logp_new - logp_old_t[b])
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t[b]
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * (ret_t[b] - v_pred).pow(2).mean()
                # 额外统计
                approx_kl = (logp_old_t[b] - logp_new).mean()
                clipfrac = (torch.abs(ratio - 1.0) > CLIP_EPS).float().mean()
                # 总损失（注意熵项是减号）
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * ent.mean()
                red_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(red_pi.parameters(), GRAD_CLIP)
                red_opt.step()

                # 累加到本迭代统计
                pg_acc += pg_loss.item()
                vf_acc += v_loss.item()
                ent_acc += ent.mean().item()
                kl_acc += approx_kl.item()
                cf_acc += clipfrac.item()
                mb_count += 1

        # b) 计算 value 的 explained variance（用整个 buffer）
        with torch.no_grad():
            v_all = red_pi.critic(obs_t).squeeze(-1)
            diff = (ret_t - v_all).cpu().numpy()
            ret_np_all = ret_t.cpu().numpy()
            ev = 1.0 - (diff.var() / (ret_np_all.var() + 1e-8))

        # c) 平均并打印
        pg_mean = pg_acc / max(1, mb_count)
        vf_mean = vf_acc / max(1, mb_count)
        ent_mean = ent_acc / max(1, mb_count)
        kl_mean = kl_acc / max(1, mb_count)
        cf_mean = cf_acc / max(1, mb_count)
        avg_team = float(np.mean(score_meter)) if len(score_meter) else 0.0
        per_agent = " ".join([f"{aid}={float(np.mean(agent_meters[aid])):.2f}"
                              for aid in red_ids if len(agent_meters[aid])])

        # 本批次“步均奖励”（看采样质量）
        batch_mean_rew = float(rew_t.mean().item()) if rew_t.numel() > 0 else 0.0
        print(
            f"[Iter {it:04d}] episodes={global_episodes} env_steps={ENV_STEPS_TOTAL} "
            f"avg_ep_ret_team(last{score_meter.maxlen})={avg_team:.2f}  "
            f"per-agent(last{score_meter.maxlen}): {per_agent}  "
            f"pg={pg_mean:.4f} vf={vf_mean:.4f} ent={ent_mean:.4f} "
            f"kl={kl_mean:.4f} clipfrac={cf_mean:.3f} EV={ev:.3f} "
            f"batch_mean_rew={batch_mean_rew:.2f}"
        )
        # —— 打印协同成功率（如果开了 CCC 奖励）——
        if USE_CCC_RSHAPE:
            succ, total, rate = get_ccc_stats(reset=True)
            print(f"[CCC] success={succ} / total={total}, rate={rate:.3f}")
        if it % SAVE_EVERY == 0:
            save_ckpt(it)

if __name__ == "__main__":
    main()