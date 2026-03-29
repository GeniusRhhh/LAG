# -*- coding: utf-8 -*-
"""
1v1 POP-T Hybrid（模板 + 连续参数）训练脚本

红方：PPOPolicyHybrid（模板选择 + 连续参数）
蓝方：已有 PPOPolicy（作为对手）
环境：SingleCombatEnv，配置文件
       envs/JSBSim/configs/1v1/ShootMissile/HierarchySelfplay.yaml
"""
import os, sys, numpy as np, torch
from collections import deque
from pathlib import Path
import argparse

# ========= 工程根路径（一定要在用 ROOT 之前定义）=========
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
ROOT = PROJECT_ROOT

# ==== 输出与目录管理 ====
EXP_NAME   = "hybrid_1v1_exp"   # 可改/可做成命令行参数
SAVE_EVERY = 1                  # 每多少个 Iter 存一次 ckpt
BASE_OUT   = os.path.join(
    ROOT, "scripts", "results",
    "SingleCombat", "1v1", "ShootMissile", "HybridRL", EXP_NAME
)

LOG_EVERY_EP   = 1       # 每隔多少局记录一局，例如每10局一条
MAX_LOG_EPIS   = 50       # 最多记录多少个训练episode，防止写爆硬盘
def _next_run_dir(base_dir: str) -> Path:
    os.makedirs(base_dir, exist_ok=True)
    names = [d for d in os.listdir(base_dir) if d.startswith("run")]
    ids = []
    for n in names:
        try:
            ids.append(int(n[3:]))  # 提取数字部分
        except Exception:
            pass
    nxt = (max(ids) + 1) if ids else 1
    run_dir = Path(base_dir) / f"run{nxt}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# ========= 路径 =========
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "Tactical_Rule_Template") not in sys.path:
    sys.path.append(os.path.join(ROOT, "Tactical_Rule_Template"))

# ========= 环境 / 雷达 =========
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.core.radar import RadarModel

# ========= 红方（Hybrid）=========
from algorithms.hybrid.ppo_policy_hybrid import PPOPolicyHybrid
from algorithms.hybrid.rollout_buffer_hybrid import HybridRolloutBuffer
from Tactical_Rule_Template.ruleset.rl_template_driver import (
    list_templates, max_param_dim, param_mask_by_id,
    step_red_agent_pure_rl, reset_agent_state,
    idx2name, reset_team_ctx,
)

# ========= 蓝方（独立 PPO）=========
from config import get_config
from algorithms.ppo.ppo_policy import PPOPolicy  # 工程里的 PPOPolicy（非 MAPPO）

# ----------------- PPO 超参 -----------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA      = 0.99
LAMBDA     = 0.95
LR         = 3e-4
CLIP_EPS   = 0.2
ENT_COEF   = 0.01
VF_COEF    = 0.5
MAX_ITERS  = 2000
ROLLOUT_H  = 10000          # 单回合最大步数上限（通常被环境 max_steps 截断）
BATCH_SIZE = 2048           # 每次更新前采样的红方样本数
MINI_BSZ   = 512
EPOCHS     = 4              # 每次更新的迭代轮数
GRAD_CLIP  = 0.5
PRINT_EVERY= 2
# ----- 奖励日志相关：把每局总奖励存成 .npy，后面画图用 -----
REWARD_LOG_DIR = os.path.join(BASE_OUT, "reward_logs")
os.makedirs(REWARD_LOG_DIR, exist_ok=True)

def init_return_logger(red_ids):
    """为每个红方智能体准备一个 list，用来记录每局总回报。"""
    return {aid: [] for aid in red_ids}

def dump_return_logger(return_logger, out_dir=REWARD_LOG_DIR, prefix="returns_"):
    """
    把每个智能体的每局总回报保存成 returns_<agent>.npy，
    例如 returns_A0100.npy
    """
    os.makedirs(out_dir, exist_ok=True)
    for aid, vals in return_logger.items():
        arr = np.asarray(vals, dtype=np.float32)
        np.save(os.path.join(out_dir, f"{prefix}{aid}.npy"), arr)

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
    reset -> get_obs -> 任选一架的观测长度作为 obs_dim
    """
    env.reset()
    boot_obs = env.get_obs()  # dict: {agent_id: obs_vec}
    any_id = (env.ego_ids + env.enm_ids)[0]
    return len(boot_obs[any_id])

# ----------------- 主程序 -----------------
def main():
    def save_ckpt(it):
        run_dir = _next_run_dir(BASE_OUT)
        torch.save(red_pi.actor.state_dict(), str(run_dir / "actor_latest.pt"))
        torch.save(red_pi.critic.state_dict(), str(run_dir / "critic_latest.pt"))
        print(f"[save to] {run_dir}")

    # 1) 环境 / 雷达（1v1）
    env_cfg = os.path.join(ROOT, "envs/JSBSim/configs/1v1/ShootMissile/HierarchySelfplay.yaml")
    env  = SingleCombatEnv(env_cfg)
    radar= RadarModel()

    # 2) 控制 ID 顺序
    control_ids = (env.ego_ids + env.enm_ids)[:env.num_agents]
    print("[control_ids used by task]:", control_ids)

    # 3) 观测维度
    obs_dim = get_boot_obs_dim(env)

    # 4) 红方 Hybrid 策略（模板头 + 连续参数头）
    #    —— 1v1 中屏蔽双机协同模板（如 GRINDER、PINCER 等）——
    ID2NAME = idx2name()
    DUAL_TEMPLATES = {"GRINDER", "PINCER"}  # 这里按你的双机模板名字补充

    # 仅保留“非双机模板”的全局模板索引
    SINGLE_TPL_IDS = [i for i, name in ID2NAME.items() if name not in DUAL_TEMPLATES]
    SINGLE_TPL_IDS = sorted(SINGLE_TPL_IDS)

    tpl_num = len(SINGLE_TPL_IDS)  # 策略头维度 = 单机模板个数
    z_dim = max_param_dim()
    red_pi = PPOPolicyHybrid(obs_dim=obs_dim, tpl_num=tpl_num, z_dim=z_dim, device=DEVICE)
    red_opt = torch.optim.Adam(
        list(red_pi.actor.parameters()) + list(red_pi.critic.parameters()),
        lr=LR
    )

    # 5) 蓝方（已有 PPO baseline）
    parser = get_config()
    args   = parser.parse_args([])  # 用默认超参构建对手策略
    blue_pi= PPOPolicy(args=args,
                       obs_space=env.observation_space,
                       act_space=env.action_space,
                       device=DEVICE)
    blue_pi.actor.to(DEVICE).eval()
    blue_rnn = {}  # {bid: rnn_state}

    red_ids  = pick_red_ids(env)
    blue_ids = pick_blue_ids(env)
    assert len(red_ids) == 1 and len(blue_ids) == 1, "1v1 场景应为 1 架红机 + 1 架蓝机"

    # 奖励日志：记录每个 episode 的总回报
    return_logger = init_return_logger(red_ids)

    ENV_STEPS_TOTAL = 0      # 环境推进总步数
    global_episodes = 0      # 已完成回合数
    score_meter = deque(maxlen=20)              # 最近 N 局红方总回报
    agent_meters = {aid: deque(maxlen=20) for aid in red_ids}  # 每机回报

    # ====== 训练迭代 ======
    for it in range(1, MAX_ITERS + 1):
        buf = HybridRolloutBuffer(device=DEVICE)
        steps_collected = 0
        ep_return_sum = 0.0
        ep_steps = 0

        # 开新局
        env.reset()
        reset_team_ctx()
        for aid in red_ids:
            reset_agent_state(aid)
        obs_dict = env.get_obs()
        ep_return_agent = {aid: 0.0 for aid in red_ids}

        while steps_collected < BATCH_SIZE:
            # 单回合最多 ROLLOUT_H 步（一般由 env.max_steps 更早截断）
            for t in range(ROLLOUT_H):
                actions = {}
                start = len(buf.rew)  # 当前时间步红方样本写入起点
                # ===== 红方：Hybrid 采样 + 模板执行 =====
                for aid in red_ids:
                    obs_np = np.asarray(obs_dict[aid], dtype=np.float32)
                    with torch.no_grad():
                        # tpl_local 是 [0, tpl_num-1] 的“本地索引”
                        tpl_local, z_vec, logp, v = red_pi.act(obs_np)
                    # 映射到模板库中的“全局模板 ID”
                    tpl_id = SINGLE_TPL_IDS[tpl_local]
                    mask_vec = param_mask_by_id(tpl_id, z_dim)
                    z_vec = z_vec * mask_vec
                    # 用全局 tpl_id 调用模板执行器
                    actions[aid] = step_red_agent_pure_rl(env, radar, aid, tpl_id, z_vec)
                    # buffer 里存的是“本地索引 tpl_local”，
                    # 因为策略头只认识本地编号
                    buf.add_step(obs_np, tpl_local, z_vec, mask_vec, logp, v,
                                 rew=0.0, done=0.0)

                # ===== 蓝方：固定 PPO 对手 =====
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

                # ===== 推进环境 =====
                ordered_actions = [actions[aid] for aid in control_ids]
                ret = env.step(ordered_actions)
                ENV_STEPS_TOTAL += 1

                if len(ret) == 5:
                    _, _, rewards, dones, info = ret
                else:
                    _, rewards, dones, info = ret

                # 统一从环境获取观测
                obs_dict = env.get_obs()

                # ===== 回填红方 reward/done =====
                for j, aid in enumerate(red_ids):
                    r = _scalar(rewards.get(aid, 0.0)) if isinstance(rewards, dict) \
                        else _scalar(rewards[j])

                    if isinstance(dones, dict):
                        d = 1.0 if bool(dones.get(aid, False)) else 0.0
                    elif isinstance(dones, (list, tuple, np.ndarray)):
                        d = 1.0 if bool(np.asarray(dones)[j]) else 0.0
                    else:
                        d = 0.0

                    buf.rew[start + j] = r
                    buf.done[start + j] = d

                    ep_return_sum += r
                    ep_return_agent[aid] += r

                steps_collected += len(red_ids)  # 1v1 时就是 +1
                ep_steps += 1

                # ===== 回合结束判定 =====
                if isinstance(dones, dict):
                    done_all = bool(dones.get("__all__", False))
                elif isinstance(dones, (list, tuple, np.ndarray)):
                    done_all = bool(np.all(dones))
                else:
                    done_all = bool(dones)

                if done_all:
                    score_meter.append(ep_return_sum)
                    for aid in red_ids:
                        agent_meters[aid].append(ep_return_agent[aid])
                        # 记录这一局该智能体的总奖励
                        return_logger[aid].append(ep_return_agent[aid])

                    global_episodes += 1

                    # 清回合累计 & 开新局
                    ep_return_sum = 0.0
                    ep_return_agent = {aid: 0.0 for aid in red_ids}
                    blue_rnn.clear()
                    env.reset()
                    reset_team_ctx()
                    for aid in red_ids:
                        reset_agent_state(aid)
                    obs_dict = env.get_obs()
                    break

                elif steps_collected >= BATCH_SIZE:
                    # 仅样本量够了就退出，保留当前回合状态到下个 iter
                    break

        # ===== Bootstrap 值（最后一个样本的状态）=====
        last_obs = buf.obs[-1]
        with torch.no_grad():
            v_last = float(
                red_pi.critic(
                    torch.as_tensor(last_obs[None],
                                    dtype=torch.float32,
                                    device=DEVICE)
                ).item()
            )

        # ===== 张量化 & GAE =====
        obs_t, tpl_t, z_t, m_t, logp_old_t, v_t, rew_t, done_t = buf.as_tensors()
        vals_np = np.concatenate(
            [v_t.cpu().numpy(), np.array([v_last], dtype=np.float32)],
            axis=0
        )
        rews_np = rew_t.cpu().numpy()
        done_np = done_t.cpu().numpy()
        adv_np, ret_np = compute_gae(rews_np, done_np, vals_np)
        adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=DEVICE)
        ret_t = torch.as_tensor(ret_np, dtype=torch.float32, device=DEVICE)

        # 优势标准化
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ===== PPO 更新（红方 Hybrid）=====
        N = obs_t.shape[0]
        pg_acc = 0.0; vf_acc = 0.0; ent_acc = 0.0; kl_acc = 0.0; cf_acc = 0.0; mb_count = 0

        for _ in range(EPOCHS):
            idx = np.random.permutation(N)
            for s in range(0, N, MINI_BSZ):
                b = idx[s:s + MINI_BSZ]
                logp_new, ent, v_pred = red_pi.evaluate_actions(
                    obs_t[b], tpl_t[b], z_t[b]
                )
                ratio = torch.exp(logp_new - logp_old_t[b])
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t[b]
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * (ret_t[b] - v_pred).pow(2).mean()

                approx_kl = (logp_old_t[b] - logp_new).mean()
                clipfrac = (torch.abs(ratio - 1.0) > CLIP_EPS).float().mean()

                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * ent.mean()
                red_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(red_pi.parameters(), GRAD_CLIP)
                red_opt.step()

                pg_acc += pg_loss.item()
                vf_acc += v_loss.item()
                ent_acc += ent.mean().item()
                kl_acc += approx_kl.item()
                cf_acc += clipfrac.item()
                mb_count += 1

        # ===== 统计与打印 =====
        with torch.no_grad():
            v_all = red_pi.critic(obs_t).squeeze(-1)
            diff = (ret_t - v_all).cpu().numpy()
            ret_np_all = ret_t.cpu().numpy()
            ev = 1.0 - (diff.var() / (ret_np_all.var() + 1e-8))

        pg_mean = pg_acc / max(1, mb_count)
        vf_mean = vf_acc / max(1, mb_count)
        ent_mean = ent_acc / max(1, mb_count)
        kl_mean = kl_acc / max(1, mb_count)
        cf_mean = cf_acc / max(1, mb_count)
        avg_ep = float(np.mean(score_meter)) if len(score_meter) else 0.0
        per_agent = " ".join(
            [f"{aid}={float(np.mean(agent_meters[aid])):.2f}"
             for aid in red_ids if len(agent_meters[aid])]
        )
        batch_mean_rew = float(rew_t.mean().item()) if rew_t.numel() > 0 else 0.0

        if it % PRINT_EVERY == 0:
            print(
                f"[Iter {it:04d}] episodes={global_episodes} env_steps={ENV_STEPS_TOTAL} "
                f"avg_ep_ret(last{score_meter.maxlen})={avg_ep:.2f}  "
                f"per-agent(last{score_meter.maxlen}): {per_agent}  "
                f"pg={pg_mean:.4f} vf={vf_mean:.4f} ent={ent_mean:.4f} "
                f"kl={kl_mean:.4f} clipfrac={cf_mean:.3f} EV={ev:.3f} "
                f"batch_mean_rew={batch_mean_rew:.2f}"
            )
            # 训练结束后，把奖励曲线数据保存下来
            dump_return_logger(return_logger)

        if it % SAVE_EVERY == 0:
            save_ckpt(it)


if __name__ == "__main__":
    main()
