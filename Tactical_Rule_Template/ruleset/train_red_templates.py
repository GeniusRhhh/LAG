# ruleset/train_red_templates.py
import os, sys, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- PATH 修正：保证能 import 到 env 和 ruleset ---
_CUR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_CUR, ".."))          # LAG 根
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
_TAC  = os.path.join(_ROOT, "Tactical_Rule_Template")
if _TAC not in sys.path: sys.path.insert(0, _TAC)

# --- 环境 & 桥接层 ---
from envs.JSBSim.envs import MultipleCombatEnv
from rl_template_driver import list_templates, max_param_dim, step_red_agent_pure_rl
from rl_template_driver import reset_agent_state   # 训练每局前清 ctx
from envs.JSBSim.core.radar import RadarModel  # 你已有的雷达构造器

# --- 策略头（模板ID + 连续参数 z + 值函数） ---
# 若你已有 policy_template_head.py，就用它；没有就用这个最小实现
class TemplatePolicy(nn.Module):
    def __init__(self, obs_dim, tpl_num, z_dim, hidden=256):
        super().__init__()
        self.obs_dim, self.tpl_num, self.z_dim = obs_dim, tpl_num, z_dim
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.pi_id   = nn.Linear(hidden, tpl_num)        # 模板ID -> 分类
        self.pi_mu   = nn.Linear(hidden, z_dim)          # 连续参数均值
        self.pi_logstd = nn.Parameter(torch.zeros(z_dim))# 连续参数对数方差
        self.v       = nn.Linear(hidden, 1)

    def forward(self, obs_t):
        x = self.backbone(obs_t)
        logits = self.pi_id(x)
        mu     = self.pi_mu(x)
        logstd = self.pi_logstd.expand_as(mu)
        v      = self.v(x).squeeze(-1)
        return logits, mu, logstd, v

    def act(self, obs_np):
        self.eval()
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)  # [1,obs]
        logits, mu, logstd, v = self.forward(obs_t)
        # 模板ID
        id_dist = torch.distributions.Categorical(logits=logits)
        tpl_id  = id_dist.sample()                 # [1]
        id_logp = id_dist.log_prob(tpl_id)
        # 连续 z
        std  = torch.exp(logstd)
        z_dist = torch.distributions.Normal(mu, std)
        z     = z_dist.sample()                    # [1,z_dim]
        z_logp= z_dist.log_prob(z).sum(-1)
        # 打包
        a = (tpl_id.item(), z.squeeze(0).detach().cpu().numpy())
        logp = (id_logp.item(), z_logp.item())
        val  = v.item()
        return a, logp, val

    def evaluate_actions(self, obs_t, tpl_id_t, z_t):
        logits, mu, logstd, v = self.forward(obs_t)
        id_dist = torch.distributions.Categorical(logits=logits)
        id_logp = id_dist.log_prob(tpl_id_t)
        id_ent  = id_dist.entropy()
        std     = torch.exp(logstd)
        z_dist  = torch.distributions.Normal(mu, std)
        z_logp  = z_dist.log_prob(z_t).sum(-1)
        z_ent   = z_dist.entropy().sum(-1)
        return id_logp, z_logp, id_ent + z_ent, v

# --- 简单 Rollout 缓存 ---
class Rollout:
    def __init__(self, T, obs_dim, z_dim):
        self.obs = np.zeros((T, obs_dim), np.float32)
        self.tpl_id = np.zeros((T,), np.int64)
        self.z    = np.zeros((T, z_dim), np.float32)
        self.id_logp = np.zeros((T,), np.float32)
        self.z_logp  = np.zeros((T,), np.float32)
        self.rew = np.zeros((T,), np.float32)
        self.val = np.zeros((T,), np.float32)
        self.done= np.zeros((T,), np.float32)
        self.ptr = 0
    def add(self, **kw):
        i = self.ptr; self.ptr += 1
        for k,v in kw.items():
            getattr(self, k)[i] = v

def compute_gae(rew, val, done, gamma=0.99, lam=0.95):
    T = len(rew)
    adv = np.zeros_like(rew, np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * (val[t+1] if t+1<T else 0.0) * nonterminal - val[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + val
    return adv, ret

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # === 环境 ===
    cfg = os.path.join(_ROOT, "config", "HierarchyMultipleSelfplay.yaml")
    env = MultipleCombatEnv(cfg)
    env.reset()
    # reset 之后、主循环之前
    if hasattr(env, "task") and hasattr(env.task, "radar"):
        radar = env.task.radar  # 直接复用
    else:
        radar = RadarModel()  # 环境里没有就自己建一个（用默认参数）

    # 我们只训练红方第一架（A0）演示；实际可同时训练A0/A1并求和回报
    red_ids = [k for k in env.agents.keys() if k.startswith("A")]
    A0 = red_ids[0]

    # === 动作维度 ===
    tpl_names = list_templates()
    z_dim     = max_param_dim()

    # === 策略 ===
    # 观测维度：直接用 env.get_obs()[A0] 的长度
    obs0 = env.get_obs()[A0]
    obs_dim = len(obs0)
    pi = TemplatePolicy(obs_dim, tpl_num=len(tpl_names), z_dim=z_dim).to(device)
    opt = optim.Adam(pi.parameters(), lr=3e-4)

    # === 训练循环（最小可跑版） ===
    EPISODES = 200
    HORIZON  = 256
    GAMMA, LAM, CLIP = 0.99, 0.95, 0.2

    for ep in range(EPISODES):
        # 每局清理模板上下文（很重要）
        for aid in red_ids: reset_agent_state(aid)
        env.reset()

        ro = Rollout(HORIZON, obs_dim, z_dim)
        step = 0
        done_flag = False

        while step < HORIZON and not done_flag:
            obs_dict = env.get_obs()
            actions = {}

            # —— 红方 A0 用“模板 RL” ——
            obsA = obs_dict[A0]
            (tpl_id, z_vec), (id_logp, z_logp), val = pi.act(obsA)
            actA = step_red_agent_pure_rl(env, radar, A0, tpl_id, z_vec)
            actions[A0] = actA

            # —— 红方其他机 / 蓝方：按你现有逻辑（纯 PPO 或规则） ——
            for aid in env.agents:
                if aid == A0: continue
                # 这里示例：都保持原来逻辑（不展开），你按项目已有的写法补上
                actions.setdefault(aid, [1,2,1,0])

            obs_next, rew, done, info = env.step(actions)
            done_flag = all(done.values())

            ro.add(
                obs=obsA, tpl_id=tpl_id, z=z_vec, id_logp=id_logp, z_logp=z_logp,
                rew=rew[A0], val=val, done=float(done[A0])
            )
            step += 1

        # === PPO 更新 ===
        T = ro.ptr
        adv, ret = compute_gae(ro.rew[:T], np.concatenate([ro.val[:T],[0]]), ro.done[:T], GAMMA, LAM)
        obs_t    = torch.as_tensor(ro.obs[:T], dtype=torch.float32, device=device)
        tpl_id_t = torch.as_tensor(ro.tpl_id[:T], dtype=torch.int64, device=device)
        z_t      = torch.as_tensor(ro.z[:T], dtype=torch.float32, device=device)
        old_idlp = torch.as_tensor(ro.id_logp[:T], dtype=torch.float32, device=device)
        old_zlp  = torch.as_tensor(ro.z_logp[:T], dtype=torch.float32, device=device)
        adv_t    = torch.as_tensor(adv[:T], dtype=torch.float32, device=device)
        ret_t    = torch.as_tensor(ret[:T], dtype=torch.float32, device=device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std()+1e-8)

        EPOCHS, BATCH = 4, 128
        idx = np.arange(T)
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, T, BATCH):
                j = idx[s:s+BATCH]
                id_logp, z_logp, ent, v = pi.evaluate_actions(
                    obs_t[j], tpl_id_t[j], z_t[j]
                )
                ratio_id = torch.exp(id_logp - old_idlp[j])
                ratio_z  = torch.exp(z_logp  - old_zlp[j])
                ratio    = ratio_id + ratio_z - 1.0  # 简单融合；也可分别裁剪后求和

                surr1 = ratio * adv_t[j]
                surr2 = torch.clamp(ratio, 1.0-CLIP, 1.0+CLIP) * adv_t[j]
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (ret_t[j] - v).pow(2).mean()
                entropy_bonus = ent.mean()

                loss = actor_loss + 0.5*value_loss - 0.01*entropy_bonus

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
                opt.step()

        if (ep+1) % 10 == 0:
            print(f"[EP {ep+1}] R_mean={ro.rew[:T].sum():.2f}  T={T}")
            os.makedirs(os.path.join(_TAC, "model"), exist_ok=True)
            torch.save(pi.state_dict(), os.path.join(_TAC, "model", "red_template_head.pt"))

if __name__ == "__main__":
    train()
