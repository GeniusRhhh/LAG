# algorithms/hybrid/ppo_policy_hybrid.py
import torch, torch.nn as nn
import numpy as np
from .ppo_actor_hybrid import PPOActorHybrid
# 从桥接层拿模板信息与掩码工具
from Tactical_Rule_Template.ruleset.rl_template_driver import list_templates, max_param_dim, param_mask_by_id
USE_CC_SVB = True
class CCSVBTeamCritic(nn.Module):
    """
    CC-IVB 团队 Critic：
      - trunk: 提取特征 h
      - gate: 生成混合权重 w ∈ Δ^2
      - v_heads: 两个团队价值专家头 (v0, v1) → 混合成 V_team
      - b_head: 输出个体基线向量 b(s) = [b0, b1]
    forward(x)    → V_team(x)  (B,1)
    value_heads(x)→ (V_team(B,), b_heads(B,2))
    """
    def __init__(self, obs_dim):
        super().__init__()
        self.trunk   = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.gate    = nn.Sequential(nn.Linear(128, 2), nn.Softmax(dim=-1))
        self.v_heads = nn.ModuleList([nn.Linear(128, 1), nn.Linear(128, 1)])
        self.b_head  = nn.Linear(128, 2)

    def _h_w(self, x):
        h = self.trunk(x)          # (B,128)
        w = self.gate(h)           # (B,2)
        return h, w

    def forward(self, x):
        h, w = self._h_w(x)
        v0 = self.v_heads[0](h)    # (B,1)
        v1 = self.v_heads[1](h)    # (B,1)
        v  = w[:, :1] * v0 + w[:, 1:] * v1   # (B,1)
        return v

    @torch.no_grad()
    def value_heads(self, x):
        h, w = self._h_w(x)
        v0 = self.v_heads[0](h)    # (B,1)
        v1 = self.v_heads[1](h)    # (B,1)
        v  = w[:, :1] * v0 + w[:, 1:] * v1   # (B,1)
        b  = self.b_head(h)        # (B,2) → [b0, b1]
        return v.squeeze(-1), b

class PPOPolicyHybrid(nn.Module):
    def __init__(self, obs_dim, tpl_num=None, z_dim=None, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.tpl_names = list_templates()
        self.tpl_num   = len(self.tpl_names) if tpl_num is None else tpl_num
        self.z_dim     = max_param_dim()    if z_dim   is None else z_dim
        self.actor = PPOActorHybrid(obs_dim, self.tpl_num, self.z_dim).to(self.device)

        if USE_CC_SVB:
            # CC-IVB 模式：critic 是一个组合模块，forward 给 V_team，内部也带 b_heads
            self.critic = CCSVBTeamCritic(obs_dim).to(self.device)
        else:
            # 原始单头 critic（保持老逻辑）
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 1)
            ).to(self.device)

    def value_heads(self, obs_np: np.ndarray):
        """
        统一接口：
          - 开 CC-IVB ：返回 (V_team(B,), b_heads(B,2))
          - 关 CC-IVB ：退化为 (v(B,), stack([v,v], dim=-1))，即 b0=b1=v（占位）
        """
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if x.ndim == 1: x = x.unsqueeze(0)
        if USE_CC_SVB and hasattr(self.critic, "value_heads"):
            return self.critic.value_heads(x)
        else:
            v = self.critic(x).squeeze(-1)  # (B,)
            b = torch.stack([v, v], dim=-1)  # (B,2) 占位，保证下游维度一致
            return v, b

    @torch.no_grad()
    def act(self, obs_np: np.ndarray):
        """
        obs_np: (obs_dim,) 或 (B,obs_dim)
        return: tpl_id(int or np.ndarray), z_vec(np.ndarray[M]), logp(float), v(float)
        """
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if x.ndim == 1: x = x.unsqueeze(0)  # (1,D)
        dist = self.actor(x)

        # 先采样模板
        tpl = dist.cat.sample()  # (B,)
        # 生成掩码 (B,M)
        masks = []
        for tid in tpl.tolist():
            m = param_mask_by_id(tid, self.z_dim)
            masks.append(m)
        mask = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.float32, device=self.device)

        # 再采样 z，并施加掩码
        _, z = dist.sample(mask=mask)  # z 已被 *mask
        logp = dist.log_prob(tpl, z, mask=mask)      # 只算有效前K维
        v = self.critic(x).squeeze(-1)            # (B,)

        tpl_np = tpl.squeeze(0).cpu().numpy() if tpl.shape[0] > 1 else int(tpl.item())
        z_np   = z.squeeze(0).cpu().numpy()  if z.shape[0] > 1 else z.squeeze(0).cpu().numpy()
        logp_f = float(logp.mean().item())
        v_f    = float(v.mean().item())
        return tpl_np, z_np, logp_f, v_f

    def evaluate_actions(self, obs_batch, tpl_batch, z_batch):
        """
        训练时用：给一批 (obs, tpl_id, z)，返回 logp_new、entropy、V(obs)
        会为每个样本构造自己的 mask，确保只对各自模板的前K维回传梯度。
        """
        x   = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        tpl = torch.as_tensor(tpl_batch, dtype=torch.long,    device=self.device)
        z   = torch.as_tensor(z_batch,   dtype=torch.float32, device=self.device)

        dist = self.actor(x)

        # 构造 (B,M) 掩码
        masks = []
        for tid in tpl.tolist():
            m = param_mask_by_id(tid, self.z_dim)
            masks.append(m)
        mask = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.float32, device=self.device)

        # 用掩码计算 logp / entropy
        logp = dist.log_prob(tpl, z, mask=mask)      # (B,)
        ent  = dist.entropy(mask=mask)               # (B,)
        v = self.critic(x).squeeze(-1)            # (B,)
        return logp, ent, v