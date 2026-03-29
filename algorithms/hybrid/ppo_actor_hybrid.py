# algorithms/hybrid/ppo_actor_hybrid.py
import torch, torch.nn as nn
from .act_hybrid import HybridDist

class PPOActorHybrid(nn.Module):
    def __init__(self, obs_dim, tpl_num, z_dim, hid=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU()
        )
        self.head_tpl  = nn.Linear(hid, tpl_num)   # 模板ID logits
        self.head_mean = nn.Linear(hid, z_dim)     # 连续 z 的均值
        self.logstd    = nn.Parameter(torch.zeros(z_dim))  # 全局 logstd

    def forward(self, obs):
        feat = self.body(obs)
        logits  = self.head_tpl(feat)
        z_mean  = self.head_mean(feat)
        z_logsd = self.logstd.expand_as(z_mean)
        return HybridDist(logits, z_mean, z_logsd)
