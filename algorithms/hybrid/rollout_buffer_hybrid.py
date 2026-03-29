# -*- coding: utf-8 -*-
# 作用：把采样到的红方样本按步扁平存起来（每个红方agent各一条）
from dataclasses import dataclass, field
import numpy as np
import torch

@dataclass
class HybridRolloutBuffer:
    device: torch.device
    obs:  list = field(default_factory=list)   # (obs_dim,)
    tpl:  list = field(default_factory=list)   # () int
    z:    list = field(default_factory=list)   # (z_dim,)
    mask: list = field(default_factory=list)   # (z_dim,)
    logp: list = field(default_factory=list)   # () float
    val:  list = field(default_factory=list)   # () float
    rew:  list = field(default_factory=list)   # () float
    done: list = field(default_factory=list)   # () float

    def clear(self):
        self.obs.clear(); self.tpl.clear(); self.z.clear(); self.mask.clear()
        self.logp.clear(); self.val.clear(); self.rew.clear(); self.done.clear()

    def __len__(self):
        return len(self.rew)

    def add_step(self, obs_np, tpl_id, z_vec, mask_vec, logp, v, rew=0.0, done=0.0):
        self.obs.append(np.asarray(obs_np, dtype=np.float32))
        self.tpl.append(int(tpl_id))
        self.z.append(np.asarray(z_vec, dtype=np.float32))
        self.mask.append(np.asarray(mask_vec, dtype=np.float32))
        self.logp.append(float(logp))
        self.val.append(float(v))
        self.rew.append(float(rew))
        self.done.append(float(done))

    def as_tensors(self):
        obs_t  = torch.as_tensor(np.asarray(self.obs,  dtype=np.float32), device=self.device)
        tpl_t  = torch.as_tensor(np.asarray(self.tpl,  dtype=np.int64  ), device=self.device)
        z_t    = torch.as_tensor(np.asarray(self.z,    dtype=np.float32), device=self.device)
        m_t    = torch.as_tensor(np.asarray(self.mask, dtype=np.float32), device=self.device)
        logp_t = torch.as_tensor(np.asarray(self.logp, dtype=np.float32), device=self.device)
        val_t  = torch.as_tensor(np.asarray(self.val,  dtype=np.float32), device=self.device)
        rew_t  = torch.as_tensor(np.asarray(self.rew,  dtype=np.float32), device=self.device)
        done_t = torch.as_tensor(np.asarray(self.done, dtype=np.float32), device=self.device)
        return obs_t, tpl_t, z_t, m_t, logp_t, val_t, rew_t, done_t
