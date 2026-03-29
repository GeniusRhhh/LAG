# algorithms/hybrid/act_hybrid.py
import torch
import torch.nn as nn
import torch.distributions as D
from typing import Union

class HybridDist:
    """
    混合动作分布 = Categorical(模板ID) × DiagNormal(连续z向量)
    支持对连续部分传入 mask (B, M)：逐维加权 logprob/entropy，屏蔽无效维度的梯度。
    """
    def __init__(self, logits, z_mean, z_logstd):
        # logits: (B, T)     z_mean: (B, M)   z_logstd: (B, M)
        self.cat  = D.Categorical(logits=logits)
        self.mean = z_mean
        self.std  = (z_logstd).exp()
        self.norm = D.Independent(D.Normal(self.mean, self.std), 1)  # 仅用于无mask时

    def sample(self, mask: Union[torch.Tensor, None] = None):
        """
        返回 (tpl_id: (B,), z:(B,M))
        若传 mask，则把 z *= mask（无效维=0，执行时/回传都不会用到）
        """
        tpl = self.cat.sample()
        eps = torch.randn_like(self.mean)
        z = self.mean + eps * self.std
        if mask is not None:
            z = z * mask
        return tpl, z

    def log_prob(self, tpl, z, mask: Union[torch.Tensor, None] = None):
        """
        若传 mask，则对 DiagNormal 的逐维 logprob 乘 mask 再求和；
        没传 mask 时退回 Independent 的整体 logprob。
        """
        lp_cat = self.cat.log_prob(tpl)  # (B,)
        if mask is None:
            lp_z = self.norm.log_prob(z)  # (B,)
        else:
            # 手动逐维 logprob：sum_i mask_i * log N(z_i | mean_i, std_i)
            var = self.std * self.std
            # log N
            lp_per_dim = -0.5 * ((z - self.mean) ** 2 / (var + 1e-8) + 2.0 * torch.log(self.std + 1e-8) + torch.log(torch.tensor(2.0*3.1415926, device=z.device)))
            lp_z = (lp_per_dim * mask).sum(dim=-1)  # (B,)
        return lp_cat + lp_z

    def entropy(self, mask: Union[torch.Tensor, None] = None):
        ent_cat = self.cat.entropy()  # (B,)
        if mask is None:
            ent_z = self.norm.entropy()  # (B,)
        else:
            # DiagNormal 的逐维熵：0.5*(1+ln(2πσ^2))，再乘 mask 求和
            ent_per_dim = 0.5 * (1.0 + torch.log(2.0*torch.tensor(3.1415926, device=self.std.device)) + 2.0 * torch.log(self.std + 1e-8))
            ent_z = (ent_per_dim * mask).sum(dim=-1)
        return ent_cat + ent_z