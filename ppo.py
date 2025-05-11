"""Minimal PPO‑clip objective with shared policy/value head."""
import torch, torch.nn as nn

class PPOClipLoss(nn.Module):
    def __init__(self, eps=0.2, vf_coef=0.5, ent_coef=0.01):
        super().__init__(); self.eps, self.vf_coef, self.ent_coef = eps, vf_coef, ent_coef

    def forward(self, logp: torch.Tensor, logp_old: torch.Tensor,
                returns: torch.Tensor, values: torch.Tensor,
                entropy: torch.Tensor):
        ratio   = (logp - logp_old).exp()               # [B]
        adv     = (returns - values.detach())           # [B]
        pg_core = torch.min(ratio*adv, torch.clamp(ratio,1-self.eps,1+self.eps)*adv)
        policy_loss = -pg_core.mean()
        value_loss  = 0.5*(returns - values).pow(2).mean()
        ent_loss    = -entropy.mean()
        return policy_loss + self.vf_coef*value_loss + self.ent_coef*ent_loss, 