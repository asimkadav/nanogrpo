"""Gradient‑Reweighted Preference Optimisation (GRPO) loss (token‑wise)."""
import torch, torch.nn as nn

class GRPOLoss(nn.Module):
    def __init__(self, kl_coef: float = 0.02, eps: float = 1e-8):
        super().__init__(); self.kl_coef, self.eps = kl_coef, eps

    def forward(self, logp_tokens: torch.Tensor, logp_ref: torch.Tensor,
                rewards: torch.Tensor):
        """Args
        -----
        logp_tokens : [B,T] token‑wise log‑prob from policy
        logp_ref    : [B]   sequence log‑prob from frozen ref model
        rewards     : [B]   scalar reward for each completion
        Returns: total loss, pg loss, kl loss, |adv| mean, entropy token wise
        """
        # token‑level entropy
        probs = logp_tokens.exp()
        ent   = -(probs * logp_tokens).sum(-1).mean()   # average over B,T
        # batch‑norm advantage (scalar per sequence → broadcast)
        adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False)+self.eps)  # [B]
        pg  = -(adv.unsqueeze(1).detach() * logp_tokens).mean()                    # [ ]
        kl  = self.kl_coef * (logp_tokens.sum(-1) - logp_ref).mean()               # [ ]
        loss= pg + kl
        return loss, pg, kl, ent, adv.abs().mean() 