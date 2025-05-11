"""Tiny Transformer Reward Model reusing nanoGPT blocks."""
import torch, torch.nn as nn
from model import GPTConfig, GPT     # local model.py

class RewardModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        gpt_cfg = GPTConfig(cfg['vocab_size'], cfg['block_size'], n_layer=cfg['n_layer'],
                            n_head=cfg['n_head'], n_embd=cfg['n_embd'])
        self.transformer = GPT(gpt_cfg)
        self.value_head  = nn.Linear(cfg['n_embd'], 1, bias=False)

    def forward(self, x, y):
        # concat prompt + completion; shift right like GPT training
        toks = torch.cat([x, y], dim=1)[:, :-1]  # remove final token for causal
        emb  = self.transformer(toks)            # [B,T,E]
        v    = self.value_head(emb).mean(dim=1)  # [B]
        return v.squeeze(-1) 