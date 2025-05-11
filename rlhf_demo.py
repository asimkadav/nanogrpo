"""
Self-contained RLHF demo with GRPO and PPO implementations.
This example creates dummy data and runs both algorithms for comparison.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Create logs directory
os.makedirs('logs', exist_ok=True)

# GRPO Loss Implementation
class GRPOLoss(nn.Module):
    def __init__(self, kl_coef: float = 0.02, eps: float = 1e-8):
        super().__init__(); self.kl_coef, self.eps = kl_coef, eps

    def forward(self, logp_tokens: torch.Tensor, logp_ref: torch.Tensor, rewards: torch.Tensor):
        # Token-level entropy
        probs = logp_tokens.exp()
        ent = -(probs * logp_tokens).sum(-1).mean()  # average over B,T
        
        # Batch-norm advantage (scalar per sequence → broadcast)
        adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + self.eps)  # [B]
        pg = -(adv.unsqueeze(1).detach() * logp_tokens).mean()  # [ ]
        kl = self.kl_coef * (logp_tokens.sum(-1) - logp_ref).mean()  # [ ]
        loss = pg + kl
        return loss, pg, kl, ent, adv.abs().mean()

# PPO-Clip Loss Implementation
class PPOClipLoss(nn.Module):
    def __init__(self, eps=0.2, vf_coef=0.5, ent_coef=0.01):
        super().__init__(); self.eps, self.vf_coef, self.ent_coef = eps, vf_coef, ent_coef

    def forward(self, logp: torch.Tensor, logp_old: torch.Tensor, 
                returns: torch.Tensor, values: torch.Tensor, entropy: torch.Tensor):
        ratio = (logp - logp_old).exp()  # [B]
        adv = (returns - values.detach())  # [B]
        pg_core = torch.min(ratio*adv, torch.clamp(ratio, 1-self.eps, 1+self.eps)*adv)
        policy_loss = -pg_core.mean()
        value_loss = 0.5*(returns - values).pow(2).mean()
        ent_loss = -entropy.mean()
        return policy_loss + self.vf_coef*value_loss + self.ent_coef*ent_loss,

# CSV Logger for metrics
class CSVLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'w', newline='')
        import csv
        self.writer = csv.writer(self.f)
        self.writer.writerow(['step', 'loss', 'mean_r', 'kl', 'entropy', 'token_adv'])
    
    def write(self, step, loss, r, kl, ent, adv):
        self.writer.writerow([step, loss, r, kl, ent, adv])
        self.f.flush()
    
    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()

def train_grpo(steps=100):
    """Run a simplified GRPO training loop with synthetic data."""
    print("Training with GRPO...")
    # Parameters
    batch_size = 4
    seq_len = 10
    device = 'cpu'
    kl_coef = 0.02
    
    # Setup logger
    logger = CSVLogger('logs/grpo.csv')
    
    # GRPO loss function
    loss_fn = GRPOLoss(kl_coef)
    
    for step in range(steps):
        # Generate dummy data
        logp_tokens = torch.randn(batch_size, seq_len, device=device)
        logp_ref = torch.randn(batch_size, device=device)
        rewards = torch.rand(batch_size, device=device)
        
        # Compute loss
        loss, pg, kl, ent, adv = loss_fn(logp_tokens, logp_ref, rewards)
        
        # Log metrics
        if step % 10 == 0:
            logger.write(step, loss.item(), rewards.mean().item(), kl.item(), ent.item(), adv.item())
            print(f"{step:>6} GRPO loss {loss:.3f} R {rewards.mean():.2f} KL {kl:.3f} H {ent:.2f}")
    
    return 'logs/grpo.csv'

def train_ppo(steps=100):
    """Run a simplified PPO training loop with synthetic data."""
    print("\nTraining with PPO...")
    # Parameters
    batch_size = 4
    device = 'cpu'
    eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    
    # Setup logger
    logger = CSVLogger('logs/ppo.csv')
    
    # PPO loss function
    loss_fn = PPOClipLoss(eps, vf_coef, ent_coef)
    
    for step in range(steps):
        # Generate dummy data
        logp = torch.randn(batch_size, device=device)
        logp_old = logp + 0.1 * torch.randn_like(logp)  # slightly different
        returns = torch.rand(batch_size, device=device)
        values = returns + 0.2 * torch.randn_like(returns)  # prediction with some error
        
        # Generate varying entropy instead of constant
        entropy = 0.3 + 0.4 * torch.rand(1, device=device)  # random between 0.3 and 0.7
        
        # Compute loss
        loss, = loss_fn(logp, logp_old, returns, values, entropy)
        
        # Simulate KL divergence and token advantage with realistic values
        kl_div = 0.05 * torch.sin(torch.tensor(step / 10.0)).item()  # oscillating KL
        token_adv = 0.2 + 0.2 * torch.cos(torch.tensor(step / 5.0)).item()  # varying advantage
        
        # Log metrics - now with varying values for all metrics
        if step % 10 == 0:
            logger.write(step, loss.item(), returns.mean().item(), kl_div, entropy.item(), token_adv)
            print(f"{step:>6} PPO loss {loss:.3f} R {returns.mean():.2f} KL {kl_div:.3f} H {entropy.item():.2f}")
    
    return 'logs/ppo.csv'

def plot_metrics(grpo_log, ppo_log, metric='mean_r', output=None):
    """Plot comparison of metrics between GRPO and PPO."""
    print(f"\nPlotting {metric} comparison...")
    
    # Plot rewards over time
    plt.figure(figsize=(10, 6))
    for p, ls in [(grpo_log, '-'), (ppo_log, '--')]:
        df = pd.read_csv(p)
        label = p.split('/')[-1].split('.')[0]
        plt.plot(df['step'], df[metric], ls, label=label)
    
    plt.xlabel('Step')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Comparison of {metric.replace("_", " ").title()} between GRPO and PPO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output:
        plt.savefig(output)
        print(f"Plot saved to {output}")
    else:
        output = f'logs/{metric}_comparison.png'
        plt.savefig(output)
        print(f"Plot saved to {output}")

def main():
    # Run simplified GRPO training
    grpo_log = train_grpo()
    
    # Run simplified PPO training
    ppo_log = train_ppo()
    
    # Plot comparison of metrics
    plot_metrics(grpo_log, ppo_log, 'mean_r', 'logs/reward_comparison.png')
    plot_metrics(grpo_log, ppo_log, 'loss', 'logs/loss_comparison.png')
    plot_metrics(grpo_log, ppo_log, 'entropy', 'logs/entropy_comparison.png')
    plot_metrics(grpo_log, ppo_log, 'kl', 'logs/kl_comparison.png')

if __name__ == '__main__':
    main() 