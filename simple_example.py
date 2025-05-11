"""
Simple example to demonstrate GRPO and PPO losses without using actual models.
This creates dummy data and runs simplified versions to create comparable metrics.
"""
import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from grpo import GRPOLoss
from ppo import PPOClipLoss
from utils.metrics import CSVLogger

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

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
        entropy = torch.tensor(0.5, device=device)
        
        # Compute loss
        loss, = loss_fn(logp, logp_old, returns, values, entropy)
        
        # Log metrics
        if step % 10 == 0:
            logger.write(step, loss.item(), returns.mean().item(), 0, entropy.item(), 0)
            print(f"{step:>6} PPO loss {loss:.3f} R {returns.mean():.2f} H {entropy.item():.2f}")
    
    return 'logs/ppo.csv'

def plot_metrics(grpo_log, ppo_log, metric='mean_r'):
    """Plot comparison of metrics between GRPO and PPO."""
    print("\nPlotting comparison...")
    
    # Plot rewards over time
    plt.figure(figsize=(10, 6))
    for p, ls in [(grpo_log, '-'), (ppo_log, '--')]:
        df = pd.read_csv(p)
        label = p.split('/')[-1].split('.')[0]
        plt.plot(df['step'], df[metric], ls, label=label)
    
    plt.xlabel('step')
    plt.ylabel(metric)
    plt.title(f'Comparison of {metric} between GRPO and PPO')
    plt.legend()
    plt.savefig(f'logs/{metric}_comparison.png')
    print(f"Plot saved to logs/{metric}_comparison.png")

def main():
    # Run simplified GRPO training
    grpo_log = train_grpo()
    
    # Run simplified PPO training
    ppo_log = train_ppo()
    
    # Plot comparison of metrics
    plot_metrics(grpo_log, ppo_log, 'mean_r')  # Compare rewards
    plot_metrics(grpo_log, ppo_log, 'loss')    # Compare loss

if __name__ == '__main__':
    main() 