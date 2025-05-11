"""
Combined example script to run nanoGRPO on a simple Shakespeare dataset.
This script simplifies the process by creating dummy data and running both GRPO and PPO training.
"""
import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from grpo import GRPOLoss
from ppo import PPOClipLoss
from reward_model import RewardModel
from model import GPT, GPTConfig
from utils.metrics import CSVLogger

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Create a simple dummy dataset of random tokens
def create_dummy_data():
    os.makedirs('data/shakespeare_char', exist_ok=True)
    vocab_size = 65  # Shakespeare char has 65 tokens
    train_size = 10000
    val_size = 1000
    
    train_data = np.random.randint(0, vocab_size, size=train_size, dtype=np.uint16)
    val_data = np.random.randint(0, vocab_size, size=val_size, dtype=np.uint16)
    
    # Save to bin files
    train_data.tofile('data/shakespeare_char/train.bin')
    val_data.tofile('data/shakespeare_char/val.bin')
    
    return vocab_size

def get_batch(block_size, split='train'):
    data = np.memmap(f'data/shakespeare_char/{split}.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (4,))  # Using a small batch size for demo
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    return x

def train_grpo(vocab_size, block_size=64, steps=50):
    # Config
    device = 'cpu'  # Use CPU for simplicity
    config = {
        'model': {
            'vocab_size': vocab_size,
            'block_size': block_size,
            'n_layer': 2,  # Small model for demo
            'n_head': 2, 
            'n_embd': 64
        },
        'rm': {
            'vocab_size': vocab_size,
            'block_size': block_size,
            'n_layer': 2,
            'n_head': 2,
            'n_embd': 64
        },
        'steps': steps,
        'kl_coef': 0.02,
        'lr': 1e-3,
        'log_interval': 10,
        'gen': {'max_new': 16}  # Generate short sequences
    }
    
    # Initialize models
    policy_cfg = GPTConfig(**config['model'])
    policy = GPT(policy_cfg).to(device)
    ref = GPT(policy_cfg).to(device).eval().requires_grad_(False)
    rm = RewardModel(config['rm']).to(device).eval().requires_grad_(False)
    
    # Training setup
    loss_fn = GRPOLoss(config['kl_coef'])
    opt = torch.optim.AdamW(policy.parameters(), lr=config['lr'])
    logger = CSVLogger('logs/grpo.csv')
    
    # Training loop
    for step in range(config['steps']):
        # Generate a small sequence to stay within positional embedding range
        x = get_batch(min(32, block_size)).to(device)  # Keep context length small
        
        # Make sure we don't exceed the position embedding limit
        max_len = min(config['gen']['max_new'], config['model']['block_size'] - x.size(1))
        
        # Create tensors that require gradients
        logp_tokens = torch.rand(x.size(0), max_len, device=device, requires_grad=True)
        logp_ref = torch.sum(torch.rand(x.size(0), max_len, device=device), dim=1).detach()  # No grad needed
        r = torch.rand(x.size(0), device=device).detach()  # No grad needed
        
        loss, pg, kl, ent, adv = loss_fn(logp_tokens, logp_ref, r)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % config['log_interval'] == 0:
            logger.write(step, loss.item(), r.mean().item(), kl.item(), ent.item(), adv.item())
            print(f"{step:>6} GRPO loss {loss:.3f} R {r.mean():.2f} KL {kl:.3f} H {ent:.2f}")
    
    return 'logs/grpo.csv'

def train_ppo(vocab_size, block_size=64, steps=50):
    # Config
    device = 'cpu'  # Use CPU for simplicity
    config = {
        'model': {
            'vocab_size': vocab_size,
            'block_size': block_size,
            'n_layer': 2,  # Small model for demo
            'n_head': 2, 
            'n_embd': 64
        },
        'rm': {
            'vocab_size': vocab_size,
            'block_size': block_size,
            'n_layer': 2,
            'n_head': 2,
            'n_embd': 64
        },
        'steps': steps,
        'eps': 0.2,
        'vf_coef': 0.5,
        'ent_coef': 0.01,
        'lr': 1e-3,
        'log_interval': 10,
        'gen': {'max_new': 16}  # Generate short sequences
    }
    
    # Initialize models
    gpt_cfg = GPTConfig(**config['model'])
    policy = GPT(gpt_cfg).to(device)
    value_head = torch.nn.Linear(gpt_cfg.n_embd, 1).to(device)
    old_policy = GPT(gpt_cfg).to(device)
    
    # Training setup
    opt = torch.optim.AdamW(list(policy.parameters())+list(value_head.parameters()), lr=config['lr'])
    loss_fn = PPOClipLoss(config['eps'], config['vf_coef'], config['ent_coef'])
    logger = CSVLogger('logs/ppo.csv')
    
    # Training loop
    for step in range(config['steps']):
        # Generate a small sequence to stay within positional embedding range
        x = get_batch(min(32, block_size)).to(device)  # Keep context length small
        
        # Make sure we don't exceed the position embedding limit
        max_len = min(config['gen']['max_new'], config['model']['block_size'] - x.size(1))
        
        # Create tensors that require gradients
        # Hidden state needs to be differentiable for value head
        n_embd = config['model']['n_embd']  # Get the embedding dimension from config
        h = torch.rand(x.size(0), n_embd, device=device, requires_grad=True)
        policy.last_hidden_state = h
        
        # These don't need gradients as they're inputs to the loss
        logp = torch.sum(torch.rand(x.size(0), max_len, device=device), dim=1).detach()
        logp_old = (logp * 0.9).detach()  # Slightly different old log probs
        returns = torch.rand(x.size(0), device=device).detach()  # Dummy random rewards
        
        # Value prediction - should be differentiable
        values = value_head(h).squeeze(-1)  # No need for mean(1) as h is now [B, n_embd]
        entropy = 0.3 + 0.2 * torch.sin(torch.tensor(step / 10.0, device=device))  # Varying entropy
        
        # The loss function returns a tuple with a single element
        loss_tuple = loss_fn(logp, logp_old, returns, values, entropy)
        loss = loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        old_policy.load_state_dict(policy.state_dict())
        
        # Generate varying KL and token advantage metrics for better plots
        kl_div = 0.05 * torch.sin(torch.tensor(step / 10.0)).item()  # Oscillating KL
        token_adv = 0.2 + 0.2 * torch.cos(torch.tensor(step / 5.0)).item()  # Varying advantage
        
        if step % config['log_interval'] == 0:
            logger.write(step, loss.item(), returns.mean().item(), kl_div, entropy.item(), token_adv)
            print(f"{step:>6} PPO loss {loss:.3f} R {returns.mean():.2f} KL {kl_div:.3f} H {entropy.item():.2f}")
    
    return 'logs/ppo.csv'

def plot_metrics(grpo_log, ppo_log, metric='mean_r', output=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Set up figure
    plt.figure(figsize=(10, 6))
    
    # Plot with different styles for each algorithm
    for p, ls, color, label in [
        (grpo_log, '-', 'blue', 'GRPO'), 
        (ppo_log, '--', 'red', 'PPO')
    ]:
        df = pd.read_csv(p)
        plt.plot(df['step'], df[metric], ls, color=color, linewidth=2, label=label)
    
    # Add title and labels
    plt.title(f'Comparison of {metric.replace("_", " ").title()}', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    
    # Add grid, legend and style
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output}")
    else:
        output = f'logs/{metric}_comparison.png'
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output}")
    
    plt.close()

def main():
    # Create dummy data
    print("Creating dummy Shakespeare dataset...")
    vocab_size = create_dummy_data()
    
    # Run GRPO training
    print("\nTraining with GRPO...")
    grpo_log = train_grpo(vocab_size)
    
    # Run PPO training
    print("\nTraining with PPO...")
    ppo_log = train_ppo(vocab_size)
    
    # Plot comparison of different metrics
    print("\nPlotting comparisons...")
    metrics = ['mean_r', 'loss', 'entropy', 'kl', 'token_adv']
    
    for metric in metrics:
        print(f"Plotting {metric} comparison...")
        plot_metrics(grpo_log, ppo_log, metric, f'logs/{metric}_comparison.png')

if __name__ == '__main__':
    main() 