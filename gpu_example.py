#!/usr/bin/env python3
"""
GPU-accelerated example of GRPO and PPO training.
This script demonstrates how to use GPU acceleration with nanoGRPO.
"""
import os
import torch
import numpy as np
import argparse
from grpo import GRPOLoss
from ppo import PPOClipLoss
from reward_model import RewardModel
from model import GPT, GPTConfig
from utils.metrics import CSVLogger

def create_dummy_data(data_dir='data/shakespeare_char'):
    """Create a simple dummy dataset of random tokens."""
    os.makedirs(data_dir, exist_ok=True)
    vocab_size = 65  # Shakespeare char has 65 tokens
    train_size = 10000
    val_size = 1000
    
    train_data = np.random.randint(0, vocab_size, size=train_size, dtype=np.uint16)
    val_data = np.random.randint(0, vocab_size, size=val_size, dtype=np.uint16)
    
    # Save to bin files
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    
    print(f"Created dummy data at {data_dir}")
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    return vocab_size, train_path, val_path

def get_batch(data_path, block_size, batch_size=4, device='cpu'):
    """Get a batch of data from the dataset."""
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    return x.to(device)

def train_grpo(vocab_size, train_path, val_path, device='cuda', num_steps=100):
    """Train a model with GRPO on GPU."""
    print(f"Using device: {device}")
    
    # Config
    config = {
        'model': {
            'vocab_size': vocab_size,
            'block_size': 64,
            'n_layer': 4,
            'n_head': 4,
            'n_embd': 128,
            'device': device
        },
        'rm': {
            'vocab_size': vocab_size,
            'block_size': 64,
            'n_layer': 4,
            'n_head': 4,
            'n_embd': 128
        },
        'steps': num_steps,
        'batch_size': 16,
        'kl_coef': 0.02,
        'lr': 1e-4,
        'log_interval': 10,
        'gen': {'max_new': 16}
    }
    
    # Initialize models on GPU
    policy_cfg = GPTConfig(**config['model'])
    policy = GPT(policy_cfg)
    print(f"Policy model is on: {next(policy.parameters()).device}")
    
    ref = GPT(policy_cfg).eval().requires_grad_(False)
    rm = RewardModel(config['rm']).to(device).eval().requires_grad_(False)
    
    # Training setup
    loss_fn = GRPOLoss(config['kl_coef'])
    opt = torch.optim.AdamW(policy.parameters(), lr=config['lr'])
    os.makedirs('logs', exist_ok=True)
    logger = CSVLogger('logs/grpo_gpu.csv')
    
    # Training loop
    block_size = config['model']['block_size']
    batch_size = config['batch_size']
    
    # CUDA events for timing
    if device == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
    for step in range(config['steps']):
        # Get batch
        x = get_batch(train_path, min(32, block_size), batch_size, device)
        
        # Ensure we don't exceed position embedding limit
        max_len = min(config['gen']['max_new'], config['model']['block_size'] - x.size(1))
        
        # Generate sequence with the policy
        with torch.set_grad_enabled(True):
            # Forward pass through policy and reference
            # Create target data for CrossEntropy calculation (next token prediction)
            targets = torch.roll(x, shifts=-1, dims=1)
            targets[:, -1] = -1  # Mask the last token
            
            # Forward pass to get logits and loss
            policy_logits, policy_loss = policy(x, targets=targets)
            
            with torch.no_grad():
                _, ref_loss = ref(x, targets=targets)
            
            # Create logprobs for GRPO (using a simple approximation for the example)
            logp_tokens = torch.ones(batch_size, max_len, device=device, requires_grad=True) * (-policy_loss.item() / max_len)
            logp_ref = torch.ones(batch_size, device=device).detach() * (-ref_loss.item())
            
            # Simulate rewards (typically from reward model)
            r = torch.rand(batch_size, device=device).detach()
            
            # Calculate GRPO loss
            loss, pg, kl, ent, adv = loss_fn(logp_tokens.sum(-1), logp_ref, r)
            
            # Backward pass and optimization
            opt.zero_grad()
            loss.backward()
            
            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            
            opt.step()
        
        # Log metrics
        if step % config['log_interval'] == 0:
            logger.write(step, loss.item(), r.mean().item(), kl.item(), ent.item(), adv.mean().item())
            print(f"{step:>6} GRPO loss {loss:.3f} R {r.mean():.2f} KL {kl:.3f} H {ent:.2f}")
    
    # Record GPU timing
    if device == 'cuda':
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
        print(f"Total training time: {elapsed_time:.2f} seconds")
    
    return 'logs/grpo_gpu.csv'

def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated GRPO example")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--data-dir', type=str, default='data/shakespeare_char', help='Data directory')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots')
    args = parser.parse_args()
    
    # Check if CUDA is available and warn if not
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    
    # Create dummy data
    vocab_size, train_path, val_path = create_dummy_data(args.data_dir)
    
    # Train with GRPO on specified device
    log_file = train_grpo(vocab_size, train_path, val_path, device=args.device, num_steps=args.steps)
    
    # Plot results if not skipped
    if not args.skip_plots:
        try:
            from plot import plot_metrics
            metrics = ['mean_r', 'kl', 'entropy', 'loss']
            for metric in metrics:
                output_file = f'logs/gpu_{metric}_comparison.png'
                # Call the plot script directly
                plot_metrics([log_file], metric=metric, output=output_file)
                print(f"Plot saved to {output_file}")
        except ImportError:
            print("Could not import plot_metrics, skipping plotting.")

if __name__ == '__main__':
    main() 