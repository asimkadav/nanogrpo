#!/usr/bin/env python3
"""
Train CLIP using standard training (not GRPO) as a demonstration of working functionality.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import clip
from torchvision import datasets

def load_data(args):
    """Load and preprocess data for CLIP training."""
    # Get CLIP's preprocessing
    _, preprocess = clip.load(args.model_name, device=args.device, jit=False)
    
    # Create dataset
    if args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root=os.path.expanduser("~/.cache"),
            download=True,
            train=True,
            transform=preprocess
        )
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    return train_loader, train_dataset.classes

def get_text_features(model, classes, template="a photo of a {}.", device='cuda'):
    """Generate text features for all classes."""
    with torch.no_grad():
        text_inputs = torch.cat([clip.tokenize(template.format(c)) for c in classes]).to(device)
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def train_clip(args):
    """Train CLIP with standard cross-entropy loss."""
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Load CLIP model
    print(f"Loading CLIP model: {args.model_name}")
    clip_model, _ = clip.load(args.model_name, device=device, jit=False)
    
    # Freeze the text encoder, train only the visual encoder
    for param in clip_model.transformer.parameters():
        param.requires_grad = False
    
    # Freeze logit scale to avoid instability
    clip_model.logit_scale.requires_grad = False
    
    # Load data
    train_loader, classes = load_data(args)
    
    # Generate text features for each class
    text_features = get_text_features(clip_model, classes, device=args.device)
    
    # Optimizer with gradient clipping built in
    optimizer = torch.optim.AdamW(
        [p for p in clip_model.visual.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-6  # Larger epsilon for better numerical stability
    )
    
    # Training loop
    print("Starting training...")
    total_steps = 0
    nan_steps = 0
    
    for epoch in range(args.epochs):
        clip_model.visual.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, (images, labels) in enumerate(progress_bar):
            total_steps += 1
            images = images.to(device)
            labels = labels.to(device)
            
            try:
                # Zero gradients
                optimizer.zero_grad()
                
                # Get image features and ensure they are normalized
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    image_features = clip_model.encode_image(images)
                    # Ensure normalization is done outside of autocast
                
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
                
                # Compute similarity with all class texts
                logit_scale = clip_model.logit_scale.exp().clamp(0, 100)  # Clamp to avoid excessive scaling
                logits = logit_scale * image_features @ text_features.t()
                
                # Check for NaN in logits
                if torch.isnan(logits).any():
                    print(f"NaN detected in logits at step {total_steps}")
                    nan_steps += 1
                    continue
                
                # Cross-entropy loss with label smoothing for stability
                loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss detected at step {total_steps}: {loss.item()}")
                    nan_steps += 1
                    continue
                
                # Compute accuracy
                with torch.no_grad():
                    _, predicted = logits.max(1)
                    correct = (predicted == labels).float().sum()
                    batch_accuracy = correct / images.size(0)
                
                # Backward and optimize
                loss.backward()
                
                # Clip gradients explicitly for stability
                torch.nn.utils.clip_grad_norm_(clip_model.visual.parameters(), max_norm=args.grad_clip)
                
                # Check for NaN gradients
                grad_is_nan = False
                for param in clip_model.visual.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        grad_is_nan = True
                        break
                
                if grad_is_nan:
                    print(f"NaN/Inf gradient detected at step {total_steps}")
                    nan_steps += 1
                    continue
                    
                optimizer.step()
                
                # Log progress
                if step % args.log_interval == 0:
                    progress_bar.set_postfix(
                        loss=f"{loss.item():.3f}",
                        acc=f"{batch_accuracy.item():.3f}",
                        nan_steps=nan_steps
                    )
            
            except Exception as e:
                print(f"Error at step {total_steps}: {e}")
                nan_steps += 1
                continue
        
        # Print training statistics
        print(f"Epoch {epoch+1} training completed. Total steps: {total_steps}, NaN steps: {nan_steps}")
        
        # Evaluate on the entire dataset
        clip_model.visual.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc=f"Evaluating epoch {epoch+1}"):
                images = images.to(device)
                labels = labels.to(device)
                
                # Get image features
                image_features = clip_model.encode_image(images)
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
                
                # Compute similarity with all class texts
                logit_scale = clip_model.logit_scale.exp().clamp(0, 100)
                logits = logit_scale * image_features @ text_features.t()
                
                # Get predictions
                _, predicted = logits.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        # Print epoch results
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} accuracy: {accuracy:.2f}%")
    
    print("Training complete!")

def main():
    parser = argparse.ArgumentParser(description="Simple CLIP Training")
    
    # Model and dataset
    parser.add_argument('--model-name', type=str, default="ViT-B/32", help='CLIP model to use')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset to use (cifar100)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.2, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--log-interval', type=int, default=5, help='Log interval')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Train CLIP
    train_clip(args)

if __name__ == '__main__':
    main() 