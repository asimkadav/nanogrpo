#!/usr/bin/env python3
"""
Train CLIP using GRPO.
This script demonstrates how to apply GRPO to finetune CLIP models.
Supports optional LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip
from torchvision import transforms, datasets
from grpo import GRPOLoss
from utils.metrics import CSVLogger

# Custom GRPO Loss for classification tasks
class ClassificationGRPOLoss(nn.Module):
    def __init__(self, kl_coef: float = 0.02, eps: float = 1e-4):
        super().__init__()
        self.kl_coef = kl_coef
        self.eps = eps  # Larger epsilon for classification tasks

    def forward(self, logp_pol: torch.Tensor, logp_ref: torch.Tensor,
                rewards: torch.Tensor):
        """
        Args:
            logp_pol : [B,1] log-prob of correct class from policy
            logp_ref : [B,1] log-prob of correct class from reference model
            rewards  : [B]   rewards (1 for correct prediction, 0 otherwise, or confidence)
        
        Returns: 
            total loss, pg loss, kl loss, entropy (dummy), |adv| mean
        """
        # Ensure inputs are valid
        batch_size = rewards.size(0)
        rewards = rewards.detach()  # Detach rewards
        logp_ref = logp_ref.detach()  # Detach reference log probs
        
        # Ensure no zeros in rewards with a small epsilon
        rewards = rewards + self.eps
        
        # Use direct log-prob * reward (REINFORCE-style)
        policy_term = -torch.mean(logp_pol.squeeze(-1) * rewards)
        
        # Safe KL calculation
        kl_div = (logp_pol.squeeze(-1) - logp_ref.squeeze(-1)).clamp(min=-10, max=10)
        kl_term = self.kl_coef * torch.mean(kl_div)
        
        # Combine terms
        loss = policy_term + kl_term
        
        # Calculate a simple advantage for logging
        rewards_mean = rewards.mean().item()
        adv = rewards - rewards_mean
        
        # Dummy entropy for compatibility
        ent = torch.zeros(1, device=loss.device)
        
        return loss, policy_term, kl_term, ent, adv.abs().mean()

# Import peft for LoRA support if available
try:
    from peft import LoraConfig, get_peft_model, TaskType
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print("Warning: PEFT library not found. LoRA will not be available.")
    print("To enable LoRA, install peft: pip install peft>=0.5.0")

class CLIPWrapper(nn.Module):
    """
    Wrapper around CLIP model to access the image and text encoders separately
    and make it compatible with GRPO training.
    """
    def __init__(self, clip_model, device, use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.05, freeze_logit_scale=False):
        super().__init__()
        self.model = clip_model
        self.device = device
        self.use_lora = use_lora
        
        # Freeze the text encoder as reference model
        for param in self.model.transformer.parameters():
            param.requires_grad = False
            
        # Optionally freeze the logit scale
        if freeze_logit_scale:
            print("Freezing policy model logit scale")
            self.model.logit_scale.requires_grad_(False)
            
        # Apply LoRA if requested and available
        if self.use_lora and LORA_AVAILABLE:
            self._apply_lora(lora_r, lora_alpha, lora_dropout)
        else:
            # Traditional fine-tuning - train the whole visual encoder
            if not self.use_lora:
                print("Using full fine-tuning for visual encoder")
            for param in self.model.visual.parameters():
                param.requires_grad = True
    
    def _apply_lora(self, lora_r, lora_alpha, lora_dropout):
        """Helper method to apply LoRA with proper error handling"""
        print(f"Applying LoRA to CLIP visual encoder (r={lora_r}, alpha={lora_alpha})")
        
        # Freeze all parameters first
        for param in self.model.visual.parameters():
            param.requires_grad = False
        
        # For ViT-based models, apply LoRA to attention layers
        if hasattr(self.model.visual, 'transformer'):
            lora_applied = False
            
            # Strategy 1: Use common layer names
            try:
                print("Trying LoRA with common layer names")
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="FEATURE_EXTRACTION"
                )
                self.model.visual = get_peft_model(self.model.visual, lora_config)
                lora_applied = True
            except ValueError:
                print("First LoRA strategy failed")
            
            # Strategy 2: Automatic detection
            if not lora_applied:
                try:
                    print("Trying LoRA with automatic target detection")
                    lora_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type="FEATURE_EXTRACTION"
                    )
                    self.model.visual = get_peft_model(self.model.visual, lora_config)
                    lora_applied = True
                except ValueError:
                    print("Second LoRA strategy failed")
            
            # Fallback: Full fine-tuning
            if not lora_applied:
                print("LoRA strategies failed, falling back to full fine-tuning")
                for param in self.model.visual.parameters():
                    param.requires_grad = True
            else:
                # Count trainable parameters
                trainable_params = sum(p.numel() for p in self.model.visual.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.visual.parameters())
                print(f"LoRA trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total)")
        else:
            # For non-ViT architectures, train the whole encoder
            print("Warning: Model architecture not supported for LoRA (defaulting to full fine-tuning)")
            for param in self.model.visual.parameters():
                param.requires_grad = True
            
    def encode_image(self, image):
        return self.model.encode_image(image)
        
    def encode_text(self, text):
        with torch.no_grad():
            return self.model.encode_text(text)
            
    def compute_similarity(self, image_features, text_features):
        # Normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits
        
    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        return self.compute_similarity(image_features, text_features)

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
    else:  # Default to ImageNet
        train_path = os.path.join(args.data_path, 'train')
        train_dataset = datasets.ImageFolder(
            train_path,
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

def compute_entropy(logits, temperature=1.0):
    """
    Compute entropy of probability distribution with temperature scaling.
    
    Args:
        logits: Raw logits tensor
        temperature: Temperature for softmax (higher values make distribution more uniform)
    
    Returns:
        Mean entropy across batch
    """
    # Apply temperature scaling to make distributions less extreme
    scaled_logits = logits / temperature
    
    # Compute probabilities with more stable softmax
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Add epsilon to prevent log(0) issues (slightly larger for better stability)
    eps = 1e-6
    
    # Compute entropy and handle NaN/Inf values
    entropy_per_example = -(probs * (probs + eps).log()).sum(dim=-1)
    
    # Replace NaN/Inf with zeros and return mean
    return torch.nan_to_num(entropy_per_example).mean()

def train_clip_with_grpo(args):
    """Train CLIP using GRPO for alignment."""
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Check LoRA option compatibility
    if args.use_lora and not LORA_AVAILABLE:
        print("Warning: LoRA requested but peft library not available. Falling back to full fine-tuning.")
        args.use_lora = False
    
    # Load CLIP model
    print(f"Loading CLIP model: {args.model_name}")
    clip_model, _ = clip.load(args.model_name, device=device, jit=False)
    
    # Always freeze logit scale for stability (like in simplified_clip_train.py)
    clip_model.logit_scale.requires_grad = False
    
    # Create model wrapper
    model = CLIPWrapper(
        clip_model, 
        device, 
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_logit_scale=True  # Always freeze for stability
    )
    
    # Load data
    train_loader, classes = load_data(args)
    
    # Generate text features for each class
    text_features = get_text_features(clip_model, classes, device=args.device)
    
    # Optimizer with improved stability
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-6  # Larger epsilon for better numerical stability
    )
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(train_loader),
        eta_min=args.lr * 0.1  # Minimum learning rate will be 10% of initial
    )
    
    # Create a reference model (frozen)
    ref_model, _ = clip.load(args.model_name, device=device, jit=False)
    for param in ref_model.parameters():
        param.requires_grad = False
    # Always freeze the reference model logit scale
    ref_model.logit_scale.requires_grad_(False)
    
    # Use the classification-specific GRPO loss
    grpo_loss = ClassificationGRPOLoss(args.kl_coef)
    
    # Logger
    model_suffix = f"_{args.model_name.replace('/', '_')}"
    lora_suffix = f"_lora{args.lora_r}" if args.use_lora else ""
    log_file = f'logs/clip_grpo{model_suffix}{lora_suffix}.csv'
    logger = CSVLogger(log_file)
    
    # Training loop
    total_steps = 0
    nan_steps = 0
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (images, labels) in enumerate(progress_bar):
            total_steps += 1
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass through policy model with mixed precision for efficiency
                with torch.autocast(device_type='cuda', enabled=args.use_amp):
                    image_features = model.encode_image(images)
                
                # Normalize features (outside of autocast for better precision)
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
                
                # Compute similarity with ALL class texts (policy)
                logit_scale = model.model.logit_scale.exp().clamp(0, 100)  # Clamp to avoid excessive scaling
                all_logits = logit_scale * image_features @ text_features.t()  # [B, N_classes]
                
                # Reference model forward pass
                with torch.no_grad():
                    ref_image_features = ref_model.encode_image(images)
                    ref_image_features = ref_image_features / (ref_image_features.norm(dim=-1, keepdim=True) + 1e-8)
                    logit_scale_ref = ref_model.logit_scale.exp().clamp(0, 100)
                    ref_logits_full = logit_scale_ref * ref_image_features @ text_features.t()
                
                # Get batch size
                batch_size = images.size(0)
                
                # Get predictions from policy model
                predictions = all_logits.argmax(dim=1)
                
                # Compute reward: use hard or smooth rewards based on args
                if args.smooth_rewards:
                    # Smooth reward based on confidence in correct class
                    # Add temperature scaling for better numerical stability
                    temperature = 0.1  # Lower temperature makes the distribution more peaked
                    scaled_logits = all_logits / temperature
                    r = F.softmax(scaled_logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
                else:
                    # Binary reward (1 if correct, 0 otherwise)
                    r = (predictions == labels).float()
                
                # Add a small epsilon to rewards to avoid zero rewards
                r = r + 1e-6
                
                # Get log probabilities of ground truth classes
                # Use log_softmax for better numerical stability
                logp_pol = F.log_softmax(all_logits, dim=-1).gather(1, labels.unsqueeze(1))
                logp_ref = F.log_softmax(ref_logits_full, dim=-1).gather(1, labels.unsqueeze(1))
                
                # Compute GRPO loss
                loss, pg_loss, kl, entropy, advantages = grpo_loss(logp_pol, logp_ref, r)
                
                # Check for NaN values and handle them
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected at step {total_steps}, replacing with zero loss")
                    nan_steps += 1
                    
                    # Use cross-entropy loss as fallback for stability
                    loss = F.cross_entropy(all_logits, labels, label_smoothing=0.1)
                    kl = torch.tensor(0.0, device=device)
                    entropy = torch.tensor(0.0, device=device)
                    advantages = torch.tensor(0.0, device=device)
                
                # Calculate true entropy for logging
                true_entropy = compute_entropy(all_logits, temperature=2.0)  # Higher temperature for more meaningful entropy values
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                
                # Check for NaN gradients
                grad_is_nan = False
                for param in model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        grad_is_nan = True
                        break
                
                if grad_is_nan:
                    print(f"Warning: NaN/Inf gradient detected at step {total_steps}, skipping update")
                    nan_steps += 1
                    continue
                
                # Update model
                optimizer.step()
                
                # Step the learning rate scheduler
                scheduler.step()
                
                # Compute accuracy for logging
                accuracy = (predictions == labels).float().mean().item()
                
                # Log metrics
                if step % args.log_interval == 0:
                    # Get current learning rate
                    current_lr = scheduler.get_last_lr()[0]
                    
                    logger.write(
                        epoch * len(train_loader) + step,
                        loss.item(),
                        accuracy,
                        kl.item(),
                        true_entropy.item(),
                        advantages.mean().item() if isinstance(advantages, torch.Tensor) else advantages
                    )
                    
                    progress_bar.set_postfix(
                        loss=f"{loss.item():.3f}",
                        acc=f"{accuracy:.3f}",
                        kl=f"{kl.item():.3f}",
                        ent=f"{true_entropy.item():.2f}",
                        nan=nan_steps,
                        lr=f"{current_lr:.1e}"
                    )
            
            except Exception as e:
                print(f"Error at step {total_steps}: {e}")
                nan_steps += 1
                continue
        
        # Print training statistics
        print(f"Epoch {epoch+1} training completed. Total steps: {total_steps}, NaN steps: {nan_steps}")
                
        # Note: The evaluation phase uses cached text features from the initial model
        # If the policy's logit scale drifts, this might not be optimal for evaluation
        if args.eval_after_epoch:
            model.eval()
            correct = 0
            total = 0
            eval_steps = 0
            
            with torch.no_grad():
                for images, labels in tqdm(train_loader, desc=f"Evaluating epoch {epoch+1}"):
                    try:
                        eval_steps += 1
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        # Get image features
                        image_features = model.encode_image(images)
                        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
                        
                        # Compute similarities
                        logit_scale = model.model.logit_scale.exp().clamp(0, 100)
                        similarities = logit_scale * image_features @ text_features.t()
                        
                        # Get predictions
                        _, predicted = similarities.max(1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                    except Exception as e:
                        print(f"Error during evaluation step {eval_steps}: {e}")
                        continue
            
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1} accuracy: {accuracy:.2f}%")
    
    # Save final model
    if args.save_model:
        os.makedirs('models', exist_ok=True)
        model_suffix = f"{args.model_name.replace('/', '_')}"
        lora_suffix = f"_lora{args.lora_r}" if args.use_lora else ""
        model_path = f'models/clip_grpo_{model_suffix}{lora_suffix}.pt'
        
        if args.use_lora and LORA_AVAILABLE:
            # If using LoRA, save only the LoRA weights
            model.model.visual.save_pretrained(model_path)
            print(f"LoRA weights saved to {model_path}")
        else:
            # Otherwise save the full model state
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    print("Training complete!")
    return log_file

def main():
    parser = argparse.ArgumentParser(description="Train CLIP with GRPO and optional LoRA")
    
    # Model and dataset
    parser.add_argument('--model-name', type=str, default="ViT-B/32", help='CLIP model to use')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset to use (cifar100 or imagenet)')
    parser.add_argument('--data-path', type=str, default='./data', help='Path to ImageNet data')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.2, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=5.0, help='Gradient clipping (5-10 recommended)')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--eval-after-epoch', action='store_true', help='Evaluate after each epoch')
    parser.add_argument('--freeze-logit-scale', action='store_true', help='Freeze policy model logit scale')
    parser.add_argument('--smooth-rewards', action='store_true', help='Use confidence-based smooth rewards instead of binary rewards')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    
    # GRPO parameters
    parser.add_argument('--kl-coef', type=float, default=0.01, help='KL coefficient')
    
    # LoRA parameters
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA for fine-tuning')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank (r)')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha scaling factor')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--save-model', action='store_true', help='Save model')
    parser.add_argument('--plot-metrics', action='store_true', help='Plot training metrics')
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Train CLIP with GRPO
    log_file = train_clip_with_grpo(args)
    
    # Plot results
    if args.plot_metrics:
        try:
            from plot import plot_metrics
            metrics = ['mean_r', 'loss', 'kl', 'entropy']
            for metric in metrics:
                output_file = f'logs/clip_grpo_{args.model_name.replace("/", "_")}_{metric}.png'
                plot_metrics([log_file], metric=metric, output=output_file)
                print(f"{metric.capitalize()} plot saved to {output_file}")
        except ImportError:
            print("Could not import plot_metrics, skipping plotting.")

if __name__ == '__main__':
    main() 