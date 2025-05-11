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

# Import peft for LoRA support if available
try:
    from peft import LoraConfig, get_peft_model
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
    def __init__(self, clip_model, device, use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.model = clip_model
        self.device = device
        self.use_lora = use_lora
        
        # Freeze the text encoder as reference model
        for param in self.model.transformer.parameters():
            param.requires_grad = False
            
        # Apply LoRA if requested and available
        if self.use_lora and LORA_AVAILABLE:
            print(f"Applying LoRA to CLIP visual encoder (r={lora_r}, alpha={lora_alpha})")
            
            # Freeze all parameters first
            for param in self.model.visual.parameters():
                param.requires_grad = False
            
            # For ViT-based models, apply LoRA to attention layers
            if hasattr(self.model.visual, 'transformer'):
                # Try to apply LoRA with different target module strategies
                lora_applied = False
                
                # Strategy 1: Get list of attention layer names in the model
                target_modules = []
                for name, _ in self.model.visual.named_modules():
                    if name.endswith('.attn.out_proj') or name.endswith('.attn.in_proj'):
                        target_modules.append(name.split('.')[-2] + '.' + name.split('.')[-1])
                
                if target_modules:
                    print(f"Applying LoRA to modules: {target_modules}")
                    try:
                        # For ViT architecture (ViT-B/32, ViT-L/14, etc.)
                        lora_config = LoraConfig(
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            target_modules=target_modules,
                            lora_dropout=lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM"  # This is the default for attention models
                        )
                        
                        # Apply LoRA to the model
                        self.model.visual = get_peft_model(self.model.visual, lora_config)
                        lora_applied = True
                    except ValueError as e:
                        print(f"First LoRA strategy failed: {e}")
                
                # Strategy 2: Use out_proj only
                if not lora_applied:
                    try:
                        print("Trying LoRA with 'out_proj' only")
                        lora_config = LoraConfig(
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            target_modules=["out_proj"],
                            lora_dropout=lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM"
                        )
                        self.model.visual = get_peft_model(self.model.visual, lora_config)
                        lora_applied = True
                    except ValueError as e:
                        print(f"Second LoRA strategy failed: {e}")
                
                # Strategy 3: Let PEFT library find the linear layers automatically
                if not lora_applied:
                    try:
                        print("Trying LoRA with automatic target detection")
                        lora_config = LoraConfig(
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            target_modules=None,  # Auto-detect Linear layers
                            lora_dropout=lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM"
                        )
                        self.model.visual = get_peft_model(self.model.visual, lora_config)
                        lora_applied = True
                    except ValueError as e:
                        print(f"Third LoRA strategy failed: {e}")
                
                # If all LoRA strategies fail, fall back to full fine-tuning
                if not lora_applied:
                    print("All LoRA strategies failed, falling back to full fine-tuning")
                    for param in self.model.visual.parameters():
                        param.requires_grad = True
                else:
                    # Count trainable parameters
                    trainable_params = sum(p.numel() for p in self.model.visual.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in self.model.visual.parameters())
                    print(f"LoRA trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total)")
            
            else:
                # For non-ViT architectures (ResNet-based), no LoRA so train the whole encoder
                print("Warning: LoRA config not applied - model architecture not supported (defaulting to full fine-tuning)")
                for param in self.model.visual.parameters():
                    param.requires_grad = True
        else:
            # Traditional fine-tuning - train the whole visual encoder
            if not self.use_lora:
                print("Using full fine-tuning for visual encoder")
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
    
    # Create model wrapper
    model = CLIPWrapper(
        clip_model, 
        device, 
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Load data
    train_loader, classes = load_data(args)
    
    # Generate text features for each class
    text_features = get_text_features(clip_model, classes, device=args.device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create a reference model (frozen)
    ref_model, _ = clip.load(args.model_name, device=device, jit=False)
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # GRPO loss
    grpo_loss = GRPOLoss(args.kl_coef)
    
    # Logger
    model_suffix = f"_{args.model_name.replace('/', '_')}"
    lora_suffix = f"_lora{args.lora_r}" if args.use_lora else ""
    log_file = f'logs/clip_grpo{model_suffix}{lora_suffix}.csv'
    logger = CSVLogger(log_file)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get text descriptions for current batch
            batch_text_features = text_features[labels]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass through policy model
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Policy logprobs (current model)
            logit_scale = model.model.logit_scale.exp()
            logits = logit_scale * image_features @ batch_text_features.t()
            logp_tokens = F.log_softmax(logits, dim=-1)
            
            # Reference logprobs (frozen original model)
            with torch.no_grad():
                ref_image_features = ref_model.encode_image(images)
                ref_image_features = ref_image_features / ref_image_features.norm(dim=-1, keepdim=True)
                ref_logits = logit_scale * ref_image_features @ batch_text_features.t()
                logp_ref = F.log_softmax(ref_logits, dim=-1).sum(-1)
            
            # Reward is accuracy (diagonal elements of the similarity matrix)
            batch_size = images.size(0)
            r = torch.zeros(batch_size, device=device)
            for i in range(batch_size):
                # Reward is 1 if the correct class has the highest similarity
                predicted_class = logits[i].argmax().item()
                r[i] = 1.0 if predicted_class == labels[i].item() else 0.0
            
            # Compute GRPO loss
            loss, pg_loss, kl, entropy, advantages = grpo_loss(logp_tokens.sum(-1), logp_ref, r)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                
            # Update model
            optimizer.step()
            
            # Log metrics
            if step % args.log_interval == 0:
                # Compute accuracy
                accuracy = r.mean().item()
                
                logger.write(
                    epoch * len(train_loader) + step,
                    loss.item(),
                    accuracy,
                    kl.item(),
                    entropy.item(),
                    advantages.mean().item()
                )
                
                progress_bar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    acc=f"{accuracy:.3f}",
                    kl=f"{kl.item():.3f}",
                    ent=f"{entropy.item():.2f}"
                )
        
        # Evaluate on the entire dataset after each epoch
        if args.eval_after_epoch:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(train_loader, desc=f"Evaluating epoch {epoch+1}"):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Get image features
                    image_features = model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarities
                    logit_scale = model.model.logit_scale.exp()
                    similarities = logit_scale * image_features @ text_features.t()
                    
                    # Get predictions
                    _, predicted = similarities.max(1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            
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
    return logger.filepath

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
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--eval-after-epoch', action='store_true', help='Evaluate after each epoch')
    
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