"""
Training script for Hecto Car Classification
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from dataset import create_folds, make_loaders
from models import create_model, EMA, count_parameters
from augmentations import get_transforms
from utils import (
    seed_everything, load_config, save_config,
    get_optimizer, get_scheduler, one_hot,
    calculate_metrics, AverageMeter, save_checkpoint,
    get_lr
)


def train_epoch(model, loader, criterion, optimizer, scheduler, 
                warmup_scheduler, warmup_steps, global_step, 
                ema=None, device='cuda', amp=True):
    """
    Train for one epoch
    
    Args:
        model: Model instance
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        warmup_scheduler: Warmup scheduler
        warmup_steps: Number of warmup steps
        global_step: Global training step counter
        ema: EMA instance (optional)
        device: Device to use
        amp: Use automatic mixed precision
        
    Returns:
        Average loss, updated global step
    """
    model.train()
    losses = AverageMeter()
    
    # Create GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device == 'cuda')
    
    pbar = tqdm(loader, desc='Training')
    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=amp and device == 'cuda'):
            logits = model(imgs)
            loss = criterion(logits, targets)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Update schedulers
        if global_step < warmup_steps:
            warmup_scheduler.step()
        else:
            scheduler.step()
        global_step += 1
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Update metrics
        losses.update(loss.item(), imgs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lr': f'{get_lr(optimizer):.6f}'
        })
    
    return losses.avg, global_step


def validate(model, loader, criterion, num_classes, device='cuda'):
    """
    Validate model
    
    Args:
        model: Model instance
        loader: Validation data loader
        criterion: Loss function
        num_classes: Number of classes
        device: Device to use
        
    Returns:
        Metrics dictionary, predictions, labels
    """
    model.eval()
    losses = AverageMeter()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(imgs)
            
            # Calculate loss
            targets = one_hot(labels, num_classes)
            loss = criterion(logits, targets)
            losses.update(loss.item(), imgs.size(0))
            
            # Store predictions
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    # Concatenate all predictions
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_probs, num_classes)
    metrics['val_loss'] = losses.avg
    
    return metrics, all_probs, all_labels


def train_fold(fold, df, cfg, save_dir):
    """
    Train a single fold
    
    Args:
        fold: Fold index
        df: DataFrame with fold assignments
        cfg: Configuration dictionary
        save_dir: Directory to save models
        
    Returns:
        Best validation metrics
    """
    # Set seed
    seed_everything(cfg['training']['seed'])
    
    # Device
    device = cfg['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠️ CUDA not available, using CPU")
    
    # Create transforms
    train_tf = get_transforms(cfg, phase='train')
    val_tf = get_transforms(cfg, phase='val')
    
    # Create data loaders
    train_loader, val_loader, _ = make_loaders(fold, df, train_tf, val_tf, cfg)
    
    # Create model
    model = create_model(cfg).to(device)
    print(f"📊 Model parameters: {count_parameters(model):,}")
    
    # Create optimizer
    optimizer = get_optimizer(model, cfg)
    
    # Create scheduler
    total_steps = len(train_loader) * cfg['training']['epochs']
    scheduler, warmup_scheduler, warmup_steps = get_scheduler(optimizer, cfg, total_steps)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # EMA
    ema = None
    if cfg['ema']['enabled']:
        ema = EMA(model, decay=cfg['ema']['decay'])
    
    # Training loop
    best_metrics = {'log_loss': float('inf')}
    global_step = 0
    
    for epoch in range(cfg['training']['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Fold {fold}")
        print(f"{'='*50}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, warmup_scheduler, warmup_steps,
            global_step, ema, device, cfg['training']['amp']
        )
        
        # Validate
        if ema is not None:
            ema.apply_shadow()
        
        val_metrics, _, _ = validate(
            model, val_loader, criterion,
            cfg['data']['num_classes'], device
        )
        
        if ema is not None:
            ema.restore()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val Log Loss: {val_metrics['log_loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Top-5 Accuracy: {val_metrics['top5_accuracy']:.4f}")
        
        # Save best model
        if val_metrics['log_loss'] < best_metrics['log_loss']:
            best_metrics = val_metrics
            
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': cfg
            }
            
            if ema is not None:
                checkpoint['ema'] = ema.shadow
            
            save_path = os.path.join(save_dir, f'best_model_fold{fold}.pth')
            save_checkpoint(checkpoint, save_path)
            print(f"✨ New best model! Log Loss: {best_metrics['log_loss']:.4f}")
    
    print(f"\n🏆 Best Fold {fold} Log Loss: {best_metrics['log_loss']:.4f}")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Hecto Car Classification Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train (0-4)')
    parser.add_argument('--train_all_folds', action='store_true',
                        help='Train all 5 folds')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(cfg, save_dir / 'config.yaml')
    
    # Create folds
    print("📂 Creating data folds...")
    df, class_names = create_folds(
        cfg['data']['train_dir'],
        cfg['training']['num_folds'],
        cfg['training']['seed']
    )
    print(f"✅ Created {cfg['training']['num_folds']} folds with {len(class_names)} classes")
    
    # Train folds
    if args.train_all_folds:
        # Train all folds
        all_metrics = []
        for fold in range(cfg['training']['num_folds']):
            metrics = train_fold(fold, df, cfg, save_dir)
            all_metrics.append(metrics)
        
        # Print summary
        print("\n" + "="*50)
        print("Training Complete - All Folds Summary")
        print("="*50)
        for i, metrics in enumerate(all_metrics):
            print(f"Fold {i}: Log Loss = {metrics['log_loss']:.4f}")
        
        avg_logloss = np.mean([m['log_loss'] for m in all_metrics])
        print(f"\nAverage Log Loss: {avg_logloss:.4f}")
        
    else:
        # Train single fold
        fold = args.fold if args.fold is not None else 0
        train_fold(fold, df, cfg, save_dir)


if __name__ == '__main__':
    main()