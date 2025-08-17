"""
Utility functions for training and evaluation
"""

import os
import random
import numpy as np
import torch
import yaml
from sklearn.metrics import log_loss


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_optimizer(model, cfg):
    """
    Create optimizer with differential learning rates
    
    Args:
        model: PyTorch model
        cfg: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    opt_cfg = cfg['optimizer']
    
    # Separate backbone and head parameters
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': opt_cfg['base_lr']},
        {'params': head_params, 'lr': opt_cfg['base_lr'] * opt_cfg['head_lr_multiplier']}
    ]
    
    # Create optimizer
    if opt_cfg['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=opt_cfg['base_lr'],
            betas=opt_cfg['betas'],
            weight_decay=opt_cfg['weight_decay']
        )
    elif opt_cfg['type'] == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=opt_cfg['base_lr'],
            betas=opt_cfg['betas'],
            weight_decay=opt_cfg['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_cfg['type']}")
    
    return optimizer


def get_scheduler(optimizer, cfg, total_steps):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        cfg: Configuration dictionary
        total_steps: Total training steps
        
    Returns:
        Scheduler instance
    """
    sch_cfg = cfg['scheduler']
    
    if sch_cfg['type'] == 'cosine':
        # Cosine annealing with warmup
        warmup_steps = sch_cfg['warmup_epochs'] * (total_steps // cfg['training']['epochs'])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=sch_cfg['min_lr']
        )
        
        # Warmup scheduler
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return scheduler, warmup_scheduler, warmup_steps
    
    else:
        raise ValueError(f"Unknown scheduler type: {sch_cfg['type']}")


def one_hot(labels, num_classes):
    """
    Convert labels to one-hot encoding
    
    Args:
        labels: Tensor of labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded tensor
    """
    y = torch.zeros((labels.size(0), num_classes), device=labels.device)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    return y


def calculate_metrics(y_true, y_pred, num_classes):
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    # Calculate log loss
    logloss = log_loss(y_true, y_pred, labels=list(range(num_classes)))
    
    # Calculate accuracy
    y_pred_labels = np.argmax(y_pred, axis=1)
    accuracy = (y_pred_labels == y_true).mean()
    
    # Calculate top-5 accuracy
    top5_preds = np.argsort(y_pred, axis=1)[:, -5:]
    top5_acc = np.mean([y_true[i] in top5_preds[i] for i in range(len(y_true))])
    
    metrics = {
        'log_loss': logloss,
        'accuracy': accuracy,
        'top5_accuracy': top5_acc
    }
    
    return metrics


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state
        filename: Path to save checkpoint
    """
    torch.save(state, filename)
    print(f"✅ Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        filename: Path to checkpoint file
        model: Model instance
        optimizer: Optimizer instance (optional)
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filename, map_location='cpu')
    
    # Load model state
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"✅ Checkpoint loaded: {filename}")
    return checkpoint


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']