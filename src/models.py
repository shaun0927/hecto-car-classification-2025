"""
Model architectures for Hecto Car Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange


class GeM(nn.Module):
    """
    Generalized Mean Pooling
    
    Args:
        p: Initial pooling parameter (default: 3.0)
        eps: Small constant for numerical stability
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=(x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).flatten(1)


class SubCenterArcFace(nn.Module):
    """
    Sub-center ArcFace head for fine-grained classification
    
    Args:
        in_features: Input feature dimension
        out_classes: Number of output classes
        k: Number of sub-centers per class
        s: Scale factor for logits
    """
    def __init__(self, in_features, out_classes, k=3, s=30.0):
        super().__init__()
        self.out_classes = out_classes
        self.k = k
        self.s = s
        
        # Weight shape: [out_classes * k, in_features]
        self.weight = nn.Parameter(torch.randn(out_classes * k, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Normalize features and weights
        x = F.normalize(x, dim=1)
        W = F.normalize(self.weight, dim=1)
        
        # Compute cosine similarity
        cos = F.linear(x, W)  # (B, out_classes * k)
        
        # Reshape to (B, classes, k) and take max over sub-centers
        cos = rearrange(cos, "b (c k) -> b c k", c=self.out_classes, k=self.k)
        cos_max, _ = torch.max(cos, dim=2)  # (B, classes)
        
        # Scale logits
        logits = cos_max * self.s
        return logits


class CarNet(nn.Module):
    """
    Main model architecture for car classification
    
    Args:
        n_classes: Number of classes
        backbone_name: Name of timm backbone
        k: Number of sub-centers for ArcFace
        gem_p: Initial GeM pooling parameter
        scale: ArcFace scale factor
    """
    def __init__(
        self,
        n_classes,
        backbone_name="convnext_base",
        k=3,
        gem_p=3.0,
        scale=30.0
    ):
        super().__init__()
        
        # Create backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        in_features = self.backbone.num_features
        
        # Pooling and head
        self.pool = GeM(p=gem_p)
        self.head = SubCenterArcFace(in_features, n_classes, k=k, s=scale)
        
    def forward(self, x):
        # Extract features
        feat = self.backbone.forward_features(x)  # (B, C, H, W)
        
        # Pool features
        feat = self.pool(feat)  # (B, C)
        
        # Classification head
        logits = self.head(feat)  # (B, n_classes)
        
        return logits
    
    def extract_features(self, x):
        """Extract features without classification head"""
        feat = self.backbone.forward_features(x)
        feat = self.pool(feat)
        return feat


def create_model(cfg):
    """
    Factory function to create model from configuration
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Model instance
    """
    model = CarNet(
        n_classes=cfg['data']['num_classes'],
        backbone_name=cfg['model']['backbone'],
        k=cfg['model']['num_subcenters'],
        gem_p=cfg['model']['gem_p'],
        scale=cfg['model']['scale_factor']
    )
    
    return model


class EMA:
    """
    Exponential Moving Average for model weights
    
    Args:
        model: PyTorch model
        decay: EMA decay rate
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register model parameters for EMA"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + \
                              self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)