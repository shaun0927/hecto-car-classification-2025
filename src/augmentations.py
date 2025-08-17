"""
Data augmentation pipelines for training and validation
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(cfg, phase="train"):
    """
    Get augmentation pipeline based on phase
    
    Args:
        cfg: Configuration dictionary
        phase: 'train' or 'val'
        
    Returns:
        Albumentations Compose object
    """
    img_size = cfg['augmentation']['image_size']
    mean = cfg['augmentation']['mean']
    std = cfg['augmentation']['std']
    
    if phase == "train":
        aug_cfg = cfg['augmentation']['train']
        
        transforms = [
            # Random Resized Crop
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=aug_cfg['random_resized_crop']['scale'],
                ratio=aug_cfg['random_resized_crop']['ratio'],
                p=1.0
            ),
            
            # Color augmentations (one of the two)
            A.OneOf([
                A.ColorJitter(
                    brightness=aug_cfg['color_jitter']['brightness'],
                    contrast=aug_cfg['color_jitter']['contrast'],
                    saturation=aug_cfg['color_jitter']['saturation'],
                    hue=aug_cfg['color_jitter']['hue'],
                    p=1.0
                ),
                A.ShiftScaleRotate(
                    shift_limit=aug_cfg['shift_scale_rotate']['shift_limit'],
                    scale_limit=aug_cfg['shift_scale_rotate']['scale_limit'],
                    rotate_limit=aug_cfg['shift_scale_rotate']['rotate_limit'],
                    p=1.0
                ),
            ], p=1.0),
            
            # Horizontal Flip
            A.HorizontalFlip(p=aug_cfg['horizontal_flip']['p']),
            
            # Coarse Dropout (Cutout)
            A.CoarseDropout(
                max_holes=aug_cfg['coarse_dropout']['max_holes'],
                max_height=aug_cfg['coarse_dropout']['max_height'],
                max_width=aug_cfg['coarse_dropout']['max_width'],
                min_holes=aug_cfg['coarse_dropout']['min_holes'],
                min_height=aug_cfg['coarse_dropout']['min_height'],
                min_width=aug_cfg['coarse_dropout']['min_width'],
                fill_value=0,
                p=aug_cfg['coarse_dropout']['p']
            ),
        ]
        
    else:  # validation or test
        val_cfg = cfg['augmentation']['val']
        transforms = [
            A.Resize(
                height=val_cfg['resize']['height'],
                width=val_cfg['resize']['width']
            ),
        ]
    
    # Add normalization and tensor conversion
    transforms.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)


def get_tta_transforms(cfg):
    """
    Get Test Time Augmentation transforms
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        List of augmentation pipelines for TTA
    """
    img_size = cfg['augmentation']['image_size']
    mean = cfg['augmentation']['mean']
    std = cfg['augmentation']['std']
    
    tta_transforms = []
    
    # Original
    tta_transforms.append(
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    )
    
    # Horizontal Flip
    tta_transforms.append(
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    )
    
    # Different scales
    for scale in [0.9, 1.1]:
        size = int(img_size * scale)
        tta_transforms.append(
            A.Compose([
                A.Resize(height=size, width=size),
                A.CenterCrop(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        )
    
    return tta_transforms


def mixup_data(x, y, alpha=1.0):
    """
    Apply MixUp augmentation
    
    Args:
        x: Input images
        y: Input labels (one-hot)
        alpha: Beta distribution parameter
        
    Returns:
        Mixed images and labels
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y