"""
Dataset and DataLoader implementations for Hecto Car Classification
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold


class CarDataset(Dataset):
    """
    Custom Dataset for car image classification
    
    Args:
        df: DataFrame with columns ['img_path', 'label', 'fold']
        transform: Albumentations transform object
        is_test: Boolean flag for test mode
    """
    def __init__(self, df, transform=None, is_test=False):
        self.paths = df["img_path"].tolist()
        self.labels = None if is_test else df["label"].tolist()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Read and convert image
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform is not None:
            img = self.transform(image=img)["image"]

        # Return based on mode
        if self.is_test:
            return img, self.paths[idx]
        else:
            label = self.labels[idx]
            return img, label


def create_folds(data_dir, n_folds=5, seed=42):
    """
    Create stratified k-fold split for training data
    
    Args:
        data_dir: Path to training data directory
        n_folds: Number of folds
        seed: Random seed
        
    Returns:
        DataFrame with fold assignments
    """
    # Get class names from directory structure
    class_names = sorted([d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))])
    cls2id = {c: i for i, c in enumerate(class_names)}
    
    # Collect all image paths
    records = []
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                records.append([os.path.join(cls_dir, fname), cls2id[cls]])
    
    # Create DataFrame
    df = pd.DataFrame(records, columns=["img_path", "label"])
    
    # Add fold column
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold
    
    return df, class_names


def rand_bbox(W, H, lam):
    """
    Generate random bounding box for CutMix
    
    Args:
        W: Image width
        H: Image height
        lam: Lambda value from beta distribution
        
    Returns:
        Bounding box coordinates (bbx1, bby1, bbx2, bby2)
    """
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_cutmix_collate(n_classes, alpha=1.0, prob=0.5):
    """
    Factory function for CutMix collate function
    
    Args:
        n_classes: Number of classes
        alpha: Beta distribution parameter
        prob: Probability of applying CutMix
        
    Returns:
        Collate function for DataLoader
    """
    def _collate(batch):
        imgs, labels = list(zip(*batch))
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels)

        # Convert to one-hot
        onehot = torch.zeros(imgs.size(0), n_classes, dtype=torch.float32)
        onehot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply CutMix
        if np.random.rand() < prob:
            lam = np.random.beta(alpha, alpha)
            rand_idx = torch.randperm(imgs.size(0))
            shuffled_imgs = imgs[rand_idx]
            shuffled_onehot = onehot[rand_idx]

            _, H, W = imgs.shape[1:]
            bbx1, bby1, bbx2, bby2 = rand_bbox(W, H, lam)

            # Mix images
            imgs[:, :, bby1:bby2, bbx1:bbx2] = shuffled_imgs[:, :, bby1:bby2, bbx1:bbx2]
            
            # Adjust lambda based on actual box size
            lam_adj = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            
            # Mix labels
            onehot = onehot * lam_adj + shuffled_onehot * (1. - lam_adj)

        return imgs, onehot
    
    return _collate


def make_loaders(fold, df_full, train_tf, val_tf, cfg):
    """
    Create train, validation, and test data loaders
    
    Args:
        fold: Fold index for validation
        df_full: Full DataFrame with fold assignments
        train_tf: Training transforms
        val_tf: Validation transforms
        cfg: Configuration dictionary
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split data
    train_df = df_full[df_full.fold != fold].reset_index(drop=True)
    val_df = df_full[df_full.fold == fold].reset_index(drop=True)
    
    # Create datasets
    train_set = CarDataset(train_df, transform=train_tf)
    val_set = CarDataset(val_df, transform=val_tf)
    
    # Get number of classes
    n_classes = cfg['data']['num_classes']
    
    # Create CutMix collate function
    cutmix_fn = get_cutmix_collate(
        n_classes=n_classes,
        alpha=cfg['cutmix']['alpha'],
        prob=cfg['cutmix']['prob']
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        collate_fn=cutmix_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch])
        )
    )
    
    # Create test loader
    test_paths = sorted([
        os.path.join(cfg['data']['test_dir'], f)
        for f in os.listdir(cfg['data']['test_dir'])
        if f.lower().endswith('.jpg')
    ])
    test_df = pd.DataFrame({"img_path": test_paths})
    test_set = CarDataset(test_df, transform=val_tf, is_test=True)
    
    test_loader = DataLoader(
        test_set,
        batch_size=cfg['inference']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            [b[1] for b in batch]
        )
    )
    
    print(f"📊 Fold {fold} | train={len(train_set)} | val={len(val_set)} | test={len(test_set)}")
    
    return train_loader, val_loader, test_loader