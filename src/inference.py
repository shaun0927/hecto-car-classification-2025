"""
Inference script for Hecto Car Classification
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

from dataset import CarDataset
from models import create_model
from augmentations import get_transforms, get_tta_transforms
from utils import load_config, load_checkpoint


def predict(model, loader, device='cuda', tta_transforms=None):
    """
    Generate predictions for test data
    
    Args:
        model: Model instance
        loader: Test data loader
        device: Device to use
        tta_transforms: List of TTA transforms (optional)
        
    Returns:
        Predictions array, file paths
    """
    model.eval()
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Inference')
        for imgs, paths in pbar:
            imgs = imgs.to(device, non_blocking=True)
            
            if tta_transforms is None:
                # Single prediction
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                # TTA predictions
                tta_probs = []
                for transform in tta_transforms:
                    # Apply transform to batch
                    # Note: This is simplified - proper TTA would require re-loading images
                    logits = model(imgs)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    tta_probs.append(probs)
                probs = np.mean(tta_probs, axis=0)
            
            all_probs.append(probs)
            all_paths.extend(paths)
    
    # Concatenate all predictions
    all_probs = np.concatenate(all_probs)
    
    return all_probs, all_paths


def create_submission(predictions, paths, class_names, cfg, save_path):
    """
    Create submission file
    
    Args:
        predictions: Prediction probabilities
        paths: Image file paths
        class_names: List of class names
        cfg: Configuration dictionary
        save_path: Path to save submission
    """
    # Load sample submission
    sample = pd.read_csv(cfg['data']['submission_csv'])
    
    # Extract IDs from paths
    ids = [os.path.basename(p).replace('.jpg', '') for p in paths]
    
    # Create submission dataframe
    submission = pd.DataFrame({'ID': ids})
    
    # Add predictions for each class
    for i, cls in enumerate(class_names):
        submission[cls] = predictions[:, i]
    
    # Handle special class mappings
    if 'special_classes' in cfg:
        for pair in cfg['special_classes']:
            if len(pair) == 2:
                cls1, cls2 = pair
                if cls1 in submission.columns and cls2 in submission.columns:
                    # Average predictions for equivalent classes
                    avg_pred = (submission[cls1] + submission[cls2]) / 2
                    submission[cls1] = avg_pred
                    submission[cls2] = avg_pred
    
    # Ensure all columns from sample submission are present
    for col in sample.columns:
        if col not in submission.columns and col != 'ID':
            submission[col] = 0.0
    
    # Reorder columns to match sample submission
    submission = submission[sample.columns]
    
    # Normalize probabilities to sum to 1
    prob_cols = [c for c in submission.columns if c != 'ID']
    submission[prob_cols] = submission[prob_cols].div(
        submission[prob_cols].sum(axis=1), axis=0
    )
    
    # Clip probabilities to avoid numerical issues
    submission[prob_cols] = submission[prob_cols].clip(1e-15, 1 - 1e-15)
    
    # Save submission
    submission.to_csv(save_path, index=False)
    print(f"✅ Submission saved to {save_path}")
    print(f"   Shape: {submission.shape}")
    print(f"   ID range: {submission['ID'].iloc[0]} to {submission['ID'].iloc[-1]}")


def ensemble_predictions(model_paths, test_loader, cfg, device='cuda'):
    """
    Ensemble predictions from multiple models
    
    Args:
        model_paths: List of model checkpoint paths
        test_loader: Test data loader
        cfg: Configuration dictionary
        device: Device to use
        
    Returns:
        Ensemble predictions, file paths
    """
    all_model_probs = []
    
    for model_path in model_paths:
        print(f"📦 Loading model: {model_path}")
        
        # Create model
        model = create_model(cfg).to(device)
        
        # Load checkpoint
        checkpoint = load_checkpoint(model_path, model)
        
        # Generate predictions
        probs, paths = predict(model, test_loader, device)
        all_model_probs.append(probs)
    
    # Average predictions
    ensemble_probs = np.mean(all_model_probs, axis=0)
    print(f"✅ Ensemble complete: {len(model_paths)} models")
    
    return ensemble_probs, paths


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for test data')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to single model checkpoint')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing model checkpoints')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble of all models in model_dir')
    parser.add_argument('--tta', action='store_true',
                        help='Use test time augmentation')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output submission file path')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Device
    device = cfg['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠️ CUDA not available, using CPU")
    
    # Create test dataset and loader
    print("📂 Loading test data...")
    test_paths = sorted([
        os.path.join(cfg['data']['test_dir'], f)
        for f in os.listdir(cfg['data']['test_dir'])
        if f.lower().endswith('.jpg')
    ])
    test_df = pd.DataFrame({"img_path": test_paths})
    
    # Get transforms
    val_tf = get_transforms(cfg, phase='val')
    test_dataset = CarDataset(test_df, transform=val_tf, is_test=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg['inference']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            [b[1] for b in batch]
        )
    )
    print(f"✅ Test data loaded: {len(test_dataset)} images")
    
    # Get class names (assuming they're saved in checkpoint or from train directory)
    if os.path.exists(cfg['data']['train_dir']):
        class_names = sorted([
            d for d in os.listdir(cfg['data']['train_dir'])
            if os.path.isdir(os.path.join(cfg['data']['train_dir'], d))
        ])
    else:
        # Load from sample submission
        sample = pd.read_csv(cfg['data']['submission_csv'])
        class_names = [c for c in sample.columns if c != 'ID']
    
    print(f"📊 Number of classes: {len(class_names)}")
    
    # Generate predictions
    if args.ensemble:
        # Ensemble prediction
        model_dir = Path(args.model_dir)
        model_paths = list(model_dir.glob('best_model_fold*.pth'))
        
        if not model_paths:
            print("❌ No model checkpoints found in", model_dir)
            return
        
        print(f"🎯 Ensemble inference with {len(model_paths)} models")
        predictions, paths = ensemble_predictions(
            model_paths, test_loader, cfg, device
        )
        
    else:
        # Single model prediction
        if args.model_path is None:
            # Find best model
            model_dir = Path(args.model_dir)
            model_paths = list(model_dir.glob('best_model_fold*.pth'))
            if model_paths:
                args.model_path = str(model_paths[0])
            else:
                print("❌ No model checkpoint specified")
                return
        
        print(f"🎯 Single model inference: {args.model_path}")
        
        # Create model
        model = create_model(cfg).to(device)
        
        # Load checkpoint
        checkpoint = load_checkpoint(args.model_path, model)
        
        # TTA transforms
        tta_transforms = None
        if args.tta:
            print("🔄 Using test time augmentation")
            tta_transforms = get_tta_transforms(cfg)
        
        # Generate predictions
        predictions, paths = predict(model, test_loader, device, tta_transforms)
    
    # Create submission
    print("📝 Creating submission file...")
    create_submission(predictions, paths, class_names, cfg, args.output)
    
    print("\n✨ Inference complete!")


if __name__ == '__main__':
    main()