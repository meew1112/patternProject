#!/usr/bin/env python3
"""
Example script for training MedViT with a custom dataset.

This script demonstrates how to:
1. Load a custom dataset with CSV metadata
2. Configure the model
3. Train the model
4. Save checkpoints

Usage:
    python train_custom_dataset.py
    
Or with custom arguments:
    python train_custom_dataset.py --model_name MedViT_small --epochs 50 --batch_size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score


# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import CustomDataset, build_transform
from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from tqdm import tqdm


def plot_training_curves(train_losses, test_losses, test_accs, test_aucs, output_path):
    """Plot train/test loss, test accuracy, and test AUC."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Training & Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, test_accs, label='Test Acc', color='green')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_aucs, label='Test AUC', color='red')
    plt.title('Test AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def main(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build transforms
    train_transform, test_transform = build_transform(args)
    
    # Load custom dataset
    print("\nLoading custom dataset...")
    dataset_dir = args.dataset_dir
    metadata_csv = os.path.join(dataset_dir, 'meta67.csv')
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    # Build one global class mapping from the full metadata so all splits share identical label IDs
    metadata = pd.read_csv(metadata_csv)
    if 'diagnostic' not in metadata.columns:
        raise ValueError("Metadata CSV must contain a 'diagnostic' column")
    global_classes = sorted(metadata['diagnostic'].dropna().unique().tolist())
    global_class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}
    
    # Load train, val, and test datasets
    train_dataset = CustomDataset(
        dataset_dir, 
        metadata_csv, 
        split='train', 
        transform=train_transform,
        class_to_idx=global_class_to_idx,
        classes=global_classes
    )
    
    val_dataset = CustomDataset(
        dataset_dir, 
        metadata_csv, 
        split='val', 
        transform=test_transform,
        class_to_idx=global_class_to_idx,
        classes=global_classes
    )
    
    test_dataset = CustomDataset(
        dataset_dir, 
        metadata_csv, 
        split='test', 
        transform=test_transform,
        class_to_idx=global_class_to_idx,
        classes=global_classes
    )
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Compute class weights for handling class imbalance
    print("\nComputing class weights for imbalanced data...")
    labels = [train_dataset.class_to_idx[row['diagnostic']] for _, row in train_dataset.samples.iterrows()]
    from collections import Counter
    counts = Counter(labels)
    class_sample_count = np.array([counts.get(i, 1) for i in range(num_classes)])
    print(f"Class distribution: {dict(zip(train_dataset.classes, class_sample_count))}")

    class_weights = None
    sampler = None
    if args.imbalance == 'weights' or args.imbalance == 'both':
        class_weights = torch.tensor(class_sample_count.max() / class_sample_count, dtype=torch.float).to(device)
        print(f"Class weights (for loss): {class_weights.cpu().numpy()}")

    if args.imbalance == 'sampler' or args.imbalance == 'both':
        sample_weights = np.array([1.0 / class_sample_count[label] for label in labels])
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print("Using WeightedRandomSampler to oversample minority classes.\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    print(f"\nInitializing {args.model_name}...")
    model_classes = {
        'MedViT_tiny': MedViT_tiny,
        'MedViT_small': MedViT_small,
        'MedViT_base': MedViT_base,
        'MedViT_large': MedViT_large
    }
    
    if args.model_name not in model_classes:
        raise ValueError(f"Unknown model: {args.model_name}")
    
    model_class = model_classes[args.model_name]
    model = model_class(num_classes=num_classes)
    model = model.to(device)
    
    # Loss function (with class weights) and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    best_acc = 0.0
    epochs_without_improvement = 0
    history_train_losses = []
    history_val_losses = []
    history_val_accs = []
    history_val_aucs = []
    
    for epoch in range(args.epochs):
        # Train phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix(
                loss=train_loss / (train_bar.n + 1),
                acc=100 * train_correct / train_total
            )
        
        scheduler.step()
        
        # Validation phase (on val set during training)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_targets = []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
                val_targets.append(labels.cpu().numpy())
                
                val_bar.set_postfix(
                    loss=val_loss / (val_bar.n + 1),
                    acc=100 * val_correct / val_total
                )
        
        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        val_probs = np.concatenate(val_probs, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        try:
            if val_probs.shape[1] == 2:
                val_auc = roc_auc_score(val_targets, val_probs[:, 1])
            else:
                val_auc = roc_auc_score(val_targets, val_probs, multi_class='ovr')
        except ValueError:
            val_auc = 0.0

        history_train_losses.append(avg_train_loss)
        history_val_losses.append(avg_val_loss)
        history_val_accs.append(val_acc / 100.0)
        history_val_aucs.append(val_auc)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"  Val AUC:    {val_auc:.4f}")

        metrics_plot_path = os.path.join(os.path.dirname(args.checkpoint_path) or '.', 'training_metrics_plot.png')
        plot_training_curves(history_train_losses, history_val_losses, history_val_accs, history_val_aucs, metrics_plot_path)
        
        # Save checkpoint if best val accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0
            print(f"  -> Saving best checkpoint (val_acc: {best_acc:.2f}%)")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'classes': train_dataset.classes,
                'model_name': args.model_name,
            }
            torch.save(checkpoint, args.checkpoint_path)
        else:
            epochs_without_improvement += 1

        if args.patience is not None and epochs_without_improvement >= args.patience:
            print(
                f"\nEarly stopping triggered after {args.patience} epoch(s) without improvement. "
                f"Best val accuracy: {best_acc:.2f}%"
            )
            break
    
    print(f"\nTraining completed!")
    print(f"Best val accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.checkpoint_path}")
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    model.load_state_dict(torch.load(args.checkpoint_path)['model_state_dict'])
    model.eval()
    test_correct = 0
    test_total = 0
    test_probs = []
    test_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Test Evaluation"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            test_targets.append(labels.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    test_probs = np.concatenate(test_probs, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    try:
        if test_probs.shape[1] == 2:
            test_auc = roc_auc_score(test_targets, test_probs[:, 1])
        else:
            test_auc = roc_auc_score(test_targets, test_probs, multi_class='ovr')
    except ValueError:
        test_auc = 0.0
    
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Final Test AUC: {test_auc:.4f}")
    print("="*70)
    
    return best_acc


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MedViT with custom dataset'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='MedViT_tiny',
        choices=['MedViT_tiny', 'MedViT_small', 'MedViT_base', 'MedViT_large'],
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='./dataset',
        help='Path to dataset directory containing train/, test/, val/ folders and meta67.csv'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training and testing'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint/medvit_custom.pth',
        help='Path to save model checkpoint'
    )

    parser.add_argument(
        '--imbalance',
        type=str,
        default='weights',
        choices=['weights', 'sampler', 'both', 'none'],
        help='How to handle class imbalance: weights, sampler, both, or none'
    )

    parser.add_argument(
        '--use_augmix',
        type=lambda x: str(x).lower() in ['1', 'true', 'yes', 'y'],
        default=True,
        help='Enable AugMix during training'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Stop training if test accuracy does not improve for this many epochs. Set to 0 or a negative value to disable early stopping.'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        choices=['adam', 'adamw', 'sgd'],
        help='Optimizer to use'
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay for optimizer'
    )

    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for SGD'
    )
    
  
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print("="*70)
    print("MedViT Custom Dataset Training")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Imbalance handling: {args.imbalance}")
    print(f"Use AugMix: {args.use_augmix}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Weight decay: {args.weight_decay}")
    if args.optimizer == 'sgd':
        print(f"Momentum: {args.momentum}")
    print("="*70 + "\n")

    if args.patience is not None and args.patience <= 0:
        args.patience = None
    
    best_acc = main(args)
