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
from torch.utils.data import DataLoader
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import CustomDataset, build_transform
from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from tqdm import tqdm


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
    
    # Load train and test datasets
    train_dataset = CustomDataset(
        dataset_dir, 
        metadata_csv, 
        split='train', 
        transform=train_transform
    )
    
    test_dataset = CustomDataset(
        dataset_dir, 
        metadata_csv, 
        split='test', 
        transform=test_transform
    )
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
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
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
        
        # Test phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Test]")
        with torch.no_grad():
            for images, labels in test_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                test_bar.set_postfix(
                    loss=test_loss / (test_bar.n + 1),
                    acc=100 * test_correct / test_total
                )
        
        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {avg_test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        
        # Save checkpoint if best accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            epochs_without_improvement = 0
            print(f"  -> Saving best checkpoint (acc: {best_acc:.2f}%)")
            
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
                f"Best test accuracy: {best_acc:.2f}%"
            )
            break
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.checkpoint_path}")
    
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
        '--patience',
        type=int,
        default=10,
        help='Stop training if test accuracy does not improve for this many epochs. Set to 0 or a negative value to disable early stopping.'
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
    print("="*70 + "\n")

    if args.patience is not None and args.patience <= 0:
        args.patience = None
    
    best_acc = main(args)
