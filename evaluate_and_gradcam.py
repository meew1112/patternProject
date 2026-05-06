#!/usr/bin/env python3
"""
Interactive evaluation script: Test model and visualize confusion matrix with Grad-CAM.

Usage:
    python evaluate_and_gradcam.py --checkpoint ./checkpoint_best.pth --model_name MedViT_base --dataset_dir ./dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from datasets import CustomDataset, build_transform
from confusion_gradcam import analyze_confusion_pair


MODEL_CLASSES = {
    'MedViT_tiny': MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base': MedViT_base,
    'MedViT_large': MedViT_large
}


def _extract_state_dict(checkpoint):
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    if 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def load_model(checkpoint_path, model_name, num_classes, device='cuda'):
    """Load trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    saved_model_name = checkpoint.get('model_name')

    candidate_names = []
    for candidate in [saved_model_name, model_name, *MODEL_CLASSES.keys()]:
        if candidate and candidate not in candidate_names:
            candidate_names.append(candidate)

    last_error = None
    for candidate_name in candidate_names:
        if candidate_name not in MODEL_CLASSES:
            continue

        model = MODEL_CLASSES[candidate_name](num_classes=num_classes)
        try:
            model.load_state_dict(state_dict)
            if saved_model_name and saved_model_name != model_name:
                print(f"Checkpoint specifies {saved_model_name}; using it instead of --model_name {model_name}.")
            elif candidate_name != model_name:
                print(f"Using {candidate_name} because it matches the checkpoint.")
            return model.to(device).eval()
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(
        "Could not match checkpoint weights to any MedViT variant. "
        f"Last error: {last_error}"
    )


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set and return predictions, targets, and confusion matrix."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Compute confusion matrix and metrics
    cm = confusion_matrix(all_targets, all_preds)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return all_preds, all_targets, all_probs, cm, accuracy, precision, recall, f1


def plot_confusion_matrix(cm, class_names, figsize=(10, 8)):
    """Plot and display confusion matrix."""
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Normalized Confusion Matrix')
    
    plt.tight_layout()
    return fig, ax


def interactive_gradcam_selection(cm, class_names, model, test_dataset, device, 
                                  checkpoint_path, model_name, dataset_dir, output_dir):
    """Interactive selection of confusion matrix cells for Grad-CAM visualization."""
    
    num_classes = len(class_names)
    print("\n" + "="*70)
    print("INTERACTIVE GRAD-CAM VISUALIZATION")
    print("="*70)
    print(f"\nConfusion matrix shape: {cm.shape}")
    print(f"Available classes: {class_names}")
    print("\nSelect confusion matrix cells to visualize Grad-CAM.")
    print("Format: Enter 'row col' (e.g., '0 1' for true_class=0, pred_class=1)")
    print("Type 'quit' to exit.\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    while True:
        user_input = input("Enter [true_class pred_class] or 'quit': ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting Grad-CAM visualization.")
            break
        
        try:
            parts = user_input.split()
            if len(parts) != 2:
                print("Invalid input. Please enter two numbers separated by space.")
                continue
            
            true_class = int(parts[0])
            pred_class = int(parts[1])
            
            if true_class < 0 or true_class >= num_classes or pred_class < 0 or pred_class >= num_classes:
                print(f"Invalid class indices. Please enter values between 0 and {num_classes-1}.")
                continue
            
            # Get misclassifications for this pair
            misclassifications = []
            for idx, (img_path, label) in enumerate(test_dataset.data):
                # This requires knowing predictions - for now we'll recompute
                pass
            
            print(f"\nGenerating Grad-CAM for true_class={true_class} ({class_names[true_class]}) "
                  f"-> pred_class={pred_class} ({class_names[pred_class]})...")
            
            # Analyze confusion pair with Grad-CAM
            results = analyze_confusion_pair(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                true_class=true_class,
                pred_class=pred_class,
                cm=cm,
                misclassifications=[],  # Will be computed internally
                label_names=class_names,
                n_samples=3,
                save_dir=output_dir,
                dataset_dir=dataset_dir,
                target_mode="predicted"
            )
            
            if results:
                print(f"✓ Generated Grad-CAM visualizations for {len(results)} samples:")
                for i, result in enumerate(results):
                    print(f"  {i+1}. {result.get('image_path', 'N/A')} -> {result.get('cam_path', 'N/A')}")
                print(f"\nSaved to: {output_dir}\n")
            else:
                print(f"No misclassifications found for this pair.\n")
        
        except ValueError:
            print("Invalid input. Please enter two integers.")
        except Exception as e:
            print(f"Error: {e}\n")


def main(args):
    """Main evaluation function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Build transforms
    _, test_transform = build_transform(args)
    
    # Load test dataset
    print("Loading test dataset...")
    dataset_dir = args.dataset_dir
    metadata_csv = os.path.join(dataset_dir, 'meta67.csv')
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    
    test_dataset = CustomDataset(
        data_dir=dataset_dir,
        metadata_csv=metadata_csv,
        split='TEST',
        transform=test_transform
    )
    
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {class_names}\n")
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = load_model(args.checkpoint, args.model_name, num_classes, device)
    print("Model loaded successfully!\n")
    
    # Evaluate model
    print("Evaluating model on test set...")
    preds, targets, probs, cm, accuracy, precision, recall, f1 = evaluate_model(
        model, test_loader, device
    )
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}\n")
    
    # Display confusion matrix
    print("="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    fig, ax = plot_confusion_matrix(cm, class_names)
    
    # Save and show
    output_dir = os.path.join(args.output_dir, 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    
    plt.show()
    
    # Interactive Grad-CAM selection
    interactive_gradcam_selection(
        cm=cm,
        class_names=class_names,
        model=model,
        test_dataset=test_dataset,
        device=device,
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        dataset_dir=dataset_dir,
        output_dir=output_dir
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained MedViT model with interactive Grad-CAM'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='MedViT_small',
        choices=['MedViT_tiny', 'MedViT_small', 'MedViT_base', 'MedViT_large'],
        help='Model architecture (must match the checkpoint)'
    )
    
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='./dataset',
        help='Path to dataset directory with meta67.csv'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./analysis_outputs',
        help='Directory to save visualizations'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
