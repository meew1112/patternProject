#!/usr/bin/env python3
"""
Example: Using analyze_confusion_pair for Grad-CAM error analysis.

Shows how to:
1. Evaluate model and collect misclassifications
2. Use analyze_confusion_pair to visualize specific confusion patterns
"""

import torch
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from datasets import CustomDataset, build_transform
from confusion_gradcam import analyze_confusion_pair


MODEL_CLASSES = {
    'MedViT_tiny': MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base': MedViT_base,
    'MedViT_large': MedViT_large,
}


def extract_state_dict(checkpoint):
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    if 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def detect_model_name(checkpoint, state_dict, num_classes, device):
    saved_name = checkpoint.get('model_name')
    candidate_names = []
    for candidate in [saved_name, *MODEL_CLASSES.keys()]:
        if candidate and candidate not in candidate_names:
            candidate_names.append(candidate)

    for candidate_name in candidate_names:
        model = MODEL_CLASSES[candidate_name](num_classes=num_classes)
        try:
            model.load_state_dict(state_dict)
            return candidate_name, model.to(device).eval()
        except RuntimeError:
            continue

    raise RuntimeError(
        "Could not match the checkpoint to any supported MedViT variant. "
        "Make sure the checkpoint was saved with one of the MedViT_* architectures."
    )


def main():
    # ============================================================================
    # STEP 1: Evaluate model and collect predictions + misclassifications
    # ============================================================================
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint_path = './checkpoint_best.pth'
    dataset_dir = './dataset'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    num_classes = checkpoint.get('num_classes', 6)
    class_names = checkpoint.get('classes', [f'class_{i}' for i in range(num_classes)])

    model_name, model = detect_model_name(checkpoint, state_dict, num_classes, device)

    print(f"Loaded {model_name}: {num_classes} classes")
    print(f"Classes: {class_names}\n")
    
    # Load test dataset
    _, test_transform = build_transform(type('Args', (), {'dataset': 'custom'}))
    metadata_csv = os.path.join(dataset_dir, 'meta67.csv')
    
    test_dataset = CustomDataset(
        data_dir=dataset_dir,
        metadata_csv=metadata_csv,
        split='TEST',
        transform=test_transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    # ============================================================================
    # STEP 2: Evaluate and collect misclassifications
    # ============================================================================
    
    print("Evaluating model...")
    all_preds = []
    all_targets = []
    misclassifications = []
    
    with torch.no_grad():
        sample_index = 0
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
            # Track misclassifications
            for i, (pred, target) in enumerate(zip(preds, labels)):
                if pred.item() != target.item():
                    img_idx = sample_index + i
                    img_id = test_dataset.samples.iloc[img_idx]['img_id']
                    
                    misclassifications.append({
                        'true_label': target.item(),
                        'pred_label': pred.item(),
                        'image_path': os.path.join(dataset_dir, 'test', img_id),
                        'img_id': img_id,
                        'confidence': outputs[i, pred.item()].item()
                    })

                    sample_index += len(labels)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"\nTotal test samples: {len(all_targets)}")
    print(f"Misclassifications: {len(misclassifications)}")
    print(f"Confusion matrix shape: {cm.shape}\n")
    
    # ============================================================================
    # STEP 3: Use analyze_confusion_pair for specific confusion patterns
    # ============================================================================
    
    output_dir = './gradcam_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Analyze true_class=0 → pred_class=1
    print("="*70)
    print("EXAMPLE 1: Analyzing confusion pattern")
    print("="*70)
    
    true_label_idx = 0
    pred_label_idx = 1
    n_samples = 3
    
    if cm[true_label_idx, pred_label_idx] > 0:
        print(f"\nAnalyzing: {class_names[true_label_idx]} → {class_names[pred_label_idx]}")
        print(f"Number of samples with this confusion: {cm[true_label_idx, pred_label_idx]}\n")
        
        results = analyze_confusion_pair(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            true_class=true_label_idx,
            pred_class=pred_label_idx,
            cm=cm,
            misclassifications=misclassifications,
            label_names=class_names,
            n_samples=n_samples,
            save_dir=output_dir,
            dataset_dir=dataset_dir,
            target_mode="predicted"
        )
        
        print(f"\nGenerated {len(results)} Grad-CAM visualizations:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.get('img_id', 'N/A')}")
            print(f"     CAM saved to: {result.get('cam_path', 'N/A')}")
    else:
        print(f"\nNo misclassifications of {class_names[true_label_idx]} as {class_names[pred_label_idx]}")
    
    # ============================================================================
    # STEP 4: Analyze all confusion pairs (optional)
    # ============================================================================
    
    print("\n" + "="*70)
    print("Confusion matrix summary:")
    print("="*70)
    print(f"\nClass mapping:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    
    print(f"\nMisclassification patterns:")
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                print(f"  {class_names[i]:10s} → {class_names[j]:10s}: {cm[i, j]:3d} samples")


if __name__ == '__main__':
    main()
