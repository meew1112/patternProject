#!/usr/bin/env python3
"""
Inference script: Use the trained MedViT model to make predictions on new images.

This script shows how to:
1. Load a trained model
2. Preprocess images
3. Make predictions
4. Get confidence scores

Usage:
    # Single image
    python inference.py --checkpoint ./checkpoint/medvit_custom.pth --image ./test_image.png
    
    # Batch of images in a directory
    python inference.py --checkpoint ./checkpoint/medvit_custom.pth --image_dir ./test_images/
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large


def load_model(checkpoint_path, model_name='MedViT_small', device='cuda'):
    """Load trained model from checkpoint."""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model info from checkpoint
    num_classes = checkpoint.get('num_classes', 5)
    classes = checkpoint.get('classes', [f'class_{i}' for i in range(num_classes)])
    
    print(f"Loading model...")
    print(f"  - Model: {model_name}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Classes: {classes}")
    print(f"  - Best accuracy during training: {checkpoint.get('best_acc', 'N/A')}")
    
    # Initialize model
    model_classes = {
        'MedViT_tiny': MedViT_tiny,
        'MedViT_small': MedViT_small,
        'MedViT_base': MedViT_base,
        'MedViT_large': MedViT_large
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_classes.keys())}")
    
    model = model_classes[model_name](num_classes=num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, classes


def get_transform():
    """Get image preprocessing transforms."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    return transform


def predict_image(model, image_path, transform, classes, device='cuda'):
    """Make prediction for a single image."""
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = classes[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()
    
    return predicted_class, confidence, probabilities.cpu().numpy()


def main(args):
    """Main inference function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model, classes = load_model(args.checkpoint, args.model_name, device)
    
    # Get transforms
    transform = get_transform()
    
    print(f"\nModel loaded successfully!")
    print(f"Ready for inference\n")
    print("="*70)
    
    # Handle single image
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        
        print(f"\nProcessing image: {args.image}")
        predicted_class, confidence, probs = predict_image(
            model, args.image, transform, classes, device
        )
        
        print(f"\nPrediction Results:")
        print(f"  Predicted class: {predicted_class}")
        print(f"  Confidence: {confidence*100:.2f}%")
        print(f"\nClass probabilities:")
        for cls, prob in zip(classes, probs):
            bar_length = int(prob * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"  {cls:15s}: {bar} {prob*100:6.2f}%")
    
    # Handle image directory
    elif args.image_dir:
        if not os.path.isdir(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            return
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = [f for f in os.listdir(args.image_dir) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"No images found in {args.image_dir}")
            return
        
        print(f"\nProcessing {len(image_files)} images from {args.image_dir}\n")
        
        results = []
        for img_file in image_files:
            img_path = os.path.join(args.image_dir, img_file)
            try:
                predicted_class, confidence, probs = predict_image(
                    model, img_path, transform, classes, device
                )
                results.append((img_file, predicted_class, confidence))
                print(f"✓ {img_file:40s} → {predicted_class:15s} ({confidence*100:6.2f}%)")
            except Exception as e:
                print(f"✗ {img_file:40s} → Error: {e}")
        
        # Summary
        print("\n" + "="*70)
        print("Summary:")
        print(f"Successfully processed: {len(results)}/{len(image_files)} images")
        
        # Class distribution
        from collections import Counter
        class_dist = Counter([r[1] for r in results])
        print(f"\nClass distribution:")
        for cls, count in class_dist.most_common():
            pct = 100 * count / len(results)
            print(f"  {cls}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Inference with trained MedViT model'
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
        '--image',
        type=str,
        help='Path to single image for inference'
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        help='Path to directory containing images for batch inference'
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("Please provide either --image or --image_dir")
    
    if args.image and args.image_dir:
        parser.error("Please provide either --image or --image_dir, not both")
    
    return args


if __name__ == '__main__':
    args = parse_args()
    
    print("="*70)
    print("MedViT Inference")
    print("="*70)
    
    main(args)
