#!/usr/bin/env python3
"""
Test script to verify the custom dataset implementation is working correctly.

This script runs a quick sanity check to ensure:
1. All required packages are installed
2. Dataset structure is valid
3. CustomDataset class can load images
4. Model can run a forward pass

Run this before training to catch issues early.
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed."""
    print("Testing package imports...")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'medmnist': 'MedMNIST',
        'timm': 'TIMM',
        'tqdm': 'TQDM',
    }
    
    failed = []
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)
    
    if failed:
        print(f"\n✗ Missing packages: {', '.join(failed)}")
        print("\nInstall with: pip install torch torchvision pillow pandas timm medmnist")
        return False
    
    print("  ✓ All packages installed\n")
    return True


def test_dataset_structure(dataset_dir='./dataset'):
    """Test if dataset has correct folder structure."""
    print(f"Testing dataset structure ({dataset_dir})...")
    
    if not os.path.exists(dataset_dir):
        print(f"  ✗ Dataset directory not found: {dataset_dir}")
        return False
    
    required_folders = ['train', 'test', 'val']
    required_files = ['meta67.csv']
    
    all_ok = True
    
    # Check folders
    for folder in required_folders:
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.exists(folder_path):
            num_images = len([f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  ✓ {folder}/ - {num_images} images")
        else:
            print(f"  ✗ {folder}/ - NOT FOUND")
            all_ok = False
    
    # Check files
    for file in required_files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_csv_format(dataset_dir='./dataset', csv_file='meta67.csv'):
    """Test if CSV has correct format."""
    print(f"\nTesting CSV format ({csv_file})...")
    
    import pandas as pd
    
    csv_path = os.path.join(dataset_dir, csv_file)
    
    try:
        df = pd.read_csv(csv_path)
        print(f"  ✓ CSV loaded - {len(df)} entries")
        
        required_cols = ['img_id', 'diagnostic', 'SPLIT']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ✗ Missing columns: {missing_cols}")
            return False
        
        print(f"  ✓ Has required columns")
        
        # Check split values
        split_values = df['SPLIT'].unique()
        expected_splits = {'TRAIN', 'TEST', 'VAL'}
        invalid_splits = [s for s in split_values if s not in expected_splits]
        
        if invalid_splits:
            print(f"  ⚠ Invalid SPLIT values: {invalid_splits}")
            print(f"     Expected: TRAIN, TEST, VAL")
        
        split_counts = df['SPLIT'].value_counts()
        for split in ['TRAIN', 'TEST', 'VAL']:
            if split in split_counts.index:
                print(f"    - {split}: {split_counts[split]} samples")
        
        # Check classes
        classes = df['diagnostic'].unique()
        print(f"  ✓ Found {len(classes)} classes: {', '.join(sorted(classes))}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error reading CSV: {e}")
        return False


def test_custom_dataset(dataset_dir='./dataset', csv_file='meta67.csv'):
    """Test if CustomDataset can load data."""
    print("\nTesting CustomDataset class...")
    
    import torch
    from torchvision import transforms
    from PIL import Image
    import pandas as pd
    
    # Import CustomDataset
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from datasets import CustomDataset
        print("  ✓ CustomDataset imported")
    except ImportError as e:
        print(f"  ✗ Failed to import CustomDataset: {e}")
        return False
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Try loading train split
    try:
        train_dataset = CustomDataset(dataset_dir, 
                                     os.path.join(dataset_dir, csv_file),
                                     split='train', 
                                     transform=transform)
        print(f"  ✓ Loaded train split - {len(train_dataset)} samples")
        print(f"    Classes: {train_dataset.classes}")
        
        # Try loading one sample
        image, label = train_dataset[0]
        print(f"  ✓ Sample loaded - image shape: {image.shape}, label: {label}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """Test if model can do a forward pass."""
    print("\nTesting model forward pass...")
    
    import torch
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from MedViT import MedViT_tiny
        print("  ✓ Model imported")
    except ImportError as e:
        print(f"  ✗ Failed to import model: {e}")
        return False
    
    try:
        # Create dummy input
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MedViT_tiny(num_classes=5).to(device)
        print(f"  ✓ Model created on {device}")
        
        # Try forward pass
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  ✓ Forward pass successful - output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("MedViT Custom Dataset - Implementation Test")
    print("="*70 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Package imports", test_imports()))
    
    if not results[-1][1]:
        print("\n✗ Package installation failed. Please install requirements.")
        return False
    
    # Test 2: Dataset structure
    results.append(("Dataset structure", test_dataset_structure()))
    
    # Test 3: CSV format
    results.append(("CSV format", test_csv_format()))
    
    # Test 4: CustomDataset
    results.append(("CustomDataset loading", test_custom_dataset()))
    
    # Test 5: Model forward
    results.append(("Model forward pass", test_model_forward()))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary:")
    print("="*70)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
        if not result:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to train!")
        print("\nNext steps:")
        print("  1. Run: python prepare_dataset.py --dataset_dir ./dataset")
        print("  2. Run: python train_custom_dataset.py --dataset_dir ./dataset")
        print("\nFor more information, see QUICKSTART.md")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        print("\nFor troubleshooting, see CUSTOM_DATASET_GUIDE.md")
    
    print("="*70 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
