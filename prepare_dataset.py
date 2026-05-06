#!/usr/bin/env python3
"""
Utility script to prepare and validate your dataset for training with MedViT.

This script helps you:
1. Check if your dataset structure is correct
2. Generate metadata CSV if needed
3. Validate that images exist and labels are correct
4. Show dataset statistics

Usage:
    python prepare_dataset.py --dataset_dir ./dataset
"""

import os
import pandas as pd
import argparse
from pathlib import Path
from PIL import Image
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


def validate_dataset_structure(dataset_dir):
    """Check if dataset has the required folder structure."""
    print("Checking dataset structure...")
    
    required_dirs = ['train', 'test', 'val']
    all_exist = True
    
    for split in required_dirs:
        split_path = os.path.join(dataset_dir, split)
        if os.path.exists(split_path):
            num_images = len([f for f in os.listdir(split_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  ✓ {split}/ - {num_images} images found")
        else:
            print(f"  ✗ {split}/ - NOT FOUND")
            all_exist = False
    
    return all_exist


def validate_csv(dataset_dir, csv_file='meta67.csv'):
    """Validate if CSV file exists and has required columns."""
    print(f"\nChecking CSV file ({csv_file})...")
    
    csv_path = os.path.join(dataset_dir, csv_file)
    if not os.path.exists(csv_path):
        print(f"  ✗ CSV file not found at {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"  ✓ CSV file found with {len(df)} entries")
        
        # Check required columns
        required_cols = ['img_id', 'diagnostic', 'SPLIT']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ✗ Missing columns: {missing_cols}")
            print(f"     Available columns: {list(df.columns)}")
            return None
        
        print(f"  ✓ Has required columns: img_id, diagnostic, SPLIT")
        
        # Check for missing values in required columns
        for col in required_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"  ⚠ Column '{col}' has {missing} missing values")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error reading CSV: {e}")
        return None


def validate_images(dataset_dir, df):
    """Check if images referenced in CSV actually exist."""
    print("\nValidating image files...")
    
    missing_images = []
    invalid_images = []
    
    for idx, row in df.iterrows():
        img_id = row['img_id']
        split = row['SPLIT'].upper()
        split_dir = split.lower()
        
        img_path = os.path.join(dataset_dir, split_dir, img_id)
        
        if not os.path.exists(img_path):
            missing_images.append((img_id, split))
        else:
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception as e:
                invalid_images.append((img_id, str(e)))
    
    if missing_images:
        print(f"  ✗ {len(missing_images)} images not found:")
        for img_id, split in missing_images[:5]:  # Show first 5
            print(f"     - {split}/{img_id}")
        if len(missing_images) > 5:
            print(f"     ... and {len(missing_images)-5} more")
    else:
        print(f"  ✓ All {len(df)} images found")
    
    if invalid_images:
        print(f"  ✗ {len(invalid_images)} images are corrupted or invalid:")
        for img_id, error in invalid_images[:5]:  # Show first 5
            print(f"     - {img_id}: {error}")
    else:
        print(f"  ✓ All images are valid")
    
    return len(missing_images) == 0 and len(invalid_images) == 0


def show_dataset_statistics(dataset_dir, df):
    """Display statistics about the dataset."""
    print("\nDataset Statistics:")
    
    # Split distribution
    split_counts = df['SPLIT'].value_counts()
    print(f"  Split distribution:")
    for split, count in split_counts.items():
        pct = 100 * count / len(df)
        print(f"    - {split}: {count} samples ({pct:.1f}%)")
    
    # Class distribution
    class_counts = df['diagnostic'].value_counts()
    print(f"\n  Class distribution (overall):")
    for cls, count in class_counts.items():
        pct = 100 * count / len(df)
        print(f"    - {cls}: {count} samples ({pct:.1f}%)")
    
    # Class distribution per split
    print(f"\n  Class distribution per split:")
    for split in ['TRAIN', 'TEST', 'VAL']:
        split_data = df[df['SPLIT'] == split]
        if len(split_data) > 0:
            print(f"    {split}:")
            for cls, count in split_data['diagnostic'].value_counts().items():
                pct = 100 * count / len(split_data)
                print(f"      - {cls}: {count} ({pct:.1f}%)")
    
    # Image format distribution
    print(f"\n  Image format distribution:")
    formats = Counter()
    for idx, row in df.iterrows():
        img_id = row['img_id']
        ext = os.path.splitext(img_id)[1].lower()
        formats[ext] += 1
    
    for ext, count in formats.most_common():
        pct = 100 * count / len(df)
        print(f"    - {ext}: {count} images ({pct:.1f}%)")


def generate_csv(dataset_dir, output_file='meta67.csv'):
    """Generate CSV metadata file from dataset structure."""
    print(f"\nGenerating CSV file from dataset structure...")
    
    data = []
    
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"  Skipping {split}/ (not found)")
            continue
        
        for img_file in os.listdir(split_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Try to extract class from filename
                # You may need to customize this based on your naming convention
                parts = img_file.split('_')
                
                # Default: use a placeholder class
                diagnostic = 'UNKNOWN'
                
                # If your filenames have pattern like CLASS_ID.png
                if len(parts) > 0:
                    potential_class = parts[0]
                    if potential_class.isupper() and len(potential_class) <= 4:
                        diagnostic = potential_class
                
                data.append({
                    'img_id': img_file,
                    'diagnostic': diagnostic,
                    'SPLIT': split.upper()
                })
    
    if not data:
        print(f"  ✗ No images found in dataset structure")
        return False
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(dataset_dir, output_file)
    df.to_csv(csv_path, index=False)
    
    print(f"  ✓ Generated CSV with {len(df)} entries")
    print(f"  ✓ Saved to {csv_path}")
    print(f"\n  ⚠ IMPORTANT: Please review and update the 'diagnostic' column in the CSV")
    print(f"     The generated diagnostic values may not be accurate.")
    print(f"     Edit the CSV to set correct class labels for each image.")
    
    return True


def main(args):
    """Main validation function."""
    dataset_dir = args.dataset_dir
    
    print("="*70)
    print("Dataset Preparation and Validation Tool")
    print("="*70)
    print(f"Dataset directory: {dataset_dir}\n")
    
    if not os.path.exists(dataset_dir):
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return False
    
    # Check structure
    structure_ok = validate_dataset_structure(dataset_dir)
    
    # Check/create CSV
    csv_file = 'meta67.csv'
    df = validate_csv(dataset_dir, csv_file)
    
    if df is None:
        print(f"\nCSV file not valid. Would you like to generate it? (y/n): ", end='')
        response = input().strip().lower()
        if response == 'y':
            generate_csv(dataset_dir, csv_file)
            df = validate_csv(dataset_dir, csv_file)
        else:
            return False
    
    if df is None:
        print("Cannot proceed without valid CSV file.")
        return False
    
    # Validate images
    images_ok = validate_images(dataset_dir, df)
    
    # Show statistics
    show_dataset_statistics(dataset_dir, df)
    
    # Summary
    print("\n" + "="*70)
    if structure_ok and images_ok:
        print("✓ Dataset validation PASSED")
        print("\nYour dataset is ready for training!")
        print("\nTo train with this dataset, run:")
        print(f"  python train_custom_dataset.py --dataset_dir {dataset_dir}")
        print("\nOr with main.py:")
        print(f"  python main.py --dataset Custom --dataset_dir {dataset_dir}")
    else:
        print("✗ Dataset validation FAILED")
        print("\nPlease fix the issues above before training.")
    print("="*70)
    
    return structure_ok and images_ok


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare and validate dataset for MedViT training'
    )
    
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='./dataset',
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--generate_csv',
        action='store_true',
        help='Generate CSV file from dataset structure'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    success = main(args)
    exit(0 if success else 1)
