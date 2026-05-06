# Quick Start Guide: Train MedViT with Your Custom Dataset

## What I've Implemented for You

I've created a complete pipeline to train the MedViT model with your custom dataset:

### 1. **Custom Dataset Loader** (`datasets.py`)
   - New `CustomDataset` class that reads images + CSV metadata
   - Automatically handles train/val/test splits
   - Integrates with the existing build_dataset function

### 2. **Training Scripts**
   - **`train_custom_dataset.py`** - Standalone training script with full control
   - **Updated `main.py`** - Now supports `--dataset Custom` option

### 3. **Utility Scripts**
   - **`prepare_dataset.py`** - Validate and prepare your dataset
   - **`inference.py`** - Make predictions on new images with trained model

### 4. **Documentation**
   - **`CUSTOM_DATASET_GUIDE.md`** - Comprehensive guide with troubleshooting

---

## Quick Setup (3 Steps)

### Step 1: Organize Your Dataset
Your dataset directory should look like:
```
dataset/
├── train/           # Training images
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── test/            # Test images
│   └── ...
├── val/             # Validation images (optional)
│   └── ...
└── meta67.csv       # Metadata file
```

### Step 2: Prepare Your CSV Metadata
Your `meta67.csv` should have these columns:
```
img_id,diagnostic,SPLIT
PAT_1516_1765_530.png,NEV,TRAIN
PAT_46_881_939.png,BCC,VAL
PAT_1545_1867_547.png,ACK,TRAIN
...
```

**Column meanings:**
- `img_id`: Filename of the image
- `diagnostic`: Class label (e.g., BCC, NEV, ACK, SEK, SCC, or your own classes)
- `SPLIT`: Must be TRAIN, VAL, or TEST (uppercase)

### Step 3: Validate and Train

```bash
# Validate your dataset
python prepare_dataset.py --dataset_dir ./dataset

# Train the model
python train_custom_dataset.py \
    --model_name MedViT_small \
    --dataset_dir ./dataset \
    --epochs 100 \
    --batch_size 32 \
    --checkpoint_path ./checkpoint/my_model.pth
```

---

## Training Commands

### Using the standalone script (recommended):
```bash
python train_custom_dataset.py \
    --model_name MedViT_small \
    --dataset_dir ./dataset \
    --batch_size 32 \
    --lr 0.0001 \
    --epochs 100 \
    --checkpoint_path ./checkpoint/model.pth
```

### Using main.py:
```bash
python main.py \
    --dataset Custom \
    --dataset_dir ./dataset \
    --model_name MedViT_small \
    --batch_size 32 \
    --epochs 100
```

### Model Options:
- `MedViT_tiny` - Fastest, lowest memory
- `MedViT_small` - **Recommended for most cases**
- `MedViT_base` - Larger, better accuracy
- `MedViT_large` - Largest, highest accuracy

---

## After Training: Make Predictions

```bash
# Single image
python inference.py \
    --checkpoint ./checkpoint/my_model.pth \
    --image ./test_image.png

# Batch of images
python inference.py \
    --checkpoint ./checkpoint/my_model.pth \
    --image_dir ./test_images/
```

---

## Dataset Statistics
Check your dataset before training:
```bash
python prepare_dataset.py --dataset_dir ./dataset
```

This will show you:
- ✓ All required folders exist
- ✓ CSV file is valid
- ✓ All images are accessible
- Statistics on class distribution
- Train/Val/Test split sizes

---

## Common Issues & Solutions

### "ModuleNotFoundError: No module named 'PIL'"
```bash
pip install pillow
```

### "CUDA out of memory"
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use smaller model: `--model_name MedViT_tiny`

### "Dataset directory not found"
- Use absolute path: `--dataset_dir /full/path/to/dataset`
- Or ensure relative path is correct

### "Metadata CSV not found"
- Make sure `meta67.csv` is in your dataset root folder
- Check the filename is exactly `meta67.csv`

### "No samples found for split TRAIN"
- Check CSV SPLIT column values are uppercase: TRAIN, TEST, VAL
- Verify images actually exist in corresponding folders

---

## Performance Tips

**For Faster Training:**
- Use `MedViT_tiny` with `--batch_size 64`
- Reduce epochs: `--epochs 50`
- Reduce learning rate: `--lr 0.001`

**For Better Accuracy:**
- Use `MedViT_large` with `--batch_size 16`
- Train longer: `--epochs 200`
- Lower learning rate: `--lr 0.00005`

---

## File Structure of Implementation

```
MedViTV2/
├── datasets.py                      # ✨ Updated with CustomDataset class
├── main.py                          # ✨ Updated with --dataset_dir argument
├── train_custom_dataset.py          # 🆕 Standalone training script
├── prepare_dataset.py               # 🆕 Dataset validation utility
├── inference.py                     # 🆕 Prediction script
├── CUSTOM_DATASET_GUIDE.md          # 🆕 Detailed guide
├── dataset/
│   ├── train/                       # Your training images
│   ├── test/                        # Your test images
│   ├── val/                         # Your validation images
│   └── meta67.csv                   # Your metadata
└── checkpoint/
    └── my_model.pth                 # Saved model
```

---

## Key Features

✅ **Flexible:** Works with any number of classes  
✅ **Robust:** Validates dataset structure before training  
✅ **Easy to Use:** Simple command-line interface  
✅ **Production Ready:** Inference script included  
✅ **Well Documented:** Comprehensive guides and examples  

---

## Example Workflow

```bash
# 1. Organize your data
cp -r /path/to/images/train ./dataset/train
cp -r /path/to/images/test ./dataset/test
cp -r /path/to/images/val ./dataset/val

# 2. Create and place CSV
# (See CUSTOM_DATASET_GUIDE.md for CSV format)

# 3. Validate
python prepare_dataset.py --dataset_dir ./dataset

# 4. Train
python train_custom_dataset.py \
    --model_name MedViT_small \
    --dataset_dir ./dataset \
    --epochs 100

# 5. Evaluate
python inference.py \
    --checkpoint ./checkpoint/medvit_custom.pth \
    --image_dir ./dataset/test/
```

---

## Next Steps

1. Read [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md) for detailed documentation
2. Run `prepare_dataset.py` to validate your data
3. Run `train_custom_dataset.py` to train the model
4. Use `inference.py` to make predictions

Happy training! 🚀
