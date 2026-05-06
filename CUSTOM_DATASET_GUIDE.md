# Using Your Custom Dataset with MedViT

This guide explains how to train the MedViT model with your own custom dataset.

## Dataset Structure

Your dataset should be organized as follows:

```
dataset/
├── train/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── test/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── val/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── meta67.csv
```

## CSV Metadata Format

Your `meta67.csv` file should contain at least the following columns:
- `img_id`: The filename of the image (e.g., "PAT_1516_1765_530.png")
- `diagnostic`: The class label (e.g., "BCC", "NEV", "ACK", "SEK", "SCC")
- `SPLIT`: The data split - must be one of "TRAIN", "TEST", or "VAL"

Additional columns are ignored and can include patient information, clinical features, etc.

Example CSV structure:
```
,patient_id,lesion_id,age,gender,diagnostic,img_id,SPLIT
0,PAT_1516,1765,8,,NEV,PAT_1516_1765_530.png,TRAIN
1,PAT_46,881,55,FEMALE,BCC,PAT_46_881_939.png,VAL
2,PAT_1545,1867,77,,ACK,PAT_1545_1867_547.png,TRAIN
```

## Classes Supported

The implementation automatically detects all unique classes from your CSV's `diagnostic` column. In your dataset:
- **BCC**: Basal Cell Carcinoma
- **NEV**: Nevus
- **ACK**: Actinic Keratosis
- **SEK**: Seborrheic Keratosis
- **SCC**: Squamous Cell Carcinoma

You can modify this for your own classification tasks - just update the `diagnostic` column with your class names.

## Training with Custom Dataset

### Option 1: Using Command Line (Recommended)

```bash
python main.py \
    --model_name MedViT_small \
    --dataset Custom \
    --dataset_dir ./dataset \
    --batch_size 32 \
    --lr 0.0001 \
    --epochs 100 \
    --checkpoint_path ./checkpoint/custom_model.pth
```

**Arguments:**
- `--model_name`: Model architecture (MedViT_tiny, MedViT_small, MedViT_base, MedViT_large)
- `--dataset`: Must be set to "Custom" for custom datasets
- `--dataset_dir`: Path to your dataset directory (default: ./dataset)
- `--batch_size`: Batch size for training (default: 24)
- `--lr`: Learning rate (default: 0.0001)
- `--epochs`: Number of training epochs (default: 100)
- `--checkpoint_path`: Path to save the trained model

### Option 2: Using Python Script

See `train_custom_dataset.py` for a complete example.

## Preparing Your Dataset

### Step 1: Organize your images
```python
import shutil
import os

# Create directory structure
os.makedirs('dataset/train', exist_ok=True)
os.makedirs('dataset/val', exist_ok=True)
os.makedirs('dataset/test', exist_ok=True)

# Move your images to appropriate folders
# Example:
# shutil.copy('path/to/image.png', 'dataset/train/image.png')
```

### Step 2: Create metadata CSV

Use a script like this to generate your CSV:

```python
import pandas as pd
import os

data = []
for split in ['train', 'val', 'test']:
    split_path = f'dataset/{split}'
    if os.path.exists(split_path):
        for img_file in os.listdir(split_path):
            if img_file.endswith('.png'):
                # Extract label from filename or from your own mapping
                label = 'YOUR_CLASS_LABEL'  # Replace with your actual class
                data.append({
                    'img_id': img_file,
                    'diagnostic': label,
                    'SPLIT': split.upper()
                })

df = pd.DataFrame(data)
df.to_csv('dataset/meta67.csv', index=False)
print(f"Created metadata CSV with {len(df)} entries")
```

## Model Checkpoints

The model checkpoint will be saved at the path specified by `--checkpoint_path`. Each checkpoint contains:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Best accuracy achieved
- Epoch number

To load and use the saved model:

```python
import torch
from MedViT import MedViT_small

# Load model
model = MedViT_small(num_classes=5)  # Replace 5 with your number of classes
checkpoint = torch.load('./checkpoint/custom_model.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

# Now use model for inference
```

## Image Requirements

- **Format**: PNG, JPG, or other image formats supported by PIL
- **Size**: Images are automatically resized to 224×224 during training
- **Color**: Images should be RGB (will be converted automatically)
- **Resolution**: Recommended minimum 256×256 for best results

## Data Augmentation

The training pipeline includes:
- Random resized crop (224×224)
- AugMix augmentation (alpha=0.4)
- Random horizontal flip (p=0.4)
- Normalization (mean=0.5, std=0.5)

## Common Issues

### "Metadata CSV not found"
- Make sure `meta67.csv` is in your dataset root directory
- Check the exact filename matches

### "Dataset directory not found"
- Verify the path to your dataset
- Use absolute paths if relative paths don't work: `/full/path/to/dataset`

### "No samples found for split"
- Check that your CSV has the correct SPLIT values (TRAIN, TEST, VAL in uppercase)
- Verify image files exist in the corresponding folders (train/, test/, val/)

### Memory issues
- Reduce batch size: `--batch_size 16` or lower
- Consider using a smaller model: `--model_name MedViT_tiny`

## Performance Optimization

For faster training:
1. Use smaller model: `--model_name MedViT_tiny`
2. Increase batch size if your GPU allows: `--batch_size 64`
3. Reduce number of epochs initially: `--epochs 50`

For better accuracy:
1. Use larger model: `--model_name MedViT_large`
2. Train longer: `--epochs 200`
3. Use lower learning rate: `--lr 0.00005`

## Evaluation

After training, you can evaluate the model on the test set using the scripts in the Tutorials folder or modify the training script to include validation metrics like AUC, F1-score, etc.
