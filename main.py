import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import requests
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary
from datasets import build_dataset
from distutils.util import strtobool
from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import natten
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
#from MedViTV1 import MedViT_small, MedViT_base, MedViT_large


model_classes = {
    'MedViT_tiny': MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base': MedViT_base,
    'MedViT_large': MedViT_large
}

model_urls = {
    "MedViT_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0",
    "MedViT_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViT_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViT_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0"
}

def download_checkpoint(url, path):
    print(f"Downloading checkpoint from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    print(f"Checkpoint downloaded and saved to {path}")

# Define the MNIST training routine
def train_mnist(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path, data_flag, task):
    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, datax in enumerate(train_bar):
            images, labels = datax
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            
            if task == 'multi-label, binary-class':
                labels = labels.to(torch.float32)
                loss = loss_function(outputs, labels)
            else:
                labels = labels.squeeze().long()
                loss = loss_function(outputs.squeeze(0), labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
        
        net.eval()
        y_score = torch.tensor([])
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                inputs, targets = val_data
                outputs = net(inputs.to(device))
                
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
                
                y_score = torch.cat((y_score, outputs.cpu()), 0)
                
        y_score = y_score.detach().numpy()
        evaluator = Evaluator(data_flag, 'test', size=224, root='./data')
        metrics = evaluator.evaluate(y_score)
        
        val_accurate, _ = metrics
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  auc: {metrics[0]:.3f}  acc: {metrics[1]:.3f}')
        #print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')
        if val_accurate > best_acc:
            print('\nSaving checkpoint...')
            best_acc = val_accurate
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)

    print('Finished Training')

# Define the non-MNIST training routine
def specificity_per_class(conf_matrix):
    """Calculates specificity for each class."""
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp))
    return specificity

def overall_accuracy(conf_matrix):
    """Calculates overall accuracy for multi-class."""
    tp_tn_sum = conf_matrix.trace()  # Sum of all diagonal elements (TP for all classes)
    total_sum = conf_matrix.sum()  # Sum of all elements in the matrix
    return tp_tn_sum / total_sum


def get_class_names(dataset):
    """Best-effort extraction of class names from wrapped datasets."""
    base_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    if hasattr(base_dataset, 'classes'):
        return list(base_dataset.classes)
    if hasattr(base_dataset, 'class_to_idx'):
        return list(base_dataset.class_to_idx.keys())
    return []


def plot_training_curves(train_losses, test_losses, test_accs, test_aucs, output_path):
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


def plot_confusion_matrix(conf_matrix, class_names, output_path, normalize=True):
    matrix = conf_matrix.astype(np.float32)
    if normalize and matrix.sum() > 0:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    tick_positions = np.arange(len(class_names))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = f'{matrix[i, j]:.2f}' if normalize else f'{int(conf_matrix[i, j])}'
            ax.text(j, i, value, ha='center', va='center', color='white' if matrix[i, j] > threshold else 'black')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def get_gradcam_target_layer(model):
    if hasattr(model, 'features'):
        for layer in reversed(model.features):
            if hasattr(layer, 'norm2'):
                return layer.norm2
            if hasattr(layer, 'norm1'):
                return layer.norm1
            if hasattr(layer, 'norm'):
                return layer.norm
    if hasattr(model, 'norm'):
        return model.norm
    return None


def save_gradcam_for_misclassified_sample(model, loader, device, class_names, output_path):
    target_layer = get_gradcam_target_layer(model)
    if target_layer is None:
        print('Grad-CAM skipped: no valid target layer found.')
        return

    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])
    sample_found = False

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).squeeze().long()

        with torch.no_grad():
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)

        mismatch = predictions.ne(targets)
        if not mismatch.any():
            continue

        sample_index = torch.where(mismatch)[0][0].item()
        input_tensor = inputs[sample_index:sample_index + 1]
        true_label = int(targets[sample_index].item())
        pred_label = int(predictions[sample_index].item())

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(true_label)])[0]
        raw_image = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        raw_image = np.clip(raw_image * 0.5 + 0.5, 0, 1)
        cam_image = show_cam_on_image(raw_image, grayscale_cam, use_rgb=True)

        plt.imsave(output_path, cam_image)
        print(f'--> Grad-CAM saved to {output_path} (true={class_names[true_label]}, pred={class_names[pred_label]})')
        sample_found = True
        break

    if not sample_found:
        print('Grad-CAM skipped: no misclassified validation sample found.')


def evaluate_with_metrics(model, loader, n_classes, criterion, device):
    model.eval()
    all_probs = []
    all_targets = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze().long()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    epoch_loss = running_loss / len(loader.dataset)
    preds_classes = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_targets, preds_classes)

    try:
        if n_classes == 2:
            auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    except ValueError:
        auc = 0.0

    conf_matrix = confusion_matrix(all_targets, preds_classes, labels=list(range(n_classes)))
    return epoch_loss, acc, auc, conf_matrix, preds_classes, all_targets

def train_other(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path, class_names, output_dir):
    best_acc = 0.0
    history_train_losses = []
    history_test_losses = []
    history_test_accs = []
    history_test_aucs = []

    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        # Training Loop
        for step, datax in enumerate(train_bar):
            images, labels = datax
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
        
        test_loss, test_acc, auc, conf_matrix, preds_classes, all_labels = evaluate_with_metrics(
            net, test_loader, len(class_names), loss_function, device
        )

        precision = precision_score(all_labels, preds_classes, average='weighted', zero_division=0)
        recall = recall_score(all_labels, preds_classes, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, preds_classes, average='weighted', zero_division=0)
        specificity = specificity_per_class(conf_matrix)
        avg_specificity = sum(specificity) / len(specificity) if len(specificity) else 0.0
        overall_acc = overall_accuracy(conf_matrix) if conf_matrix.sum() else 0.0

        history_train_losses.append(running_loss / len(train_loader))
        history_test_losses.append(test_loss)
        history_test_accs.append(test_acc)
        history_test_aucs.append(auc)

        metrics_plot_path = os.path.join(output_dir, 'training_metrics_plot.png')
        confusion_plot_path = os.path.join(output_dir, f'confusion_matrix_epoch_{epoch + 1:03d}.png')
        gradcam_path = os.path.join(output_dir, f'gradcam_epoch_{epoch + 1:03d}.png')

        plot_training_curves(history_train_losses, history_test_losses, history_test_accs, history_test_aucs, metrics_plot_path)
        plot_confusion_matrix(conf_matrix, class_names, confusion_plot_path, normalize=True)

        print(f'[epoch {epoch + 1}] train_loss: {history_train_losses[-1]:.3f} '
              f'test_loss: {test_loss:.3f} test_accuracy: {test_acc:.4f} precision: {precision:.4f} '
              f'recall: {recall:.4f} specificity: {avg_specificity:.4f} '
              f'f1_score: {f1:.4f} auc: {auc:.4f} overall_accuracy: {overall_acc:.4f}')
        
        #print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')
        
        # Save best model
        if test_acc > best_acc:
            print('\nSaving checkpoint...')
            best_acc = test_acc
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)
            save_gradcam_for_misclassified_sample(net, test_loader, device, class_names, gradcam_path)

    print('Finished Training')

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    model_name = args.model_name
    dataset_name = args.dataset
    pretrained = args.pretrained
    if args.dataset.endswith('mnist'):
        info = INFO[args.dataset]
        task = info['task']
        if task == "multi-label, binary-class":
            loss_function = nn.BCEWithLogitsLoss()
        else:
            loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss()
    model_class = model_classes.get(model_name)

    # if not model_class:
    #     raise ValueError(f"Model {model_name} is not recognized. Available models: {list(model_classes.keys())}")

    batch_size = args.batch_size
    lr = args.lr
    
    train_dataset, test_dataset, nb_classes = build_dataset(args=args)
    class_names = get_class_names(train_dataset)
    if len(class_names) != nb_classes:
        class_names = [str(idx) for idx in range(nb_classes)]
    val_num = len(test_dataset)
    train_num = len(train_dataset)
    
    # scheduler max iteration
    eta = args.epochs * train_num // args.batch_size

    # Select model
    if model_name in model_classes:
        model_class = model_classes[model_name]
        net = model_class(num_classes=nb_classes).cuda()
        if pretrained:
            checkpoint_path = args.checkpoint_path
            if not os.path.exists(checkpoint_path):
                checkpoint_url = model_urls.get(model_name)
                if not checkpoint_url:
                    raise ValueError(f"Checkpoint URL for model {model_name} not found.")
                download_checkpoint(checkpoint_url, f'./{model_name}.pth')
                checkpoint_path = f'./{model_name}.pth'

            checkpoint = torch.load(checkpoint_path)
            state_dict = net.state_dict()
            for k in ['proj_head.0.weight', 'proj_head.0.bias']:
                if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            net.load_state_dict(checkpoint, strict=False)
    else:
        net = timm.create_model(model_name, pretrained=pretrained, num_classes=nb_classes).cuda()

    
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=[0.9, 0.999], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eta, eta_min=5e-6)
    
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
    
    print(train_dataset)
    print("===================")
    print(test_dataset)

    epochs = args.epochs
    best_acc = 0.0
    save_path = f'./{model_name}_{dataset_name}.pth'
    output_dir = os.path.join('.', 'analysis_outputs')
    train_steps = len(train_loader)

    if dataset_name.endswith('mnist'):
        
        train_mnist(epochs, net, train_loader, test_loader,
        optimizer, scheduler, loss_function, device, save_path, dataset_name, task)
    else:
        train_other(epochs, net, train_loader, test_loader,
        optimizer, scheduler, loss_function, device, save_path, class_names, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for MedViT models.')
    parser.add_argument('--model_name', type=str, default='MedViT_tiny', help='Model name to use.')
    #tissuemnist, pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, breastmnist, bloodmnist,
    #organamnist, organcmnist, organsmnist'
    parser.add_argument('--dataset', type=str, default='PAD', help='Dataset to use.')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='Path to the custom dataset directory (for --dataset Custom).')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--pretrained', type=lambda x: bool(strtobool(x)), default=False, help="Whether to use pretrained weights (True/False).")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/MedViT_tiny.pth', help='Path to the checkpoint file.')

    args = parser.parse_args()
    main(args)

# python main.py --model_name 'convnext_tiny' --dataset 'PAD'
