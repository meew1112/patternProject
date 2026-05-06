"""Grad-CAM analysis helpers for confusion-matrix error inspection.

Usage example:

    from MedViTV2 import analyze_confusion_pair

    results = analyze_confusion_pair(
        model_name="MedViT_small",
        checkpoint_path="./best_checkpoint.pth",
        true_class=0,
        pred_class=1,
        cm=cm,
        misclassifications=misc,
        label_names=names,
        n_samples=3,
        save_dir="./confusion_analysis",
    )

Expected `misclassifications` formats:
    - pandas.DataFrame with columns like true_label, pred_label, image_path
    - list[dict] with keys like true_label, pred_label, image_path
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from MedViT import MedViT_base, MedViT_large, MedViT_small, MedViT_tiny


MODEL_REGISTRY = {
    "MedViT_tiny": MedViT_tiny,
    "MedViT_small": MedViT_small,
    "MedViT_base": MedViT_base,
    "MedViT_large": MedViT_large,
}


def _to_index(value: Any, label_names: Sequence[str]) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        if value in label_names:
            return int(label_names.index(value))
        if value.isdigit():
            return int(value)
    raise ValueError(f"Could not resolve label value {value!r} to an index")


def _get_entry_value(entry: Any, key_candidates: Sequence[str]) -> Any:
    if isinstance(entry, dict):
        for key in key_candidates:
            if key in entry:
                return entry[key]
    else:
        for key in key_candidates:
            if hasattr(entry, key):
                return getattr(entry, key)
    raise KeyError(f"Could not find any of {key_candidates} in misclassification entry")


def _iter_misclassifications(misclassifications: Any) -> Iterable[Any]:
    if hasattr(misclassifications, "iterrows"):
        for _, row in misclassifications.iterrows():
            yield row
        return
    if isinstance(misclassifications, (list, tuple)):
        for entry in misclassifications:
            yield entry
        return
    raise TypeError("misclassifications must be a pandas DataFrame or a list/tuple of records")


def _extract_image_path(entry: Any, dataset_dir: Optional[str]) -> str:
    image_path = _get_entry_value(entry, ["image_path", "img_path", "path", "file_path"])
    if dataset_dir and not os.path.isabs(str(image_path)):
        candidate = os.path.join(dataset_dir, str(image_path))
        if os.path.exists(candidate):
            return candidate
    return str(image_path)


def _load_model(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_name {model_name!r}. Available: {list(MODEL_REGISTRY)}")

    model = MODEL_REGISTRY[model_name](num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint

    model_state = model.state_dict()
    compatible_state = {}
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value

    model.load_state_dict({**model_state, **compatible_state}, strict=False)
    model = model.to(device)
    model.eval()
    return model


def _get_target_layer(model):
    if hasattr(model, "features"):
        for layer in reversed(model.features):
            for attr in ("norm2", "norm1", "norm"):
                if hasattr(layer, attr):
                    return getattr(layer, attr)
    if hasattr(model, "norm"):
        return model.norm
    return None


def _build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _load_rgb_image(image_path: str):
    image = Image.open(image_path).convert("RGB")
    return image


def analyze_confusion_pair(
    model_name: str,
    checkpoint_path: str,
    true_class: int,
    pred_class: int,
    cm,
    misclassifications,
    label_names: Sequence[str],
    n_samples: int = 3,
    save_dir: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    target_mode: str = "predicted",
):
    """Run Grad-CAM on samples belonging to one confusion-matrix cell.

    Args:
        model_name: MedViT model name.
        checkpoint_path: Path to a saved checkpoint.
        true_class: Row index in the confusion matrix.
        pred_class: Column index in the confusion matrix.
        cm: Confusion matrix array. Used for validation and reporting.
        misclassifications: DataFrame/list of misclassified records.
        label_names: Class names aligned to indices.
        n_samples: Maximum number of samples to visualize.
        save_dir: Directory to save overlays. If None, nothing is saved.
        dataset_dir: Base dataset directory used to resolve relative image paths.
        target_mode: "predicted" to explain the model's wrong prediction, "true" to target the true label.

    Returns:
        List of dicts with sample metadata and generated image paths.
    """

    if true_class < 0 or true_class >= len(label_names):
        raise IndexError(f"true_class {true_class} is out of range for {len(label_names)} labels")
    if pred_class < 0 or pred_class >= len(label_names):
        raise IndexError(f"pred_class {pred_class} is out of range for {len(label_names)} labels")

    save_path = Path(save_dir) if save_dir else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_name, checkpoint_path, len(label_names), device)
    target_layer = _get_target_layer(model)
    if target_layer is None:
        raise RuntimeError("Could not locate a valid target layer for Grad-CAM")

    cam = GradCAM(model=model, target_layers=[target_layer])
    transform = _build_transform()

    selected_rows: List[Any] = []
    for entry in _iter_misclassifications(misclassifications):
        entry_true = _to_index(_get_entry_value(entry, ["true_label", "label", "y_true", "target"]), label_names)
        entry_pred = _to_index(_get_entry_value(entry, ["pred_label", "predicted_label", "y_pred", "prediction"]), label_names)
        if entry_true == true_class and entry_pred == pred_class:
            selected_rows.append(entry)
        if len(selected_rows) >= n_samples:
            break

    if not selected_rows:
        cm_value = None
        if cm is not None:
            cm_array = np.asarray(cm)
            if cm_array.ndim == 2 and true_class < cm_array.shape[0] and pred_class < cm_array.shape[1]:
                cm_value = int(cm_array[true_class, pred_class])
        print(
            f"No misclassification samples found for {label_names[true_class]} → {label_names[pred_class]}"
            + (f" (cm cell count={cm_value})" if cm_value is not None else "")
        )
        return []

    results: List[Dict[str, Any]] = []
    for index, entry in enumerate(selected_rows, start=1):
        image_path = _extract_image_path(entry, dataset_dir)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = _load_rgb_image(image_path)
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            predicted_idx = int(torch.argmax(probs).item())
            confidence = float(probs[predicted_idx].item())

        target_idx = pred_class if target_mode == "predicted" else true_class
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_idx)])[0]

        raw_image = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        raw_image = np.clip(raw_image * 0.5 + 0.5, 0, 1)
        cam_image = show_cam_on_image(raw_image, grayscale_cam, use_rgb=True)

        output_file = None
        if save_path is not None:
            output_file = save_path / f"{label_names[true_class]}_to_{label_names[pred_class]}_{index:02d}.png"
            Image.fromarray(cam_image).save(output_file)

        results.append(
            {
                "image_path": image_path,
                "true_label": label_names[true_class],
                "pred_label": label_names[pred_class],
                "predicted_idx": predicted_idx,
                "confidence": confidence,
                "target_mode": target_mode,
                "cam_path": str(output_file) if output_file else None,
            }
        )

    print(
        f"Grad-CAM generated for {len(results)} sample(s) in cell {label_names[true_class]} → {label_names[pred_class]}"
    )
    return results
