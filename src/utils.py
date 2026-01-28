"""
DT-SegNet Utility Functions
===========================
Helper functions for training, inference, and visualization.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import random

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


logger = logging.getLogger(__name__)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_info(model: nn.Module, verbose: bool = False) -> Dict[str, Any]:
    """Get model information."""
    n_params = count_parameters(model)
    n_layers = len(list(model.modules()))
    
    info = {
        'name': model.__class__.__name__,
        'parameters': n_params,
        'layers': n_layers,
        'parameters_mb': n_params * 4 / 1024 / 1024,  # Assuming float32
    }
    
    if verbose:
        logger.info(f"Model: {info['name']}")
        logger.info(f"Parameters: {info['parameters']:,} ({info['parameters_mb']:.2f} MB)")
        logger.info(f"Layers: {info['layers']}")
    
    return info


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Returns metadata from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    return metadata


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    save_path: str,
    **kwargs
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


# =============================================================================
# Visualization
# =============================================================================

def visualize_detection(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12)
):
    """
    Visualize detection results.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        boxes: Detection boxes (N, 4) in xyxy format
        scores: Optional confidence scores (N,)
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        if scores is not None:
            ax.text(x1, y1 - 5, f'{scores[i]:.2f}', color='red', fontsize=10)
    
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12)
):
    """
    Visualize segmentation results.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        mask: Segmentation mask (H, W)
        alpha: Overlay alpha
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0], figsize[1] // 3))
    
    # Original image
    if image.ndim == 2:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='jet')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    if image.ndim == 2:
        overlay = np.stack([image] * 3, axis=-1)
        overlay = overlay / overlay.max() if overlay.max() > 0 else overlay
    else:
        overlay = image.copy().astype(float)
        overlay = overlay / overlay.max() if overlay.max() > 0 else overlay
    
    mask_rgb = np.zeros((*mask.shape, 3))
    mask_rgb[mask == 1] = [1, 0, 0]  # Red for precipitates
    
    overlay = overlay * (1 - alpha) + mask_rgb * alpha
    overlay = np.clip(overlay, 0, 1)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_dtsegnet_output(
    image: np.ndarray,
    mask: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize end-to-end DT-SegNet output.
    
    Args:
        image: Input image
        mask: Segmentation mask
        boxes: Optional detection boxes
        scores: Optional detection scores
        save_path: Optional path to save figure
        figsize: Figure size
    """
    ncols = 4 if boxes is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    
    # Original
    if image.ndim == 2:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    # Detections (if available)
    col = 1
    if boxes is not None:
        if image.ndim == 2:
            axes[col].imshow(image, cmap='gray')
        else:
            axes[col].imshow(image)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[col].add_patch(rect)
            if scores is not None:
                axes[col].text(x1, y1 - 5, f'{scores[i]:.2f}', color='red', fontsize=8)
        
        axes[col].set_title('Detections')
        axes[col].axis('off')
        col += 1
    
    # Segmentation mask
    axes[col].imshow(mask, cmap='jet')
    axes[col].set_title('Segmentation')
    axes[col].axis('off')
    col += 1
    
    # Overlay
    if image.ndim == 2:
        overlay = np.stack([image / 255.0] * 3, axis=-1)
    else:
        overlay = image.astype(float) / 255.0
    
    mask_rgb = np.zeros((*mask.shape, 3))
    mask_rgb[mask == 1] = [1, 0, 0]
    
    overlay = overlay * 0.7 + mask_rgb * 0.3
    overlay = np.clip(overlay, 0, 1)
    
    axes[col].imshow(overlay)
    axes[col].set_title('Overlay')
    axes[col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_confusion_matrix(
    pred: np.ndarray, 
    target: np.ndarray, 
    num_classes: int = 2
) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = np.sum((target == i) & (pred == j))
    return cm


def compute_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    """Compute metrics from confusion matrix."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    
    # Per-class metrics
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    
    # Overall metrics
    accuracy = tp.sum() / cm.sum()
    
    return {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'iou': iou.tolist(),
        'mIoU': iou.mean(),
        'mF1': f1.mean(),
    }


# =============================================================================
# Data Utilities
# =============================================================================

def create_train_val_split(
    data_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """
    Create train/val split from a directory of images.
    
    Args:
        data_dir: Directory containing images
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_paths, val_paths)
    """
    random.seed(seed)
    
    image_paths = sorted(
        list(data_dir.glob('*.png')) + 
        list(data_dir.glob('*.jpg'))
    )
    
    random.shuffle(image_paths)
    
    n_val = int(len(image_paths) * val_ratio)
    val_paths = image_paths[:n_val]
    train_paths = image_paths[n_val:]
    
    return train_paths, val_paths


def convert_yolo_to_xyxy(
    boxes: np.ndarray, 
    img_width: int, 
    img_height: int
) -> np.ndarray:
    """
    Convert YOLO format (x_center, y_center, width, height) to xyxy format.
    
    Args:
        boxes: Boxes in YOLO format (N, 4)
        img_width: Image width
        img_height: Image height
        
    Returns:
        Boxes in xyxy format (N, 4)
    """
    if boxes.shape[0] == 0:
        return boxes
    
    x_center = boxes[:, 0] * img_width
    y_center = boxes[:, 1] * img_height
    width = boxes[:, 2] * img_width
    height = boxes[:, 3] * img_height
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return np.stack([x1, y1, x2, y2], axis=1)


def convert_xyxy_to_yolo(
    boxes: np.ndarray, 
    img_width: int, 
    img_height: int
) -> np.ndarray:
    """
    Convert xyxy format to YOLO format (x_center, y_center, width, height).
    
    Args:
        boxes: Boxes in xyxy format (N, 4)
        img_width: Image width
        img_height: Image height
        
    Returns:
        Boxes in YOLO format (N, 4), normalized
    """
    if boxes.shape[0] == 0:
        return boxes
    
    x_center = (boxes[:, 0] + boxes[:, 2]) / 2 / img_width
    y_center = (boxes[:, 1] + boxes[:, 3]) / 2 / img_height
    width = (boxes[:, 2] - boxes[:, 0]) / img_width
    height = (boxes[:, 3] - boxes[:, 1]) / img_height
    
    return np.stack([x_center, y_center, width, height], axis=1)
