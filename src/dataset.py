"""
DT-SegNet Dataset Utilities
===========================
Dataset loaders for detection and segmentation training.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
# Detection Dataset (YOLO Format)
# =============================================================================

class DetectionDataset(Dataset):
    """
    Dataset for object detection training.
    
    Expects YOLO format labels (class x_center y_center width height).
    
    Args:
        data_dir: Directory containing images and labels
        img_size: Target image size
        transform: Albumentations transform
        split: Dataset split ('train', 'val', 'test')
    """
    
    def __init__(
        self,
        data_dir: str,
        img_size: int = 1280,
        transform: Optional[Callable] = None,
        split: str = 'train'
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.split = split
        
        # Find all images
        split_dir = self.data_dir / split
        self.image_paths = sorted(list(split_dir.glob('*.png')) + list(split_dir.glob('*.jpg')))
        
        # Default transform
        if transform is None:
            self.transform = self._get_default_transform(split)
        else:
            self.transform = transform

    def _get_default_transform(self, split: str) -> A.Compose:
        """Get default augmentation transforms."""
        if split == 'train':
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size, 
                    min_width=self.img_size,
                    border_mode=0,
                    fill=(114, 114, 114)
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size, 
                    min_width=self.img_size,
                    border_mode=0,
                    fill=(114, 114, 114)
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def _load_labels(self, label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load YOLO format labels."""
        if not label_path.exists():
            return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
        
        labels = []
        classes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append([x_center, y_center, width, height])
                classes.append(cls)
        
        if len(labels) == 0:
            return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
        
        return np.array(labels), np.array(classes, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load labels
        label_path = img_path.with_suffix('.txt')
        bboxes, class_labels = self._load_labels(label_path)
        
        # Apply transforms
        if len(bboxes) > 0:
            transformed = self.transform(
                image=image, 
                bboxes=bboxes.tolist(), 
                class_labels=class_labels.tolist()
            )
        else:
            transformed = self.transform(
                image=image, 
                bboxes=[], 
                class_labels=[]
            )
        
        image = transformed['image']
        bboxes = transformed['bboxes']
        class_labels = transformed['class_labels']
        
        # Convert to tensors
        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            class_labels = torch.tensor(class_labels, dtype=torch.int64)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.int64)
        
        return {
            'image': image,
            'bboxes': bboxes,
            'class_labels': class_labels,
            'image_path': str(img_path)
        }


# =============================================================================
# Segmentation Dataset
# =============================================================================

class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation training.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label masks
        img_size: Target image size
        transform: Albumentations transform
        split: Dataset split ('train', 'val', 'test')
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        img_size: Tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
        split: str = 'train',
        in_channels: int = 1
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.split = split
        self.in_channels = in_channels
        
        # Find all images
        self.image_paths = sorted(list(self.images_dir.glob('*.png')) + list(self.images_dir.glob('*.jpg')))
        
        # Default transform
        if transform is None:
            self.transform = self._get_default_transform(split)
        else:
            self.transform = transform

    def _get_default_transform(self, split: str) -> A.Compose:
        """Get default augmentation transforms."""
        if split == 'train':
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=[0.5], std=[0.5]) if self.in_channels == 1 else A.Normalize(),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(mean=[0.5], std=[0.5]) if self.in_channels == 1 else A.Normalize(),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        img_path = self.image_paths[idx]
        if self.in_channels == 1:
            image = Image.open(img_path).convert('L')
            image = np.array(image)[..., np.newaxis]  # (H, W, 1)
        else:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        
        # Load label mask
        label_path = self.labels_dir / img_path.name
        if label_path.exists():
            mask = Image.open(label_path)
            mask = np.array(mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]  # Take first channel if RGB
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].long()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(img_path)
        }


# =============================================================================
# End-to-End Dataset for DT-SegNet
# =============================================================================

class DTSegNetDataset(Dataset):
    """
    Dataset for end-to-end DT-SegNet training/inference.
    
    Combines detection and segmentation labels.
    
    Args:
        data_dir: Root data directory
        seg_labels_dir: Directory containing segmentation labels
        img_size: Target image size
        split: Dataset split ('train', 'val', 'test')
        roi_dilation: Dilation factor for ROI extraction
    """
    
    def __init__(
        self,
        data_dir: str,
        seg_labels_dir: str,
        img_size: int = 1280,
        split: str = 'train',
        roi_dilation: float = 1.5,
        transform: Optional[Callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.seg_labels_dir = Path(seg_labels_dir)
        self.img_size = img_size
        self.split = split
        self.roi_dilation = roi_dilation
        
        # Find all images
        split_dir = self.data_dir / split
        self.image_paths = sorted(list(split_dir.glob('*.png')) + list(split_dir.glob('*.jpg')))
        
        # Default transform
        if transform is None:
            self.transform = self._get_default_transform(split)
        else:
            self.transform = transform

    def _get_default_transform(self, split: str) -> A.Compose:
        """Get default transforms (no bbox transforms for end-to-end)."""
        if split == 'train':
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size, 
                    min_width=self.img_size,
                    border_mode=0,
                    fill=114
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size, 
                    min_width=self.img_size,
                    border_mode=0,
                    fill=114
                ),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])

    def _compute_bboxes_from_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute YOLO-format bounding boxes from segmentation mask.
        
        Uses connected components to find individual objects.
        Returns normalized (x_center, y_center, width, height) format.
        """
        import cv2
        
        h, w = mask.shape[:2]
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        bboxes = []
        classes = []
        
        # Skip label 0 (background)
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            
            # Filter out very small components (noise)
            if area < 10:
                continue
            
            # Convert to YOLO format (normalized x_center, y_center, width, height)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bbox_w = bw / w
            bbox_h = bh / h
            
            bboxes.append([x_center, y_center, bbox_w, bbox_h])
            classes.append(0)  # Single class: precipitate
        
        if len(bboxes) == 0:
            return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
        
        return np.array(bboxes), np.array(classes, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load image (grayscale for EM images)
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        image_np = np.array(image)
        original_size = image_np.shape[:2]
        
        # Load segmentation mask (full image)
        seg_label_path = self.seg_labels_dir / f'{img_path.stem}.png'
        if seg_label_path.exists():
            seg_mask = np.array(Image.open(seg_label_path))
            if seg_mask.ndim == 3:
                seg_mask = seg_mask[:, :, 0]
            # Binarize if needed
            seg_mask = (seg_mask > 0).astype(np.uint8)
        else:
            seg_mask = np.zeros(original_size, dtype=np.uint8)
        
        # Compute detection bboxes from segmentation mask
        det_bboxes, det_classes = self._compute_bboxes_from_mask(seg_mask)
        
        # Apply transforms
        transformed = self.transform(image=image_np[..., np.newaxis], mask=seg_mask)
        image = transformed['image']
        seg_mask = transformed['mask'].long()
        
        if len(det_bboxes) > 0:
            det_bboxes = torch.tensor(det_bboxes, dtype=torch.float32)
            det_classes = torch.tensor(det_classes, dtype=torch.int64)
        else:
            det_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            det_classes = torch.zeros(0, dtype=torch.int64)
        
        return {
            'image': image,
            'seg_mask': seg_mask,
            'det_bboxes': det_bboxes,
            'det_classes': det_classes,
            'image_path': str(img_path),
            'original_size': original_size
        }


# =============================================================================
# Collate Functions
# =============================================================================

def detection_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for detection dataset."""
    images = torch.stack([item['image'] for item in batch])
    
    # Bboxes and classes need special handling (variable length)
    bboxes = [item['bboxes'] for item in batch]
    class_labels = [item['class_labels'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'bboxes': bboxes,
        'class_labels': class_labels,
        'image_paths': image_paths
    }


def segmentation_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for segmentation dataset."""
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'masks': masks,
        'image_paths': image_paths
    }


def dtsegnet_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for DT-SegNet dataset."""
    images = torch.stack([item['image'] for item in batch])
    seg_masks = torch.stack([item['seg_mask'] for item in batch])
    
    det_bboxes = [item['det_bboxes'] for item in batch]
    det_classes = [item['det_classes'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'images': images,
        'seg_masks': seg_masks,
        # Aliases for compatibility with train_detector
        'bboxes': det_bboxes,
        'det_bboxes': det_bboxes,
        'class_labels': det_classes,
        'det_classes': det_classes,
        'image_paths': image_paths,
        'original_sizes': original_sizes
    }


# =============================================================================
# DataLoader Factory Functions
# =============================================================================

def create_detection_dataloader(
    data_dir: str,
    split: str = 'train',
    img_size: int = 1280,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create detection dataloader."""
    dataset = DetectionDataset(
        data_dir=data_dir,
        img_size=img_size,
        split=split
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == 'train' else False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=split == 'train'
    )


def create_segmentation_dataloader(
    images_dir: str,
    labels_dir: str,
    split: str = 'train',
    img_size: Tuple[int, int] = (512, 512),
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    in_channels: int = 1
) -> DataLoader:
    """Create segmentation dataloader."""
    dataset = SegmentationDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        img_size=img_size,
        split=split,
        in_channels=in_channels
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == 'train' else False,
        num_workers=num_workers,
        collate_fn=segmentation_collate_fn,
        pin_memory=True,
        drop_last=split == 'train'
    )


def create_dtsegnet_dataloader(
    data_dir: str,
    seg_labels_dir: str,
    split: str = 'train',
    img_size: int = 1280,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create end-to-end DT-SegNet dataloader."""
    dataset = DTSegNetDataset(
        data_dir=data_dir,
        seg_labels_dir=seg_labels_dir,
        img_size=img_size,
        split=split
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == 'train' else False,
        num_workers=num_workers,
        collate_fn=dtsegnet_collate_fn,
        pin_memory=True,
        drop_last=split == 'train'
    )
