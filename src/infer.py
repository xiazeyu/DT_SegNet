"""
DT-SegNet Inference Script
==========================
Command-line interface for running end-to-end inference with trained models.

Usage:
    python -m src.infer --model-path ./outputs/best.pt --input ./images --output ./results
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from .model import DTSegNet, create_dtsegnet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Image Processing Utilities
# =============================================================================

def load_image(
    image_path: Path, 
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load and preprocess a grayscale image.
    
    Args:
        image_path: Path to image file
        target_size: Optional target size (H, W)
        
    Returns:
        Tuple of (preprocessed tensor, original size)
    """
    image = Image.open(image_path)
    original_size = image.size[::-1]  # (H, W)
    
    image = image.convert('L')
    image_np = np.array(image)[..., np.newaxis]  # (H, W, 1)
    
    # Resize if needed
    if target_size and (image_np.shape[0] != target_size[0] or image_np.shape[1] != target_size[1]):
        image = Image.fromarray(image_np.squeeze())
        image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        image_np = np.array(image)[..., np.newaxis]
    
    # Normalize and convert to tensor
    image_tensor = torch.from_numpy(image_np).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # (1, H, W)
    image_tensor = (image_tensor / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
    
    return image_tensor, original_size


def save_mask(
    mask: torch.Tensor, 
    output_path: Path, 
    original_size: Optional[Tuple[int, int]] = None
):
    """
    Save segmentation mask to file.
    
    Args:
        mask: Segmentation mask tensor (H, W)
        output_path: Output file path
        original_size: Optional original size to resize to
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    
    if original_size and (mask_np.shape[0] != original_size[0] or mask_np.shape[1] != original_size[1]):
        mask_img = Image.fromarray(mask_np * 255)
        mask_img = mask_img.resize((original_size[1], original_size[0]), Image.NEAREST)
        mask_np = np.array(mask_img) // 255
    
    # Save as PNG (binary mask)
    Image.fromarray(mask_np * 255).save(output_path)
    
    # Also save as numpy array
    np.save(output_path.with_suffix('.npy'), mask_np)


def save_detections(
    detections: List[Dict], 
    output_path: Path,
    image_size: Tuple[int, int]
):
    """
    Save detection results to YOLO format.
    
    Args:
        detections: List of detection dictionaries
        output_path: Output file path
        image_size: Image size (H, W)
    """
    H, W = image_size
    
    with open(output_path.with_suffix('.txt'), 'w') as f:
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['score']
            cls = det['class']
            
            # Convert to YOLO format (normalized xywh)
            x_center = (x1 + x2) / 2 / W
            y_center = (y1 + y2) / 2 / H
            width = (x2 - x1) / W
            height = (y2 - y1) / H
            
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")


# =============================================================================
# Inference Functions
# =============================================================================

def infer(
    model: DTSegNet,
    image_paths: List[Path],
    output_dir: Path,
    device: torch.device,
    img_size: int = 1280,
    save_intermediate: bool = False
) -> List[Dict[str, Any]]:
    """
    Run end-to-end DT-SegNet inference.
    
    All processing happens in memory without intermediate disk I/O.
    
    Args:
        model: DTSegNet model
        image_paths: List of image paths
        output_dir: Output directory
        device: Computation device
        img_size: Input image size
        save_intermediate: Whether to save intermediate detection results
        
    Returns:
        List of results
    """
    model.eval()
    results = []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'masks').mkdir(exist_ok=True)
    if save_intermediate:
        (output_dir / 'detections').mkdir(exist_ok=True)
    
    for image_path in tqdm(image_paths, desc='Processing'):
        # Load image
        image, original_size = load_image(image_path, target_size=(img_size, img_size))
        image = image.unsqueeze(0).to(device)
        
        # Run end-to-end inference
        with torch.no_grad():
            output = model(image, return_intermediate=save_intermediate)
        
        # Get segmentation mask
        mask = output.segmentation_mask[0]  # (H, W)
        
        # Save mask (resized to original size)
        output_path = output_dir / 'masks' / f'{image_path.stem}.png'
        save_mask(mask, output_path, original_size)
        
        result = {
            'image_path': str(image_path),
            'mask_path': str(output_path),
            'original_size': original_size
        }
        
        # Save intermediate detection results if requested
        if save_intermediate and output.detections:
            det = output.detections[0]
            detections = []
            for i in range(det.boxes.shape[0]):
                box = det.boxes[i].tolist()
                # Scale to original size
                scale_x = original_size[1] / img_size
                scale_y = original_size[0] / img_size
                detections.append({
                    'box': [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y],
                    'score': det.scores[i].item(),
                    'class': int(det.classes[i].item())
                })
            
            det_path = output_dir / 'detections' / f'{image_path.stem}'
            save_detections(detections, det_path, original_size)
            result['detections'] = detections
        
        results.append(result)
    
    return results


def batch_infer(
    model: DTSegNet,
    image_paths: List[Path],
    output_dir: Path,
    device: torch.device,
    img_size: int = 1280,
    batch_size: int = 4
) -> List[Dict[str, Any]]:
    """
    Run batched end-to-end inference for efficiency.
    
    Args:
        model: DTSegNet model
        image_paths: List of image paths
        output_dir: Output directory
        device: Computation device
        img_size: Input image size
        batch_size: Batch size for inference
        
    Returns:
        List of results
    """
    model.eval()
    results = []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'masks').mkdir(exist_ok=True)
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc='Processing batches'):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load batch
        images = []
        original_sizes = []
        for image_path in batch_paths:
            image, original_size = load_image(image_path, target_size=(img_size, img_size))
            images.append(image)
            original_sizes.append(original_size)
        
        images = torch.stack(images).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(images, return_intermediate=False)
        
        # Process outputs
        for j, image_path in enumerate(batch_paths):
            mask = output.segmentation_mask[j]
            
            output_path = output_dir / 'masks' / f'{image_path.stem}.png'
            save_mask(mask, output_path, original_sizes[j])
            
            results.append({
                'image_path': str(image_path),
                'mask_path': str(output_path),
                'original_size': original_sizes[j]
            })
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DT-SegNet End-to-End Inference')
    
    # Paths
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, default='./inference_output',
                        help='Output directory')
    
    # Model configuration (must match training config)
    parser.add_argument('--detector-size', type=str, default='l',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv5 model size')
    parser.add_argument('--segmentor-size', type=str, default='b1',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer backbone size')
    parser.add_argument('--img-size', type=int, default=1280,
                        help='Input image size')
    
    # Inference configuration
    parser.add_argument('--conf-threshold', type=float, default=0.475,
                        help='Detection confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--save-intermediate', action='store_true',
                        help='Save intermediate detection results')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, cpu, or auto)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f'Using device: {device}')
    
    # Find input images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(
            list(input_path.glob('*.png')) + 
            list(input_path.glob('*.jpg')) + 
            list(input_path.glob('*.jpeg'))
        )
    
    if len(image_paths) == 0:
        logger.error(f'No images found in {args.input}')
        return
    
    logger.info(f'Found {len(image_paths)} images')
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = create_dtsegnet(
        detector_size=args.detector_size,
        segmentor_size=args.segmentor_size,
        img_size=args.img_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # Load checkpoint
    logger.info(f'Loading model from {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Run inference
    if args.batch_size > 1:
        results = batch_infer(
            model, image_paths, output_dir, device,
            args.img_size, args.batch_size
        )
    else:
        results = infer(
            model, image_paths, output_dir, device,
            args.img_size, args.save_intermediate
        )
    
    logger.info(f'Inference complete! Results saved to {output_dir}')
    logger.info(f'Processed {len(results)} images')


if __name__ == '__main__':
    main()
