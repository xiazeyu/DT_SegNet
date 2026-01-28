# DT-SegNet v2.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

**DT-SegNet: Two-Stage Deep Learning for Precipitate Detection and Segmentation**

A PyTorch implementation of an end-to-end two-stage deep learning model combining YOLOv5-based detection and SegFormer-based segmentation for precise precipitate identification in electron microscopy images.

## ✨ Key Features

- **End-to-End Architecture**: Combines detection (YOLOv5) and segmentation (SegFormer) in a single pipeline
- **Pure PyTorch**: Fully implemented in PyTorch (no PaddlePaddle dependency)
- **Memory Efficient**: All processing happens in memory without intermediate disk I/O
- **CLI Interface**: Easy-to-use command-line tools for training and inference
- **Modern Python**: Uses `uv` for fast dependency management
- **Well Tested**: Includes comprehensive unit tests

## 📁 Project Structure

```
DT_SegNet/
├── src/
│   ├── __init__.py
│   ├── model.py      # YOLOv5, SegFormer, and DTSegNet models
│   ├── dataset.py    # Dataset loaders for detection and segmentation
│   ├── train.py      # Training script with CLI
│   └── infer.py      # Inference script with CLI
├── tests/
│   └── test_model.py # Unit tests
├── Dataset/          # Dataset directory
├── old/              # Original implementation (archived)
├── pyproject.toml    # Project configuration (uv/pip compatible)
└── README.md
```

## 🚀 Quick Start

### Installation

1. **Install uv** (recommended package manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Create virtual environment and install dependencies**:
```bash
cd DT_SegNet
uv venv
source .venv/bin/activate  # On macOS/Linux
# or: .venv\Scripts\activate  # On Windows
uv pip install -e .
```

Or using pip:
```bash
pip install -e .
```

### Run Tests

```bash
# Quick smoke test
python tests/test_model.py

# Full test suite
uv run pytest tests/ -v
```

## 📖 Usage

### Training

Train the end-to-end DT-SegNet model (two-stage detection + segmentation):

```bash
python -m src.train \
    --data-dir ./Dataset \
    --detector-size l \
    --segmentor-size b1 \
    --img-size 1280 \
    --epochs-det 100 \
    --epochs-seg 100 \
    --batch-size 8 \
    --output-dir ./outputs
```

The training process:
1. **Stage 1**: Train YOLOv5 detector for object localization
2. **Stage 2**: Train SegFormer segmentor for pixel-wise segmentation

Output files:
- `best.pt`: Best model checkpoint (by validation IoU)
- `final.pt`: Final model checkpoint after training

### Inference

Run end-to-end inference on images:

```bash
python -m src.infer \
    --model-path ./outputs/best.pt \
    --input ./test_images \
    --output ./results \
    --save-intermediate
```

Options:
- `--save-intermediate`: Save detection boxes and ROI masks separately
- `--conf-threshold`: Detection confidence threshold (default: 0.475)
- `--iou-threshold`: NMS IoU threshold (default: 0.45)

## 🏗️ Model Architecture

### YOLOv5 Detector
- CSPDarknet backbone with PANet neck
- Anchor-based detection with multi-scale outputs
- Model sizes: `n`, `s`, `m`, `l`, `x`

### SegFormer Segmentor
- Mix Vision Transformer (MiT) backbone
- MLP decoder head
- Backbone sizes: `b0`, `b1`, `b2`, `b3`, `b4`, `b5`

### End-to-End Pipeline
1. **Detection**: YOLOv5 localizes precipitate regions
2. **ROI Extraction**: Dilated bounding boxes are cropped (in memory)
3. **Segmentation**: SegFormer segments each ROI
4. **Merging**: ROI masks are merged into full-size output

## 📊 Dataset Format

### Segmentation Labels
- PNG mask files in `labels/` directory
- Binary: 0 = background, 255 = precipitate
- Bounding boxes are automatically computed from segmentation masks

### Directory Structure
```
Dataset/
├── train/
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── val/
│   ├── 5.png
│   └── ...
├── test/
│   └── ...
└── labels/
    ├── 1.png
    ├── 2.png
    ├── 5.png
    └── ...
```

## ⚙️ Configuration

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--detector-size` | `l` | YOLOv5 model size (n/s/m/l/x) |
| `--segmentor-size` | `b1` | SegFormer backbone (b0-b5) |
| `--img-size` | `1280` | Input image size |
| `--batch-size` | `8` | Training batch size |
| `--lr` | `1e-4` | Learning rate |
| `--epochs-det` | `100` | Detector training epochs |
| `--epochs-seg` | `100` | Segmentor training epochs |

### Inference Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--conf-threshold` | `0.475` | Detection confidence threshold |
| `--iou-threshold` | `0.45` | NMS IoU threshold |
| `--batch-size` | `1` | Inference batch size |

## 🔧 Development

### Run Tests
```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run specific test
uv run pytest tests/test_model.py::TestDTSegNet -v
```

### Code Formatting
```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/
```

## 📚 API Reference

### Create Models

```python
from src.model import create_detector, create_segmentor, create_dtsegnet

# Create detector
detector = create_detector(num_classes=1, model_size='l', img_size=1280)

# Create segmentor
segmentor = create_segmentor(num_classes=2, backbone_size='b1', in_channels=1)

# Create end-to-end model
model = create_dtsegnet(
    detector_size='l',
    segmentor_size='b1',
    conf_threshold=0.475,
    iou_threshold=0.45
)
```

### Run Inference

```python
import torch
from src.model import create_dtsegnet

device = torch.device('cuda')
model = create_dtsegnet().to(device)

# Load complete model checkpoint
checkpoint = torch.load('outputs/best.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

# Input: grayscale image (B, 1, H, W)
image = torch.randn(1, 1, 1280, 1280).to(device)

with torch.no_grad():
    output = model(image, return_intermediate=True)

# Output
mask = output.segmentation_mask  # (B, H, W)
detections = output.detections   # List of DetectionResult
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{dtsegnet2023,
  title={Accurate identification and measurement of the precipitate area by two-stage deep neural networks in novel chromium-based alloys},
  journal={Physical Chemistry Chemical Physics},
  year={2023},
  doi={10.1039/D3CP00402C}
}
```

## 🔄 Migration from v1.0

The v2.0 release is a complete rewrite with the following changes:

| v1.0 | v2.0 |
|------|------|
| Conda | uv/pip |
| PaddlePaddle + PyTorch | Pure PyTorch |
| Jupyter notebooks | CLI scripts |
| Intermediate disk I/O | In-memory processing |
| Separate models | Unified model.py |

To migrate, simply use the new CLI interface. The dataset format remains compatible.
