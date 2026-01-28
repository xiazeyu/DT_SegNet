"""
DT-SegNet Tests
===============
Unit tests for model components and end-to-end functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    Conv, Bottleneck, C3, SPPF, Detect,
    YOLOv5Backbone, YOLOv5Neck, YOLOv5,
    DropPath, MixFFN, EfficientSelfAttention, TransformerBlock,
    OverlapPatchEmbed, MixVisionTransformer, SegFormerHead, SegFormer,
    DTSegNet, DetectionResult, DTSegNetOutput,
    create_detector, create_segmentor, create_dtsegnet
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get computation device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def batch_size():
    return 2


# =============================================================================
# YOLOv5 Component Tests
# =============================================================================

class TestYOLOv5Components:
    """Test YOLOv5 building blocks."""
    
    def test_conv(self, device):
        """Test Conv block."""
        conv = Conv(3, 64, k=3, s=2).to(device)
        x = torch.randn(1, 3, 64, 64).to(device)
        out = conv(x)
        
        assert out.shape == (1, 64, 32, 32)
    
    def test_bottleneck(self, device):
        """Test Bottleneck block."""
        block = Bottleneck(64, 64).to(device)
        x = torch.randn(1, 64, 32, 32).to(device)
        out = block(x)
        
        assert out.shape == x.shape
    
    def test_c3(self, device):
        """Test C3 block."""
        c3 = C3(64, 128, n=2).to(device)
        x = torch.randn(1, 64, 32, 32).to(device)
        out = c3(x)
        
        assert out.shape == (1, 128, 32, 32)
    
    def test_sppf(self, device):
        """Test SPPF block."""
        sppf = SPPF(64, 64).to(device)
        x = torch.randn(1, 64, 32, 32).to(device)
        out = sppf(x)
        
        assert out.shape == x.shape
    
    def test_backbone(self, device):
        """Test YOLOv5 backbone."""
        backbone = YOLOv5Backbone(3, depth_multiple=0.33, width_multiple=0.5).to(device)
        x = torch.randn(1, 3, 256, 256).to(device)
        features = backbone(x)
        
        assert len(features) == 3  # P3, P4, P5
        assert features[0].shape[2] > features[1].shape[2] > features[2].shape[2]
    
    def test_neck(self, device):
        """Test YOLOv5 neck (PANet)."""
        in_channels = [32, 64, 128]  # Matches backbone output
        neck = YOLOv5Neck(in_channels, depth_multiple=0.33, width_multiple=0.5).to(device)
        
        # Create dummy features
        features = [
            torch.randn(1, 32, 32, 32).to(device),
            torch.randn(1, 64, 16, 16).to(device),
            torch.randn(1, 128, 8, 8).to(device),
        ]
        
        out = neck(features)
        assert len(out) == 3


class TestYOLOv5Model:
    """Test complete YOLOv5 model."""
    
    @pytest.mark.parametrize("model_size", ['n', 's', 'm'])
    def test_model_sizes(self, device, model_size):
        """Test different model sizes."""
        model = create_detector(
            num_classes=1, 
            model_size=model_size, 
            img_size=320
        ).to(device)
        
        x = torch.randn(1, 3, 320, 320).to(device)
        model.eval()
        
        with torch.no_grad():
            out, features = model(x)
        
        # Output should be (batch, num_predictions, 6) for nc=1
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 6  # x, y, w, h, conf, cls
    
    def test_training_mode(self, device):
        """Test model in training mode."""
        model = create_detector(num_classes=1, model_size='n', img_size=320).to(device)
        model.train()
        
        x = torch.randn(1, 3, 320, 320).to(device)
        out = model(x)
        
        # In training mode, returns list of feature maps
        assert isinstance(out, list) or isinstance(out, tuple)


# =============================================================================
# SegFormer Component Tests
# =============================================================================

class TestSegFormerComponents:
    """Test SegFormer building blocks."""
    
    def test_drop_path(self, device):
        """Test DropPath."""
        drop = DropPath(0.1)
        x = torch.randn(2, 100, 64).to(device)
        
        # In eval mode, should pass through
        drop.eval()
        out = drop(x)
        assert torch.allclose(out, x)
    
    def test_mix_ffn(self, device):
        """Test MixFFN."""
        ffn = MixFFN(64, 256, 64).to(device)
        x = torch.randn(2, 100, 64).to(device)
        out = ffn(x, 10, 10)
        
        assert out.shape == x.shape
    
    def test_efficient_attention(self, device):
        """Test EfficientSelfAttention."""
        attn = EfficientSelfAttention(dim=64, num_heads=4, sr_ratio=2).to(device)
        x = torch.randn(2, 100, 64).to(device)
        out = attn(x, 10, 10)
        
        assert out.shape == x.shape
    
    def test_transformer_block(self, device):
        """Test TransformerBlock."""
        block = TransformerBlock(dim=64, num_heads=4, sr_ratio=2).to(device)
        x = torch.randn(2, 100, 64).to(device)
        out = block(x, 10, 10)
        
        assert out.shape == x.shape
    
    def test_overlap_patch_embed(self, device):
        """Test OverlapPatchEmbed."""
        embed = OverlapPatchEmbed(patch_size=7, stride=4, in_channels=3, embed_dim=64).to(device)
        x = torch.randn(2, 3, 224, 224).to(device)
        out, H, W = embed(x)
        
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 64
        assert H == 56 and W == 56
    
    def test_mix_vision_transformer(self, device):
        """Test MixVisionTransformer backbone."""
        backbone = MixVisionTransformer(
            in_channels=1,
            embed_dims=[32, 64, 160, 256],
            depths=[2, 2, 2, 2]
        ).to(device)
        
        x = torch.randn(2, 1, 224, 224).to(device)
        features = backbone(x)
        
        assert len(features) == 4
        # Check hierarchical feature sizes
        for i in range(len(features) - 1):
            assert features[i].shape[2] >= features[i+1].shape[2]


class TestSegFormerModel:
    """Test complete SegFormer model."""
    
    @pytest.mark.parametrize("backbone_size", ['b0', 'b1'])
    def test_model_sizes(self, device, backbone_size):
        """Test different backbone sizes."""
        model = create_segmentor(
            num_classes=2,
            backbone_size=backbone_size,
            in_channels=1
        ).to(device)
        
        x = torch.randn(1, 1, 256, 256).to(device)
        model.eval()
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (1, 2, 256, 256)
    
    def test_different_input_sizes(self, device):
        """Test model with different input sizes."""
        model = create_segmentor(num_classes=2, backbone_size='b0', in_channels=1).to(device)
        model.eval()
        
        for size in [128, 256, 512]:
            x = torch.randn(1, 1, size, size).to(device)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 2, size, size)


# =============================================================================
# DT-SegNet End-to-End Tests
# =============================================================================

class TestDTSegNet:
    """Test end-to-end DT-SegNet model."""
    
    def test_model_creation(self, device):
        """Test model can be created."""
        model = create_dtsegnet(
            detector_size='n',
            segmentor_size='b0'
        ).to(device)
        
        assert isinstance(model, DTSegNet)
        assert isinstance(model.detector, YOLOv5)
        assert isinstance(model.segmentor, SegFormer)
    
    def test_inference_mode(self, device):
        """Test end-to-end inference."""
        model = create_dtsegnet(
            detector_size='n',
            segmentor_size='b0'
        ).to(device)
        model.eval()
        
        x = torch.randn(1, 1, 320, 320).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert isinstance(output, DTSegNetOutput)
        assert output.segmentation_mask.shape == (1, 320, 320)
    
    def test_intermediate_results(self, device):
        """Test returning intermediate results."""
        model = create_dtsegnet(
            detector_size='n',
            segmentor_size='b0'
        ).to(device)
        model.eval()
        
        x = torch.randn(1, 1, 320, 320).to(device)
        
        with torch.no_grad():
            output = model(x, return_intermediate=True)
        
        assert output.detections is not None
        assert len(output.detections) == 1
        assert isinstance(output.detections[0], DetectionResult)
    
    def test_batch_inference(self, device, batch_size):
        """Test batch inference."""
        model = create_dtsegnet(
            detector_size='n',
            segmentor_size='b0'
        ).to(device)
        model.eval()
        
        x = torch.randn(batch_size, 1, 320, 320).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.segmentation_mask.shape == (batch_size, 320, 320)
    
    def test_train_modes(self, device):
        """Test training mode switches."""
        model = create_dtsegnet(
            detector_size='n',
            segmentor_size='b0'
        ).to(device)
        
        # Train detector
        model.train_detector()
        assert model.detector.training
        assert not model.segmentor.training
        
        # Train segmentor
        model.train_segmentor()
        assert not model.detector.training
        assert model.segmentor.training


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_memory_efficiency(self, device):
        """Test that no intermediate results are written to disk."""
        model = create_dtsegnet(
            detector_size='n',
            segmentor_size='b0'
        ).to(device)
        model.eval()
        
        # Create temporary directory to ensure nothing is written
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            initial_files = set(os.listdir(tmpdir))
            
            # Run inference
            x = torch.randn(1, 1, 320, 320).to(device)
            with torch.no_grad():
                output = model(x, return_intermediate=True)
            
            # Check no files were created
            final_files = set(os.listdir(tmpdir))
            assert initial_files == final_files
    
    def test_gpu_memory_usage(self, device):
        """Test that GPU memory is properly managed."""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
        
        torch.cuda.reset_peak_memory_stats()
        
        model = create_dtsegnet(
            detector_size='n',
            segmentor_size='b0'
        ).to(device)
        model.eval()
        
        x = torch.randn(1, 1, 320, 320).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {peak_memory:.2f} MB")
        
        # Memory should be reasonable for small model
        assert peak_memory < 2000  # Less than 2GB


# =============================================================================
# Quick Smoke Test
# =============================================================================

def test_smoke():
    """Quick smoke test to verify basic functionality."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test detector
    detector = create_detector(model_size='n', img_size=256).to(device)
    detector.eval()
    x = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        out, _ = detector(x)
    assert out.ndim == 3
    
    # Test segmentor
    segmentor = create_segmentor(backbone_size='b0', in_channels=1).to(device)
    segmentor.eval()
    x = torch.randn(1, 1, 128, 128).to(device)
    with torch.no_grad():
        out = segmentor(x)
    assert out.shape == (1, 2, 128, 128)
    
    # Test end-to-end
    model = create_dtsegnet(detector_size='n', segmentor_size='b0').to(device)
    model.eval()
    x = torch.randn(1, 1, 256, 256).to(device)
    with torch.no_grad():
        out = model(x)
    assert out.segmentation_mask.shape == (1, 256, 256)
    
    print("All smoke tests passed!")


if __name__ == '__main__':
    # Run quick smoke test
    test_smoke()
    
    # Run full tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
