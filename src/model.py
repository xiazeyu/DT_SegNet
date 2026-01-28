"""
DT-SegNet Model Architecture
============================
End-to-end two-stage model combining:
- Stage 1: YOLOv5-based detection for precipitate localization
- Stage 2: SegFormer-based segmentation for precise boundary delineation

All computations happen in memory without intermediate disk I/O.
"""

import math
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utility Functions
# =============================================================================

def autopad(k: int, p: Optional[int] = None) -> int:
    """Pad to 'same' for convolution."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x: int, divisor: int) -> int:
    """Returns nearest x divisible by divisor."""
    return math.ceil(x / divisor) * divisor


# =============================================================================
# YOLOv5 Building Blocks
# =============================================================================

class Conv(nn.Module):
    """Standard convolution with BatchNorm and SiLU activation."""
    
    def __init__(
        self, 
        c1: int, 
        c2: int, 
        k: int = 1, 
        s: int = 1, 
        p: Optional[int] = None, 
        g: int = 1, 
        act: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    
    def __init__(
        self, 
        c1: int, 
        c2: int, 
        shortcut: bool = True, 
        g: int = 1, 
        e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    
    def __init__(
        self, 
        c1: int, 
        c2: int, 
        n: int = 1, 
        shortcut: bool = True, 
        g: int = 1, 
        e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    """Concatenate tensors along dimension."""
    
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


class Detect(nn.Module):
    """YOLOv5 Detect head for object detection."""
    
    stride: torch.Tensor
    
    def __init__(self, nc: int = 1, anchors: tuple = (), ch: tuple = ()):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (x, y, w, h, conf, cls)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors per layer
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
                y = x[i].sigmoid()
                xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        
        return (torch.cat(z, 1), x) if not self.training else x

    def _make_grid(self, nx: int, ny: int, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape)
        anchor_grid = (self.anchors[i] * self.stride[i]).view(1, self.na, 1, 1, 2).expand(shape)
        return grid, anchor_grid


# =============================================================================
# YOLOv5 Detector Model
# =============================================================================

class YOLOv5Backbone(nn.Module):
    """YOLOv5 backbone (CSPDarknet)."""
    
    def __init__(
        self, 
        in_channels: int = 3, 
        depth_multiple: float = 1.0, 
        width_multiple: float = 1.0
    ):
        super().__init__()
        
        # Compute channels
        def ch(c): return max(round(c * width_multiple), 1)
        def n(num): return max(round(num * depth_multiple), 1)
        
        # Focus layer replaced with regular Conv for simplicity
        self.stem = Conv(in_channels, ch(64), 6, 2, 2)
        
        # Backbone
        self.layer1 = nn.Sequential(
            Conv(ch(64), ch(128), 3, 2),
            C3(ch(128), ch(128), n(3))
        )
        self.layer2 = nn.Sequential(
            Conv(ch(128), ch(256), 3, 2),
            C3(ch(256), ch(256), n(6))
        )
        self.layer3 = nn.Sequential(
            Conv(ch(256), ch(512), 3, 2),
            C3(ch(512), ch(512), n(9))
        )
        self.layer4 = nn.Sequential(
            Conv(ch(512), ch(1024), 3, 2),
            C3(ch(1024), ch(1024), n(3)),
            SPPF(ch(1024), ch(1024), 5)
        )
        
        self.out_channels = [ch(256), ch(512), ch(1024)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        p3 = self.layer2(x)  # P3
        p4 = self.layer3(p3)  # P4
        p5 = self.layer4(p4)  # P5
        return [p3, p4, p5]


class YOLOv5Neck(nn.Module):
    """YOLOv5 neck (PANet)."""
    
    def __init__(
        self, 
        in_channels: List[int], 
        depth_multiple: float = 1.0, 
        width_multiple: float = 1.0
    ):
        super().__init__()
        
        def n(num): return max(round(num * depth_multiple), 1)
        
        c3, c4, c5 = in_channels
        
        # Top-down path (FPN)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Lateral connections and top-down path
        self.lateral_c5 = Conv(c5, c4, 1, 1)
        self.fpn_p4 = C3(c4 * 2, c4, n(3), shortcut=False)  # concat c4 + upsampled c5
        
        self.lateral_c4 = Conv(c4, c3, 1, 1)
        self.fpn_p3 = C3(c3 * 2, c3, n(3), shortcut=False)  # concat c3 + upsampled c4
        
        # Bottom-up path (PAN)
        self.downsample_p3 = Conv(c3, c3, 3, 2)
        self.pan_n4 = C3(c3 * 2, c4, n(3), shortcut=False)  # concat downsampled p3 + fpn_p4 output
        
        self.downsample_n4 = Conv(c4, c4, 3, 2)
        self.pan_n5 = C3(c4 * 2, c5, n(3), shortcut=False)  # concat downsampled n4 + lateral_c5 output
        
        self.out_channels = [c3, c4, c5]

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features
        
        # Top-down path (FPN)
        # C5 -> C4
        lat_c5 = self.lateral_c5(p5)  # c5 -> c4
        fpn_p4 = self.fpn_p4(torch.cat([self.up(lat_c5), p4], 1))  # (c4 + c4) -> c4
        
        # C4 -> C3
        lat_c4 = self.lateral_c4(fpn_p4)  # c4 -> c3
        fpn_p3 = self.fpn_p3(torch.cat([self.up(lat_c4), p3], 1))  # (c3 + c3) -> c3
        
        # Bottom-up path (PAN)
        # N3 -> N4
        down_p3 = self.downsample_p3(fpn_p3)  # c3 -> c3 (with stride 2)
        pan_n4 = self.pan_n4(torch.cat([down_p3, lat_c4], 1))  # (c3 + c3) -> c4
        
        # N4 -> N5
        down_n4 = self.downsample_n4(pan_n4)  # c4 -> c4 (with stride 2)
        pan_n5 = self.pan_n5(torch.cat([down_n4, lat_c5], 1))  # (c4 + c4) -> c5
        
        return [fpn_p3, pan_n4, pan_n5]


class YOLOv5(nn.Module):
    """
    YOLOv5 detection model.
    
    Args:
        num_classes: Number of object classes (default: 1 for precipitate detection)
        img_size: Input image size (default: 1280)
        model_size: Model variant ('n', 's', 'm', 'l', 'x')
    """
    
    # Model size configurations: (depth_multiple, width_multiple)
    SIZE_CONFIG = {
        'n': (0.33, 0.25),
        's': (0.33, 0.50),
        'm': (0.67, 0.75),
        'l': (1.00, 1.00),
        'x': (1.33, 1.25),
    }
    
    # Anchors for different scales
    ANCHORS = [
        [10, 13, 16, 30, 33, 23],       # P3/8
        [30, 61, 62, 45, 59, 119],      # P4/16
        [116, 90, 156, 198, 373, 326],  # P5/32
    ]
    
    def __init__(
        self, 
        num_classes: int = 1, 
        img_size: int = 1280, 
        model_size: str = 'l',
        in_channels: int = 3
    ):
        super().__init__()
        
        assert model_size in self.SIZE_CONFIG, f"Model size must be one of {list(self.SIZE_CONFIG.keys())}"
        depth_multiple, width_multiple = self.SIZE_CONFIG[model_size]
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Build model
        self.backbone = YOLOv5Backbone(in_channels, depth_multiple, width_multiple)
        self.neck = YOLOv5Neck(self.backbone.out_channels, depth_multiple, width_multiple)
        self.detect = Detect(num_classes, self.ANCHORS, self.neck.out_channels)
        
        # Compute strides - need to do forward pass in training mode
        # In training mode, detect returns list directly (not tuple)
        s = 256
        self.detect.stride = torch.tensor([8., 16., 32.])  # Default strides for P3, P4, P5
        # Verify by running forward pass
        self.train()
        with torch.no_grad():
            dummy_feats = self.backbone(torch.zeros(1, in_channels, s, s))
            dummy_neck = self.neck(dummy_feats)
            # Compute strides from neck output (before detect)
            computed_strides = torch.tensor([s / f.shape[-2] for f in dummy_neck])
            self.detect.stride = computed_strides
        self._check_anchor_order()
        self._initialize_biases()

    def _check_anchor_order(self):
        """Check anchor order against stride order."""
        a = self.detect.anchors.prod(-1).mean(-1).view(-1)
        da = a[-1] - a[0]
        ds = self.detect.stride[-1] - self.detect.stride[0]
        if da and (da.sign() != ds.sign()):
            self.detect.anchors[:] = self.detect.anchors.flip(0)

    def _initialize_biases(self):
        """Initialize detection biases."""
        for mi, s in zip(self.detect.m, self.detect.stride):
            b = mi.bias.view(self.detect.na, -1)
            b.data[:, 4] += math.log(8 / (self.img_size / s) ** 2)
            b.data[:, 5:5 + self.num_classes] += math.log(0.6 / (self.num_classes - 0.99999))
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = self.backbone(x)
        features = self.neck(features)
        return self.detect(features)


# =============================================================================
# SegFormer Building Blocks (PyTorch Implementation)
# =============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class DWConvSegFormer(nn.Module):
    """Depth-wise convolution for SegFormer MLP."""
    
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixFFN(nn.Module):
    """Mix Feed-Forward Network with depth-wise conv."""
    
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConvSegFormer(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientSelfAttention(nn.Module):
    """Efficient self-attention with spatial reduction."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """SegFormer Transformer block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        sr_ratio: int = 1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Overlapping Patch Embedding."""
    
    def __init__(
        self, 
        patch_size: int = 7, 
        stride: int = 4, 
        in_channels: int = 3, 
        embed_dim: int = 768
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=stride, 
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    """Mix Vision Transformer (MiT) backbone for SegFormer."""
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [64, 128, 320, 512],
        num_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        depths: List[int] = [3, 4, 6, 3],
        sr_ratios: List[int] = [8, 4, 2, 1]
    ):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_channels, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])
        
        # Transformer blocks
        cur = 0
        self.block1 = nn.ModuleList([
            TransformerBlock(
                embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias,
                drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[0]
            ) for i in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        
        cur += depths[0]
        self.block2 = nn.ModuleList([
            TransformerBlock(
                embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias,
                drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[1]
            ) for i in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        
        cur += depths[1]
        self.block3 = nn.ModuleList([
            TransformerBlock(
                embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias,
                drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[2]
            ) for i in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        
        cur += depths[2]
        self.block4 = nn.ModuleList([
            TransformerBlock(
                embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias,
                drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[3]
            ) for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        outs = []
        
        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        
        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        
        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        
        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        
        return outs


class SegFormerHead(nn.Module):
    """SegFormer MLP decoder head."""
    
    def __init__(
        self, 
        in_channels: List[int], 
        embedding_dim: int = 256, 
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.linear_c4 = nn.Sequential(
            nn.Linear(in_channels[3], embedding_dim),
        )
        self.linear_c3 = nn.Sequential(
            nn.Linear(in_channels[2], embedding_dim),
        )
        self.linear_c2 = nn.Sequential(
            nn.Linear(in_channels[1], embedding_dim),
        )
        self.linear_c1 = nn.Sequential(
            nn.Linear(in_channels[0], embedding_dim),
        )
        
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        
        self.dropout = nn.Dropout2d(dropout)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        n, _, h1, w1 = c1.shape
        
        # MLP projection
        _c4 = self.linear_c4(c4.flatten(2).transpose(1, 2))
        _c4 = _c4.transpose(1, 2).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=(h1, w1), mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3.flatten(2).transpose(1, 2))
        _c3 = _c3.transpose(1, 2).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=(h1, w1), mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2.flatten(2).transpose(1, 2))
        _c2 = _c2.transpose(1, 2).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=(h1, w1), mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1.flatten(2).transpose(1, 2))
        _c1 = _c1.transpose(1, 2).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        # Fuse
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        
        return x


# =============================================================================
# SegFormer Model
# =============================================================================

class SegFormer(nn.Module):
    """
    SegFormer segmentation model.
    
    Args:
        num_classes: Number of segmentation classes (default: 2 for binary)
        backbone_size: Backbone variant ('b0', 'b1', 'b2', 'b3', 'b4', 'b5')
        embedding_dim: MLP decoder embedding dimension
        in_channels: Number of input channels
    """
    
    # Backbone configurations
    BACKBONE_CONFIG = {
        'b0': {'embed_dims': [32, 64, 160, 256], 'depths': [2, 2, 2, 2], 'num_heads': [1, 2, 5, 8]},
        'b1': {'embed_dims': [64, 128, 320, 512], 'depths': [2, 2, 2, 2], 'num_heads': [1, 2, 5, 8]},
        'b2': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 4, 6, 3], 'num_heads': [1, 2, 5, 8]},
        'b3': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 4, 18, 3], 'num_heads': [1, 2, 5, 8]},
        'b4': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 8, 27, 3], 'num_heads': [1, 2, 5, 8]},
        'b5': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 6, 40, 3], 'num_heads': [1, 2, 5, 8]},
    }
    
    def __init__(
        self,
        num_classes: int = 2,
        backbone_size: str = 'b1',
        embedding_dim: int = 256,
        in_channels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert backbone_size in self.BACKBONE_CONFIG, f"Backbone must be one of {list(self.BACKBONE_CONFIG.keys())}"
        config = self.BACKBONE_CONFIG[backbone_size]
        
        self.num_classes = num_classes
        self.backbone = MixVisionTransformer(
            in_channels=in_channels,
            embed_dims=config['embed_dims'],
            num_heads=config['num_heads'],
            depths=config['depths']
        )
        self.decode_head = SegFormerHead(
            in_channels=config['embed_dims'],
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        features = self.backbone(x)
        out = self.decode_head(features)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


# =============================================================================
# End-to-End DT-SegNet Model
# =============================================================================

@dataclass
class DetectionResult:
    """Detection result for a single image."""
    boxes: torch.Tensor  # (N, 4) in xyxy format
    scores: torch.Tensor  # (N,)
    classes: torch.Tensor  # (N,)


@dataclass
class DTSegNetOutput:
    """Output of DT-SegNet model."""
    segmentation_mask: torch.Tensor  # Full-size segmentation mask
    detections: Optional[List[DetectionResult]] = None
    roi_masks: Optional[List[torch.Tensor]] = None


class DTSegNet(nn.Module):
    """
    DT-SegNet: End-to-End Two-Stage Deep Learning Model.
    
    Combines YOLOv5 detection and SegFormer segmentation in an end-to-end manner.
    All intermediate results are kept in memory without disk I/O.
    
    Args:
        num_classes_det: Number of detection classes (default: 1)
        num_classes_seg: Number of segmentation classes (default: 2)
        detector_size: YOLOv5 model size ('n', 's', 'm', 'l', 'x')
        segmentor_size: SegFormer backbone size ('b0', 'b1', 'b2', 'b3', 'b4', 'b5')
        img_size: Input image size for detection
        roi_dilation: Dilation factor for ROI cropping
        conf_threshold: Detection confidence threshold
        iou_threshold: NMS IoU threshold
    """
    
    def __init__(
        self,
        num_classes_det: int = 1,
        num_classes_seg: int = 2,
        detector_size: str = 'l',
        segmentor_size: str = 'b1',
        img_size: int = 1280,
        roi_dilation: float = 1.5,
        conf_threshold: float = 0.475,
        iou_threshold: float = 0.45,
        in_channels: int = 1  # Grayscale for EM images
    ):
        super().__init__()
        
        self.roi_dilation = roi_dilation
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # Detection model (3-channel for YOLO compatibility)
        self.detector = YOLOv5(
            num_classes=num_classes_det,
            img_size=img_size,
            model_size=detector_size,
            in_channels=3  # YOLOv5 expects RGB, we'll repeat grayscale
        )
        
        # Segmentation model (1-channel for grayscale ROIs)
        self.segmentor = SegFormer(
            num_classes=num_classes_seg,
            backbone_size=segmentor_size,
            in_channels=in_channels
        )

    def detect(self, x: torch.Tensor) -> List[DetectionResult]:
        """
        Run detection on input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of DetectionResult for each image in batch
        """
        # If grayscale, repeat to 3 channels for YOLO
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        self.detector.eval()
        with torch.no_grad():
            pred, _ = self.detector(x)
        
        # Apply NMS
        results = []
        for det in pred:
            # det: (num_predictions, 6) -> [x1, y1, x2, y2, conf, cls]
            if det.shape[0] == 0:
                results.append(DetectionResult(
                    boxes=torch.empty(0, 4, device=x.device),
                    scores=torch.empty(0, device=x.device),
                    classes=torch.empty(0, device=x.device)
                ))
                continue
            
            # Filter by confidence
            det = det[det[:, 4] >= self.conf_threshold]
            
            if det.shape[0] == 0:
                results.append(DetectionResult(
                    boxes=torch.empty(0, 4, device=x.device),
                    scores=torch.empty(0, device=x.device),
                    classes=torch.empty(0, device=x.device)
                ))
                continue
            
            # NMS
            boxes = det[:, :4]
            scores = det[:, 4]
            classes = det[:, 5]
            
            keep = self._nms(boxes, scores, self.iou_threshold)
            
            results.append(DetectionResult(
                boxes=boxes[keep],
                scores=scores[keep],
                classes=classes[keep]
            ))
        
        return results

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Non-Maximum Suppression."""
        return torch.ops.torchvision.nms(boxes, scores, iou_threshold)

    def _extract_rois(
        self, 
        image: torch.Tensor, 
        boxes: torch.Tensor, 
        dilation: float
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]]]:
        """
        Extract dilated ROIs from image based on detection boxes.
        
        Args:
            image: Single image tensor (C, H, W)
            boxes: Detection boxes (N, 4) in xyxy format
            dilation: Dilation factor for ROIs
            
        Returns:
            Tuple of (list of ROI tensors, list of original box coordinates)
        """
        _, H, W = image.shape
        rois = []
        coords = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            
            # Compute dilated box
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            new_w, new_h = w * dilation, h * dilation
            
            x1_d = max(0, int(cx - new_w / 2))
            y1_d = max(0, int(cy - new_h / 2))
            x2_d = min(W, int(cx + new_w / 2))
            y2_d = min(H, int(cy + new_h / 2))
            
            roi = image[:, y1_d:y2_d, x1_d:x2_d]
            rois.append(roi)
            coords.append((x1_d, y1_d, x2_d, y2_d))
        
        return rois, coords

    def segment_rois(
        self, 
        rois: List[torch.Tensor], 
        target_size: Optional[Tuple[int, int]] = None
    ) -> List[torch.Tensor]:
        """
        Segment a list of ROIs.
        
        Args:
            rois: List of ROI tensors
            target_size: Optional target size for resizing ROIs
            
        Returns:
            List of segmentation masks
        """
        if len(rois) == 0:
            return []
        
        masks = []
        self.segmentor.eval()
        
        with torch.no_grad():
            for roi in rois:
                if roi.numel() == 0:
                    masks.append(torch.zeros(1, dtype=torch.long, device=roi.device))
                    continue
                
                # Add batch dimension
                roi_input = roi.unsqueeze(0)
                
                # Resize if needed
                orig_size = roi_input.shape[2:]
                if target_size and orig_size != target_size:
                    roi_input = F.interpolate(roi_input, size=target_size, mode='bilinear', align_corners=False)
                
                # Run segmentation
                logits = self.segmentor(roi_input)
                
                # Resize back if needed
                if target_size and orig_size != target_size:
                    logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)
                
                # Get prediction
                mask = logits.argmax(dim=1).squeeze(0)
                masks.append(mask)
        
        return masks

    def _merge_masks(
        self, 
        masks: List[torch.Tensor], 
        coords: List[Tuple[int, int, int, int]], 
        output_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Merge ROI masks into full-size output mask.
        
        Args:
            masks: List of ROI segmentation masks
            coords: List of ROI coordinates (x1, y1, x2, y2)
            output_size: Output mask size (H, W)
            
        Returns:
            Merged segmentation mask
        """
        H, W = output_size
        device = masks[0].device if masks else torch.device('cpu')
        output = torch.zeros(H, W, dtype=torch.long, device=device)
        
        for mask, (x1, y1, x2, y2) in zip(masks, coords):
            h, w = y2 - y1, x2 - x1
            if mask.shape != (h, w):
                mask = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0), 
                    size=(h, w), 
                    mode='nearest'
                ).squeeze().long()
            
            # Merge using OR logic for overlapping regions
            output[y1:y2, x1:x2] = torch.maximum(output[y1:y2, x1:x2], mask)
        
        return output

    def forward(
        self, 
        x: torch.Tensor, 
        return_intermediate: bool = False
    ) -> DTSegNetOutput:
        """
        End-to-end forward pass.
        
        Args:
            x: Input images (B, C, H, W)
            return_intermediate: Whether to return intermediate results
            
        Returns:
            DTSegNetOutput containing segmentation masks and optionally intermediate results
        """
        B, C, H, W = x.shape
        
        # Step 1: Detection
        detections = self.detect(x)
        
        # Step 2: Extract ROIs and segment for each image
        all_masks = []
        all_roi_masks = [] if return_intermediate else None
        
        for i in range(B):
            image = x[i]
            det = detections[i]
            
            if det.boxes.shape[0] == 0:
                # No detections, return empty mask
                all_masks.append(torch.zeros(H, W, dtype=torch.long, device=x.device))
                if return_intermediate:
                    all_roi_masks.append([])
                continue
            
            # Extract ROIs
            rois, coords = self._extract_rois(image, det.boxes, self.roi_dilation)
            
            # Segment ROIs
            roi_masks = self.segment_rois(rois)
            
            if return_intermediate:
                all_roi_masks.append(roi_masks)
            
            # Merge masks
            merged_mask = self._merge_masks(roi_masks, coords, (H, W))
            all_masks.append(merged_mask)
        
        # Stack masks
        segmentation_mask = torch.stack(all_masks, dim=0)
        
        return DTSegNetOutput(
            segmentation_mask=segmentation_mask,
            detections=detections if return_intermediate else None,
            roi_masks=all_roi_masks
        )

    def train_detector(self):
        """Set detector to training mode."""
        self.detector.train()
        self.segmentor.eval()
    
    def train_segmentor(self):
        """Set segmentor to training mode."""
        self.detector.eval()
        self.segmentor.train()


# =============================================================================
# Model Factory Functions
# =============================================================================

def create_detector(
    num_classes: int = 1,
    model_size: str = 'l',
    img_size: int = 1280,
    pretrained: bool = False
) -> YOLOv5:
    """Create a YOLOv5 detector."""
    model = YOLOv5(
        num_classes=num_classes,
        img_size=img_size,
        model_size=model_size
    )
    return model


def create_segmentor(
    num_classes: int = 2,
    backbone_size: str = 'b1',
    in_channels: int = 1,
    pretrained: bool = False
) -> SegFormer:
    """Create a SegFormer segmentor."""
    model = SegFormer(
        num_classes=num_classes,
        backbone_size=backbone_size,
        in_channels=in_channels
    )
    return model


def create_dtsegnet(
    detector_size: str = 'l',
    segmentor_size: str = 'b1',
    pretrained: bool = False,
    **kwargs
) -> DTSegNet:
    """Create an end-to-end DT-SegNet model."""
    model = DTSegNet(
        detector_size=detector_size,
        segmentor_size=segmentor_size,
        **kwargs
    )
    return model


if __name__ == '__main__':
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test detector
    print("Testing YOLOv5 detector...")
    detector = create_detector(model_size='s').to(device)
    x = torch.randn(1, 3, 640, 640).to(device)
    out = detector(x)
    print(f"Detector output shape: {out[0].shape}")
    
    # Test segmentor
    print("\nTesting SegFormer segmentor...")
    segmentor = create_segmentor(backbone_size='b0', in_channels=1).to(device)
    x = torch.randn(1, 1, 256, 256).to(device)
    out = segmentor(x)
    print(f"Segmentor output shape: {out.shape}")
    
    # Test end-to-end model
    print("\nTesting DTSegNet...")
    model = create_dtsegnet(detector_size='s', segmentor_size='b0').to(device)
    x = torch.randn(1, 1, 640, 640).to(device)
    model.eval()
    out = model(x, return_intermediate=True)
    print(f"DTSegNet output mask shape: {out.segmentation_mask.shape}")
    
    print("\nAll tests passed!")
