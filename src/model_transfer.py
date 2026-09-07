"""Transfer-learning model for richness regression.

`GaiaTransferModel` wraps the pretrained ViTSpatialSpectral encoder with a small
MLP regression head. Extracted from the (now legacy) BioSCape `train_production.py`
so the EAGLE training/inference scripts don't drag in the S3 / NetCDF stack.
"""
import os
import sys

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vit_spatial_spectral import ViTSpatialSpectral


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class GaiaTransferModel(nn.Module):
    def __init__(self, num_targets=1, patch_size=16):
        super().__init__()
        # Backbone (Frozen)
        self.encoder = ViTSpatialSpectral(
            image_size=patch_size, spatial_patch_size=1, spectral_patch_size=10,
            num_classes=num_targets, dim=96, depth=4, heads=8, mlp_dim=64,
            dropout=0.1, emb_dropout=0.1, channels=200, spectral_pos=torch.arange(20),
            spectral_pos_embed=True, blockwise_patch_embed=True, spectral_only=False, pixelwise=False
        )
        # Small MLP head (96 -> 64 -> 1) with GELU
        self.encoder.mlp_head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets)
        )

    def load_foundation_weights(self, checkpoint_path, device):
        print(f"{Colors.OKBLUE}[*] Loading foundation weights from {checkpoint_path}{Colors.ENDC}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'mlp_head' in k or 'mask_token' in k: continue
            new_key = k.replace("encoder.", "")
            if new_key == "pos_embed":
                ckpt_grid = int(v.shape[1]**0.5)
                target_grid = int(self.encoder.pos_embed.shape[1]**0.5)
                if ckpt_grid != target_grid:
                    v = v.permute(0, 2, 1).reshape(1, v.shape[2], ckpt_grid, ckpt_grid)
                    v = torch.nn.functional.interpolate(v, size=(target_grid, target_grid), mode='bicubic', align_corners=False)
                    v = v.reshape(1, v.shape[1], target_grid * target_grid).permute(0, 2, 1)
            new_state_dict[new_key] = v
        self.encoder.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        # 1. Feature extraction: returns (B, 16, 16, num_targets)
        out = self.encoder(x)
        # 2. Spatial Average Pooling (averaging predicted richness across 16x16 patch)
        return out.mean(dim=(1, 2))
