import torch
import torch.nn as nn
from .encoder import SpectralSpatialEmbedder, ViTBackbone

class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder (MAE) / SimMIM styled pre-training model.
    Masks a portion of the 3D tokens and reconstructs the raw pixel values.
    """
    def __init__(
        self, 
        in_channels=1, 
        embed_dim=256, 
        patch_size=(20, 8, 8), 
        img_size=(200, 32, 32),
        encoder_depth=6,
        encoder_heads=8,
        mask_ratio=0.75
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.embedder = SpectralSpatialEmbedder(in_channels, embed_dim, patch_size, img_size)
        self.backbone = ViTBackbone(embed_dim, encoder_depth, encoder_heads)
        
        # The number of pixels in each 3D patch to reconstruct
        self.pixels_per_patch = patch_size[0] * patch_size[1] * patch_size[2]
        
        # SimMIM-style mask token (learnable vector replacing masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Decoder: 2-layer MLP to reconstruct raw pixels (as recommended for SimMIM)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.pixels_per_patch)
        )

    def forward(self, x, mask=None):
        """
        x: (B, C, H, W) hyperspectral input
        mask: Optional binary mask of shape (B, N) where 1 indicates masked tokens
        """
        if x.dim() == 4:
            x_unsqueezed = x.unsqueeze(1) # (B, 1, C, H, W)
        else:
            x_unsqueezed = x
            x = x.squeeze(1) # Base (B, C, H, W) for target generation
            
        b = x.shape[0]
        
        # 1. Embed all patches
        tokens = self.embedder(x_unsqueezed) # (B, N, D)
        n = tokens.shape[1]
        
        # 2. Generate random mask if not provided
        if mask is None:
            # Generate random noise and take top-k to mask
            noise = torch.rand(b, n, device=x.device)
            # Find the threshold for masking
            k = int(n * self.mask_ratio)
            mask = torch.zeros(b, n, device=x.device)
            # Get indices of top k noise values
            _, indices = torch.topk(noise, k, dim=-1)
            # Scatter 1s into mask
            mask.scatter_(1, indices, 1.0)
            
        # 3. Apply mask token to masked positions
        # mask is (B, N)
        mask_expanded = mask.unsqueeze(-1).type_as(tokens) # (B, N, 1)
        
        # Replace masked tokens with self.mask_token
        tokens = tokens * (1 - mask_expanded) + self.mask_token * mask_expanded
        
        # 4. Pass through ViT backbone
        encoded_tokens = self.backbone(tokens) # (B, N+1, D)
        
        # 5. Decode (skip the CLS token at index 0)
        decoded_patches = self.decoder(encoded_tokens[:, 1:, :]) # (B, N, pixels_per_patch)
        
        return decoded_patches, mask

    def get_reconstruction_loss(self, x, decoded_patches, mask):
        """
        Compute MSE loss only on the masked patches.
        x: original image (B, C, H, W)
        """
        b = x.shape[0]
        # Unfold original image into patches to compare with decoded_patches
        # We need to extract the target patches in the same order as the tokens.
        # Patch size info
        pc, ph, pw = self.patch_size
        
        # Unfold:
        # First reshape C, H, W into blocks
        nc = x.shape[1] // pc
        nh = x.shape[2] // ph
        nw = x.shape[3] // pw
        
        # (B, nc, pc, nh, ph, nw, pw)
        target = x.view(b, nc, pc, nh, ph, nw, pw)
        # Permute to group patches together: (B, nc, nh, nw, pc, ph, pw)
        target = target.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        # Flatten patches: (B, N, pc*ph*pw) where N = nc*nh*nw
        target = target.view(b, -1, self.pixels_per_patch)
        
        # MSE Loss
        loss = (decoded_patches - target) ** 2
        loss = loss.mean(dim=-1) # (B, N)
        
        # Average loss only over masked patches
        # mask is (B, N) with 1 for masked, 0 for visible
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

class RichnessRegressor(nn.Module):
    """
    Downstream regressor built on top of the Masked Spectral-Spatial Encoder.
    Outputs species richness counts (e.g., Birds, Frogs, Insects).
    """
    def __init__(
        self, 
        in_channels=1, 
        embed_dim=256, 
        patch_size=(20, 8, 8), 
        img_size=(200, 32, 32),
        encoder_depth=6,
        encoder_heads=8,
        num_targets=3,
        dropout=0.1
    ):
        super().__init__()
        
        # Load encoder
        self.embedder = SpectralSpatialEmbedder(in_channels, embed_dim, patch_size, img_size)
        self.backbone = ViTBackbone(embed_dim, encoder_depth, encoder_heads)
        
        # Regression head (MLP taking the CLS token)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_targets)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, num_targets) estimated richness
        """
        if x.dim() == 4:
            x = x.unsqueeze(1) # (B, 1, C, H, W)
            
        # 1. Embed all patches
        tokens = self.embedder(x) # (B, N, D)
        
        # 2. Pass through ViT backbone
        encoded_tokens = self.backbone(tokens) # (B, N+1, D)
        
        # 3. Extract CLS token (index 0)
        cls_token = encoded_tokens[:, 0, :]
        
        # 4. Regress richness
        preds = self.head(cls_token)
        
        return preds

if __name__ == '__main__':
    dummy_input = torch.rand(2, 200, 32, 32)
    mae = MaskedAutoencoder(
        embed_dim=256, 
        patch_size=(20, 8, 8), 
        img_size=(200, 32, 32),
        mask_ratio=0.75
    )
    
    decoded, mask = mae(dummy_input)
    print(f"Decoded shape: {decoded.shape}")     # (2, 160, 20*8*8=1280)
    print(f"Mask shape: {mask.shape}")           # (2, 160)
    
    loss = mae.get_reconstruction_loss(dummy_input, decoded, mask)
    print(f"Reconstruction Loss: {loss.item():.4f}")
    
    # Test Regressor
    regressor = RichnessRegressor(
        embed_dim=256, 
        patch_size=(20, 8, 8), 
        img_size=(200, 32, 32),
        num_targets=3
    )
    richness_preds = regressor(dummy_input)
    print(f"Richness Predictions shape: {richness_preds.shape}") # (2, 3)
