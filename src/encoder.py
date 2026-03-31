import torch
import torch.nn as nn

class SpectralSpatialEmbedder(nn.Module):
    """
    Tokenizes a hyperspectral image block by applying a 3D convolution to extract 
    non-overlapping spectral-spatial patches.
    """
    def __init__(self, in_channels=1, embed_dim=256, patch_size=(10, 8, 8), img_size=(200, 32, 32)):
        super().__init__()
        self.patch_size = patch_size # (pc, ph, pw)
        self.img_size = img_size     # (C, H, W)
        self.embed_dim = embed_dim
        
        # Calculate number of patches in each dimension
        self.num_patches_c = img_size[0] // patch_size[0]
        self.num_patches_h = img_size[1] // patch_size[1]
        self.num_patches_w = img_size[2] // patch_size[2]
        self.num_patches = self.num_patches_c * self.num_patches_h * self.num_patches_w
        
        # 3D convolution for patch embedding
        self.proj = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional embeddings
        # We can use a 1D sequence embedding. Or separable 3D pos embeddings.
        # For simplicity and standard ViT compatibility, we use a single 1D learned embedding for the sequence.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x is (B, C, H, W). We need to add the channel dimension for Conv3d.
        # So it becomes (B, 1, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
            
        # Extract patches
        # output is (B, embed_dim, num_patches_c, num_patches_h, num_patches_w)
        x = self.proj(x)
        
        # Flatten to (B, embed_dim, Num_Patches)
        x = x.flatten(2)
        
        # Swap axes to standard sequence format: (B, Num_Patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add positional embedding
        x = x + self.pos_embed
        return x

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class ViTBackbone(nn.Module):
    """
    Standard Vision Transformer encoder running on spectral-spatial tokens.
    """
    def __init__(self, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        # Note: CLS token is handled differently in MAE setups (often omitted during pretraining, 
        # or appended later). For Gaia, we use a CLS token or global mean pooling. We'll use a CLS token.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x is (B, N, D) sequence of embeddings
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Concatenate CLS token at the beginning
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, D)
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x

if __name__ == '__main__':
    # Define a dummy batch of 2 elements from Indian Pines: size (B, 200, 32, 32)
    dummy_input = torch.rand(2, 200, 32, 32)
    
    # 20 spectral per token -> 10 groups
    # 8x8 spatial per token -> 4x4 patches
    # Total tokens = 10 * 4 * 4 = 160 tokens
    embedder = SpectralSpatialEmbedder(patch_size=(20, 8, 8), img_size=(200, 32, 32), embed_dim=256)
    
    tokens = embedder(dummy_input)
    print(f"Tokens output shape: {tokens.shape}") # Should be (2, 160, 256)
    
    backbone = ViTBackbone(embed_dim=256, depth=4, num_heads=8)
    encoded = backbone(tokens)
    print(f"Encoded output shape: {encoded.shape}") # Should be (2, 161, 256) (160 + CLS)
