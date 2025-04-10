import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from base_model import BaseTensorCNN

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, num_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # B, num_heads, N, N
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTTensorToTensor(BaseTensorCNN):
    def __init__(self, input_channels, output_channels, image_size=224, patch_size=16, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        super(ViTTensorToTensor, self).__init__(input_channels, output_channels)
        
    def _create_model(self):
        # Calculate number of patches
        assert self.image_size % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        num_patches = (self.image_size // self.patch_size) ** 2
        
        # Create model components
        self.patch_embed = PatchEmbedding(self.input_channels, self.embed_dim, self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(self.dropout)
        
        # Initialize positional embedding with sine-cosine positional encoding
        self._init_pos_embed()
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout
            )
            for _ in range(self.depth)
        ])
        
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # Decoder to convert sequence back to 2D feature map
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.output_channels, kernel_size=1)
        )
        
        # Instead of returning self, create a dummy nn.Module that represents the "complete" model
        # The actual forward pass will be handled by the forward method of this class
        return nn.Identity()
    
    def _init_pos_embed(self):
        # Initialize positional embeddings with sinusoidal position encoding
        # This is called after pos_embed is defined in _create_model
        num_patches = (self.image_size // self.patch_size) ** 2
        num_positions = num_patches + 1  # Add 1 for cls token
        
        position = torch.arange(0, num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        pos_embed = torch.zeros(num_positions, self.embed_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))
    
    def get_model_name(self):
        return f"ViT-B{self.depth}-TensorToTensor"
    
    def forward(self, x):
        # Get input shape for later reconstruction
        B, C, H, W = x.shape
        
        # Check if the input size matches what the model was initialized with
        # If not, we need to regenerate the positional embeddings
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches = h_patches * w_patches
        expected_patches = (self.image_size // self.patch_size) ** 2
        
        if num_patches != expected_patches:
            # Update image_size to match the actual input dimensions
            self.image_size = max(H, W)
            # Regenerate positional embeddings for the new size
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
            self._init_pos_embed()
        
        # Create patch embeddings
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add cls token
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_token, x], dim=1)  # (B, 1 + num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Remove cls token
        x = x[:, 1:]  # (B, num_patches, embed_dim)
        
        # Reshape sequence back to image-like structure
        x = rearrange(x, 'b (h w) c -> b c h w', h=h_patches, w=w_patches)
        
        # Upsample to original resolution using bilinear upsampling
        x = nn.Upsample(scale_factor=self.patch_size, mode='bilinear', align_corners=False)(x)
        
        # Apply decoder
        x = self.decoder(x)
        
        return x