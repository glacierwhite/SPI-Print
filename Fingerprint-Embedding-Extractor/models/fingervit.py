import torch
import torch.nn as nn
from einops import rearrange

class FingerViT(nn.Module):
    def __init__(self, img_size=64, patch=8, dim=256, depth=6, heads=8, mlp_dim=512, emb_dim=512):
        super().__init__()
        self.patch = patch
        n_patches = (img_size // patch) ** 2

        self.to_patch = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)

        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, dim))

        encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=depth)

        self.proj = nn.Linear(dim, emb_dim)

    def forward(self, x):
        x = self.to_patch(x)                # [B, dim, 8, 8]
        x = rearrange(x, 'b c h w -> b (h w) c')

        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos

        x = self.transformer(x)
        cls_token = x[:, 0]
        emb = self.proj(cls_token)
        return nn.functional.normalize(emb)