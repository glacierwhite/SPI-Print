import torch
import torch.nn as nn

class FingerViT(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch=8,
        in_ch=1,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=256,
        emb_dim=64,
        dropout=0.15
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch, in_ch, dim)
        n_patches = self.patch_embed.n_patches

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, emb_dim)

    def forward(self, x):
        x = self.patch_embed(x)          # [B, N, dim]

        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token = x[:, 0]
        emb = self.proj(cls_token)
        return nn.functional.normalize(emb)
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch=8, in_ch=1, dim=128):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2

        self.proj = nn.Conv2d(
            in_ch, dim,
            kernel_size=patch,
            stride=patch
        )

    def forward(self, x):
        x = self.proj(x)                 # [B, dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, dim]
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x