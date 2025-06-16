# Adopted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .base_models import *


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        b, n = qkv[0].shape[:2]
        q = qkv[0].view(b, n, self.heads, -1).permute(0, 2, 1, 3)
        k = qkv[1].view(b, n, self.heads, -1).permute(0, 2, 1, 3)
        v = qkv[2].view(b, n, self.heads, -1).permute(0, 2, 1, 3)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for layer in self.layers:
            attn = layer[0]
            ff = layer[1]
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(BaseIconModel):
    """Vision transformers for ICON columns"""

    def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
        super().__init__(x3d_mean, x3d_std, x2d_mean, x2d_std)
        height_out = args.height_out
        channel_2d = args.channel_2d
        channel_3d = args.channel_3d
        channel_out = args.channel_out

        dim = args.vit_dim
        depth = args.vit_depth
        heads = args.vit_heads
        emb_dropout = args.vit_emb_dropout
        self.scale_output = args.scale_output
        dim_head = args.vit_head_dim
        mlp_dim = args.vit_mlp_dim
        dropout = args.vit_dropout
        self.smoothing_kernel = args.smoothing_kernel
        self.beta = args.beta
        self.beta_height = args.beta_height
        self.beta_height_sw = args.beta_height_sw
        self.beta_height_lw = args.beta_height_lw

        self.to_patch_embedding2 = nn.Sequential(
            nn.Linear(channel_2d, dim), nn.LayerNorm(dim)
        )
        self.to_patch_embedding3 = nn.Sequential(
            nn.Linear(channel_3d, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, height_out, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Linear(dim, channel_out)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x3d, x2d):
        x3d = self.normalizer3d(x3d)
        x2d_org = x2d.clone()
        x2d = self.normalizer2d(x2d)

        x3d = self.to_patch_embedding3(x3d)
        x2d = self.to_patch_embedding2(x2d)

        x = torch.cat([x2d[:, None, :], x3d], dim=-2)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.mlp_head(x)
        if self.scale_output:
            x = self.sigmoid(x)
            x = self._scale_output(x, x2d_org).squeeze()
        else:
            x = self.relu(x)

        if self.smoothing_kernel is not None:
            x = self._smooth(x, self.smoothing_kernel)
        if self.beta is not None:
            x = self.exponential_decay(x)
        return x
