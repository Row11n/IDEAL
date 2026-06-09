from copy import deepcopy

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Layer(nn.Module):
    def __init__(self, dim, dim_head, mlp_dim, num_head=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_head, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop1(attn_out)
        return x + self.ffn(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout=0.0, xformer=False):
        super().__init__()
        if layer_type != "normal":
            raise ValueError("IDEAL only supports the normal transformer layer.")
        layer = Layer(dim, dim_head, mlp_dim, num_head, dropout)
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(depth)])

    def __len__(self):
        return len(self.layers)

    def forward(self, x, slots=None, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
