import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..engine.ema import requires_grad
from ..engine.util import instantiate_from_config
from .transformer import Transformer


class ShallowSemanticAggregator(nn.Module):
    """Cross-attention fusion: deep semantic tokens query shallow visual tokens."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, semantic: torch.Tensor, shallow: Optional[torch.Tensor]) -> torch.Tensor:
        if shallow is None:
            return semantic
        query = self.query_norm(semantic)
        key_value = self.kv_norm(shallow)
        attn_out, _ = self.attn(query, key_value, key_value, need_weights=False)
        semantic = semantic + attn_out
        return semantic + self.ffn(self.ffn_norm(semantic))


class DirectFeatureEncoder(nn.Module):
    """Frozen VFM encoder plus IDEAL shallow-to-deep fusion before quantization."""

    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 16,
        dim: int = 1024,
        encoder_variant: str = "direct",
        visual_encoder_config=None,
        fusion_mode: str = "attn_fusion_spatial",
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.backbone = instantiate_from_config(visual_encoder_config)
        requires_grad(self.backbone, False)
        self.backbone.eval()

        self.fusion_mode = (fusion_mode or "attn_fusion_spatial").lower()
        valid_modes = {"semantic", "attn_fusion", "attn_fusion_spatial"}
        if self.fusion_mode not in valid_modes:
            raise ValueError(f"Unsupported IDEAL fusion mode '{fusion_mode}'.")
        self.attn_fuse = ShallowSemanticAggregator(dim, num_heads=8, mlp_ratio=2.0, dropout=0.1)

    def freeze_visual_encoder(self):
        requires_grad(self.backbone, False)

    def freeze(self):
        self.eval()
        requires_grad(self, False)

    def forward(self, imgs):
        with torch.no_grad():
            shallow, deep = self.backbone(imgs)
        slots = deep
        if slots.dim() != 3:
            slots = rearrange(slots, "b c h w -> b (h w) c")
        if self.fusion_mode != "semantic":
            slots = self.attn_fuse(slots, shallow)
        return slots, shallow, deep

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "pool_tokens"):
            return self.backbone.pool_tokens(tokens)
        raise RuntimeError("Backbone does not expose pooling capability.")


class DirectSemanticDecoder(nn.Module):
    """Feature decoder that reconstructs both deep semantic and shallow visual VFM tokens."""

    def __init__(
        self,
        layer_type,
        image_size,
        patch_size,
        dim,
        n_carrier,
        depth,
        num_head,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        num_register_tokens=4,
        semantic_dim=1024,
        fusion_mode: str = "attn_fusion_spatial",
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.dim = dim
        self.num_patches = (image_size // patch_size) ** 2
        scale = dim ** -0.5

        self.slot_position_embedding = nn.Parameter(torch.randn(1, n_carrier, dim) * scale)
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim) * scale)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * scale)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.num_register_tokens = num_register_tokens

        self.transformer = Transformer(layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout, xformer=False)
        self.norm_post = nn.LayerNorm(dim)
        self.project_semantic = nn.Linear(dim, semantic_dim)
        self.project_shallow = nn.Linear(dim, semantic_dim)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, slots):
        bs, carrier_tokens, _ = slots.shape
        memory = slots + self.slot_position_embedding[:, :carrier_tokens]
        cls_token = repeat(self.cls_token + self.cls_pos_embedding, "f ... -> (b f) ...", b=bs)
        register_tokens = repeat(self.register_tokens, "f ... -> (b f) ...", b=bs)
        seq = torch.cat((cls_token, memory, register_tokens), dim=1)
        seq = self.transformer(self.norm_post(seq))

        latent = seq[:, 1:1 + carrier_tokens]
        semantic_feat = self.project_semantic(latent)
        shallow_feat = self.project_shallow(latent)

        token_count = semantic_feat.size(1)
        h_src = int(math.sqrt(token_count))
        h_tgt = int(math.sqrt(self.num_patches))
        semantic_todec = semantic_feat.view(bs, h_src, h_src, -1).permute(0, 3, 1, 2).contiguous()
        if h_src != h_tgt:
            semantic_todec = F.interpolate(semantic_todec, size=(h_tgt, h_tgt), mode="bicubic", align_corners=False)
        return semantic_todec, semantic_feat, shallow_feat
