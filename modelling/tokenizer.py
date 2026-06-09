from dataclasses import dataclass, field, fields
from typing import List, Optional

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from ideal.tokenizer.vq_model import ModelArgs as IDEALInternalArgs
from ideal.tokenizer.vq_model import VQModel as IDEALInternalModel


@dataclass
class ModelArgs:
    image_size: int = 384
    codebook_size: int = 16384
    codebook_embed_dim: int = 64
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.01
    vq_loss_ratio: float = 1.0
    num_codebooks: int = 1
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 1024
    dropout_p: float = 0.0
    dec_patch_size: int = 16
    transformer_config: Optional[str] = "configs/tokenizer/ideal-transformer.yaml"
    codebook_slots_embed_dim: int = 64
    decoder_up_type: str = "CNN"
    semantic_cos_weight: float = 1.0
    semantic_l2_weight: float = 1.0
    shallow_cos_weight: float = 1.0
    shallow_l2_weight: float = 1.0


def _model_args_from_kwargs(kwargs):
    valid = {f.name for f in fields(ModelArgs)}
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return ModelArgs(**filtered)


class IDEALModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        transformer_cfg = config.transformer_config
        if transformer_cfg is None:
            raise ValueError("IDEAL requires 'transformer_config' in ModelArgs.")
        if isinstance(transformer_cfg, str):
            if not os.path.isfile(transformer_cfg):
                raise FileNotFoundError(f"Transformer config not found at '{transformer_cfg}'.")
            transformer_cfg = OmegaConf.load(transformer_cfg)

        internal_args = IDEALInternalArgs(
            codebook_size=config.codebook_size,
            codebook_embed_dim=config.codebook_embed_dim,
            codebook_l2_norm=config.codebook_l2_norm,
            codebook_show_usage=config.codebook_show_usage,
            commit_loss_beta=config.commit_loss_beta,
            entropy_loss_ratio=config.entropy_loss_ratio,
            encoder_ch_mult=config.encoder_ch_mult,
            decoder_ch_mult=config.decoder_ch_mult,
            z_channels=config.z_channels,
            dropout_p=config.dropout_p,
            transformer_config=transformer_cfg,
            codebook_slots_embed_dim=config.codebook_slots_embed_dim,
            image_size=config.image_size,
            patch_size=config.dec_patch_size,
            in_channels=3,
            decoder_up_type=config.decoder_up_type,
        )

        self.model = IDEALInternalModel(internal_args)
        self.decoder = self.model.decoder
        self.quantize = self.model.slot_quantize

    def encode(self, x: torch.Tensor):
        (quant_slots, semantic_feat, spatial_feat), emb_loss, q_indices = self.model.encode(x)
        info = (
            semantic_feat,
            None,
            q_indices.view(x.size(0), 1, -1),
            spatial_feat,
            None,
        )
        return quant_slots, emb_loss, info

    def decode(self, quant: torch.Tensor, *_, **__):
        dec, _, _ = self.model.decode(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        if shape is None:
            flat = code_b.reshape(code_b.size(0), -1)
            shape = (flat.size(0), flat.size(1), self.config.codebook_slots_embed_dim)
            code_b = flat
            channel_first = False
        dec, _, _ = self.model.decode_code(code_b, shape, channel_first)
        return dec

    @torch.no_grad()
    def decode_from_ids(self, indices: torch.Tensor):
        if indices.dim() == 3 and indices.size(1) == 1:
            indices = indices.squeeze(1)
        flat = indices.reshape(indices.size(0), -1)
        qz_shape = (flat.size(0), flat.size(1), self.config.codebook_slots_embed_dim)
        dec, _, _ = self.model.decode_from_indices(flat.contiguous().view(-1), qz_shape, channel_first=False)
        return dec

    @torch.no_grad()
    def decode_codes_to_img(self, codes: torch.Tensor, target_size: int):
        images = self.decode_from_ids(codes)
        if images.shape[-1] != target_size:
            images = F.interpolate(images, size=(target_size, target_size), mode="bicubic", align_corners=False)
        images = images.detach() * 127.5 + 128
        return torch.clamp(images, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

    @torch.no_grad()
    def encode_to_ids(self, images: torch.Tensor, as_list: bool = False):
        was_training = self.training
        if was_training:
            self.eval()
        _, _, info = self.encode(images)
        indices = info[2].squeeze(1).contiguous()
        if was_training:
            self.train()
        return indices.tolist() if as_list else indices

    def forward(self, input: torch.Tensor):
        (dec, semantic_feat, spatial_feat, semantic_recon, spatial_recon), diff, q_indices = self.model(input)
        if self.training and semantic_feat is not None and semantic_recon is not None:
            sem_cos = 1 - F.cosine_similarity(semantic_recon, semantic_feat, dim=-1)
            sem_cos = sem_cos.mean() * self.config.semantic_cos_weight
            sem_l2 = F.mse_loss(semantic_recon, semantic_feat) * self.config.semantic_l2_weight
            diff += (sem_cos, sem_l2)
        if self.training and spatial_feat is not None and spatial_recon is not None:
            shallow_cos = 1 - F.cosine_similarity(spatial_recon, spatial_feat, dim=-1)
            shallow_cos = shallow_cos.mean() * self.config.shallow_cos_weight
            shallow_l2 = F.mse_loss(spatial_recon, spatial_feat) * self.config.shallow_l2_weight
            diff += (shallow_cos, shallow_l2)
        info = (
            semantic_feat,
            semantic_recon,
            q_indices.view(input.size(0), 1, -1),
            spatial_feat,
            spatial_recon,
        )
        return dec, diff, info


def IDEAL(**kwargs):
    return IDEALModel(_model_args_from_kwargs(kwargs))


VQ_models = {
    "IDEAL": IDEAL,
}
