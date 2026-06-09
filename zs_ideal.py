"""Evaluate zero-shot ImageNet classification accuracy for IDEAL tokens."""
import os
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from open_clip import (
    create_model_from_pretrained,
    get_tokenizer,
    build_zero_shot_classifier,
    IMAGENET_CLASSNAMES,
    OPENAI_IMAGENET_TEMPLATES,
)

from omegaconf import OmegaConf
from modelling.tokenizer import VQ_models
from ideal.modules.encoders.pretrained_enc_arch_timm import TimmEmbedder
from utils.misc import load_model_state_dict


# =========================
# Config
# =========================
IMAGENET_VAL_DIR = Path("ImageNet/val")
IMAGENET_CLASS_INDEX_JSON = "imagenet_class_index.json"

HF_HOME = "weights/local_hf_cache"
HF_TEXT_REPO = "timm/ViT-L-16-SigLIP2-384"

# IDEAL / VQ
VQ_CKPT = Path("weights/ideal-tokenizer.pth")
TOKENIZER_CONFIG_FILE = Path("configs/tokenizer/ideal-tokenizer.yaml")
# SigLIP2 pool helper (timm image tower head/pooling)
TIMM_MODEL_NAME = "vit_large_patch16_siglip_384.v2_webli"
TIMM_CHECKPOINT_PATH = Path("weights/vit_large_patch16_siglip_384.v2_webli/model.safetensors")

BATCH_SIZE = 64
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# logits mode (跟你成功脚本保持一致)
LOGITS_MODE = "siglip"  # "siglip" or "fixed_100"


# =========================
# Helpers
# =========================
def build_wnid_to_standard_index(class_index_json_path: str) -> Dict[str, int]:
    with open(class_index_json_path, "r", encoding="utf-8") as f:
        class_index = json.load(f)
    return {v[0]: int(k) for k, v in class_index.items()}


def load_text_side_from_hf_cache(device: torch.device):
    """
    使用 hf-hub 的官方加载，但强制离线命中本地 cache。
    这条路径能保证 preprocess/tokenizer/model_cfg 与线上一致。
    """
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    text_model, preprocess = create_model_from_pretrained(f"hf-hub:{HF_TEXT_REPO}")
    tokenizer = get_tokenizer(f"hf-hub:{HF_TEXT_REPO}")
    text_model = text_model.to(device).eval()
    return text_model, tokenizer, preprocess


def load_vq_model(device: torch.device):
    tokenizer_config = OmegaConf.load(TOKENIZER_CONFIG_FILE)

    model = VQ_models["IDEAL"](
        codebook_size=tokenizer_config.codebook_size,
        z_channels=tokenizer_config.z_channels,
        codebook_embed_dim=tokenizer_config.codebook_embed_dim,
        codebook_l2_norm=tokenizer_config.codebook_l2_norm,
        commit_loss_beta=tokenizer_config.commit_loss_beta,
        entropy_loss_ratio=tokenizer_config.entropy_loss_ratio,
        codebook_slots_embed_dim=tokenizer_config.codebook_slots_embed_dim,
        transformer_config=tokenizer_config.transformer_config,
    )

    print(f"Loading VQ checkpoint from: {VQ_CKPT}")
    checkpoint = torch.load(VQ_CKPT, map_location="cpu")

    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
        print("Using 'ema' weights from checkpoint.")
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
        print("Using 'model' weights from checkpoint.")
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
        print("Using 'state_dict' weights from checkpoint.")
    else:
        raise RuntimeError("Unknown checkpoint format (no ema/model/state_dict).")

    state = load_model_state_dict(model_weight)
    for key in list(state):
        if "codebook_used" in key:
            state.pop(key)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[VQ] missing keys: {len(missing)} (show first 10)")
        print(missing[:10])
    if unexpected:
        print(f"[VQ] unexpected keys: {len(unexpected)} (show first 10)")
        print(unexpected[:10])

    model = model.to(device).eval()
    return model


def load_pool_helper(device: torch.device):
    pool_helper = TimmEmbedder(
        model_name=TIMM_MODEL_NAME,
        pretrained=True,
        feature_levels=(23,),
        img_size=384,
        ckpt_path=str(TIMM_CHECKPOINT_PATH),
    ).to(device)
    pool_helper.eval()
    return pool_helper


@torch.no_grad()
def encode_image_with_ideal(vq_model, pool_helper, images: torch.Tensor) -> torch.Tensor:
    """
    IDEAL zero-shot image embedding:
      images -> vq_model -> semantic_recon -> pool_helper.pool_tokens -> normalize
    """
    (dec, semantic_feat, spatial_feat, semantic_recon, spatial_recon), _, _ = vq_model(images)
    pooled = pool_helper.pool_tokens(semantic_recon)
    return F.normalize(pooled, dim=-1)


@torch.no_grad()
def evaluate_ideal_zeroshot(
    vq_model,
    pool_helper,
    text_model,
    tokenizer,
    preprocess,
    device: torch.device,
) -> Tuple[float, float]:
    # text classifier (D, 1000)
    classifier = build_zero_shot_classifier(
        model=text_model,
        tokenizer=tokenizer,
        classnames=list(IMAGENET_CLASSNAMES),
        templates=list(OPENAI_IMAGENET_TEMPLATES),
        num_classes_per_batch=10,
        device=device,
        use_tqdm=True,
    )

    # dataset uses SigLIP2 official preprocess (你已验证这点非常关键)
    dataset = datasets.ImageFolder(str(IMAGENET_VAL_DIR), transform=preprocess)

    # remap wnid folder index -> standard 0..999
    wnid_to_std = build_wnid_to_standard_index(IMAGENET_CLASS_INDEX_JSON)
    old_to_std = {old_idx: wnid_to_std[wnid] for old_idx, wnid in enumerate(dataset.classes)}
    dataset.samples = [(p, old_to_std[y]) for (p, y) in dataset.samples]
    dataset.targets = [y for _, y in dataset.samples]

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if LOGITS_MODE == "siglip":
        scale = text_model.logit_scale.exp()
        bias = text_model.logit_bias
    else:
        scale = None
        bias = None

    top1_correct = 0
    top5_correct = 0
    total = 0

    use_amp = (device.type == "cuda")

    for images, labels in tqdm(loader, desc="IDEAL Zero-shot eval"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            img_feats = encode_image_with_ideal(vq_model, pool_helper, images)  # (B, D)
            sims = img_feats @ classifier  # (B, 1000)

            if LOGITS_MODE == "siglip":
                logits = sims * scale + bias
            elif LOGITS_MODE == "fixed_100":
                logits = sims * 100.0
            else:
                raise ValueError(f"Unknown LOGITS_MODE: {LOGITS_MODE}")

        pred = logits.topk(5, dim=1).indices
        top1_correct += (pred[:, 0] == labels).sum().item()
        top5_correct += (pred == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    return top1_correct / total * 100.0, top5_correct / total * 100.0


def main():
    device = torch.device(DEVICE)
    print("device:", device)
    print("HF_HOME:", HF_HOME, "| offline=1")
    print("Text repo:", HF_TEXT_REPO)
    print("LOGITS_MODE:", LOGITS_MODE)

    # 1) Text side (official hf-hub loader, offline cache)
    text_model, tokenizer, preprocess = load_text_side_from_hf_cache(device)
    print("SigLIP2 preprocess:", preprocess)

    # 2) IDEAL components
    vq_model = load_vq_model(device)
    pool_helper = load_pool_helper(device)

    # 3) Evaluate IDEAL zero-shot
    top1, top5 = evaluate_ideal_zeroshot(
        vq_model=vq_model,
        pool_helper=pool_helper,
        text_model=text_model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        device=device,
    )

    print("========================================")
    print("IDEAL Zero-shot ImageNet val")
    print(f"Top-1: {top1:.2f}%")
    print(f"Top-5: {top5:.2f}%")
    print("========================================")


if __name__ == "__main__":
    main()
