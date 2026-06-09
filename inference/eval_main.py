import os
import sys
sys.path.append(os.getcwd())

from contextlib import nullcontext
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from evaluator import VQGANEvaluator
from modelling.tokenizer import VQ_models
from train.train_tokenizer import build_parser
from utils.data import center_crop_arr
from utils.misc import load_model_state_dict, str2bool


def _add_unique_argument(parser, *args, **kwargs):
    option_strings = [opt for opt in args if isinstance(opt, str) and opt.startswith("-")]
    if any(opt in parser._option_string_actions for opt in option_strings):
        return
    parser.add_argument(*args, **kwargs)


def parse_args():
    parser = build_parser()
    _add_unique_argument(parser, "--weights", type=str, default="weights/model.pth", help="Path to model-only weights (.pth).")
    _add_unique_argument(parser, "--device", type=str, default=None)
    _add_unique_argument(parser, "--per-proc-batch-size", type=int, default=None, help="Batch size per process when running DDP eval.")
    _add_unique_argument(parser, "--distributed", type=str2bool, default=False, help="Enable distributed evaluation (NCCL).")
    _add_unique_argument(parser, "--batch-size", type=int, default=32)
    _add_unique_argument(parser, "--eval-image-size", type=int, default=None)

    # ---- NEW: log directory (one txt per config+weights) ----
    _add_unique_argument(
        parser,
        "--log-dir",
        type=str,
        default=None,
        help="Directory to write per-experiment evaluation logs (one txt per config+weights).",
    )

    args = parser.parse_args()
    if args.config and os.path.isfile(args.config):
        from ruamel import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**(config_args or {}))
        args = parser.parse_args()
    if not hasattr(args, "eval_image_size") or args.eval_image_size is None:
        args.eval_image_size = args.image_size
    return args


def build_model(args, device):
    ModelClass = VQ_models[args.vq_model]
    model = ModelClass(
        image_size=args.image_size,
        z_channels=args.z_channels,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        vq_loss_ratio=args.vq_loss_ratio,
        kl_loss_weight=args.kl_loss_weight,
        dropout_p=args.dropout_p,
        enc_type=args.enc_type,
        encoder_model=args.encoder_model,
        dec_type=args.dec_type,
        decoder_model=args.decoder_model,
        num_latent_tokens=args.num_latent_tokens,
        enc_tuning_method=args.encoder_tuning_method,
        dec_tuning_method=args.decoder_tuning_method,
        enc_pretrained=args.encoder_pretrained,
        dec_pretrained=args.decoder_pretrained,
        enc_local_ckpt=args.encoder_local_ckpt,
        dec_local_ckpt=args.decoder_local_ckpt,
        repa_local_ckpt=args.repa_local_ckpt,
        enc_patch_size=args.encoder_patch_size,
        dec_patch_size=args.decoder_patch_size,
        tau=args.tau,
        repa=args.repa,
        repa_model=args.repa_model,
        repa_patch_size=args.repa_patch_size,
        repa_proj_dim=args.repa_proj_dim,
        repa_loss_weight=args.repa_loss_weight,
        repa_align=args.repa_align,
        num_codebooks=args.num_codebooks,
        enc_token_drop=args.enc_token_drop,
        enc_token_drop_max=args.enc_token_drop_max,
        cls_recon=args.cls_recon,
        cls_recon_weight=args.cls_recon_weight,
        semantic_target_layers=args.semantic_target_layers,
        semantic_qam_layers=args.semantic_qam_layers,
        semantic_cos_weight=args.semantic_cos_weight,
        semantic_l2_weight=args.semantic_l2_weight,
        shallow_cos_weight=args.shallow_cos_weight,
        shallow_l2_weight=args.shallow_l2_weight,
        aux_dec_model=args.aux_decoder_model,
        aux_loss_mask=args.aux_loss_mask,
        aux_hog_dec=args.aux_hog_decoder,
        aux_dino_dec=args.aux_dino_decoder,
        aux_clip_dec=args.aux_clip_decoder,
        aux_supcls_dec=args.aux_supcls_decoder,
        to_pixel=args.to_pixel,
        decoder_up_type=args.decoder_up_type,
        transformer_config=args.transformer_config,
        codebook_slots_embed_dim=args.codebook_slots_embed_dim,
    ).to(device)
    model.eval()
    return model


def load_model_weights(model, weights_path):
    payload = torch.load(weights_path, map_location="cpu")
    state = payload.get("model", payload)
    state = load_model_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Eval] Missing keys: {missing}")
    if unexpected:
        print(f"[Eval] Unexpected keys: {unexpected}")


def build_dataloader(args, distributed=False, world_size=1, rank=0):
    eval_root = args.eval_data_path if args.eval_data_path else args.data_path
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(eval_root, transform=transform)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if args.per_proc_batch_size is not None:
            batch_size = args.per_proc_batch_size
        elif args.batch_size is not None:
            batch_size = max(1, args.batch_size // world_size)
        else:
            batch_size = max(1, args.global_batch_size // world_size)
    else:
        sampler = None
        batch_size = args.batch_size or args.global_batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, len(dataset), sampler


def _resize_for_eval(batch: torch.Tensor, target_size: Optional[int]):
    """Resize [-1, 1] tensors with PIL bicubic to align with training evaluator."""
    tensor_01 = torch.clamp((batch + 1.0) * 0.5, 0.0, 1.0).float()
    if target_size is None or tensor_01.size(-1) == target_size:
        return tensor_01
    np_imgs = tensor_01.permute(0, 2, 3, 1).detach().cpu().numpy()
    resized = []
    for img in np_imgs:
        pil_img = Image.fromarray((img * 255.0).round().astype(np.uint8))
        pil_img = pil_img.resize((target_size, target_size), Image.BICUBIC)
        resized.append(np.asarray(pil_img).astype(np.float32) / 255.0)
    resized_np = np.stack(resized, axis=0)
    resized_tensor = torch.from_numpy(resized_np).permute(0, 3, 1, 2).to(batch.device)
    return resized_tensor


@torch.no_grad()
def run_vqgan_eval(model, loader, evaluator, device, target_size, amp_dtype):
    model.eval()
    evaluator.reset_metrics()
    psnr_sum = 0.0
    ssim_sum = 0.0
    num_samples = 0
    iterator = tqdm(loader, desc="Evaluating", leave=False)

    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else None

    for images, _ in iterator:
        images = images.to(device, non_blocking=True)
        context = autocast_ctx(dtype=amp_dtype) if autocast_ctx is not None else nullcontext()
        with context:
            recons_imgs, _, info = model(images)

        indices = None
        if isinstance(info, (list, tuple)) and len(info) > 2 and info[2] is not None:
            indices = info[2].flatten()

        real_01 = _resize_for_eval(images, target_size)
        fake_01 = _resize_for_eval(recons_imgs, target_size)
        evaluator.update(real_images=real_01, fake_images=fake_01, codebook_indices=indices)

        real_np = real_01.permute(0, 2, 3, 1).detach().cpu().numpy()
        fake_np = fake_01.permute(0, 2, 3, 1).detach().cpu().numpy()
        for r_img, f_img in zip(real_np, fake_np):
            psnr_sum += psnr_loss(r_img, f_img, data_range=1.0)
            ssim_sum += ssim_loss(r_img, f_img, channel_axis=-1, data_range=1.0)
            num_samples += 1

    scores = evaluator.result()
    scores["PSNR"] = float(psnr_sum / max(1, num_samples))
    scores["SSIM"] = float(ssim_sum / max(1, num_samples))
    return scores


def _safe_filename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in (s or ""))


def _stem(p: Optional[str]) -> str:
    if not p:
        return "none"
    return os.path.splitext(os.path.basename(p))[0]


def _append_per_experiment_log(args, dataset_len: int, scores: dict):
    """
    Create one txt per (config_stem, weights_stem) under args.log_dir and append one line per run.
    """
    if not getattr(args, "log_dir", None):
        return None

    os.makedirs(args.log_dir, exist_ok=True)

    config_stem = _safe_filename(_stem(getattr(args, "config", None)))
    weights_stem = _safe_filename(_stem(getattr(args, "weights", None)))
    log_filename = f"{config_stem}__{weights_stem}.txt"
    log_path = os.path.join(args.log_dir, log_filename)

    key_order = ["InceptionScore", "rFID", "CodebookUsage", "CodebookEntropy", "PSNR", "SSIM"]
    metric_str = " ".join([f"{k}={scores[k]:.4f}" for k in key_order if k in scores])

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eval_root = (args.eval_data_path if args.eval_data_path else args.data_path)

    line = (
        f"[{now}] "
        f"config={getattr(args, 'config', None)} "
        f"weights={getattr(args, 'weights', None)} "
        f"eval_root={eval_root} "
        f"n={dataset_len} "
        f"image_size={args.image_size} "
        f"eval_image_size={getattr(args, 'eval_image_size', args.image_size)} "
        f"{metric_str}\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)

    return log_path


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.global_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.global_seed)

    loader, dataset_len, _ = build_dataloader(args, distributed=False, world_size=1, rank=0)
    model = build_model(args, device)
    load_model_weights(model, args.weights)

    evaluator = VQGANEvaluator(
        device=device,
        enable_rfid=True,
        enable_inception_score=True,
        enable_codebook_usage_measure=True,
        enable_codebook_entropy_measure=True,
        num_codebook_entries=args.codebook_size,
    )
    dtype_map = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    amp_dtype = dtype_map.get(args.mixed_precision, torch.float32)
    scores = run_vqgan_eval(
        model=model,
        loader=loader,
        evaluator=evaluator,
        device=device,
        target_size=getattr(args, "eval_image_size", args.image_size),
        amp_dtype=amp_dtype,
    )

    print(f"[Eval] Completed offline evaluation on {dataset_len:,} images.")
    key_order = ["InceptionScore", "rFID", "CodebookUsage", "CodebookEntropy", "PSNR", "SSIM"]
    for key in key_order:
        if key in scores:
            print(f"{key}: {scores[key]:.4f}")

    log_path = _append_per_experiment_log(args, dataset_len, scores)
    if log_path is not None:
        print(f"[Eval] Appended metrics to: {log_path}")


if __name__ == "__main__":
    main()
