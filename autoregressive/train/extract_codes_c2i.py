import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from autoregressive.dataset.dataset_with_path import ImageFolderWithPath
from modelling.tokenizer import VQ_models
from utils.data import center_crop_arr
from utils.distributed import init_distributed_mode
from utils.misc import load_model_state_dict, str2bool


def build_tokenizer(args, device):
    cfg = OmegaConf.load(args.tokenizer_config)
    model = VQ_models[args.vq_model](
        image_size=cfg.image_size,
        z_channels=cfg.z_channels,
        codebook_size=cfg.codebook_size,
        codebook_embed_dim=cfg.codebook_embed_dim,
        codebook_l2_norm=cfg.codebook_l2_norm,
        commit_loss_beta=cfg.commit_loss_beta,
        entropy_loss_ratio=cfg.entropy_loss_ratio,
        vq_loss_ratio=cfg.vq_loss_ratio,
        num_codebooks=cfg.num_codebooks,
        semantic_cos_weight=cfg.semantic_cos_weight,
        semantic_l2_weight=cfg.semantic_l2_weight,
        shallow_cos_weight=cfg.shallow_cos_weight,
        shallow_l2_weight=cfg.shallow_l2_weight,
        decoder_up_type=cfg.decoder_up_type,
        transformer_config=cfg.transformer_config,
        codebook_slots_embed_dim=cfg.codebook_slots_embed_dim,
    ).to(device)
    payload = torch.load(args.vq_ckpt, map_location="cpu")
    state = load_model_state_dict(payload.get("model", payload.get("ema", payload)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing and args.rank == 0:
        print(f"[extract] Missing tokenizer keys: {missing}")
    if unexpected and args.rank == 0:
        print(f"[extract] Unexpected tokenizer keys: {unexpected}")
    model.eval()
    return model


def append_h5(handle, codes, labels, paths):
    codes = np.asarray(codes, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int64)
    paths = np.asarray(paths, dtype=object)
    path_dtype = h5py.string_dtype("utf-8")
    n, code_len = codes.shape

    if "code" not in handle:
        handle.create_dataset("code", data=codes, maxshape=(None, code_len), chunks=True)
        handle.create_dataset("label", data=labels, maxshape=(None,), chunks=True)
        handle.create_dataset("path", data=paths, maxshape=(None,), chunks=True, dtype=path_dtype)
        return

    start = handle["code"].shape[0]
    end = start + n
    for key, values in [("code", codes), ("label", labels), ("path", paths)]:
        handle[key].resize((end,) if key != "code" else (end, code_len))
        handle[key][start:end] = values


def main(args):
    if args.distributed:
        init_distributed_mode(args)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        args.rank = rank
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = 0 if torch.cuda.is_available() else "cpu"
        args.rank = 0

    torch.manual_seed(args.global_seed * world_size + rank)
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(args, device)

    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolderWithPath(args.data_path, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if args.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    Path(args.code_path).mkdir(parents=True, exist_ok=True)
    out_file = Path(args.code_path) / f"codes_rank{rank:03d}.h5"
    ptdtype = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.mixed_precision]

    with h5py.File(out_file, "w") as handle:
        for images, labels, paths in tqdm(loader, disable=(rank != 0), desc="Extract IDEAL codes"):
            images = images.to(device, non_blocking=True)
            views = [images]
            if args.include_flip:
                views.append(torch.flip(images, dims=[-1]))
            batch = torch.cat(views, dim=0)

            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=(device.type == "cuda" and args.mixed_precision != "none"),
                dtype=ptdtype,
            ):
                indices = tokenizer.encode_to_ids(batch)

            num_views = len(views)
            labels_np = labels.numpy().repeat(num_views)
            paths_rep = [p for p in paths for _ in range(num_views)]
            append_h5(handle, indices.cpu().numpy(), labels_np, paths_rep)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(f"[extract] Wrote IDEAL code shards to {args.code_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, default="data/imagenet_codes")
    parser.add_argument("--tokenizer-config", type=str, default="configs/tokenizer/ideal-tokenizer.yaml")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="IDEAL")
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--include-flip", type=str2bool, default=True)
    parser.add_argument("--distributed", type=str2bool, default=True)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
