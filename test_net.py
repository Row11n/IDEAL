# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import warnings
warnings.filterwarnings('ignore', message='Overwriting.*in registry')

import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch, os, pdb
import os.path as osp
import math, torch, time
import torch.nn.functional as F
import torch.distributed as dist
import tensorflow.compat.v1 as tf
from evaluations.c2i.evaluator import Evaluator
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from modelling.tokenizer import VQ_models
from ideal.engine.misc import (is_main_process, get_rank, get_world_size, concat_all_gather)
import random
from utils.misc import load_model_state_dict

def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def build_vq_model(args, device):
    ModelClass = VQ_models[args.vq_model]
    tokenizer_config = OmegaConf.load(args.tokenizer_config)

    model = ModelClass(
        image_size=tokenizer_config.image_size,
        z_channels=tokenizer_config.z_channels,
        codebook_size=tokenizer_config.codebook_size,
        codebook_embed_dim=tokenizer_config.codebook_embed_dim,
        codebook_l2_norm=tokenizer_config.codebook_l2_norm,
        commit_loss_beta=tokenizer_config.commit_loss_beta,
        entropy_loss_ratio=tokenizer_config.entropy_loss_ratio,
        vq_loss_ratio=tokenizer_config.vq_loss_ratio,
        num_codebooks=tokenizer_config.num_codebooks,
        semantic_cos_weight=tokenizer_config.semantic_cos_weight,
        semantic_l2_weight=tokenizer_config.semantic_l2_weight,
        shallow_cos_weight=tokenizer_config.shallow_cos_weight,
        shallow_l2_weight=tokenizer_config.shallow_l2_weight,
        decoder_up_type=tokenizer_config.decoder_up_type,
        transformer_config=tokenizer_config.transformer_config,
        codebook_slots_embed_dim=tokenizer_config.codebook_slots_embed_dim,
    ).to(device)
    model.eval()
    return model


def load_vq_model_weights(vq_model, weights_path):
    payload = torch.load(weights_path, map_location="cpu")
    state = payload.get("model", payload) # a possible model key in ckpt
    state = load_model_state_dict(state)
    missing, unexpected = vq_model.load_state_dict(state, strict=False)
    if missing:
        print(f"[tokenizer] Missing keys: {missing}")
    if unexpected:
        print(f"[tokenizer] Unexpected keys: {unexpected}")
    del state

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    
    #* Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    #* Setup DDP:
    dist.init_process_group("nccl")
    rank = get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={get_world_size()}.")

    #* Create and load IDEAL tokenizer
    vq_model = build_vq_model(args, device)
    load_vq_model_weights(vq_model, args.vq_ckpt)

    #* Create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.latent_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        class_dropout_prob=args.class_dropout_prob
    ).to(device=device, dtype=precision)
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")

    if args.from_fsdp: # fsdp
        model_weight = checkpoint
    elif "ema" in checkpoint:  # ddp
        model_weight = checkpoint['ema']
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")

    m1, u1 = gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True,
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    else:
        ckpt_string_name = osp.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    vq_filename = osp.basename(args.vq_ckpt).split('.')[0]
    folder_name = f"{ckpt_string_name}-{vq_filename}-size-{args.image_size}-size-{args.image_size_eval}-{args.vq_model}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)
        os.makedirs(sample_folder_dir,exist_ok=True)
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * get_world_size()
    
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if is_main_process():
        print(f"Total number of images that will be sampled: {total_samples}")
   
    assert total_samples % get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total, count, nums = 0, 0, 0

    gen_samples = []
    for idx in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        
        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            )
        samples = vq_model.decode_from_ids(index_sample)
        if samples.shape[-1] != args.image_size_eval:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode="bicubic")
        samples = samples.detach() * 127.5 + 128
        samples = torch.clamp(samples, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    
        samples = concat_all_gather(samples)
        # samples = samples.cpu().numpy()
        samples = samples.to('cpu', dtype=torch.uint8).numpy()
        gen_samples.append(samples)  
        # # Save samples to disk as individual .png files
        # for i, sample in enumerate(samples):
        #     index = i * get_world_size() + rank + total
        #     Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        # total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if is_main_process():

        gen_samples = np.concatenate(gen_samples, axis=0)[:50_000]
        print(f'generated samples: {gen_samples.shape[0]}')

        config = tf.ConfigProto(
                allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True

        evaluator = Evaluator(tf.Session(config=config),batch_size=64)
        evaluator.warmup()


        print("computing reference batch activations...")
        ref_acts = evaluator.read_activations(args.ref_batch)
        print("computing/reading reference batch statistics...")
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)
        

        print("computing sample batch activations...")
        sample_acts = evaluator.read_activations(gen_samples)
        print("computing/reading sample batch statistics...")
        sample_stats, sample_stats_spatial = evaluator.read_statistics(gen_samples, sample_acts)
        print("Computing evaluations...")        
        FID = sample_stats.frechet_distance(ref_stats)
        sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)
        print("FID:", FID)
        print("sFID:", sFID)
        IS = evaluator.compute_inception_score(sample_acts[0])
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        print("Inception Score:", IS)
        print("Precision:", prec)
        print("Recall:", recall)

        txt_path = sample_folder_dir + '.txt'
        print("writing to {}".format(txt_path))
        with open(txt_path, 'w') as f:
            print("Inception Score:", IS, file=f)
            print("FID:", FID, file=f)
            print("sFID:", sFID, file=f)
            print("Precision:", prec, file=f)
            print("Recall:", recall, file=f)

        print("Done.")
    
    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-config", type=str, default=None, help="YAML config file used to specify tokenizer parameters.")    
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i'], default="c2i", help="class-conditional ImageNet generation")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--transformer-config-file", type=str, default="configs/vit_transformer.yaml")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="IDEAL")
    parser.add_argument("--latent-size", type=int, default=16, help="Latent spatial size.")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-slots-embed-dim", type=int, default=8, help="codebook dimension for queries quantization")
    parser.add_argument("--image-size", type=int, choices=[256,336, 384, 512], default=384)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--class-dropout-prob", type=float, default=0.1, help="cls drop probability")
    parser.add_argument("--z-channels", type=int, default=512,)
    parser.add_argument("--ref-batch", type=str, default='weights/VIRTUAL_imagenet256_labeled.npz', help="path to reference batch npz file")
    args = parser.parse_args()
    main(args)
