# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch, pdb
import numpy as np
from PIL import Image
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from omegaconf import OmegaConf
import torch.distributed as dist
import os, time, inspect, argparse
import ruamel.yaml as yaml
from einops import repeat, rearrange
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from utils.data import random_crop_arr, center_crop_arr
from ideal.engine.logger import create_logger
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from ideal.engine.ema import update_ema, requires_grad
from autoregressive.models.gpt import GPT_models
from autoregressive.dataset.build import build_dataset
from utils.distributed import init_distributed_mode
from modelling.tokenizer import VQ_models
from ideal.engine.misc import is_main_process, all_reduce_mean, concat_all_gather,get_world_size, get_rank
from utils.misc import load_model_state_dict, str2bool
from PIL import Image  
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random

def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

#################################################################################
#                             TOKENIZER HELPER                                  #
#################################################################################

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

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

def collate_fn(dd):

    assert ('images' in dd) & ('targets' in dd)

    rank = get_rank()
    device = rank % torch.cuda.device_count()

    images = torch.from_numpy(np.stack(dd['images'], axis=0)).flatten(0, 1).to(device)
    labels = torch.from_numpy(np.stack(dd['targets'], axis=0)).flatten().to(device)

    return images, labels

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    #* Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    rank = get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * get_world_size() + rank
    seed_everything(seed)
    torch.cuda.set_device(device)
    
    #* Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        experiment_dir = args.results_dir
        checkpoint_dir = osp.join(args.results_dir, 'snapshot')
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger_dir = osp.join(args.results_dir, 'logs')
        os.makedirs(logger_dir, exist_ok=True)
        logger = create_logger(logger_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    #* training args & training enviroment
    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={get_world_size()}.")


    #* Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    # latent_size = args.image_size // args.downsample_size
    latent_size = args.latent_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        class_dropout_prob=args.class_dropout_prob,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")


    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    #* Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # #* Setup dataset/dataloader:
    # dataset = ImageNetDataset(args.anno_file, args.image_size, True)

    # Setup data:
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    local_bsz = int(args.global_batch_size // dist.get_world_size())
    loader = DataLoader(
        dataset,
        batch_size=local_bsz,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Setup learning rate scheduler
    total_steps = args.epochs * len(loader)
    logger.info(f"Dataset loaded. {len(dataset)=}, global_bsz={args.global_batch_size}, {local_bsz=}, {len(loader)=}, {total_steps=}")

    #* Prepare models for training:
    if args.gpt_ckpt:
        assert osp.exists(args.gpt_ckpt), f'Please ensure the existence of {args.gpt_ckpt}'
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")

        model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = round(train_steps / len(loader))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    
    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    
    #* initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))

    logger.info(f"Training for {args.epochs} epochs...")
    model_dirs = osp.realpath(__file__).split('/')
    this_model_dir = '/'.join(model_dirs[2:-1])
            
    for epoch in range(start_epoch, args.epochs):

        logger.info(f"Beginning epoch {epoch}...")
       
        with tqdm(loader, dynamic_ncols=True, disable=not is_main_process()) as train_dl:

            for x, y, _ in train_dl:
                x = x.to(device, non_blocking=True).long()
                y = y.to(device, non_blocking=True).long()
                z_indices = x.reshape(x.shape[0], -1)
                c_indices = y.reshape(-1)

                assert z_indices.shape[0] == c_indices.shape[0]
                with torch.cuda.amp.autocast(dtype=ptdtype):  
                    _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
                # backward pass, with gradient scaling if training in fp16         
                scaler.scale(loss).backward()
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
                if args.ema:
                    if isinstance(model, DDP):
                        gpt_model = model.module
                    else:
                        gpt_model = model
                    update_ema(ema, gpt_model)

                # Log loss values:
                train_steps += 1
                #* Reduce loss over all processes:
                torch.cuda.synchronize()
                avg_loss = all_reduce_mean(loss)
                world_size = get_world_size()

                if (train_steps % args.log_every == 0) & is_main_process():
                    
                    train_dl.set_postfix(
                        ordered_dict={
                            "epoch"          : epoch,
                            "iters"          : train_steps,
                            "gpt_loss"       : avg_loss.item(),
                            'this_model_dir' : this_model_dir,
                            "world_size"     : world_size,
                        }
                    )

                #* Save checkpoint.
                if (train_steps > 0) & (train_steps % args.ckpt_every == 0) & is_main_process():
                        
                    checkpoint = {
                        "model": gpt_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args,
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                        
                    ckpt_path = osp.join(checkpoint_dir, '{}-{}.pt'.format(args.gpt_model, train_steps))
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint  to {ckpt_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info("Done!")
    dist.destroy_process_group()

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/ar/AR-base.yaml', help="config file used to specify parameters")
    parser.add_argument("--tokenizer-config", type=str, default=None, help="YAML config file used to specify tokenizer parameters.")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--crop-range", type=float, default=1.1)
    parser.add_argument('--pipe-name', type=str, default='')
    #* GPT model
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i'], default="c2i", help="class-conditional ImageNet generation")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--latent-size", type=int, default=16, help="Latent spatial size.")
    
    parser.add_argument("--ema", type=str2bool, default=True, help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--class-dropout-prob", type=float, default=0.1, help="dropout_probability of class label")    
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--results-dir", type=str, default="output")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--code-path", type=str, default="code-to-path")
    parser.add_argument("--image-size", type=int, choices=[256, 336, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])

    # Tokenizer
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="IDEAL")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=64, help="codebook dimension for vector quantization")
    parser.add_argument("--z-channels", type=int, default=1024,)
    return parser


if __name__ == "__main__":
    parser = build_parser()
  
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    args = parser.parse_args()
    main(args)
