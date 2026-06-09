# IDEAL: In-DEpth ALignment

Official code for **IDEAL: In-DEpth ALignment Makes A Discrete Representation AutoEncoder**.

IDEAL builds a discrete representation autoencoder from a frozen vision foundation model. It extracts shallow and deep SigLIP2 features, fuses them with cross-attention before vector quantization, reconstructs both feature depths, and decodes the deep reconstructed feature to pixels.

## Repository Layout

- `ideal/`: IDEAL tokenizer internals, including frozen SigLIP2 feature extraction, shallow/deep fusion, quantization, feature decoder, and pixel decoder.
- `modelling/tokenizer.py`: public IDEAL tokenizer wrapper used by training, evaluation, and AR sampling scripts.
- `train/train_tokenizer.py`: tokenizer training entry point.
- `inference/`: tokenizer reconstruction and metric evaluation.
- `autoregressive/`: ImageNet class-conditional AR model, dataset, generation, and token-code extraction.
- `train_c2i.py` and `test_net.py`: AR training and ImageNet generation/evaluation entry points.
- `configs/`: IDEAL tokenizer and ImageNet AR configs.

## Environment

```bash
conda env create -f environment.yaml
conda activate ideal
```

If you need to install manually, `bash_scripts/create_env.sh` contains the minimal conda setup used by the project.

## Expected Weights

Place external weights under `weights/`:

- `weights/vit_large_patch16_siglip_384.v2_webli/model.safetensors`: frozen SigLIP2 image tower used by the tokenizer.
- `weights/siglip2_openclip/`: optional text-side SigLIP2 files for zero-shot evaluation.
- `weights/ideal-tokenizer.pth`: trained IDEAL tokenizer checkpoint.
- AR checkpoints are passed explicitly through `--gpt-ckpt`.

## Train IDEAL Tokenizer

```bash
NPROC_PER_NODE=8 bash bash_scripts/tokenizer/train_tokenizer.sh
```

Main config: `configs/tokenizer/ideal-tokenizer.yaml`.

## Evaluate Reconstruction

```bash
bash bash_scripts/tokenizer/run_eval.sh \
  configs/tokenizer/ideal-tokenizer.yaml \
  weights/ideal-tokenizer.pth \
  results/ideal-tokenizer \
  256
```

## Extract IDEAL Codes for AR

```bash
torchrun --nproc_per_node=8 autoregressive/train/extract_codes_c2i.py \
  --data-path ImageNet/train \
  --code-path data/imagenet_codes \
  --tokenizer-config configs/tokenizer/ideal-tokenizer.yaml \
  --vq-ckpt weights/ideal-tokenizer.pth
```

The extractor writes sharded `.h5` files with `code`, `label`, and `path` datasets. `train_c2i.py` consumes the directory through `dataset: imagenet_code`.

## Train Class-Conditional AR

```bash
NPROC_PER_NODE=8 bash bash_scripts/AR/train_AR-B.sh
NPROC_PER_NODE=8 bash bash_scripts/AR/train_AR-L.sh
NPROC_PER_NODE=8 bash bash_scripts/AR/train_AR-XXL.sh
NPROC_PER_NODE=8 bash bash_scripts/AR/train_AR-3B.sh
```

The AR configs live under `configs/ar/`.

## Generate and Evaluate

```bash
torchrun --nproc_per_node=4 test_net.py \
  --tokenizer-config configs/tokenizer/ideal-tokenizer.yaml \
  --vq-ckpt weights/ideal-tokenizer.pth \
  --gpt-ckpt path/to/gpt.pt \
  --gpt-model GPT-B \
  --vq-model IDEAL \
  --latent-size 24 \
  --image-size 384 \
  --image-size-eval 256
```

`bash_scripts/AR/eval/` contains cfg sweep helpers for common model sizes.
