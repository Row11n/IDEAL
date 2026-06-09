source activate ideal
CONFIG=${1:-configs/tokenizer/ideal-tokenizer.yaml}
WEIGHTS=${2:-weights/ideal-tokenizer.pth}
OUTDIR=${3:-results/ideal-tokenizer}
EVAL_SIZE=${4:-256}

SAMPLE_DIR="$OUTDIR/png_eval${EVAL_SIZE}"
mkdir -p "$SAMPLE_DIR"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  inference/eval.py \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --device "cuda:0" \
  --distributed false \
  --save-png true \
  --sample-dir "$SAMPLE_DIR" \
  --npz-count 50000 \
  --eval-image-size "$EVAL_SIZE" \
  --fid true \
  --isc true
