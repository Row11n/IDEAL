source activate ideal
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
WANDB_MODE=offline torchrun --nproc_per_node="${NPROC_PER_NODE}" train/train_tokenizer.py --config configs/tokenizer/ideal-tokenizer.yaml
