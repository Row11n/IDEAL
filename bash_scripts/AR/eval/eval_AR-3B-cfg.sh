

# Activate the expected conda env (matches existing single-run script).
source activate ideal

: "${MODEL_TYPE:=GPT-3B}"  # override via env if needed
: "${TOKENIZER_CONFIG:=configs/tokenizer/ideal-tokenizer.yaml}"
: "${VQ_CKPT:=weights/ideal-tokenizer.pth}"
: "${GPT_CKPT:=output/GPT-3B-1501200.pt}"
: "${SAMPLE_DIR:=samples/cfg-GPT-3B-1501200}"
: "${NPROC:=4}"

# Hard-coded cfg sweep range.
# 1.75 1.7 1.65 1.6
CFG_VALUES=(1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)

run_one_cfg() {
  local cfg_scale="$1"
  echo ">>> Running test_net.py with cfg-scale=${cfg_scale}"
  torchrun --master-port 29501 --nproc_per_node="${NPROC}" test_net.py \
    --tokenizer-config "${TOKENIZER_CONFIG}" \
    --vq-ckpt "${VQ_CKPT}" \
    --gpt-ckpt "${GPT_CKPT}" \
    --compile \
    --gpt-model "${MODEL_TYPE}" \
    --image-size 384 \
    --sample-dir "${SAMPLE_DIR}" \
    --image-size-eval 256 \
    --cfg-scale "${cfg_scale}" \
    --precision bf16 \
    --per-proc-batch-size 64 \
    --latent-size 24 \
    --vq-model IDEAL
}

for cfg in "${CFG_VALUES[@]}"; do
  run_one_cfg "${cfg}"
done
