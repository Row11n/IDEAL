
# Activate env before launching torchrun commands.
source activate ideal

# Hard-code the checkpoint directory you want to sweep.
CKPT_DIR="output/GPT-B-nodrop/snapshot"

if [ ! -d "${CKPT_DIR}" ]; then
  echo "Checkpoint directory not found: ${CKPT_DIR}" >&2
  exit 1
fi

: "${MODEL_TYPE:=GPT-B}"
: "${TOKENIZER_CONFIG:=configs/tokenizer/ideal-tokenizer.yaml}"
: "${VQ_CKPT:=weights/ideal-tokenizer.pth}"
: "${CFG_SCALE:=1.75}"
: "${SAMPLE_DIR:=samples/GPT-B/GPT-B-nodrop}"
: "${IMAGE_SIZE:=384}"
: "${IMAGE_SIZE_EVAL:=256}"
: "${PRECISION:=bf16}"
: "${PER_PROC_BATCH:=128}"
: "${LATENT_SIZE:=24}"
: "${VQ_MODEL:=IDEAL}"
: "${NPROC:=4}"

mapfile -t CKPT_FILES < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name '*.pt' | sort)

if [ "${#CKPT_FILES[@]}" -eq 0 ]; then
  echo "No .pt checkpoints found under ${CKPT_DIR}" >&2
  exit 1
fi

for ckpt in "${CKPT_FILES[@]}"; do
  echo ">>> Running test_net.py with checkpoint ${ckpt}"
  torchrun --nproc_per_node="${NPROC}" test_net.py \
    --tokenizer-config "${TOKENIZER_CONFIG}" \
    --vq-ckpt "${VQ_CKPT}" \
    --gpt-ckpt "${ckpt}" \
    --compile \
    --gpt-model "${MODEL_TYPE}" \
    --image-size "${IMAGE_SIZE}" \
    --sample-dir "${SAMPLE_DIR}" \
    --image-size-eval "${IMAGE_SIZE_EVAL}" \
    --cfg-scale "${CFG_SCALE}" \
    --precision "${PRECISION}" \
    --per-proc-batch-size "${PER_PROC_BATCH}" \
    --latent-size "${LATENT_SIZE}" \
    --vq-model "${VQ_MODEL}"
done
