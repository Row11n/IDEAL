source activate ideal
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
model_type='GPT-XXL' # 'GPT-B' 'GPT-XL' 'GPT-XXL' 'GPT-2B' 'GPT-3B'
torchrun --nproc_per_node="${NPROC_PER_NODE}" train_c2i.py --config configs/ar/AR-XXL.yaml --tokenizer-config configs/tokenizer/ideal-tokenizer.yaml --gpt-type c2i --gpt-model ${model_type} \
    --global-batch-size 256
