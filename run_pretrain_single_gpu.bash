#!/bin/bash

# Set PYTHONPATH to include the current directory
export PYTHONPATH=.

# Output directory for checkpoints and logs
OUTPUT_DIR="./output/dinov3_hf_pretrain"
mkdir -p $OUTPUT_DIR

# Config file path
CONFIG_FILE="dinov3/configs/train/dinov3_hf_pretrain.yaml"

# Run training on a single GPU
# We use torchrun with --nproc_per_node=1 for single GPU training
python -m torch.distributed.run --nproc_per_node=1 dinov3/train/train.py \
    --config-file $CONFIG_FILE \
    --output-dir $OUTPUT_DIR \
    train.batch_size_per_gpu=16 \
    train.num_workers=4 \
    train.compile=false
