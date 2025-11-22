#!/bin/bash
#SBATCH --job-name=dinov3_pretrain
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=a100_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=10-00:00:00
#SBATCH --mem=128G

# Initialize micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate dinov3

# Set PYTHONPATH
export PYTHONPATH=.

# Configuration
CONFIG_FILE="dinov3/configs/train/dinov3_hf_pretrain.yaml"
OUTPUT_DIR="./output/dinov3_hf_pretrain_multi_gpu"
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Get number of GPUs
NUM_GPUS=4

echo "Starting training on $NUM_GPUS GPUs"

# Run training with torch.distributed.run
# We use --nproc_per_node=$NUM_GPUS to spawn one process per GPU
python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) \
    dinov3/train/train.py \
    --config-file $CONFIG_FILE \
    --output-dir $OUTPUT_DIR \
    train.batch_size_per_gpu=16 \
    train.num_workers=8 \
    train.compile=false
