#!/bin/bash
#SBATCH -p gpu-b200 --gres=gpu:8
#SBATCH --mem=128G

#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 4:00:00

nvidia-smi

source ~/.bashrc
conda activate easy2hard-vllm

module load cuda/12.4
#module unload cuda

export CUR_DIR= #Todo: set your working directory here
export WANDB_PROJECT= #TODO: set your wandb project name here
export HF_HOME= #TODO: set your HF cache path
export HF_TOKEN= #TODO: set your HF token here
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_SHM_DISABLE=0     # usually fine; set to 1 only if SHM issues on your cluster
export NCCL_IGNORE_CPU_AFFINITY=1

cd $CUR_DIR
which python

# Grab all MIG UUIDs
ALL_MIGS=$(nvidia-smi -L | awk -F'UUID: ' '/MIG/ {print $2}' | sed 's/)//')

# Select every other UUID (1st, 3rd, 5th, …)
SEL_MIGS=$(echo "$ALL_MIGS" | awk 'NR % 2 == 1')

# Export as comma-separated CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(echo $SEL_MIGS | tr ' ' ,)

# Check
echo $CUDA_VISIBLE_DEVICES

#python sft.py $@
accelerate launch --config_file accelerate_deepspeed_3_b200.yaml --main_process_port $1 sft.py ${@:2}