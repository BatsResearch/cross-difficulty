#!/bin/bash
#SBATCH -p gpu-b200 --gres=gpu:1
#SBATCH --mem=48G

#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 4:00:00

# ⚠️⚠️⚠️ IF YOU CHANGE NUMBER OF GPUS ABOVE, CHANGE THE tensor_parallel_size IN THE COMMAND BELOW ⚠️⚠️⚠️

source ~/.bashrc
conda activate easy2hard-vllm

module unload cuda

export CUR_DIR= #TODO: set this to the working directory
export WANDB_PROJECT= #TODO: set your wandb project name here
export HF_HOME= #TODO: set your HF cache path
export HF_TOKEN= #TODO: set your HF token here
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_SHM_DISABLE=0     # usually fine; set to 1 only if SHM issues on your cluster
export NCCL_IGNORE_CPU_AFFINITY=1

export VLLM_DISABLE_COMPILE_CACHE=1

export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=12

cd $CUR_DIR

unset CUDA_VISIBLE_DEVICES

lm_eval --model vllm \
  --model_args "pretrained=${1},dtype=bfloat16" \
  --task "${2}" \
  --batch_size auto \
  --seed 42 \
  --apply_chat_template \
  --log_samples \
  --output_path "${3}"