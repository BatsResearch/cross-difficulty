#!/bin/bash
#SBATCH --partition batch
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 30:00

source ~/.bashrc
conda activate easy2hard-vllm

module load cuda/12.4

export CUR_DIR= #TODO: set your working directory here
export WANDB_PROJECT= #TODO: set your wandb project name here
export HF_HOME= #TODO: set your HF cache path
export HF_TOKEN= #TODO: set your HF token here

cd $CUR_DIR
which python

CKPT_DIR="$1"
MODEL="$2"

CONVERTER_SCRIPT="convert-fp32-to-huggingface-format.py"

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "⚠️  Skipping missing checkpoint: $CKPT_DIR"
  continue
fi

echo "🔄 Converting ZeRO-3 checkpoint to FP32: $CKPT_DIR"

if [[ ! -f "$CKPT_DIR/zero_to_fp32.py" ]]; then
  echo "❌ Missing zero_to_fp32.py in $CKPT_DIR"
  continue
fi

python "$CKPT_DIR/zero_to_fp32.py" \
  "$CKPT_DIR" \
  "$CKPT_DIR/pytorch_model.bin" \
  --max_shard_size 20GB

echo "💾 Saving HF-style model into same folder: $CKPT_DIR"
python "$CONVERTER_SCRIPT" \
  --model_path "$CKPT_DIR" \
  --model_id "$MODEL" \
  --save_model_path "$CKPT_DIR"

echo "🧹 Cleaning up old weights, keeping only HuggingFace format..."

# Remove ZeRO checkpoint files
rm -rf "$CKPT_DIR"/global_step*
rm -f "$CKPT_DIR"/zero_to_fp32.py
rm -f "$CKPT_DIR"/latest

# Remove the intermediate pytorch_model.bin (FP32 converted file)
rm -f "$CKPT_DIR"/pytorch_model.bin

# Remove any optimizer states and other training artifacts
rm -f "$CKPT_DIR"/optimizer.pt
rm -f "$CKPT_DIR"/pytorch_model.bin.index.json
rm -f "$CKPT_DIR"/rng_state_*.pth
rm -f "$CKPT_DIR"/scheduler.pt
rm -f "$CKPT_DIR"/trainer_state.json
rm -f "$CKPT_DIR"/training_args.bin

echo "✅ Done: $CKPT_DIR (cleaned up old weights)"
echo