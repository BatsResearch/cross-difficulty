#!/bin/bash

MODELS=(
#  "Qwen/Qwen2.5-1.5B-Instruct"
 "meta-llama/Llama-3.2-1B-Instruct"
#  "meta-llama/Llama-3.2-3B-Instruct"
  # "Qwen/Qwen2.5-3B-Instruct"
#  "meta-llama/Llama-3.3-70B-Instruct"
  # "meta-llama/Llama-3.1-8B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "Qwen/Qwen2.5-14B-Instruct"
)

DATASETS=(
#  "gsm8k"
#  "arc"
#  "bbh"
#  "gpqa_extended"
#  "math"
  # "musr"
  # "math-solutions-qwen"
  "math-solutions-llama"
)

LARGE_DATASETS=(
#  "ifeval"
#  "mmlu_pro"
#  "musr"
)

EPOCH=5
QUANTILE_KEY="1pl_quantile"
BASE_DIR= #TODO: specify base directory
eval_output_dir= #TODO: specify evaluation output directory

map_dataset_name_to_task() {
  key="$1"
  if [ "$key" = "gsm8k" ]; then
    echo "cross_difficulty_gsm8k"
  elif [ "$key" = "arc" ]; then
    echo "cross_difficulty_arc"
  elif [ "$key" = "ifeval" ]; then
    echo "cross_difficulty_ifeval"
  elif [ "$key" = "mmlu_pro" ]; then
    echo "cross_difficulty_mmlu_pro"
  elif [ "$key" = "gpqa_extended" ]; then
    echo "cross_difficulty_gpqa_extended"
  elif [ "$key" = "bbh" ]; then
    echo "cross_difficulty_bbh"
  elif [ "$key" = "musr" ]; then
    echo "cross_difficulty_musr"
  elif [ "$key" = "math" ]; then
    echo "cross_difficulty_math"
  elif [ "$key" = "math-solutions-qwen" ]; then
    echo "cross_difficulty_math"
  elif [ "$key" = "math-solutions-llama" ]; then
    echo "cross_difficulty_math"
  else
    echo "unknown"
  fi
}

join_array() {
  local sep="$1"
  shift
  local result=""
  for item in "$@"; do
    if [ -z "$result" ]; then
      result="$item"
    else
      result="${result}${sep}${item}"
    fi
  done
  echo "$result"
}

BINS=("0 1" "1 2" "2 3" "3 4" "4 5" "5 6" "6 7" "7 8" "8 9" "9 10")

PORT=1024

for MODEL in "${MODELS[@]}"; do
  MODEL_FAMILY=$(echo "$MODEL" | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')
  MODEL_SIZE=$(echo "$MODEL" | grep -oE '[0-9]+\.[0-9]+B|[0-9]+B' | tr '[:upper:]' '[:lower:]')
  SHORT_MODEL="${MODEL_FAMILY}-${MODEL_SIZE}"

  for DATASET in "${DATASETS[@]}"; do
    ALL_EVAL_JOBS=()
    for BIN in "${BINS[@]}"; do
      read LOWER UPPER <<< "$BIN"

      OUTPUT_DIR="$BASE_DIR/sft-${SHORT_MODEL}-${DATASET}-${LOWER}-${UPPER}-epoch-${EPOCH}-full-${QUANTILE_KEY}"
      RUN_NAME="sft-${SHORT_MODEL}-${DATASET}-${LOWER}-${UPPER}-epoch-${EPOCH}-full-${QUANTILE_KEY}_parallel"

      PORT=$((PORT + 1))
      echo "Running: $MODEL | $DATASET | Bins $LOWER-$UPPER"

      CKPT_DIR="$OUTPUT_DIR/${MODEL//\//--}_${DATASET}_42--trainer-saved"

      # 1) Train job
      TRAIN_JOB_ID=$(sbatch _sft_b200.sh \
        $PORT \
        --num_train_epochs $EPOCH \
        --model_name "$MODEL" \
        --dataset_quantile_lower_bin $LOWER \
        --dataset_quantile_upper_bin $UPPER \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --dataset "$DATASET" \
        --quantile_key $QUANTILE_KEY \
        --full_finetuning True \
        --max_seq_length 4096 \
        --per_device_train_batch_size 2 \
        --use_liger_kernel True | awk '{print $NF}')

      # 2) Conversion job dependent on training job
      CONVERT_JOB_ID=$(sbatch -d afterok:$TRAIN_JOB_ID zero2fp32-direct.sh \
        "$CKPT_DIR" \
        "$MODEL" | awk '{print $NF}')

      # 3) Evaluation job dependent on conversion job
      TASK=$(map_dataset_name_to_task "$DATASET")
      EVAL_JOB_ID=$(sbatch -d afterok:$CONVERT_JOB_ID vllm-eval-with-arguments_b200.sh \
        "$CKPT_DIR" \
        "$TASK" \
        "${eval_output_dir}/${SHORT_MODEL}-${DATASET}-${LOWER}-${UPPER}-full-${QUANTILE_KEY}-epoch-${EPOCH}/" | awk '{print $NF}')

      # echo "Job IDs: Train=$TRAIN_JOB_ID | Convert=$CONVERT_JOB_ID | Eval=$EVAL_JOB_ID"

      ALL_EVAL_JOBS+=("$EVAL_JOB_ID")
    done
    echo
    echo
  done

  for DATASET in "${LARGE_DATASETS[@]}"; do
    QUANTILE_KEY="1pl_quantile"

    ALL_EVAL_JOBS=()
    for BIN in "${BINS[@]}"; do
      read LOWER UPPER <<< "$BIN"

      OUTPUT_DIR="$BASE_DIR/sft-${SHORT_MODEL}-${DATASET}-${LOWER}-${UPPER}-epoch-${EPOCH}-full-${QUANTILE_KEY}"
      RUN_NAME="sft-${SHORT_MODEL}-${DATASET}-${LOWER}-${UPPER}-epoch-${EPOCH}-full-${QUANTILE_KEY}"

      PORT=$((PORT + 1))
      echo "Running: $MODEL | $DATASET | Bins $LOWER-$UPPER"

      CKPT_DIR="$OUTPUT_DIR/${MODEL//\//--}_${DATASET}_42--trainer-saved"

      # 1) Train job
      TRAIN_JOB_ID=$(sbatch _sft_b200.sh \
        $PORT \
        --model_name "$MODEL" \
        --dataset_quantile_lower_bin $LOWER \
        --dataset_quantile_upper_bin $UPPER \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --dataset "$DATASET" \
        --quantile_key $QUANTILE_KEY \
        --full_finetuning True \
        --per_device_train_batch_size 1 \
        --gradient_checkpointing True \
        --max_seq_length 4096 | awk '{print $NF}')
      echo "Submitted training job with ID: $TRAIN_JOB_ID."

      # 2) Conversion job dependent on training job
      CONVERT_JOB_ID=$(sbatch -d afterok:$TRAIN_JOB_ID zero2fp32-direct.sh \
        "$CKPT_DIR" \
        "$MODEL" | awk '{print $NF}')
      echo "Submitted conversion job with ID: $CONVERT_JOB_ID. Launching evaluation dependent on its completion."

      # 3) Evaluation job dependent on conversion job
      TASK=$(map_dataset_name_to_task "$DATASET")
      sbatch -d afterok:$CONVERT_JOB_ID vllm-eval-with-arguments_b200.sh \
        "$CKPT_DIR" \
        "$TASK" \
        "${eval_output_dir}/${SHORT_MODEL}-${DATASET}-${LOWER}-${UPPER}-full-${QUANTILE_KEY}-epoch-${EPOCH}/"
    done
    echo
    echo
  done
done

