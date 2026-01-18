#!/bin/bash

MODELS=(
 "Qwen/Qwen2.5-1.5B-Instruct"
#  "meta-llama/Llama-3.2-1B-Instruct"
#  "meta-llama/Llama-3.2-3B-Instruct"
#  "Qwen/Qwen2.5-3B-Instruct"
#  "meta-llama/Llama-3.1-8B-Instruct"
#  "Qwen/Qwen2.5-7B-Instruct"
#   "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen3-4B-Instruct-2507"
)

TASKS=(
    # "cross_difficulty_gsm8k"
    # "cross_difficulty_arc"
    # "cross_difficulty_ifeval"
    "cross_difficulty_mmlu_pro"
    # "cross_difficulty_gpqa_extended"
    # "cross_difficulty_bbh"
    # "cross_difficulty_musr"
    # "cross_difficulty_math"
)

map_dataset_name() {
  key="$1"
  if [ "$key" = "cross_difficulty_gsm8k" ]; then
    echo "gsm8k"
  elif [ "$key" = "cross_difficulty_arc" ]; then
    echo "arc"
  elif [ "$key" = "cross_difficulty_ifeval" ]; then
    echo "ifeval"
  elif [ "$key" = "cross_difficulty_mmlu_pro" ]; then
    echo "mmlu_pro"
  elif [ "$key" = "cross_difficulty_gpqa_extended" ]; then
    echo "gpqa_extended"
  elif [ "$key" = "cross_difficulty_bbh" ]; then
    echo "bbh"
  elif [ "$key" = "cross_difficulty_musr" ]; then
    echo "musr"
  elif [ "$key" = "cross_difficulty_math" ]; then
    echo "math"
  elif [ "$key" = "gsm8k" ]; then
    echo "gsm8k"
  elif [ "$key" = "arc" ]; then
    echo "arc"
  elif [ "$key" = "ifeval" ]; then
    echo "ifeval"
  elif [ "$key" = "mmlu_pro" ]; then
    echo "mmlu_pro"
  else
    echo "unknown"
  fi
}

output_dir= #TODO: specify output directory

for MODEL in "${MODELS[@]}"; do

    MODEL_FAMILY=$(echo "$MODEL" | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')
    MODEL_SIZE=$(echo "$MODEL" | grep -oE '[0-9]+\.[0-9]+B|[0-9]+B' | tr '[:upper:]' '[:lower:]')
    SHORT_MODEL="${MODEL_FAMILY}-${MODEL_SIZE}"

    for TASK in "${TASKS[@]}"; do
        dataset=$(map_dataset_name "$TASK")

        task="${TASK}"
        pretrained="${MODEL}"
        output_path="${output_dir}/${SHORT_MODEL}-${dataset}-zero-shot/"
        mkdir -p "${output_path}"

        echo "=== Submitting zero-shot evaluation for ${MODEL} on ${dataset} → ${output_path} ==="
        sbatch vllm-eval-with-arguments_b200.sh "${pretrained}" "${task}" "${output_path}"
    done
done