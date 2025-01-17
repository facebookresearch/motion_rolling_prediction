#!/bin/bash
# Script that evaluates the model on the eval_gap_config specified in the parameters.
# How to run:
# ./scripts/eval.sh GPU_NUM RESULTS_FOLDER MODEL_NAME none medium_hands_idp

if [ $# -lt 2 ]; then
    echo "Usage: $0 <gpu_num> <results_folder> <model_name> <config1> <config2> ..."
    exit 1
fi
gpu_num=$1
results_folder=$2
model_name=$3
shift 3
configs=("$@")
echo "Running evaluation for model: $model_name on GPU: $gpu_num"

for config in "${configs[@]}"; do
    if [ "$config" == "none" ]; then
        echo "Running evaluation without --eval_gap_config"
        CUDA_VISIBLE_DEVICES=$gpu_num buck2 run @//mode/dev-nosan //xr_body/realtime_diffusion:test \
            -- \
            --model_path manifold://xr_body/tree/personal/gbarquero/"$results_folder"/checkpoints/"$model_name"/model_latest.pt \
            --eval --eval_batch_size 64
    else
        echo "Running evaluation with --eval_gap_config=$config"
        CUDA_VISIBLE_DEVICES=$gpu_num buck2 run @//mode/dev-nosan //xr_body/realtime_diffusion:test \
            -- \
            --model_path manifold://xr_body/tree/personal/gbarquero/"$results_folder"/checkpoints/"$model_name"/model_latest.pt \
            --eval --eval_batch_size 64 --eval_gap_config "$config"
    fi
done

echo "Evaluations completed"
