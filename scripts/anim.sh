#!/bin/bash
# Script that generates the rolling animations (i.e., feature-wise plots) for the model and eval_gap_config specified in the parameters.
# How to run:
# ./scripts/anim.sh GPU_NUM RESULTS_FOLDER MODEL_NAME none medium_hands_idp

if [ $# -lt 2 ]; then
    echo "Usage: $0 <gpu_num> <results_folder> <model_name> <config1> <config2> ..."
    exit 1
fi
gpu_num=$1
results_folder=$2
model_name=$3
shift 3
configs=("$@")
echo "Generating visualizations + animations for model: $model_name on GPU: $gpu_num"

for config in "${configs[@]}"; do
    if [ "$config" == "none" ]; then
        echo "Generating visualizations + animations without --eval_gap_config"
        CUDA_VISIBLE_DEVICES=$gpu_num buck2 run @//mode/dev-nosan //xr_body/realtime_diffusion:test \
            -- \
            --model_path manifold://xr_body/tree/personal/gbarquero/"$results_folder"/checkpoints/"$model_name"/model_latest.pt \
            --vis --vis_overwrite --vis_anim rolling_cr > /dev/null 2>&1
    else
        echo "Generating visualizations + animations with --eval_gap_config=$config"
        CUDA_VISIBLE_DEVICES=$gpu_num buck2 run @//mode/dev-nosan //xr_body/realtime_diffusion:test \
            -- \
            --model_path manifold://xr_body/tree/personal/gbarquero/"$results_folder"/checkpoints/"$model_name"/model_latest.pt \
            --vis --vis_overwrite --vis_anim rolling_cr --eval_gap_config "$config" > /dev/null 2>&1
    fi
done

echo "Visualizations + animations ready!"
