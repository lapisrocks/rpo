#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
export TRANSFORMERS_CACHE='/scratch/bbrb/andyz3/cache'

export n=25
export model=$1 # llama2 or starling

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

CUDA_LAUNCH_BLOCKING=1 python -u /home/andyz3/rpo/experiments/main.py > rpo_test.txt  \
    --config="CONFIGPATH" \
    --config.attack=gcg \
    --config.train_data="DATA_PATH" \
    --config.result_prefix="../results/transfer_${model}_gcg_${n}_progressive" \
    --config.progressive_goals=False \
    --config.stop_on_success=True \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_test_data=30 \
    --config.n_steps=2000 \
    --config.test_steps=50 \
    --config.batch_size=64 \
    --config.selection_interval=100 \
