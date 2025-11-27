#!/bin/bash

# Fine-tune meta-llama/Meta-Llama-3-8B-Instruct with adapter tuning to mitigate gender bias.
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

CUDA_VISIBLE_DEVICES=0 python -u debias_adapter.py \
    --model_name_or_path "$MODEL" \
    --task_type "causal_lm" \
    --train_file "data/wikipedia-10.txt" \
    --max_seq_length 2048 \
    --line_by_line \
    --bias_type "gender" \
    --cda_mode "partial" \
    --output_dir "checkpoints/meta-llama3-8b-instruct-gender-adapter" \
    --do_train \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --lr_scheduler_type "linear" \
    --warmup_steps 500 \
    --save_strategy "epoch" \
    --evaluation_strategy "no" \
    --seed 42 \
    --down_sample 0.2 \
    --adapter_config "pfeiffer" \
    --adapter_reduction_factor 16 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --bf16 \
    > run_meta_llama3_8b_instruct_adapter.out 2>&1
