#!/bin/bash

export WANDB_API_KEY='your_wandb_key'
WANDB_PROJECT_NAME=your_project_name
RUN_NAME=your_run_name
CONFIG_PATH="configs/your_config.yaml"
OUTPUT_DIR="check_points/${RUN_NAME}"
accelerate launch --mixed_precision="bf16" \
                  --num_processes=2 --num_machines 1 --main_process_port 12312 \
                  --config_file configs/deepspeed/acc_zero2_bf16.yaml train.py \
                  --tracker_project_name $WANDB_PROJECT_NAME \
                  --output_dir=$OUTPUT_DIR \
                  --config_file=$CONFIG_PATH \
                  --train_log_interval=10000 \
                  --val_interval=40000 \
                  --val_cfg=2.0 \
                  --min_decay=0.5 \
                  --log_every 10 \
                  --seed 0 \
                  --run_name $RUN_NAME  \
                  --num_workers_per_gpu 2 \
                  --checkpointing_last_steps 5000 \
                  --autocast_fp32_on_distill \
                  # --config_file="check_points/${RUN_NAME}/config.yaml" \
                  # --resume_from_last
