#!/bin/bash

huggingface_root=/media/drive_16TB/huggingface
deepspeed llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${huggingface_root}/lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ./playground/data/pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/pretrain/images \
    --vision_tower ${huggingface_root}/openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/pretrain/cca-llava-1.5-7b-558k \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
