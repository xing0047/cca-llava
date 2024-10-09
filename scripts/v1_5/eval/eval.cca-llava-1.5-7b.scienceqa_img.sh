python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/xingyun/xingy/cca-llava/checkpoints/finetune/cca-llava-1.5-7b",attn_implementation="sdpa" \
    --tasks scienceqa_img \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix cca_llava_1.5_scienceqa_img \
    --output_path ./logs/