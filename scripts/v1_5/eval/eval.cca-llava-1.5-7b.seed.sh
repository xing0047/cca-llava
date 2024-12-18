python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="xing0047/cca-llava-1.5-7b",attn_implementation="sdpa" \
    --tasks seedbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix cca_llava_1.5_seedbench \
    --output_path ./logs/