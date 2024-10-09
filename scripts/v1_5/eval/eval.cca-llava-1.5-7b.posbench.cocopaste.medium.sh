
root=/home/xingyun
llava_root=${root}/xingy/cca-llava
img_root=${llava_root}/playground/data/posbench/cocopaste/grid6x6
save_root=${llava_root}/outputs/posbench/cocopaste/medium
model_root=${llava_root}/checkpoints/finetune
model=cca-llava-1.5-7b
model_name=${model}

echo "------------- Running for model: $model -------------"

question_file=playground/data/posbench/posbench_cocopaste_grid6x6.jsonl
answer_file=$save_root/${model_name}_posbench_cocopaste_grid6x6.jsonl
output_jpg=$save_root/${model_name}_posbench_cocopaste_grid6x6.jpg

if test -e ${answer_file}; then
    python llava/eval/eval_posbench.medium.py \
        --question-file ${question_file} \
        --result-file ${answer_file} \
        --output-jpg ${output_jpg}
else
    python llava/eval/model_vqa_pope.cca.py \
        --model-path ${model_root}/${model} \
        --question-file ${question_file} \
        --image-folder ${img_root} \
        --answers-file ${answer_file}
    python llava/eval/eval_posbench.medium.py \
        --question-file ${question_file} \
        --result-file ${answer_file} \
        --output-jpg ${output_jpg}
fi