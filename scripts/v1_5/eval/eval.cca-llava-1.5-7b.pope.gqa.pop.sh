
root=/home/xingyun
llava_root=${root}/xingy/cca-llava
img_root=/media/drive_16TB/data/gqa/images
save_root=${llava_root}/outputs/pope/gqa
model_root=${llava_root}/checkpoints/finetune
model=cca-llava-1.5-7b
model_name=${model}
pope_subset=popular

echo "------------- Running for model: $model -------------"

question_file=${llava_root}/playground/data/pope/seem/gqa/gqa_pope_seem_${pope_subset}.json
answer_file=${save_root}/${model_name}_pope_gqa_${pope_subset}.jsonl

if test -e ${answer_file}; then
    python llava/eval/eval_pope.py \
        --question-file ${question_file} \
        --result-file ${answer_file}
else
    python llava/eval/model_vqa_pope.cca.py \
        --model-path ${model_root}/${model} \
        --question-file ${question_file} \
        --image-folder ${img_root} \
        --answers-file ${answer_file}
    python llava/eval/eval_pope.py \
        --question-file ${question_file} \
        --result-file ${answer_file} 
fi