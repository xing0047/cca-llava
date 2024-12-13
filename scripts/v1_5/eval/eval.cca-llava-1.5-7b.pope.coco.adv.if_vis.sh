img_root=playground/data/coco/val2014
save_root=outputs/pope/coco
model_root=xing0047
model=cca-llava-1.5-7b
model_name=${model}
pope_subset=adversarial

echo "------------- Running for model: $model -------------"

question_file=playground/data/pope/coco/coco_pope_${pope_subset}.json
answer_file=${save_root}/${model_name}_pope_coco_${pope_subset}_if_vis.jsonl
answer_png=${save_root}/${model_name}_pope_coco_${pope_subset}_if_vis.png

if test -e ${answer_file}; then
    python llava/eval/eval_pope.if_vis.py \
        --question-file ${question_file} \
        --result-file ${answer_file} \
        --result-png ${answer_png}
else
    python llava/eval/model_vqa_pope.cca.if_vis.py \
        --model-path ${model_root}/${model} \
        --question-file ${question_file} \
        --image-folder ${img_root} \
        --answers-file ${answer_file}
    python llava/eval/eval_pope.if_vis.py \
        --question-file ${question_file} \
        --result-file ${answer_file} \
        --result-png ${answer_png}
fi