img_root=playground/data/coco/val2014
annotation_root=playground/data/coco/annotations
save_root=outputs/chair
model_root=liuhaotian
model=llava-v1.5-7b
model_name=${model}

echo "------------- Running for model: $model -------------"

mkdir -p outputs/chair
question_file=playground/data/coco_chair.jsonl
answer_file=outputs/chair/${model_name}_coco_chair_long_ans.jsonl

if test -e ${answer_file}; then
    python llava/eval/eval_chair.py \
        --cap_file ${answer_file} \
        --image_id_key image_id \
        --caption_key caption \
        --coco_path ${annotation_root}
else
    python llava/eval/model_vqa_chair.py \
        --model-path ${model_root}/${model} \
        --question-file ${question_file} \
        --image-folder ${img_root} \
        --max_new_tokens 512 \
        --answers-file ${answer_file}
    python llava/eval/eval_chair.py \
        --cap_file ${answer_file} \
        --image_id_key image_id \
        --caption_key caption \
        --coco_path  ${annotation_root}
fi
