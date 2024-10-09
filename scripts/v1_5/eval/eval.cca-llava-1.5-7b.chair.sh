
root=/home/xingyun/xingy/cca-llava
llava_root=${root}/cca-llava
img_root=/media/drive_16TB/data/coco/val2014
annotation_root=/media/drive_16TB/data/coco/annotations_trainval2014
save_root=${llava_root}/outputs/chair
model_root=/media/drive_16TB/huggingface
model=cca-llava-1.5-7b
model_name=${model}

echo "------------- Running for model: $model -------------"

question_file=/home/xingyun/xingy/cca-llava/chair_question.json
answer_file=${save_root}/${model_name}_chair.jsonl

if test -e ${answer_file}; then
    python llava/eval/eval_chair.py \
        --cap_file ${answer_file} \
        --image_id_key image_id \
        --caption_key caption \
        --coco_path  ${annotation_root}
else
    python llava/eval/model_vqa_chair.cca.py \
        --model-path ${model_root}/${model} \
        --question-file ${question_file} \
        --image-folder ${img_root} \
        --max_new_tokens 64 \
        --answers-file ${answer_file}
    python llava/eval/eval_chair.py \
        --cap_file ${answer_file} \
        --image_id_key image_id \
        --caption_key caption \
        --coco_path  ${annotation_root}

fi