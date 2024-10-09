
root=/home/yiheng/code
llava_root=${root}/cca-llava-github
img_root=/media/drive_16TB/data/AMBER/image
annotation_root=/media/drive_16TB/data/coco/annotations_trainval2014
save_root=${llava_root}/output
model_root=/media/drive_16TB/huggingface
model=llava-v1.5-7b
model_name=${model}

echo "------------- Running for model: $model -------------"

question_file_dis=${llava_root}/amber_data/query/query_discriminative.json
answer_file_dis=${save_root}/${model_name}_amber_discriminative_ans.jsonl

question_file_gen=${llava_root}/amber_data/query/query_generative.json
answer_file_gen=${save_root}/${model_name}_amber_generative_ans.jsonl

echo "------------- AMBER Discriminative Evaluation -------------"

if test -e ${answer_file_dis}; then
    python llava/eval/eval_amber.py \
        --inference_data ${answer_file_dis} \
        --evaluation_type d \
        --word_association ${llava_root}/amber_data/relation.json \
        --safe_words ${llava_root}/amber_data/safe_words.txt \
        --annotation ${llava_root}/amber_data/annotations.json \
        --metrics ${llava_root}/amber_data/metrics.txt
else
    python llava/eval/model_vqa_amber.py \
        --model-path ${model_root}/${model} \
        --question-file ${question_file_dis} \
        --image-folder ${img_root} \
        --answers-file ${answer_file_dis} \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --single-word-answer 

    python llava/eval/eval_amber.py \
        --inference_data ${answer_file_dis} \
        --evaluation_type d \
        --word_association ${llava_root}/amber_data/relation.json \
        --safe_words ${llava_root}/amber_data/safe_words.txt \
        --annotation ${llava_root}/amber_data/annotations.json \
        --metrics ${llava_root}/amber_data/metrics.txt

fi

echo "------------- AMBER Generative Evaluation -------------"

if test -e ${answer_file_gen}; then
    python llava/eval/eval_amber.py \
        --inference_data ${answer_file_gen} \
        --evaluation_type g \
        --word_association ${llava_root}/amber_data/relation.json \
        --safe_words ${llava_root}/amber_data/safe_words.txt \
        --annotation ${llava_root}/amber_data/annotations.json \
        --metrics ${llava_root}/amber_data/metrics.txt
else
    python llava/eval/model_vqa_amber.py \
        --model-path ${model_root}/${model} \
        --question-file ${question_file_gen} \
        --image-folder ${img_root} \
        --answers-file ${answer_file_gen} \
        --temperature 0 \
        --conv-mode vicuna_v1

    python llava/eval/eval_amber.py \
        --inference_data ${answer_file_gen} \
        --evaluation_type g \
        --word_association ${llava_root}/amber_data/relation.json \
        --safe_words ${llava_root}/amber_data/safe_words.txt \
        --annotation ${llava_root}/amber_data/annotations.json \
        --metrics ${llava_root}/amber_data/metrics.txt

fi
