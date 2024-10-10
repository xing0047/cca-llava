amber_root=playground/data/amber
img_root=${amber_root}/image
save_root=output/amber
model_root=xing0047
model=cca-llava-1.5-7b
model_name=${model}

echo "------------- Running for model: $model -------------"

question_file_dis=${amber_root}/query/query_discriminative.json
answer_file_dis=${save_root}/${model_name}_amber_discriminative_ans.jsonl

question_file_gen=${amber_root}/query/query_generative.json
answer_file_gen=${save_root}/${model_name}_amber_generative_ans.jsonl

echo "------------- AMBER Discriminative Evaluation -------------"

if test -e ${answer_file_dis}; then
    python llava/eval/eval_amber.py \
        --inference_data ${answer_file_dis} \
        --evaluation_type d \
        --word_association ${amber_root}/relation.json \
        --safe_words ${amber_root}/safe_words.txt \
        --annotation ${amber_root}/annotations.json \
        --metrics ${amber_root}/metrics.txt
else
    python llava/eval/model_vqa_amber.cca.py \
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
        --word_association ${amber_root}/relation.json \
        --safe_words ${amber_root}/safe_words.txt \
        --annotation ${amber_root}/annotations.json \
        --metrics ${amber_root}/metrics.txt
fi

echo "------------- AMBER Generative Evaluation -------------"

if test -e ${answer_file_gen}; then
    python llava/eval/eval_amber.py \
        --inference_data ${answer_file_gen} \
        --evaluation_type g \
        --word_association ${amber_root}/relation.json \
        --safe_words ${amber_root}/safe_words.txt \
        --annotation ${amber_root}/annotations.json \
        --metrics ${amber_root}/metrics.txt
else
    python llava/eval/model_vqa_amber.cca.py \
        --model-path ${model_root}/${model} \
        --question-file ${question_file_gen} \
        --image-folder ${img_root} \
        --answers-file ${answer_file_gen} \
        --temperature 0 \
        --conv-mode vicuna_v1

    python llava/eval/eval_amber.py \
        --inference_data ${answer_file_gen} \
        --evaluation_type g \
        --word_association ${amber_root}/relation.json \
        --safe_words ${amber_root}/safe_words.txt \
        --annotation ${amber_root}/annotations.json \
        --metrics ${amber_root}/metrics.txt

fi
