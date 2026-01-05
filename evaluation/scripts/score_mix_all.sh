lang=ch

checkpoints=("qwen-vl-chat" "qwen-vl-max" "qwen-vl-2" "gpt4-o")


for checkpoint in "${checkpoints[@]}"
do

    results_file_rag=eval_data/mix/${checkpoint}_rag_mix.jsonl
    results_file_no_rag=eval_data/mix/${checkpoint}_norag_mix.jsonl

    search_decision_file_soft=eval_data/mix/mix_1129_soft.jsonl
    search_decision_file_hard=eval_data/mix/mix_1129_hard.jsonl
    search_decision_file_human=eval_data/mix/mix_1129_human.jsonl
    search_decision_file_prompt=eval_data/mix/mix_prompt_KB.jsonl

    echo -e "\n#######${checkpoint}#######"
    python src/score_with_search_decision_all.py \
        --results_file_rag $results_file_rag \
        --results_file_no_rag $results_file_no_rag \
        --search_decision_file_hard $search_decision_file_hard \
        --search_decision_file_soft $search_decision_file_soft \
        --search_decision_file_human $search_decision_file_human \
        --search_decision_file_prompt $search_decision_file_prompt \
        --lang $lang
    echo -e "\n#######${checkpoint} END#######"

done