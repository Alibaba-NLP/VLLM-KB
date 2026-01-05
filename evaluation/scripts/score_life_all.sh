lang=ch

checkpoints=("qwen-vl-chat" "qwen-vl-max" "qwen-vl-2" "gpt4-o")


for checkpoint in "${checkpoints[@]}"
do

    results_file_rag=eval_data/life/${checkpoint}_rag_life.jsonl
    results_file_no_rag=eval_data/life/${checkpoint}_norag_life.jsonl

    search_decision_file_soft=eval_data/life/life_oss_1022_soft_KB_model.jsonl
    search_decision_file_hard=eval_data/life/life_oss_1022_hard_KB_model.jsonl
    search_decision_file_human=eval_data/life/life_oss_1022_human.jsonl
    search_decision_file_prompt=eval_data/life/life_oss_prompt_KB.jsonl

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