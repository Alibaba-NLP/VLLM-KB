Contains the neccessary files to produce the evaluation results in paper. 
The inference steps are omitted for now due to the retrieval calls sent the internet, and the output is directly upload to the `eval_data` dir.

Note that you will still need to add dashscope key to .env file at the root directory of this repo. to run the evaluation. 

```bash
DASHSCOPE=...
```

E.g., the usage to evaluate the `Mix` data (In `evaluation/`)

```bash
bash scripts/score_mix_all.sh
```
The LLM(qwen-score) metric might be different from the paper and the Accuracy meitric will be identical.