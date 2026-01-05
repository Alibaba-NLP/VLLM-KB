This directory contains the neccessary code for 
  - Training scripts under swift (& Qwen, Deepseek) environment.
  - Code for evaluating the sampled responses using LLM (Qwen-max).

Usage of scripts:
1. Download the training set and put it in `training/` dir
2. E.g., train a `qwen-vl-chat model` for `hard` knowledge boundary identification:
```bash
bash scripts/KB_hard_train_qwen_7b.sh
```

Besides, we provide the prompt and code we use to evaluate the sampled responses from VLLMs in `src/qwen_max_eval_repeat_response.py`. The prompt is shown at the top of the file. Note that you will need a qwen-max dashscope key to use the py script.
