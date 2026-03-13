import os


langs = ['ch', 'en']
model_names = [
    'chatgpt-4o-latest',
    'deepseek-vl-7b-chat',
    'deepseek-vl2',
    'qwen25-vl-7b',
    'qwen-vl-max'
]

rag_settings = [
    'norag',
    'image2image',
    'both'
]

for lang in langs:
    for model_name in model_names:
        for rag in rag_settings:
            if rag == 'norag':
                path = f'/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_{lang}/{rag}/vfreshqa_{lang}_{model_name}.jsonl'
                print(path)
                assert os.path.exists(path), print(path, 'Not exists')
            elif rag == 'image2image':
                path = f'/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_{lang}/{rag}/vfreshqa_{lang}_{rag}_bing_{model_name}.jsonl'
                print(path)
                assert os.path.exists(path), print(path, 'Not exists')
            elif rag == 'both':
                path = f'/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_{lang}/{rag}/vfreshqa_{lang}_{rag}_{model_name}.jsonl'
                print(path)
                assert os.path.exists(path), print(path, 'Not exists')

