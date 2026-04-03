import ujson as json
import os
from tqdm import tqdm
import copy
from prompt_template import PREFIX_MODEL, PREFIX_RAG, IN, IMAGE_OUT, QUERY_OUT, BOTH_OUT
import random
random.seed(42)


def _split_train_dev(data_list, dir='/mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_data/tmp'):

    os.makedirs(dir, exist_ok=True)
    train_path = os.path.join(dir, 'train.jsonl')
    dev_path = os.path.join(dir, 'dev.jsonl')
    
    random.shuffle(data_list)
    split_index = int(0.15 * len(data_list))  # 20% for dev, 80% for train
    train_fold = data_list[split_index:]
    dev_fold = data_list[:split_index]

    with open(train_path, 'w') as g:
        for i in train_fold:
            g.write(json.dumps(i, ensure_ascii=False)+'\n')
    with open(dev_path, 'w') as g:
        for i in dev_fold:
            g.write(json.dumps(i, ensure_ascii=False)+'\n')

    print('Saved to', dir)


raw_files = [
    '/mnt/nas-alinlp/zhili/RAG-Evaluation/vl_boundary/model_boundary/data_storage/final_data/checked/en/infoseek.jsonl',
    '/mnt/nas-alinlp/zhili/RAG-Evaluation/vl_boundary/model_boundary/data_storage/final_data/checked/en/vqav2.jsonl',
    '/mnt/nas-alinlp/zhili/RAG-Evaluation/vl_boundary/model_boundary/data_storage/final_data/checked/zh/wanwu_qa_vlkb_small.jsonl',
]

template = {"messages": [{"role": "user", "content": None}, {"role": "assistant", "content": None}], "images": []}
data_stat = {'in': 0, 'image_out': 0, 'query_out': 0, 'query_image_out': 0, 'query_image_in': 0}

all_data = []
for file in raw_files:
    print(file)
    with open(file) as f:
        lines = list(f.readlines())
        # if 'vqa' in file:
        #     random.shuffle(lines)
        #     lines = lines[:2000]
        
        # if 'infoseek' in file:
        #     random.shuffle(lines)
        #     lines = lines[:10000]
        
        # if 'animal' in file or 'plant' in file or 'person' in file or 'wanwu' in file:
        #     random.shuffle(lines)
        #     lines = lines[:10000]

        for line in tqdm(lines):
            data = json.loads(line)
            # breakpoint()
            query = data.get('question', False) or data.get('query', False)
            image = data.get('image', False) or data.get('image_url', False)
            assert query, print('question not specified')
            assert image, print('image not specified')

            new_data = copy.deepcopy(template)
            new_data['messages'][0]['content'] = PREFIX_MODEL.format(query=query)
            new_data['images'].append(image)

            label_model_boundary = data['label_model_boundary']
            data_stat[label_model_boundary] += 1
            if label_model_boundary == 'in':
                new_data['messages'][1]['content'] = IN
            elif label_model_boundary == 'image_out':
                new_data['messages'][1]['content'] = IMAGE_OUT
            elif label_model_boundary == 'query_out':
                new_data['messages'][1]['content'] = QUERY_OUT
            elif label_model_boundary == 'query_image_out':
                new_data['messages'][1]['content'] = BOTH_OUT
            elif label_model_boundary == 'query_image_in':
                continue
            else:
                print('Wrong label_model_boundary:', label_model_boundary)
            
            all_data.append(new_data)


random.shuffle(all_data)
all_data_even = []

even_data_stat = {'in': 0, 'image_out': 0, 'query_out': 0, 'query_image_out': 0, 'query_image_in': 0}
max_each = 30000

for i in tqdm(all_data, ncols=100):
    if i['messages'][1]['content'] == IN:
        if even_data_stat['in'] < max_each:
            all_data_even.append(i)
            even_data_stat['in'] += 1
    elif i['messages'][1]['content'] == IMAGE_OUT:
        if even_data_stat['image_out'] < max_each:
            all_data_even.append(i)
            even_data_stat['image_out'] += 1
    elif i['messages'][1]['content'] == QUERY_OUT:
        if even_data_stat['query_out'] < max_each:
            all_data_even.append(i)
            even_data_stat['query_out'] += 1
    elif i['messages'][1]['content'] == QUERY_OUT:
        if even_data_stat['query_out'] < max_each:
            all_data_even.append(i)
            even_data_stat['query_out'] += 1
    elif i['messages'][1]['content'] == BOTH_OUT:
        if even_data_stat['query_image_out'] < max_each:
            all_data_even.append(i)
            even_data_stat['query_image_out'] += 1


# _split_train_dev(all_data_even, dir='/mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_data/0526_more')
_split_train_dev(all_data_even)
print(f'All data (un-processed): {data_stat}')
print(f'Training data: {even_data_stat}')