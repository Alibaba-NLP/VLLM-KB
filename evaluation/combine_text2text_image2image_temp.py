import ujson as json
import copy


g = open('vfreshqa_ch/both/vfreshqa_ch_both.jsonl', 'w')

with open('vfreshqa_ch/image2image/vfreshqa_ch_image2image_bing.jsonl') as f1, open('vfreshqa_ch/text2text/vfreshqa_ch_text2text_google.jsonl') as f2:
    lines_image = f1.readlines()
    lines_text = f2.readlines()

    for line_i, line_t in zip(lines_image, lines_text):
        data_i = json.loads(line_i)
        data_t = json.loads(line_t)
        
        new_data = data_i
        new_data['search_data_text2text'] = data_t['search_data']
        new_data['search_data_image2image'] = copy.deepcopy(data_i['search_data'])

        del new_data['search_data']
        # breakpoint()
        g.write(
            json.dumps(new_data, ensure_ascii=False)+'\n'
        )
g.close()

g = open('vfreshqa_en/both/vfreshqa_en_both.jsonl', 'w')

with open('vfreshqa_en/image2image/vfreshqa_en_image2image_bing.jsonl') as f1, open('vfreshqa_en/text2text/vfreshqa_en_text2text_google.jsonl') as f2:
    lines_image = f1.readlines()
    lines_text = f2.readlines()

    for line_i, line_t in zip(lines_image, lines_text):
        data_i = json.loads(line_i)
        data_t = json.loads(line_t)
        
        new_data = data_i
        new_data['search_data_text2text'] = data_t['search_data']
        new_data['search_data_image2image'] = copy.deepcopy(data_i['search_data'])

        del new_data['search_data']
    
        g.write(
            json.dumps(new_data, ensure_ascii=False)+'\n'
        )
g.close()