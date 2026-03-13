output_file = 'vfreshqa_en/norag/vfreshqa_en_deepseek-vl-7b-chat.jsonl'

try:
    g = open(output_file, 'r')
    num_lines = len(g.readlines())
    print(output_file, num_lines, 'lines')
    if num_lines in [737, 715]:
        print(output_file, 'skip.', num_lines)
        g.close()
        exit()
    raise NotImplementedError
except Exception as e:
    print(e)
    pass
# finally:
    # print("Here?")

print("Writing...")
a = 1
print(a)