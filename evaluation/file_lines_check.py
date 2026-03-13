import os
import json


# 定义目录及其要求的行数
DIR_LINE_REQUIREMENTS = {
    'lifevqa': 149,
    'qwenvqa': 500,
    'vfreshqa_ch': 737,
    'vfreshqa_en': 715,
    'visual7w': 574,
    'nocaps': 500 
}

def check_file_lines(filepath, required_lines):
    if not filepath.endswith('jsonl'):
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
            if line_count != required_lines:
                print(f"  - Line count mismatch: {filepath} {line_count}/{required_lines}")

    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        # return False

def check_none_response(filepath):
    if not filepath.endswith('jsonl'):
        return
    threshold_ratio = 0.35
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = list(f.readlines())
            if 'response' in json.loads(lines[0]):
                none_count = sum(1 for line in lines if json.loads(line)['response'] == '')
                ratio = none_count / len(lines)
            
                if ratio >= threshold_ratio:
                    print('*'*100)
                    print(f"None response warning: {filepath}, {none_count}/{len(lines)}")
                    print('*'*100)

    except Exception as e:
        print(f"Error in check_none_response {filepath}: {e}")
        return 

def check_invalid_score(filepath):
    if not filepath.endswith('jsonl'):
        return
    threshold_ratio = 0.10
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = list(f.readlines())

            if 'qwen_max_score' in json.loads(lines[0]):
                invalide_score_cnt = sum(1 for line in lines if json.loads(line)['qwen_max_score'] == -1)
                ratio = invalide_score_cnt / len(lines)
            
                if ratio >= threshold_ratio:
                    print('$'*100)
                    print(f"Invalid score warning: {filepath}, {invalide_score_cnt}/{len(lines)}")
                    print('$'*100)

    except Exception as e:
        print(f"Error in check_invalid_score {filepath}: {e}")
        return 

def check_directory(directory, required_lines):
    """递归检查目录下的所有文件行数"""
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if os.path.isfile(filepath):  # 确保是文件而不是其他类型
                check_file_lines(filepath, required_lines)
                check_none_response(filepath)
                check_invalid_score(filepath)

def main():
    # 检查当前目录下的各个子目录
    for dir_name, required_lines in DIR_LINE_REQUIREMENTS.items():
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print(f"Checking directory '{dir_name}' for files with {required_lines} lines...")
            check_directory(dir_name, required_lines)
        else:
            print(f"Directory '{dir_name}' does not exist or is not a directory")

if __name__ == "__main__":
    main()