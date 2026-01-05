import json
import os

def extract_questions(input_file, output_file):

    # Initialize the top-level JSON structure
    questions_data = {
        "license": {
        "url": "http://localhost.com",
        "name": "vfreshqa"
        },
        "task_type": "Open-Ended",
        "data_subtype": "ch",
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "v1.0 of the vfreshqa dataset."
            },
        "data_type": "vfreshqa",
        "questions": []
    }

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                entry = {
                    "image_id": data["question_id"],
                    "question": data["origin_question"],
                    "question_id": data["question_id"]
                }

                # Append entry to questions list
                questions_data["questions"].append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {line}")
                continue
            except KeyError as e:
                print(f"KeyError: {e} in line: {line}")
                continue

    # Write the entire JSON structure to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=2)

    print(f"Extracted questions saved in {output_file}")
    return output_file




def generate_and_reorder_annotation_file(dataset_file, output_file):
    # 初始化顶级 JSON 结构
    annotation_data = {
        "license": {"url": "http://localhost.com", "name": "Vfreshqa_en"},
        "data_subtype": "Vfreshqa_en",
        "question_types": {"1": "All"},
        "info": {"year": 2024, "version": "1.0", "description": "v1.0 of the Vfreshqa_en dataset."},
        "data_type": "Vfreshqa_en",
        "confidence": 3,
        "annotations": []
    }

    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    ground_truth = data.get("ground_truth", [])
                    if not isinstance(ground_truth, list):
                        ground_truth = ground_truth.split('，')
                    # breakpoint()

                    # 为每个 ground_truth 条目创建一个独立的 answer 对象
                    answers = []
                    for i, answer_text in enumerate(ground_truth):
                        answer = {
                            "answer_id": i + 1,
                            "answer": answer_text,
                            "answer_confidence": "yes",
                            "raw_answer": answer_text
                        }
                        answers.append(answer)

                    entry = {
                        "question_id": index + 1,  # 按顺序分配 question_id
                        "image_id": index + 1,     # 按顺序分配 image_id
                        "question_type": data.get("question_type", "1"),
                        "answer_type": "1",
                        "answers": answers
                    }

                    # 将 entry 添加到 annotations 列表中
                    annotation_data["annotations"].append(entry)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line: {line}")
                    continue

        # 将整个 JSON 结构写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)

        print(f"Generated and reordered annotations saved in {output_file}")
        return output_file

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing file: {e}")



def get_filename_without_extension(file_path):
    # 提取文件名（包括后缀）
    filename_with_ext = os.path.basename(file_path)
    # 去掉文件后缀
    filename_without_ext, _ = os.path.splitext(filename_with_ext)
    return filename_without_ext