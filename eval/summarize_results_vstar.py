import os
import json
import csv

# ======== 你需要修改的部分 ========
# jsonl_list = [
#     "/abs/path/to/result_1.jsonl",
#     "/abs/path/to/result_2.jsonl",
#     # ... 填满所有 13 个 jsonl 文件
# ]
jsonl_dir = "/root/autodl-tmp/code/mllms_fine_grained/eval/answers/vstar/llava-v1.5-7b"
jsonl_list = os.listdir(jsonl_dir)
jsonl_list = [os.path.join(jsonl_dir, p) for p in jsonl_list]
print(jsonl_list)

output_csv = os.path.join(jsonl_dir, "results_vstar.csv")
# ==================================

# 解析函数
def load_jsonl_to_dict(jsonl_path):
    """返回 dict: {input_image_basename: (test_type, output)}"""
    result = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            img_name = os.path.splitext(os.path.basename(data["input_image"]))[0]
            test_type = data["test_type"]
            output_flag = 1 if data["output"] == 0 else 0
            result[img_name] = (test_type, output_flag)
    return result

# 读取全部 jsonl
all_results = [load_jsonl_to_dict(p) for p in jsonl_list]
json_names = [os.path.splitext(os.path.basename(p))[0] for p in jsonl_list]

# 聚合所有 input_image
all_images = sorted(set().union(*[set(r.keys()) for r in all_results]))

# 写入 CSV
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # 表头
    header = ["test_type", "input_image"] + json_names
    writer.writerow(header)

    for img in all_images:
        # 取第一个非空 test_type
        test_type = None
        row = []
        for res in all_results:
            if img in res:
                test_type = res[img][0]
                break

        row = [test_type if test_type else "", img]
        for res in all_results:
            row.append(res[img][1] if img in res else "")
        writer.writerow(row)

print(f"✅ Done! CSV saved to: {os.path.abspath(output_csv)}")
