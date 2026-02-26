import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gt_files", type=str, default="data/POPE/coco/coco_pope_popular.json"
)
parser.add_argument(
    "--gen_files",
    type=str,
    default="answer_files_POPE/llava15_coco_pope_popular_answers_no_cd.jsonl",
)
parser.add_argument(
    "--output",
    type=str,
    default="",
)
# python eval/eval_pope.py --gt_files /root/nolan/VCD_c1/experiments/data/POPE/coco/coco_pope_random.json --gen_file /root/nolan/VCD_c1/experiments/output/alpha/13b/a_1.0/llava15/llava15_coco_pope_random_answers_cd_seed55_p0.8.jsonl

args = parser.parse_args()

# open ground truth answers
gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), "r")]

# open generated answers
gen_files = [json.loads(q) for q in open(os.path.expanduser(args.gen_files), "r")]

# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
total_questions = len(gt_files)
yes_answers = 0

# compare answers
for index, line in enumerate(gt_files):
    idx = line["question_id"]
    gt_answer = line["label"]
    assert idx == gen_files[index]["question_id"]
    gen_answer = gen_files[index]["text"]
    # convert to lowercase
    gt_answer = gt_answer.lower()
    gen_answer = gen_answer.lower()
    # strip
    gt_answer = gt_answer.strip()
    gen_answer = gen_answer.strip()
    # pos = 'yes', neg = 'no'
    if gt_answer == "yes":
        if "yes" in gen_answer:
            true_pos += 1
            yes_answers += 1
        else:
            false_neg += 1
    elif gt_answer == "no":
        if "no" in gen_answer:
            true_neg += 1
        else:
            yes_answers += 1
            false_pos += 1
    else:
        print(f"Warning: unknown gt_answer: {gt_answer}")
        unknown += 1
# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / total_questions
yes_proportion = yes_answers / total_questions
unknown_prop = unknown / total_questions
# report results
# 格式化输出内容
output = f"{accuracy*100:0.2f}, {precision*100:0.2f}, {recall*100:0.2f}, {f1*100:0.2f}"

# 打印到控制台
print(output)

if not os.path.exists(args.output):
    os.makedirs(args.output)
filename = f"{args.output}/results.csv"

# 打开文件进行写入
with open(filename, "a") as file:  # 使用 'a' 模式以追加内容
    file.write(output + "\n")  # 写入数据并添加换行符以便下次写入时自动换行
