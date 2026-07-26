"""作用：提供数据集或评测结果格式转换脚本，服务训练、评测或提交流程。"""

import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()

all_answers = []
for line_idx, line in enumerate(open(args.src)):
    res = json.loads(line)
    question_id = res['question_id']
    text = res['text'].rstrip('.').lower()
    all_answers.append({"questionId": question_id, "prediction": text})

with open(args.dst, 'w') as f:
    json.dump(all_answers, f)
