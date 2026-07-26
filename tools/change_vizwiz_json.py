#!/usr/bin/env python3
"""作用：清洗 VizWiz 标注 JSON 中重复的图片扩展名，修复数据集中 image 字段的路径格式。"""

import json
import re

def fix_json_images(json_file):
    """作用：遍历 JSON 标注列表，修复 image 字段中重复的扩展名并原地写回文件。"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    fixed_count = 0
    for item in data:
        if 'image' in item:
            original = item['image']
            # 修复 .jpg.jpg 等重复扩展名
            fixed = re.sub(r'\.([a-zA-Z0-9]+)\.\1$', r'.\1', original)
            if fixed != original:
                item['image'] = fixed
                fixed_count += 1
    
    # 保存文件（覆盖原文件）
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"修复完成！共修复 {fixed_count} 个文件名")

if __name__ == "__main__":
    fix_json_images("/home/aigc_account_2/CODE/HiDe-LLaVA/instructions/VizWiz/val.json")