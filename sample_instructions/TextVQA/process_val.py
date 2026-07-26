"""作用：处理示例 instruction 数据，生成训练或验证所需的标注格式。"""

import json

# 1. 读取 JSON 文件
input_file = "/mnt/cache/xlpr_sharedata/MLLM_CL/CoIN_instructions/TextVQA/val.json"  # 替换为你的 JSON 文件路径
with open(input_file, "r") as f:
    data = json.load(f)

# 2. 遍历每个条目，修改 question_id
for item in data:
    # 获取 image 路径中的最后一个部分（即文件名）
    image_path = item["image"]
    image_name = image_path.split("/")[-1]  # 获取文件名部分
    image_name_without_extension = image_name.split(".")[0]  # 去掉 .jpg 后缀
    
    # 将 question_id 修改为 image_name_without_extension
    item["question_id"] = image_name_without_extension

# 3. 将修改后的数据保存回 JSON 文件
output_file = "val.json"  # 替换为你想保存的文件路径
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"数据已修改并保存到 {output_file}")