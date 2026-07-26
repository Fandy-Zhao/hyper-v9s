#!/bin/bash

# 1. 记录总流程开始时间
TOTAL_START=$(date +%s)
echo "================================================="
echo "🚀 开始执行所有评估测试 | 开始时间: $(date)"
echo "================================================="

bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_imagenet.sh hyper-task6 runs/checkpoints/Hyper/UCIT_IFRCAV/Task6_llava_lora_ours
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_arxivqa.sh hyper-task6 runs/checkpoints/Hyper/UCIT_IFRCAV/Task6_llava_lora_ours
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_vizwiz.sh hyper-task6 runs/checkpoints/Hyper/UCIT_IFRCAV/Task6_llava_lora_ours 
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_iconqa.sh hyper-task6 runs/checkpoints/Hyper/UCIT_IFRCAV/Task6_llava_lora_ours
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_clevr.sh hyper-task6 runs/checkpoints/Hyper/UCIT_IFRCAV/Task6_llava_lora_ours
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_flickr30k.sh hyper-task6 runs/checkpoints/Hyper/UCIT_IFRCAV/Task6_llava_lora_ours 

bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_iconqa.sh hyper-task5 runs/checkpoints/Hyper/UCIT_IFRCAV/Task5_llava_lora_ours 
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_flickr30k.sh hyper-task5 runs/checkpoints/Hyper/UCIT_IFRCAV/Task5_llava_lora_ours 
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_imagenet.sh hyper-task5 runs/checkpoints/Hyper/UCIT_IFRCAV/Task5_llava_lora_ours 
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_clevr.sh hyper-task5 runs/checkpoints/Hyper/UCIT_IFRCAV/Task5_llava_lora_ours 
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_arxivqa.sh hyper-task5 runs/checkpoints/Hyper/UCIT_IFRCAV/Task5_llava_lora_ours 

bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_iconqa.sh hyper-task4 runs/checkpoints/Hyper/UCIT_IFRCAV/Task4_llava_lora_ours
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_flickr30k.sh hyper-task4 runs/checkpoints/Hyper/UCIT_IFRCAV/Task4_llava_lora_ours 
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_imagenet.sh hyper-task4 runs/checkpoints/Hyper/UCIT_IFRCAV/Task4_llava_lora_ours 
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_clevr.sh hyper-task4 runs/checkpoints/Hyper/UCIT_IFRCAV/Task4_llava_lora_ours 

bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_iconqa.sh hyper-task3 runs/checkpoints/Hyper/UCIT_IFRCAV/Task3_llava_lora_ours
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_flickr30k.sh hyper-task3 runs/checkpoints/Hyper/UCIT_IFRCAV/Task3_llava_lora_ours
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_imagenet.sh hyper-task3 runs/checkpoints/Hyper/UCIT_IFRCAV/Task3_llava_lora_ours

bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_iconqa.sh hyper-task2 runs/checkpoints/Hyper/UCIT_IFRCAV/Task2_llava_lora_ours
bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_flickr30k.sh hyper-task2 runs/checkpoints/Hyper/UCIT_IFRCAV/Task2_llava_lora_ours

bash scripts/Hyper/Eval_UCIT_IFRCAV/eval_iconqa.sh hyper-task1 runs/checkpoints/Hyper/UCIT_IFRCAV/Task1_llava_lora_ours

# 3. 记录总流程结束时间并计算耗时
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

TOTAL_H=$((TOTAL_DURATION / 3600))
TOTAL_M=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_S=$((TOTAL_DURATION % 60))

echo "================================================="
echo "🎉 所有评估测试执行完毕 | 结束时间: $(date)"
echo "⏱️  测试总计耗时: ${TOTAL_H}小时 ${TOTAL_M}分钟 ${TOTAL_S}秒"
echo "================================================="