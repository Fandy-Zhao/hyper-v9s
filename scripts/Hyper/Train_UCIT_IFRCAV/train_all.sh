# pip install -e .
# sh /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task1.sh
# pip install -e .
# sh /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task2.sh
# pip install -e .
# sh /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task3.sh
# pip install -e .
# sh /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task4.sh
# pip install -e .
# sh /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task5.sh
# pip install -e .
# sh /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task6.sh

# bash /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task1.sh
# bash /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task2.sh
# bash /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task3.sh
# bash /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task4.sh
# bash /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task5.sh
# bash /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task6.sh

#!/bin/bash

# 记录总流程开始时间
TOTAL_START=$(date +%s)
echo "================================================="
echo "🚀 开始持续学习流程 | 开始时间: $(date)"
echo "================================================="

# 循环执行 Task 1 到 Task 6
for i in {1..6}
do
    TASK_START=$(date +%s)
    echo "-------------------------------------------------"
    echo "⏳ [Task $i] 正在执行..."
    
    # 执行你的任务脚本
    bash /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT_IFRCAV/Task${i}.sh
    
    TASK_END=$(date +%s)
    TASK_DURATION=$((TASK_END - TASK_START))
    
    # 计算时分秒
    TASK_H=$((TASK_DURATION / 3600))
    TASK_M=$(((TASK_DURATION % 3600) / 60))
    TASK_S=$((TASK_DURATION % 60))
    
    echo "✅ [Task $i] 执行完毕! 耗时: ${TASK_H}小时 ${TASK_M}分钟 ${TASK_S}秒"
done

# 记录总流程结束时间
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

TOTAL_H=$((TOTAL_DURATION / 3600))
TOTAL_M=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_S=$((TOTAL_DURATION % 60))

echo "================================================="
echo "🎉 所有任务执行完毕 | 结束时间: $(date)"
echo "⏱️  总计耗时: ${TOTAL_H}小时 ${TOTAL_M}分钟 ${TOTAL_S}秒"
echo "================================================="