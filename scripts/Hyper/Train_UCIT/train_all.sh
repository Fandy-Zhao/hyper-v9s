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
set -Eeuo pipefail
export PYTHONPATH="/home/zhaozhuofan/Hyper-LlaVA:${PYTHONPATH:-}"

usage() {
    echo "Usage: bash scripts/Hyper/Train_UCIT/train_all.sh [--output_dir BASE_DIR] [--master_port PORT] [--gpus GPU_IDS]"
    echo "Example: bash scripts/Hyper/Train_UCIT/train_all.sh --output_dir /data/ckpt/zhaozhuofan/hyper_llava/Hyper/UCIT/06_18 --master_port 29611 --gpus 4,5,6,7"
}

OUTPUT_DIR="/data/ckpt/zhaozhuofan/hyper_llava/Hyper/UCIT/06_18"
MASTER_PORT=""
GPU_IDS="0,1,2,3"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --output_dir requires a value" >&2
                usage >&2
                exit 2
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --master_port)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --master_port requires a value" >&2
                usage >&2
                exit 2
            fi
            MASTER_PORT="$2"
            shift 2
            ;;
        --gpus)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --gpus requires a value" >&2
                usage >&2
                exit 2
            fi
            GPU_IDS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$OUTPUT_DIR" ]]; then
    export HYPER_OUTPUT_DIR="${OUTPUT_DIR%/}"
    mkdir -p "$HYPER_OUTPUT_DIR"
    echo "Output base dir: $HYPER_OUTPUT_DIR"
else
    unset HYPER_OUTPUT_DIR
    echo "Output base dir: legacy per-task defaults"
fi


# 记录总流程开始时间
if [[ -z "$MASTER_PORT" ]]; then
    MASTER_PORT=$(/home/zhaozhuofan/miniconda3/envs/hyper/bin/python - <<'PORTPY'
import socket
for port in range(29601, 29999):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("no free master port found in 29601-29998")
PORTPY
)
    echo "MASTER_PORT auto-selected free master port: $MASTER_PORT"
else
    echo "MASTER_PORT: $MASTER_PORT"
fi
export MASTER_PORT
export GPU_IDS
echo "GPU_IDS: $GPU_IDS"

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
    bash /home/zhaozhuofan/Hyper-LlaVA/scripts/Hyper/Train_UCIT/Task${i}.sh
    
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
