#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" # 
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='MoELoRA'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='/mnt/haiyangguo/mywork/FCIT/CoIN/checkpoints/LLaVA/FCIT/multi_task/multask_llava_lora_ours/llava_lora_MoELoRA_epoch_9'
else
    MODELPATH=$2
fi


RESULT_DIR="./runs/results/CoIN/each_dataset/GQA"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_others \
        --model-path $MODELPATH \
        --model-base /data/ckpt/zhaozhuofan/models/llava-v1.5-7b \
        --question-file instructions/GQA/test.json \
        --image-folder /home/zhangyanqin/DATA/CoIN \
        --text-tower /data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336 \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ./scripts/convert_gqa_for_eval.py \
    --src $output_file \
    --dst $RESULT_DIR/$STAGE/testdev_balanced_predictions.json

python -m llava.eval.eval_gqa \
    --tier testdev_balanced \
    --path $RESULT_DIR/$STAGE \
    --question-dir instructions/GQA \
    --output-dir $RESULT_DIR/$STAGE
