#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
# IDX=0

if [ ! -n "$1" ] ;then
    STAGE='hide'
else
    STAGE=$1
fi

MODELPATH=$2


RESULT_DIR="runs/results/UCIT_IFRCAV/each_dataset/Flickr30k"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_answer \
        --model-path $MODELPATH \
        --model-base /data/ckpt/zhaozhuofan/models/llava-v1.5-7b \
        --question-file /data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/test_3000.json \
        --image-folder /data/dataset/zhaozhuofan/UCIT/datasets \
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

python -m llava.eval.eval_caption \
    --annotation-file /data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/val_coco_type_3000.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \
