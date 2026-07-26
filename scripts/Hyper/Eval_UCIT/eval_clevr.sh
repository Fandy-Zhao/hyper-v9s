#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
# IDX=0

if [ ! -n "$1" ] ;then
    STAGE='hyper'
else
    STAGE=$1
fi

MODELPATH=$2
RESULT_ROOT="${3:-runs/results/UCIT/each_dataset}"

EVAL_ROUTING_ARG=()
if [ -n "${EVAL_MODALITY_ROUTING_MODE:-}" ]; then
    EVAL_ROUTING_ARG=(--eval-modality-routing-mode "$EVAL_MODALITY_ROUTING_MODE")
fi


RESULT_DIR="$RESULT_ROOT/CLEVR-Math"
mkdir -p "$RESULT_DIR/$STAGE"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_answer \
        --model-path $MODELPATH \
        --model-base /data/ckpt/zhaozhuofan/models/llava-v1.5-7b \
        --question-file /data/dataset/zhaozhuofan/UCIT/instructions/CLEVR/test_3000.json \
        --image-folder /data/dataset/zhaozhuofan/UCIT/datasets \
        --text-tower /data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336 \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        ${EVAL_ROUTING_ARG[@]} \
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

python -m llava.eval.eval_deepseek_r1 \
    --annotation-file /data/dataset/zhaozhuofan/UCIT/instructions/CLEVR/test_3000.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \


