################## VICUNA ##################

PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"

MODALITY_ROUTING_MODE="${MODALITY_ROUTING_MODE:-task}"
EVAL_MODALITY_ROUTING_MODE="${EVAL_MODALITY_ROUTING_MODE:-same}"
ROUTER_HIDDEN_DIM="${ROUTER_HIDDEN_DIM:-32}"
ROUTER_DROPOUT="${ROUTER_DROPOUT:-0.0}"
ROUTER_RESIDUAL_SCALE="${ROUTER_RESIDUAL_SCALE:-1.0}"
ROUTER_LOSS_WEIGHT="${ROUTER_LOSS_WEIGHT:-0.1}"
ROUTER_REPLAY_WEIGHT="${ROUTER_REPLAY_WEIGHT:-0.0}"
ROUTER_REPLAY_SAMPLES="${ROUTER_REPLAY_SAMPLES:-4}"
ROUTER_MIN_COUNT="${ROUTER_MIN_COUNT:-128}"
PREVIOUS_TASK_MODEL_PATH="${PREVIOUS_TASK_MODEL_PATH:-runs/checkpoints/Hyper/UCIT/Task5_llava_lora_ours}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/ckpt/zhaozhuofan/hyper_llava/Hyper/UCIT/Task6_llava_lora_ours}"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29601 llava/train/train_mem_MOE.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 48 --lora_alpha 96 --mm_projector_lr 2e-5 \
    --expert_num 6 \
    --model_name_or_path /data/ckpt/zhaozhuofan/models/llava-v1.5-7b \
    --previous_task_model_path "$PREVIOUS_TASK_MODEL_PATH" \
    --version $PROMPT_VERSION \
    --data_path /data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/train_brief_4w.json \
    --image_folder /data/dataset/zhaozhuofan/UCIT/datasets \
    --vision_tower /data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336 \
    --text_tower /data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336 \
    --cur_task 5 \
    --modality_routing_mode "$MODALITY_ROUTING_MODE" \
    --eval_modality_routing_mode "$EVAL_MODALITY_ROUTING_MODE" \
    --router_hidden_dim "$ROUTER_HIDDEN_DIM" \
    --router_dropout "$ROUTER_DROPOUT" \
    --router_residual_scale "$ROUTER_RESIDUAL_SCALE" \
    --router_loss_weight "$ROUTER_LOSS_WEIGHT" \
    --router_replay_weight "$ROUTER_REPLAY_WEIGHT" \
    --router_replay_samples "$ROUTER_REPLAY_SAMPLES" \
    --router_min_count "$ROUTER_MIN_COUNT" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none



