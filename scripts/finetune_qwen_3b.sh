#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_PROJECT="hallucination"

MODEL_PATH=checkpoints/Llama-3.2-3B-Instruct
DATA_PATH=playground/LLaVA-Finetune
VISION_TOWER="convnext;eva;sam;vary;dino"

LLM_VERSION_CLEAN="llama3"
VISION_TOWER_CLEAN="ccesvd"
MM_VERSION="v3"
BASE_RUN_NAME="dromo-${VISION_TOWER_CLEAN}-${LLM_VERSION_CLEAN}-${MM_VERSION}-finetune"

deepspeed train_visionweaver.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_PATH \
    --version llava_llama_3 \
    --data_path $DATA_PATH/llava_v1_5_mix665k.json \
    --image_folder $DATA_PATH \
    --vision_tower $VISION_TOWER \
    --mm_version $MM_VERSION \
    --pretrain_mm_mlp_adapter outputs/pretrain_outputs/dromo-ccesvd-llama3-v4-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_tunable_parts "dromo_stage_2" \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir outputs/finetune_outputs/$BASE_RUN_NAME \
    --run_name $BASE_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --verbose_logging \
    --attn_implementation sdpa