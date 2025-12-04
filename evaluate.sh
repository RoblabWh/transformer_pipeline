#!/bin/bash
# microsoft/conditional-detr-resnet-50
# SenseTime/deformable-detr 
#        --num_queries 100 \ check where to add
# https://huggingface.co/docs/autotrain/object_detection_params
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/object-detection/README.md

# Force use of GPU 0 to avoid nn.DataParallel issue
export CUDA_VISIBLE_DEVICES=0

python run_object_detection.py \
    --model_name_or_path models/final/rescuedet-deformable-detr \
    --dataset_name /run/media/niklas/EDDF-B044/FireDetDataset \
    --do_train false \
    --do_eval true \
    --num_train_epochs 50 \
    --image_square_size 1333 \
    --fp16 false \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --warmup_ratio 5e-2 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 2 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last true \
    --remove_unused_columns false \
    --eval_do_concat_batches false \
    --ignore_mismatched_sizes true \
    --metric_for_best_model eval_map \
    --greater_is_better true \
    --logging_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub false \
    --push_to_hub_model_id detr-finetuned-cppe-5-10k-steps \
    --hub_strategy end \
    --seed 1337
