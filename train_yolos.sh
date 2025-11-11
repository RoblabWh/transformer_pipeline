#!/bin/bash
# microsoft/conditional-detr-resnet-50
# SenseTime/deformable-detr 
#        --num_queries 100 \ check where to add
#         this parameter unfortunatly produces errors in the framework
# Test other warmup ratios from 1% to 10% (1e-2 to 1e-1)
# Maybe decrease LR to 1e-5?
# https://huggingface.co/docs/autotrain/object_detection_params
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/object-detection/README.md
# https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py

# Force use of GPU 0 to avoid nn.DataParallel issue
export CUDA_VISIBLE_DEVICES=0

# NVIDIA RTX 5090
# --per_device_train_batch_size 3 \

# NVIDIA RTX 6000 Ada
# --per_device_train_batch_size 4 \

# https://huggingface.co/docs/transformers/model_doc/yolos

python run_object_detection.py \
    --model_name_or_path hustvl/yolos-base \
    --dataset_name RoblabWhGe/FireDetDataset \
    --do_train true \
    --do_eval true \
    --output_dir yolos-base-sensetime-hustvl-firedetv11 \
    --num_train_epochs 300 \
    --image_square_size 1333 \
    --fp16 true \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --warmup_ratio 5e-2 \
    --max_grad_norm 1e-1 \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 2 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last true \
    --remove_unused_columns false \
    --eval_do_concat_batches false \
    --ignore_mismatched_sizes true \
    --metric_for_best_model eval_map \
    --greater_is_better true \
    --load_best_model_at_end true \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub false \
    --push_to_hub_model_id detr-finetuned-cppe-5-10k-steps \
    --hub_strategy end \
    --seed 1337