#!/bin/bash
# microsoft/conditional-detr-resnet-50
# SenseTime/deformable-detr 
#        --num_queries 100 \ check where to add
# https://huggingface.co/docs/autotrain/object_detection_params
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/object-detection/README.md

# Force use of GPU 0 to avoid nn.DataParallel issue
export CUDA_VISIBLE_DEVICES=0

python run_object_detection.py \
    --model_name_or_path SenseTime/deformable-detr \
    --dataset_name RoblabWhGe/FireDetDataset \
    --output_dir models/detr-sensetime-finetuned-firedetv7_test \
    --do_train false \
    --do_eval true \
    --image_square_size 1333 \
    --fp16 true \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 1 \
    --per_device_eval_batch_size 1 \
    --dataloader_drop_last true \
    --remove_unused_columns false \
    --eval_do_concat_batches false \
    --ignore_mismatched_sizes true \
    --metric_for_best_model eval_map \
    --greater_is_better true \
    --push_to_hub false \
    --seed 1337