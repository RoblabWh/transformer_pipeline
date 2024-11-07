#!/bin/bash
# microsoft/conditional-detr-resnet-50
# SenseTime/deformable-detr 
#        --num_queries 100 \ check where to add
# https://huggingface.co/docs/autotrain/object_detection_params
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/object-detection/README.md
python run_object_detection.py \
    --model_name_or_path SenseTime/deformable-detr \
    --dataset_name RoblabWhGe/FireDetDataset \
    --do_train true \
    --do_eval true \
    --output_dir detr-sensetime-finetuned-firedetv7 \
    --num_train_epochs 300 \
    --image_square_size 1333 \
    --fp16 true \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 2 \
    --per_device_train_batch_size 4 \
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