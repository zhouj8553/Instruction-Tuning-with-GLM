CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=10002 train_glm.py \
    --model_name_or_path THUDM/glm-large-chinese \
    --data_path data/alpaca_gpt4_data_zh.json \
    --bf16 True \
    --output_dir /data/zhoujing/ckpts/glm-large-chinese/alpaca_gpt4_data_zh \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed deepspeed/ds_config_zero2_bf16.json



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=10002 train_glm.py \
    --model_name_or_path BAAI/glm-10b-chinese \
    --data_path data/alpaca_gpt4_data_zh.json \
    --bf16 True \
    --output_dir /data/zhoujing/ckpts/glm-10b-chinese/alpaca_gpt4_data_zh \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed deepspeed/ds_config_zero3_bf16_cpuoffload.json