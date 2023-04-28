# Easy-Finetune-GLM
我只想发布一个简单的代码，用于对GLM进行微调。该项目只是为了分享和娱乐。该代码是基于LLAMA alpaca实现的，数据由https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
提供。
我在glm-large-chinese和glm-10b-chinese版本上进行了实验。我所有的实验都是在4张40G A100上进行的。glm-large-chinese模型的微调大约需要20分钟，glm-10b-chinese的微调大约需要37小时。

我使用deepspeed进行了显存优化，您可以依照自己的服务器显存大小在deepspeed文件夹下选择合适的版本，主要是Zero 2/Zero 3和cpuoffload的组合。总的来说，Zero 3比Zero 2占用的显存更小，加了cpuoffload的占用显存更小，但相应地，他们需要更长的训练时间。

I just want to release a simple code for fine-tuning GLM on released data. This repo is just for sharing and fun. 

The code is implemented based on LLAMA-alpaca, and the data is released by https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM.

I conducted experiments on large and 10b version. All my experiments are conducted on 4 40G A100. It costs around 20 minutes to finetune a glm-large-chinese model, and it costs around 37 hours to finetune a glm-10b-chinese model.

I have optimized the GPU memory using deepspeed. You can choose the appropriate version in the deepspeed folder based on your server's GPU memory size, mainly a combination of zero 2/zero 3 and using CPU offload or not. Overall, Zero3 occupies smaller GPU memory than Zero2, and with the addition of CPU offload, it occupies smaller GPU memory. However, correspondingly, they require longer training time.
 

# Environment
```pip install -r requirement```

# Train
```
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
```

# Evaluate
```
CUDA_VISIBLE_DEVICES=2 python test_glm.py --model /data/zhoujing/ckpts/glm-large-chinese/alpaca_gpt4_data_zh

CUDA_VISIBLE_DEVICES=3 python test_glm.py --model /data/zhoujing/ckpts/glm-10b-chinese/alpaca_gpt4_data_zh/checkpoint-1000
```
