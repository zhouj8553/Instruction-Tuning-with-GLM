import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--int8', action='store_true')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)  # fast版LlamaTokenizer载入非常慢，测试建议用slow的就好
model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.half, load_in_8bit=args.int8, trust_remote_code=True).to('cuda:0')

s = "写一封信给高校的教授，邀请他们给我的博士论文审稿。"
'''
我有10个苹果，吃掉5个，还剩几个？
计算3*4=？
向一个6岁的小孩介绍强化学习
'''
# s = "Introduce Tsinghua University for me."
while(True):
    # sentence =  "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction: {} \n### Response: [gMASK]".format(s)
    if '10b' in args.model: 
        sentence = "以下是描述任务的说明，编写一个响应以完成请求。\n\n###提示：{}\n ###回复： [gMASK]".format(s)
    else:
        sentence = "以下是描述任务的说明，编写一个响应以完成请求。\n\n###提示：{}\n ###回复： [MASK]".format(s)
    # import pdb 
    # pdb.set_trace()
    batch = tokenizer(sentence,return_tensors="pt")
    batch = tokenizer.build_inputs_for_generation(batch, max_gen_length=512)
    batch = {key: value.cuda() for key, value in batch.items()}
    with torch.inference_mode(), torch.cuda.amp.autocast():
        print("Start")
        t0 = time.time()
        #  max_length=512, temperature=1.0, top_p=0.95, do_sample=True
        generated = model.generate(**batch, max_length=512, eos_token_id=tokenizer.eop_token_id)
        t1 = time.time()
        print(f"Output generated in {(t1-t0):.2f} seconds")
        print(s)
        print(tokenizer.decode(generated[0]).split("<|startofpiece|> ")[1].rstrip(" <|endoftext|> <|endofpiece|>'"))
        print("\n\n")
    print(">", end=" ")
    s = input()

# Some models (GLM-2B, GLM-10B, and GLM-10B-Chinese) use three different mask tokens: [MASK] for short blank filling, [sMASK] for sentence filling, and [gMASK] for left-to-right generation.
"""

CUDA_VISIBLE_DEVICES=3 python test_glm.py --model /data/zhoujing/ckpts/glm-10b-chinese/alpaca_gpt4_data_zh/checkpoint-1000

CUDA_VISIBLE_DEVICES=2 python test_glm.py --model /data/zhoujing/ckpts/glm-large-chinese/alpaca_gpt4_data_zh

"""
