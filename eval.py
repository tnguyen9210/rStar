# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import random
import torch
import transformers
from shutil import copy
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--need_load", type=bool, default=False)
    parser.add_argument("--base_mode", type=str, default="")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--save_res", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from eval_output import eval_output_file
    if args.save_res:
        dirs = os.listdir(args.save_res)
        for dir in dirs:
            try:
                gsm8k_res, _, _ = eval_output_file(os.path.join(args.save_res, dir, "gsm8k.jsonl"))
            except:
                gsm8k_res = 0
            try:
                math_res, _, _ = eval_output_file(os.path.join(args.save_res, dir, "math.jsonl"))
            except:
                math_res = 0
            try:
                aime2024_res, _, _ = eval_output_file(os.path.join(args.save_res, dir, "aime2024.jsonl"))
            except:
                aime2024_res = 0
            try:
                amc23_res, _, _ = eval_output_file(os.path.join(args.save_res, dir, "amc23.jsonl"))
            except:
                amc23_res = 0
            try:
                collegemath_res, _, _ = eval_output_file(os.path.join(args.save_res, dir, "collegemath.jsonl"))
                gaokao2023en_res, _, _ = eval_output_file(os.path.join(args.save_res, dir, "gaokao2023en.jsonl"))
                olympiadbench_res, _, _ = eval_output_file(os.path.join(args.save_res, dir, "olympiadbench.jsonl"))
            except:
                collegemath_res = 0
                gaokao2023en_res = 0
                olympiadbench_res = 0
            try:
                math500_res, _, _ = eval_output_file(os.path.join(args.save_res, dir, "math500.jsonl"))
            except:
                math500_res = 0
            try:
                omni_res, _, _ = eval_output_file(os.path.join(args.save_res, dir, "omni-math.jsonl"))
            except:
                omni_res = 0
            if gsm8k_res != 0 or math_res != 0 or aime2024_res != 0 or amc23_res != 0 or collegemath_res != 0 or gaokao2023en_res != 0 or olympiadbench_res != 0 or math500_res != 0 or omni_res != 0:
                with open(os.path.join(args.save_res, "result.json"), "a") as f:
                    f.write(f'{{"model_dir": "{dir}" ,"gsm8k_res": {gsm8k_res} , "math_res": {math_res} , "aime2024_res": {aime2024_res} , "amc23_res": {amc23_res} , "math500_res": {math500_res}}}\n')
                    f.write(f'{{"model_dir": "{dir}" ,"collegemath_res": {collegemath_res} ,"gaokao2023en_res": {gaokao2023en_res} , "olympiadbench_res": {olympiadbench_res} ,  "omni_res": {omni_res}}}\n')
        sys.exit(0)
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    tmp = ''
    if args.need_load:
        model = transformers.AutoModelForCausalLM.from_pretrained(args.base_mode)
        t = torch.load(args.model + '/policy.pt')['state']
        model.load_state_dict(t)
        tmp = '/scratch/tmp' + str(args.device) + "/" + str(random.randint(0, 10000000)) + '/'
        model.save_pretrained(tmp)
        
        token_dir = args.model[:args.model.rfind('/')]
        copy(token_dir + '/added_tokens.json', tmp)
        copy(token_dir + '/special_tokens_map.json', tmp)
        copy(token_dir + '/tokenizer_config.json', tmp)
        copy(token_dir + '/tokenizer.model', tmp)
    else:
        tmp = args.model
    print("path: ",tmp)
    if args.task == "gsm8k":
        save_dir = args.model + "/gsm8k" 
        os.system(f'python main.py --custom_cfg config/sft_eval_greedy.yaml --qaf ./eval_data/GSM8K_test.json --save_in_model {save_dir} --model_dir {tmp}')
    elif args.task == "math":
        save_dir = args.model + "/math" 
        os.system(f'python main.py --custom_cfg config/sft_eval_greedy.yaml --qaf ./eval_data/MATH_test.json --save_in_model {save_dir} --model_dir {tmp}')
    elif args.task == "aime2024":
        save_dir = args.model + "/aime2024" 
        os.system(f'python main.py --custom_cfg config/sft_eval_greedy.yaml --qaf ./eval_data/aime2024_test.json --save_in_model {save_dir} --model_dir {tmp}')
    elif args.task == "amc23":
        save_dir = args.model + "/amc23" 
        os.system(f'python main.py --custom_cfg config/sft_eval_greedy.yaml --qaf ./eval_data/amc23_test.json --save_in_model {save_dir} --model_dir {tmp}')
    elif args.task == "collegemath":
        save_dir = args.model + "/collegemath"
        os.system(f'python main.py --custom_cfg config/sft_eval_greedy.yaml --qaf ./eval_data/collegemath_test.json --save_in_model {save_dir} --model_dir {tmp}')
    elif args.task == "gaokao2023en":
        save_dir = args.model + "/gaokao2023en"
        os.system(f'python main.py --custom_cfg config/sft_eval_greedy.yaml --qaf ./eval_data/gaokao2023en_test.json --save_in_model {save_dir} --model_dir {tmp}')
    elif args.task == "olympiadbench":
        save_dir = args.model + "/olympiadbench"
        os.system(f'python main.py --custom_cfg config/sft_eval_greedy.yaml --qaf ./eval_data/olympiadbench_test.json --save_in_model {save_dir} --model_dir {tmp}')
    elif args.task == "math500":
        save_dir = args.model + "/math500"
        os.system(f'python main.py --custom_cfg config/sft_eval_greedy.yaml --qaf ./eval_data/math500_test.json --save_in_model {save_dir} --model_dir {tmp}')
    elif args.task == "omni-math":
        save_dir = args.model + "/omni-math"
        os.system(f'python main.py --custom_cfg config/sft_eval_greedy.yaml --qaf ./eval_data/omni_test.json --save_in_model {save_dir} --model_dir {tmp}')
        
    if args.need_load:
        # Delete the files in the tmp folder
        file_list = os.listdir(tmp)
        # Loop through and delete files
        for file_name in file_list:
            file_path = os.path.join(tmp, file_name)
            os.remove(file_path)