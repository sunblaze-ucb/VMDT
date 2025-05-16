import os
import sys
import json
import argparse
import time
import gc
import traceback
from multiprocessing import Pool

import torch.multiprocessing as mp
import torch
from tqdm import tqdm

from adversarial.common.utils import read_jsonl, write_jsonl, set_seed
from adversarial.v2t.v2t_utils import V2TInstance, EvaluationResult, get_mc_format, parse_mc_answer

def run_instances_on_gpu_for_model(data_part, gpu_id, model_name, frames_dir):
    """
    Run a single model on a GPU for a portion of the data.
    """
    
    torch.cuda.set_device(gpu_id)
    
    if model_name == "InternVideo2Chat8B":
        from adversarial.v2t.models.internvideo2 import InternVideo2Chat8B
        model = InternVideo2Chat8B(device=f"cuda:{gpu_id}", dtype="torch.bfloat16")
    elif model_name == "VideoChat2":
        from adversarial.v2t.models.videochat2 import VideoChat2
        model = VideoChat2(device=f"cuda:{gpu_id}", dtype="torch.bfloat16")
    elif model_name == "VideoLLaVA":
        from adversarial.v2t.models.videollava import VideoLlava
        model = VideoLlava(device=f"cuda:{gpu_id}", dtype="torch.bfloat16")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    for instance in tqdm(data_part):
        question, all_letters, gt_letters = get_mc_format(instance.question, instance.answer_choices, instance.gt)
        clean_frames = torch.load(os.path.join(frames_dir, instance.clean_frames_id, "frames.pt"))
        adv_frames = torch.load(os.path.join(frames_dir, instance.adv_frames_id, "frames.pt"))
        
        try:
            clean_response = model.generate(question, clean_frames)
            clean_score = int(set(parse_mc_answer(clean_response, instance.answer_choices, all_letters)) == set(gt_letters))
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            clean_response = ""
            clean_score = 0
            
        try:
            adv_response = model.generate(question, adv_frames)
            adv_score = int(set(parse_mc_answer(adv_response, instance.answer_choices, all_letters)) == set(gt_letters))
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            adv_response = ""
            adv_score = 0

        instance.eval_results[model_name] = EvaluationResult(
            clean_response=clean_response,
            clean_score=clean_score,
            adv_response=adv_response,
            adv_score=adv_score
        ) 
        
    return data_part

def run_instances_parallel(data, model_name, frames_dir, num_gpus=8):
    """
    Distribute data across GPUs (sequential chunking) and spawn parallel processes.
    """
    mp.set_start_method('spawn', force=True)

    chunk_size = (len(data) + num_gpus - 1) // num_gpus

    data_chunks = [
        data[i*chunk_size : (i+1)*chunk_size]
        for i in range(num_gpus)
    ]

    args = []
    for gpu_id, chunk in enumerate(data_chunks):
        args.append((chunk, gpu_id, model_name, frames_dir)) # Fix

    with Pool(processes=num_gpus) as pool:
        all_results = pool.starmap(run_instances_on_gpu_for_model, args)

    combined_results = []
    for result in all_results:
        combined_results.extend(result)

    return combined_results

def main(args):
    
    data = read_jsonl(args.input_path)
    data = [V2TInstance.parse_obj(d) for d in data]
    
    if args.n_gpus > 1:
        data = run_instances_parallel(data, args.model, args.frames_dir, args.n_gpus)
    else:
        data = run_instances_on_gpu_for_model(data, 0, args.model, args.frames_dir)
    
    clean_score = sum([instance.eval_results[args.model].clean_score for instance in data]) / len(data)
    adv_score = sum([instance.eval_results[args.model].adv_score for instance in data]) / len(data)   
    print(">> Clean Score: {:.4f}".format(clean_score), "Adv Score: {:.4f}".format(adv_score))
    
    write_jsonl([instance.dict() for instance in data], args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, default="v2t_frames/")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)