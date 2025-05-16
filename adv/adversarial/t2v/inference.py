import os
import sys
    
import json
import argparse
import time
import gc
from multiprocessing import Pool

import torch.multiprocessing as mp
import torch
from datasets import load_dataset, concatenate_datasets

from adversarial.common.utils import read_jsonl, write_jsonl, set_seed, gen_id
from adversarial.t2v.t2v_utils import T2VInstance, EvaluationResult

def run_prompt_on_gpu_for_model(data_part, gpu_id, model_name, videos_dir):
    """
    Run a single model on a GPU for a portion of the data.
    """
    
    torch.cuda.set_device(gpu_id)
    
    if model_name == "CogVideoX-2b":
        from adversarial.t2v.models.cogvideox import CogVideoX
        model = CogVideoX("2b", device=f"cuda:{gpu_id}")
    elif model_name == "CogVideoX-5b":
        from adversarial.t2v.models.cogvideox import CogVideoX
        model = CogVideoX("5b", device=f"cuda:{gpu_id}")
    elif model_name == "mochi-1-preview":
        from adversarial.t2v.models.mochi import Mochi
        model = Mochi(device=f"cuda:{gpu_id}")
    elif model_name in ["VideoCrafter2", "HunyuanVideo", "Vchitect2", "OpenSora1.2", "Luma", "Nova_reel", "Pika"]:
        from adversarial.common.video_safety_benchmark.models import load_t2v_model
        model = load_t2v_model(model_name)
    else:
        raise ValueError(f"Model {model_name} not found")

    video_outputs = model.generate_videos(data_part, videos_dir)
    
    return video_outputs

def run_prompts_parallel(data, model_name, videos_dir, num_gpus=8):
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
        args.append((chunk, gpu_id, model_name, videos_dir)) # Fix

    with Pool(processes=num_gpus) as pool:
        all_results = pool.starmap(run_prompt_on_gpu_for_model, args)

    combined_results = []
    for result in all_results:
        combined_results.extend(result)

    return combined_results

def main(args):
    
    raw = load_dataset("mmfm-trust/T2V", "adv")
    data = []
    for task_name, ds in raw.items():
        ds_with_task = ds.add_column("task", [task_name] * len(ds))
        data.append(ds_with_task)
    data = concatenate_datasets(data)
    data = [T2VInstance.parse_obj(d) for d in data]

    os.makedirs(args.vids_dir, exist_ok=True)
        
    prompts = []
    for item in data:
        if args.benign:
            prompts.append(item.benign)
        if args.adversarial:
            prompts.append(item.adversarial)
        
    if args.n_gpus > 1:
        video_outputs = run_prompts_parallel(prompts, args.model, args.vids_dir, args.n_gpus)
    else:
        video_outputs = run_prompt_on_gpu_for_model(prompts, 0, args.model, args.vids_dir)
    
    c = args.benign and args.adversarial
    
    for i in range(0, len(video_outputs), 2 if c else 1):
        
        benign_vid_id = str(video_outputs[i].video_path) if args.benign else None
        adv_vid_index = i+1 if c else i
        adv_vid_id = str(video_outputs[adv_vid_index].video_path) if args.adversarial else None
        
        data_index = i//2 if c else i
        # data[data_index].id = gen_id() # give each instance a new unique ID
        data[data_index].eval_results.setdefault(args.model, EvaluationResult())
        
        result = data[data_index].eval_results[args.model]
        result.benign_vid_id = benign_vid_id if result.benign_vid_id is None else result.benign_vid_id
        result.adv_vid_id = adv_vid_id if result.adv_vid_id is None else result.adv_vid_id
    
    write_jsonl([instance.model_dump() for instance in data], args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--benign", action="store_true")
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--vids_dir", type=str, default="vids/")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert args.benign or args.adversarial, "Must specify at least one of --benign or --adversarial"
    set_seed(args.seed)
    main(args)

