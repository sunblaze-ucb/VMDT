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
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download


from adversarial.common.utils import read_jsonl, write_jsonl, set_seed
from adversarial.v2t.v2t_utils import V2TInstance, EvaluationResult, get_mc_format, parse_mc_answer

def run_instances_on_gpu_for_model(data_part, gpu_id, model_name, vids_dir):
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
        print(vids_dir)
        question, all_letters, gt_letters = get_mc_format(instance.question, instance.choices, [instance.answer])
        benign_video_path = os.path.join(vids_dir, f"{instance.benign}.mp4")
        adv_video_path = os.path.join(vids_dir, f"{instance.adversarial}.mp4")
        
        try:
            benign_response = model.generate(question, video_path=benign_video_path)
            benign_score = int(set(parse_mc_answer(benign_response, instance.choices, all_letters)) == set(gt_letters))
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            benign_response = ""
            benign_score = 0
            
        try:
            adv_response = model.generate(question, video_path=adv_video_path)
            adv_score = int(set(parse_mc_answer(adv_response, instance.choices, all_letters)) == set(gt_letters))
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            adv_response = ""
            adv_score = 0

        instance.eval_results[model_name] = EvaluationResult(
            benign_response=benign_response,
            benign_score=benign_score,
            adv_response=adv_response,
            adv_score=adv_score
        ) 
        
    return data_part

def run_instances_parallel(data, model_name, vids_dir, num_gpus=8):
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
        args.append((chunk, gpu_id, model_name, vids_dir)) # Fix

    with Pool(processes=num_gpus) as pool:
        all_results = pool.starmap(run_instances_on_gpu_for_model, args)

    combined_results = []
    for result in all_results:
        combined_results.extend(result)

    return combined_results

def main(args):
    
    from datasets import load_dataset, concatenate_datasets
    raw = load_dataset("mmfm-trust/V2T", "adv")
    data = []
    for task_name, ds in raw.items():
        ds_with_task = ds.add_column("task", [task_name] * len(ds))
        data.append(ds_with_task)
    data = concatenate_datasets(data)
    data = [V2TInstance.parse_obj(d) for d in data][:10]
    
    if not os.path.exists(args.vids_dir):
        snapshot_download(
            repo_id="mmfm-trust/V2T",  
            repo_type="dataset",
            local_dir=args.vids_dir,              
            local_dir_use_symlinks=False,          
            allow_patterns="adv/vids/**",          
        )
    vids_dir = os.path.join(args.vids_dir, "adv/vids")
    print(vids_dir)
    
    if args.n_gpus > 1:
        data = run_instances_parallel(data, args.model, vids_dir, args.n_gpus)
    else:
        data = run_instances_on_gpu_for_model(data, 0, args.model, vids_dir)
    
    benign_score = sum([instance.eval_results[args.model].benign_score for instance in data]) / len(data)
    adv_score = sum([instance.eval_results[args.model].adv_score for instance in data]) / len(data)   
    print(">> benign Score: {:.4f}".format(benign_score), "Adv Score: {:.4f}".format(adv_score))
    
    write_jsonl([instance.model_dump() for instance in data], args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--vids_dir", type=str, default="v2t_vids/")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)