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
import pandas as pd


from VMDT.adv.common.utils import read_jsonl, write_jsonl, set_seed
from VMDT.adv.video_text.v2t_utils import V2TInstance, EvaluationResult, get_mc_format, parse_mc_answer
from VMDT.models.v2t_models import load_v2t_model

def run_instances_on_gpu_for_model(prompts, videos, model_name, gpu_id):
    """
    Run a single model on a GPU for a portion of the data.
    """
    
    torch.cuda.set_device(gpu_id)
    
    model = load_v2t_model(model_name)

    text_outputs = model.generate_texts(videos, prompts)

    return text_outputs

def run_instances_parallel(prompts, videos, model_name, num_gpus=8):
    """
    Distribute data across GPUs (sequential chunking) and spawn parallel processes.
    """
    mp.set_start_method('spawn', force=True)

    chunk_size = (len(prompts) + num_gpus - 1) // num_gpus

    # Shard both prompts and videos together
    prompt_chunks = [
        prompts[i*chunk_size : (i+1)*chunk_size]
        for i in range(num_gpus)
    ]
    
    video_chunks = [
        videos[i*chunk_size : (i+1)*chunk_size]
        for i in range(num_gpus)
    ]

    args = []
    for gpu_id, (prompt_chunk, video_chunk) in enumerate(zip(prompt_chunks, video_chunks)):
        args.append((prompt_chunk, video_chunk, model_name, gpu_id))

    with Pool(processes=num_gpus) as pool:
        all_results = pool.starmap(run_instances_on_gpu_for_model, args)

    combined_results = []
    for result in all_results:
        combined_results.extend(result)

    return combined_results

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the dataset
    raw = load_dataset("mmfm-trust/V2T", "adv")
    data = []
    for task_name, ds in raw.items():
        ds_with_task = ds.add_column("task", [task_name] * len(ds))
        data.append(ds_with_task)
    data = concatenate_datasets(data)
    data = [V2TInstance.parse_obj(d) for d in data]
    
    if not os.path.exists(args.vids_dir):
        snapshot_download(
            repo_id="mmfm-trust/V2T",  
            repo_type="dataset",
            local_dir=args.vids_dir,              
            local_dir_use_symlinks=False,          
            allow_patterns="adv/vids/**",          
        )

    args.vids_dir = os.path.join(args.vids_dir, "adv/vids/")

    # Prepare prompts and videos
    prompts, videos = [], []
    for instance in data:
        question, all_letters, gt_letters = get_mc_format(instance.question, instance.choices, [instance.answer])
        prompts.append(question)
        prompts.append(question)
        videos.append(os.path.join(args.vids_dir, f"{instance.benign}.mp4"))
        videos.append(os.path.join(args.vids_dir, f"{instance.adversarial}.mp4"))

    # Run the model
    if args.n_gpus > 1:
        outputs = run_instances_parallel(prompts, videos, args.model_id, args.n_gpus)
    else:
        outputs = run_instances_on_gpu_for_model(prompts, videos, args.model_id, 0)

    # Evaluate the model
    for i in range(0, len(outputs), 2):
        instance = data[i//2]
        question, all_letters, gt_letters = get_mc_format(instance.question, instance.choices, [instance.answer])

        benign_output = outputs[i].text_output
        benign_score = int(set(parse_mc_answer(benign_output, instance.choices, all_letters)) == set(gt_letters))

        adv_output = outputs[i+1].text_output
        adv_score = int(set(parse_mc_answer(adv_output, instance.choices, all_letters)) == set(gt_letters))

        instance.eval_results[args.model_id] = EvaluationResult(
            benign_response=benign_output,
            benign_score=benign_score,
            adv_response=adv_output,
            adv_score=adv_score
        ) 

    write_jsonl([instance.model_dump() for instance in data], os.path.join(args.output_dir, f"{args.model_id}.jsonl"))

    average_score = sum([instance.eval_results[args.model_id].adv_score for instance in data]) / len(data)

    # Update average.csv
    average_csv_path = os.path.join(args.output_dir, "average.csv")
    if os.path.exists(average_csv_path):
        # Read existing CSV and append new row
        average = pd.read_csv(average_csv_path)
        average = pd.concat([average, pd.DataFrame({
            'model': [args.model_id],
            'average': [average_score]
        })], ignore_index=True)
    else:
        # Create new DataFrame with first entry
        average = pd.DataFrame({
            'model': [args.model_id],
            'average': [average_score]
        })
    
    # Save the updated DataFrame
    average.to_csv(average_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/v2t_results/adv/")
    parser.add_argument("--vids_dir", type=str, default="VMDT/adv/v2t_vids/")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)