import os
import json
import argparse
import time
import gc
from multiprocessing import Pool

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import cv2
import numpy as np
from transformers import XCLIPProcessor, XCLIPModel

from adversarial.common.utils import read_jsonl, write_jsonl, set_seed
from adversarial.t2v.t2v_utils import T2VInstance

def load_frames_as_tensor(video_path, n_frames):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    current_frame = 0
    target_index_ptr = 0

    while video.isOpened() and target_index_ptr < len(target_indices):
        ret, frame = video.read()
        if not ret:
            break
        if current_frame == target_indices[target_index_ptr]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            target_index_ptr += 1
        current_frame += 1
    video.release()

    return list(torch.unbind(torch.tensor(np.array(frames), dtype=torch.uint8), dim=0))

@torch.no_grad()
def run_instances_on_gpu_for_model(data_part, gpu_id, model_name, videos_dir):
    """
    Run a single model on a GPU for a portion of the data.
    """
    
    device = torch.device(f"cuda:{gpu_id}")
    
    tokenizer = XCLIPProcessor.from_pretrained("microsoft/xclip-large-patch14")
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-large-patch14")
    model = XCLIPModel.from_pretrained("microsoft/xclip-large-patch14").to(device)
    
    for instance in data_part:
        if model_name not in instance.eval_results:
            continue
        
        clean_prompt = [instance.clean_prompt]
        text_inputs = tokenizer(clean_prompt, padding=True, return_tensors="pt").to(device)
        text_features = model.get_text_features(**text_inputs)
        
        clean_vid_id = instance.eval_results[model_name].clean_vid_id
        clean_vid = load_frames_as_tensor(clean_vid_id, 8)
        clean_inputs = processor(videos=clean_vid, return_tensors="pt").to(device)
        clean_features = model.get_video_features(**clean_inputs)
        
        adv_vid_id = instance.eval_results[model_name].adv_vid_id
        adv_vid = load_frames_as_tensor(adv_vid_id, 8)
        adv_inputs = processor(videos=adv_vid, return_tensors="pt").to(device)
        adv_features = model.get_video_features(**adv_inputs)
        
        clean_xclip = F.cosine_similarity(text_features, clean_features, dim=-1).mean().item()
        adv_xclip = F.cosine_similarity(text_features, adv_features, dim=-1).mean().item()
        
        instance.eval_results[model_name].clean_xclip = clean_xclip
        instance.eval_results[model_name].adv_xclip = adv_xclip
        
    return data_part

def run_instances_parallel(data, model_name, videos_dir, num_gpus=8):
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
        all_results = pool.starmap(run_instances_on_gpu_for_model, args)

    combined_results = []
    for result in all_results:
        combined_results.extend(result)

    return combined_results

def main(args):
    
    data = read_jsonl(args.input_path)
    data = [T2VInstance.parse_obj(d) for d in data]
        
    if args.n_gpus > 1:
        data = run_instances_parallel(data, args.model, args.vids_dir, args.n_gpus)
    else:
        data = run_instances_on_gpu_for_model(data, 0, args.model, args.vids_dir)
        
    write_jsonl([instance.to_dict() for instance in data], args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--vids_dir", type=str, default="vids/")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)