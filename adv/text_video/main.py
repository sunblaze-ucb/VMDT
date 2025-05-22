import argparse
import os
import asyncio
from pathlib import Path
import multiprocessing as mp
import subprocess
import time

import pandas as pd

from VMDT.adv.common.utils import read_jsonl, write_jsonl, set_seed
from VMDT.adv.text_video.t2v_utils import T2VInstance
from VMDT.adv.text_video.inference import main as run_inference
from VMDT.adv.text_video.evaluation import run_evaluations


def start_vllm_server(model, num_gpus):
    """Start vLLM server in a separate process."""
    cmd = [
        "vllm", "serve", f"{model}",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--limit-mm-per-prompt", "image=5",
        "--tensor-parallel-size", str(num_gpus),
        "--max-model-len", "4096",
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(300)  # Give it some time to initialize
    
    return process

def cleanup_vllm(vllm_process):
    if vllm_process.poll() is None:  # If process is still running
        vllm_process.terminate()
        vllm_process.wait()

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.vids_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    inference_output_path = Path(args.output_dir) / f"{args.model_id}.jsonl"
    evaluation_gpt_output_path = Path(args.output_dir) / f"{args.model_id}_gpt.jsonl"
    evaluation_qwen_output_path = Path(args.output_dir) / f"{args.model_id}_qwen.jsonl"
    
    # Step 1: Run inference
    print("\n=== Running Inference ===")
    inference_args = argparse.Namespace(
        model=args.model_id,
        output_path=inference_output_path,
        benign=True,
        adversarial=True,
        vids_dir=args.vids_dir,
        n_gpus=args.n_gpus,
        seed=args.seed
    )
    run_inference(inference_args)

    # Step 2: Run GPT evaluation
    print("\n=== Running GPT Evaluation ===")
    evaluation_gpt_args = argparse.Namespace(
        model=args.model_id,
        eval_model="gpt-4o-mini-2024-07-18",
        input_path=inference_output_path,
        output_path=evaluation_gpt_output_path,
        benign=True,
        adversarial=True,
        n_frames=args.n_frames,
        concurrency=args.concurrency
    )
    run_evaluations(evaluation_gpt_args)

    # Step 3: Start vLLM server and run Qwen evaluation
    print("\n=== Starting vLLM Server for Qwen ===")
    vllm_process = start_vllm_server()
    
    try:
        print("\n=== Running Qwen Evaluation ===")
        evaluation_qwen_args = argparse.Namespace(
            model=args.model_id,
            eval_model="Qwen/Qwen2.5-VL-72B-Instruct",
            input_path=inference_output_path,
            output_path=evaluation_qwen_output_path,
            benign=True,
            adversarial=True,
            n_frames=args.n_frames,
            concurrency=args.concurrency
        )
        run_evaluations(evaluation_qwen_args)
    finally:
        # Ensure vLLM server is terminated
        cleanup_vllm(vllm_process)

    # Step 4: Score the results
    gpt_results = read_jsonl(evaluation_gpt_output_path)
    gpt_results = [T2VInstance.parse_obj(d) for d in gpt_results]
    gpt_results = sorted(gpt_results, key=lambda x: x.id)

    qwen_results = read_jsonl(evaluation_qwen_output_path)
    qwen_results = [T2VInstance.parse_obj(d) for d in qwen_results]
    qwen_results = sorted(qwen_results, key=lambda x: x.id)

    scores = []
    for gpt, qwen in zip(gpt_results, qwen_results):
        gpt_score = gpt.eval_results[args.model_id].adv_score
        qwen_score = qwen.eval_results[args.model_id].adv_score
        scores.append((gpt_score + qwen_score) / 2)
    average_score = sum(scores) / len(scores)
    print(f"Average score: {average_score}")

    # Step 5: Update average.csv
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, choices=['Nova','Pika','Luma','OpenSora1.2','Vchitect2','VideoCrafter2','CogVideoX-5B'])
    parser.add_argument('--output_dir', type=str, default="results/t2v_results/adv/")
    parser.add_argument('--vids_dir', type=str, default="adv/t2v_vids/")
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_frames', type=int, default=5)
    parser.add_argument('--concurrency', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)