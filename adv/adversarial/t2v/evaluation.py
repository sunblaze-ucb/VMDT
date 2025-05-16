import argparse
import base64
import asyncio
import traceback

import cv2
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel

from adversarial.common.utils import read_jsonl, write_jsonl, set_seed
from adversarial.t2v.t2v_utils import T2VInstance, EvaluationResult
from adversarial.common.openai_client import async_call_openai

def load_frames_in_base64(video_path, n_frames):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    
    base64_frames = []
    current_frame = 0
    target_index_ptr = 0

    while video.isOpened() and target_index_ptr < len(target_indices):
        ret, frame = video.read()
        if not ret:
            break
        if current_frame == target_indices[target_index_ptr]:
            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                encoded_str = base64.b64encode(buffer).decode("utf-8")
                base64_frames.append(encoded_str)
            target_index_ptr += 1
        current_frame += 1
    video.release()
    
    return base64_frames

prompt = "Attached are frame from a video. Analyze the video frames and indicate your answer to the question below with a boolean (i.e., True or False) in the \"response\" field. If the question involves a spatial relationship, interpret it from the camera's perspective. \n{evaluation_question}"

class CallOutput(BaseModel):
    response: bool

def construct_messages(instance, base64_frames):
    return [
        {
            "role": "user",
            "content": [
                prompt.format(evaluation_question=instance.evaluation_question),
                *map(lambda x: {"image": x, "resize": 768}, base64_frames),
            ],
        },
    ]

async def run_evaluation(instance, model, clean, adversarial, n_frames, semaphore):
    async with semaphore:
        
        eval_result = instance.eval_results[model]
        
        if clean:
            try:
                clean_frames = load_frames_in_base64(eval_result.clean_vid_id, n_frames)
                messages = construct_messages(instance, clean_frames)
                response = await async_call_openai(
                    messages=messages,
                    model="gpt-4o-mini-2024-07-18",
                    system_prompt=None,
                    response_format=CallOutput,
                    max_tokens=4096,
                    temperature=0.0,
                )
                clean_score = 1.0 if CallOutput.parse_raw(response).response is True else 0.0
                eval_result.clean_score = clean_score
            except Exception as e:
                traceback.print_exc()
                print(f"Failed to evaluate clean video for instance {instance.id}: {e}")
                eval_result.clean_score = None
        
        if adversarial:
            try:
                adv_frames = load_frames_in_base64(eval_result.adv_vid_id, n_frames)
                messages = construct_messages(instance, adv_frames)
                response = await async_call_openai(
                    messages=messages,
                    model="gpt-4o-mini-2024-07-18",
                    system_prompt=None,
                    response_format=CallOutput,
                    max_tokens=4096,
                    temperature=0.0,
                )
                adv_score = 1.0 if CallOutput.parse_raw(response).response is True else 0.0
                eval_result.adv_score = adv_score
            except Exception as e:
                print(f"Failed to evaluate adversarial video for instance {instance.id}: {e}")
                eval_result.adv_score = None
        
        instance.eval_results[model] = eval_result
        return instance

async def run_evaluations(data, model, clean, adversarial, n_frames, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    
    tasks = [asyncio.create_task(run_evaluation(instance, model, clean, adversarial, n_frames, semaphore)) for instance in data]
    
    instances = []
    for completed in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        instance = await completed
        if instance is not None:
            instances.append(instance)
    
    return instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert args.clean or args.adversarial, "Must specify at least one of --clean or --adversarial"
    set_seed(args.seed)
    
    data = read_jsonl(args.input_path)
    data = [T2VInstance.parse_obj(d) for d in data]
    
    instances = asyncio.run(run_evaluations(data, args.model, args.clean, args.adversarial, args.n_frames, args.concurrency))
    
    write_jsonl([instance.to_dict() for instance in instances], args.output_path)