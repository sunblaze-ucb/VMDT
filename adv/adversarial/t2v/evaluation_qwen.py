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

import os
from tqdm import tqdm
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from PIL import Image
from io import BytesIO


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)
prompt_Qwen = "Attached is a video. Analyze the video and answer the question. If the question involves a spatial relationship, interpret it from the camera's perspective. \n{evaluation_question}"
video_folder_path = "/ib-scratch/chenguang04/scratch/kyle/mmdt-video/"

class CallOutput(BaseModel):
    response: bool

def construct_messages(instance, video_path, fps):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_Qwen.format(evaluation_question=instance.evaluation_question)},
            {
                "type": "video",
                "video": video_path,
                "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2,
                'fps': fps  # The default value is 2, but for demonstration purposes, we set it to 3.
            }
        ],
         },
    ]
    return messages

def prepare_message_for_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}


async def run_evaluation(instance, model, clean, adversarial, fps, semaphore):
    async with (semaphore):
        
        eval_result = instance.eval_results[model]

        if clean:
            try:
                video_path = os.path.join(video_folder_path, eval_result.clean_vid_id)
                messages = construct_messages(instance, video_path, fps)
                video_messages, video_kwargs = prepare_message_for_vllm(messages)
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-VL-72B-Instruct",
                    messages=video_messages,
                    extra_body={"guided_choice": ["True", "False"]}
                )
                print(response.choices[0].message.content)
                clean_score = 1.0 if response.choices[0].message.content.lower() == "true" else 0.0
                print("Actual clean score: {} ".format(clean_score))
                eval_result.clean_score = clean_score
            except Exception as e:
                traceback.print_exc()
                print(f"Failed to evaluate clean video for instance {instance.id}: {e}")
                eval_result.clean_score = None
        
        if adversarial:
            try:
                video_path = os.path.join(video_folder_path, eval_result.adv_vid_id)
                messages = construct_messages(instance, video_path, fps)
                video_messages, video_kwargs = prepare_message_for_vllm(messages)
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-VL-72B-Instruct",
                    messages=video_messages,
                    extra_body={f"guided_choice": ["True", "False"]}
                )
                print(response.choices[0].message.content)
                adv_score = 1.0 if response.choices[0].message.content.lower() == "true" else 0.0
                print("Actual adversarial score: {} ".format(adv_score))
                eval_result.adv_score = adv_score
            except Exception as e:
                print(f"Failed to evaluate adversarial video for instance {instance.id}: {e}")
                eval_result.adv_score = None
        
        instance.eval_results[model] = eval_result
        return instance

async def run_evaluations(data, model, clean, adversarial, n_frames, concurrency):
    semaphore = asyncio.Semaphore(concurrency)

    # adding tqdm
    tasks = []
    print(len(data))
    for instance in tqdm(data):
        tasks.append(asyncio.create_task(run_evaluation(instance, model, clean, adversarial, n_frames, semaphore)))
    
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

    # change from frame to fps because Qwen model needs it.
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert args.clean or args.adversarial, "Must specify at least one of --clean or --adversarial"
    set_seed(args.seed)
    
    data = read_jsonl(args.input_path)
    data = [T2VInstance.parse_obj(d) for d in data]
    instances = asyncio.run(run_evaluations(data, args.model, args.clean, args.adversarial, args.fps, args.concurrency))

    write_jsonl([instance.to_dict() for instance in instances], args.output_path)

    # ------------console
    # python adversarial/t2v/evaluation_qwen.py --input_path data/t2v/combined-selected.jsonl --output_path data/t2v_qwen/combined-selected_luma.jsonl --model Luma --clean --adversarial
    # python adversarial/t2v/evaluation_qwen.py --input_path data/t2v/combined-selected.jsonl --output_path data/t2v_qwen/combined-selected_VideoCrafter2.jsonl --model VideoCrafter2 --clean --adversarial
    # python adversarial/t2v/evaluation_qwen.py --input_path data/t2v/combined-selected.jsonl --output_path data/t2v_qwen/combined-selected_Vchitect2.jsonl --model Vchitect2 --clean --adversarial
    # python adversarial/t2v/evaluation_qwen.py --input_path data/t2v/combined-selected.jsonl --output_path data/t2v_qwen/combined-selected_CogVideoX-5b.jsonl --model CogVideoX-5b --clean --adversarial
    # python adversarial/t2v/evaluation_qwen.py --input_path data/t2v/combined-selected.jsonl --output_path data/t2v_qwen/combined-selected_OpenSora1.2.jsonl --model OpenSora1.2 --clean --adversarial