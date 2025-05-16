import json
import os
import argparse
from datetime import datetime
import uuid
import torch
import torch.multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm
from joblib import Memory
import os
import decord
import re
import sys

# Define a directory for caching
cache_dir = "./cache"
memory = Memory(cache_dir, verbose=0)

# need to handle separately:
# from qwen_vl2 import QwenVLClient
# from VideoLLaMA2.inference import VideoLlama2_1Client

from llava_video import LlavaVideoClient
from internvl2_5 import InternVL25Client
from gpt4o import gpt4oClient
from nova_lite import NovaLite
from nova_pro import NovaPro
from claudesonnet import claudesonnet

MODELS = {
    # "qwen_vl2_7b": partial(Qwen2VLClient, model_id="Qwen/Qwen2-VL-7B-Instruct"),  # OLD. The video parsing wasn't working.
    # "qwen_vl2_7b_clean": partial(QwenVLClient, model_id="Qwen/Qwen2-VL-7B-Instruct", use_less_tokens=False),
    # "qwen_vl2_7b_clean_less": partial(QwenVLClient, model_id="Qwen/Qwen2-VL-7B-Instruct", use_less_tokens=True),
    # "qwen_vl2_7b_clean_ctx": partial(QwenVLClient, model_id="Qwen/Qwen2-VL-7B-Instruct", use_less_tokens=False),
    # "qwen_vl25_3b": partial(QwenVLClient, model_id="Qwen/Qwen2.5-VL-3B-Instruct", use_less_tokens=False),
    # "qwen_vl25_7b_cv": partial(QwenVLClient, model_id="Qwen/Qwen2.5-VL-7B-Instruct", use_less_tokens=False),
    # "qwen_vl25_72b": partial(QwenVLClient, model_id="Qwen/Qwen2.5-VL-72B-Instruct", use_less_tokens=False),

    # "videollama2_1_7b_new": partial(VideoLlama2_1Client, model_id='DAMO-NLP-SG/VideoLLaMA2.1-7B-AV'),
    # "videollama2_72b": partial(VideoLlama2_1Client, model_id='DAMO-NLP-SG/VideoLLaMA2-72B'),

    "llava_video_7b_fixed": partial(LlavaVideoClient, model_id="lmms-lab/LLaVA-Video-7B-Qwen2"),
    "llava_video_72b_fixed": partial(LlavaVideoClient, model_id="lmms-lab/LLaVA-Video-72B-Qwen2"),

    "internvl2_5_1b": partial(InternVL25Client, model_id='OpenGVLab/InternVL2_5-1B'),
    "internvl2_5_2b": partial(InternVL25Client, model_id='OpenGVLab/InternVL2_5-2B'),
    "internvl2_5_4b": partial(InternVL25Client, model_id='OpenGVLab/InternVL2_5-4B'),
    "internvl2_5_8b": partial(InternVL25Client, model_id='OpenGVLab/InternVL2_5-8B'),
    "internvl2_5_26b": partial(InternVL25Client, model_id='OpenGVLab/InternVL2_5-26B'),
    "internvl2_5_38b": partial(InternVL25Client, model_id='OpenGVLab/InternVL2_5-38B'),
    "internvl2_5_78b": partial(InternVL25Client, model_id='OpenGVLab/InternVL2_5-78B'),

    "gpt4omini_1fps_lowdetail": partial(gpt4oClient, model='gpt-4o-mini-2024-07-18'),
    "gpt4o_1fps_lowdetail": gpt4oClient,  # gpt-4o-2024-11-20 by default

    "nova_lite_fixed": NovaLite,
    "nova_pro": NovaPro,

    "claudesonnet": claudesonnet
}

# On A6000s the number of gpus needed for each model. If not on, default 1.
MODEL_TO_GPUS = {
        "internvl2_5_38b": 2,
        "internvl2_5_78b": 4,

        "llava_video_72b_fixed": 4,
}

# copied from mmdt/utils/v2t_utils.py
def get_mc_format(question, answer_choices, gt):
    """Returns the full question formatted with multiple choice answers, the answer choices as letters, and the ground truth answer choices as letters."""
    formatted_question = f"Question: {question}\n"
    # use letters for answer choices
    all_choices, gt_choices = [], []
    for i, choice in enumerate(answer_choices):
        letter = chr(65 + i)
        if i < len(answer_choices) - 1:
            formatted_question += f"{letter}. {choice}\n"
        else:
            formatted_question += f"{letter}. {choice}"
        all_choices.append(letter)
        if choice in gt:
            gt_choices.append(letter)
    return formatted_question, all_choices, gt_choices

def parse_mc_answer(model_output, answer_choices, choice_letters, one_correct_answer=True):
    """Parse the answer to the multiple choice question. First try to find the full answer of {letter}. {answer}. If that is not possible, find the letter. Then, if neither is possible, find the first instance of any answer choice without the letter, both lower and upper case. Use regex. If cannot find anything return the empty string."""
    if not one_correct_answer:
        raise NotImplementedError("Not implemented for multiple correct answers.")
    # Remove 'Answer:' or 'ASSISTANT:' from the model output in case it is there
    model_output = re.sub(r"Answer:|ASSISTANT:", "", model_output)
    # first try to find the full answer of {letter}. {answer}
    for letter, choice in zip(choice_letters, answer_choices):
        if f"{letter}. {choice}" in model_output:
            return letter
    # if not possible, use regex to find first instance of any answer choice without the letter, both lower and upper case
    for choice in answer_choices:
        match = re.search(f"{choice}", model_output, re.IGNORECASE)
        if match:
            return choice_letters[answer_choices.index(choice)]
    # if that is not possible, find the letter
    for letter in choice_letters:
        if f"{letter}" in model_output:
            return letter
    return ""

@memory.cache(ignore=["model"])
def generate_cached_output(question, video_path, model_name, model):
    """
    Generate the model output with caching based only on question, video_path, and model_name.
    The model is ignored in the cache key bc it is a complex object.
    """
  
    output = model.generate(video_path=video_path, question=question)
    return output

@memory.cache(ignore=["model"])
def generate_cached_output_retry2(question, video_path, model_name, model):
    """
    Generate the model output with caching based only on question, video_path, and model_name.
    The model is ignored in the cache key bc it is a complex object.
    This is the version for retries where there are nones.
    """
    output = model.generate(video_path=video_path, question=question)
    return output

def load_prompts(prompt_file):
    with open(prompt_file) as f:
        data = json.load(f)
    return data

def update_progress(progress_queue, total_length, current_model_status):
    """
    Continuously updates the progress bar by checking the progress queue and tracking current models.
    """
    overall_pbar = tqdm(total=total_length, desc="Total progress", unit="instance")
    completed = 0

    # Keep track of where we are
    while completed < total_length:
        try:
            # Update overall progress
            item = progress_queue.get(timeout=5)
            if item is None:
                break
            completed += 1
            overall_pbar.update(1)
        except Exception as e:
            continue

        # Use a separate bar for the current model
        current_model_name = current_model_status.get('model_name', 'Unknown Model')
        current_model_pbar = tqdm(total=total_length, desc=f"Processing {current_model_name}", unit="instance", leave=False)
        current_model_pbar.update(completed)

    overall_pbar.close()

def run_prompt_on_gpu_for_model(data_part, gpu_id, model_class, videos_dir, progress_queue, current_model_status):
    """
    Run a single model on a GPU for a portion of the data.
    """
    # Set the device for this process
    if len(gpu_id) == 1:
        device = f"cuda:{gpu_id[0]}"
        torch.cuda.set_device(device)
    else:
        device = "cuda"

    # Initialize the model and move it to the assigned device
    model = MODELS[model_class](device=device)

    if current_model_status is not None:
        current_model_status['model_name'] = model_class

    results = []
    for d in tqdm(data_part, desc=f"Running model {model_class} on GPU {gpu_id}", unit="instance"):
        print(f"Running model {model_class} for scenario {d['scenario_name']} on prompt {d['question']} on GPU {gpu_id}")

        # Format question and run inference.
        question, all_letters, gt_letters = get_mc_format(d["question"], d["answer_choices"], d["gt"])
        # Check if path is valid
        video_path = d['video_path']
        if not os.path.exists(video_path):
            print(f"Video path {video_path} does not exist.")
            continue
        try:
            output = generate_cached_output(question, video_path, model_class, model)
        except decord._ffi.base.DECORDError as e:
            print(f"Decord error for video {video_path}: {e}")
            continue
        except Exception as e:
            print(f"Error for video {video_path}: {e}")
            continue
        if output is None:
            try:
                output = generate_cached_output_retry2(question, video_path, model_class, model)
            except Exception as e:
                print(f"Model {model_class} returned None for prompt {question}")
                continue
        print(f"{question}\nOutput: {output}")
        if isinstance(output, list):
            output = output[0]
            print(f"Output is a list. Using first element: {output}")
        parsed_output = parse_mc_answer(output, d["answer_choices"], all_letters)
        correct = set(parsed_output) == set(gt_letters)
        output_dict = {
            "output": output,
            "parsed_output": parsed_output,
            "correct": correct
        }
        d["results"] = {model_class: output_dict} if "results" not in d else {**d["results"], model_class: output_dict}
        results.append(d)

        # Update shared progress queue
        if progress_queue is not None:
            progress_queue.put(1)

    return results

def run_prompts_parallel(data, model_classes, videos_dir, num_gpus=8, models_per_gpu=4):
    """
    Distribute the workload across GPUs and spawn parallel processes for each model.
    """
    mp.set_start_method('spawn', force=True)

    # Setup a manager for shared state
    manager = Manager()
    progress_queue = manager.Queue()
    current_model_status = manager.dict()

    # Start the progress updating as an additional process
    progress_process = mp.Process(target=update_progress, args=(progress_queue, len(data) * len(model_classes), current_model_status))
    progress_process.start()

    combined_results = []
    for model_class in model_classes:
        # Split data into chunks for each GPU
        data_chunks = [data[i::num_gpus] for i in range(num_gpus)]

        # Create arguments for each GPU-model process
        args = []
        # get the specific gpus given from the cmd line
        gpu_nums = list(range(num_gpus))
        for gpu_id in gpu_nums:
            data_partitions = [data_chunks[gpu_id][i::models_per_gpu] for i in range(models_per_gpu)]
            for i in range(models_per_gpu):
                args.append((data_partitions[i], [gpu_id], model_class, videos_dir))

        # Run each process in parallel
        with Pool(processes=num_gpus * models_per_gpu) as pool:
            all_results = pool.starmap(partial(run_prompt_on_gpu_for_model, progress_queue=progress_queue, current_model_status=current_model_status), args)

        # Combine results from all processes
        for result in all_results:
            combined_results.extend(result)

    progress_queue.put(None)
    progress_process.join()

    return combined_results

def create_composite_key(entry):
    """
    Generate a composite key for a dictionary by combining all key-value pairs except for 'results'.
    """
    return tuple((k, frozenset(entry[k])) if isinstance(entry[k], list) else (k, entry[k]) for k in entry if k in ["task_name", "scenario_name", "question", "gt", "answer_choices", "video_id", "dataset"] and k != "results")

def combine_results(results):
    """
    Combines multiple result entries, removing redundancy and merging by a composite key.
    """
    combined_results = {}
    
    for entry in results:
        # Generate a composite key using all fields except 'results'
        composite_key = create_composite_key(entry)
        
        if composite_key not in combined_results:
            # Initialize the entry with all fields except 'results'
            combined_results[composite_key] = {k: entry[k] for k in entry if k != "results"}
            combined_results[composite_key]["results"] = {}
        
        # Merge the current entry's results with the existing entry
        combined_results[composite_key]["results"].update(entry["results"])
    
    # Return combined results as a list
    return list(combined_results.values())

def save_results(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

def main(prompt_file, output_file, videos_dir, models, num_gpus, models_per_gpu, num_instances_per_task, num_instances_per_task_scenario):
    original_data = load_prompts(prompt_file)
    original_data = [d for d in original_data if (d["scenario_name"] in ["NaturalSelection", "Counterfactual", "Misleading", "Distraction"])]  # bc some prompts may be empty or null, e.g., OCR.

    # Get unique tasks
    tasks = set(d["task_name"] for d in original_data)
    # This will hold your final dataset
    data = []
    # For each task, we handle scenarios
    for task in tasks:
        # Get task-specific data
        task_data = [d for d in original_data if d["task_name"] == task]
        # Limit to num_instances_per_task
        if num_instances_per_task != -1:
            task_data = task_data[:num_instances_per_task]
        # Get unique scenarios for this task
        scenarios = set(d["scenario_name"] for d in task_data)
        # For each scenario within the task, we further trim the dataset
        for scenario in scenarios:
            # Get scenario-specific data within the current task
            scenario_data = [d for d in task_data if d["scenario_name"] == scenario]
            # Limit to num_instances_per_task_scenario
            if num_instances_per_task_scenario != -1:
                scenario_data = scenario_data[:num_instances_per_task_scenario]
            print(f"Running {len(scenario_data)} instances for scenario {scenario} in task {task}")
            # Add to the final dataset
            data.extend(scenario_data)
    # Report the data count for this task
    print(f"Running {len(data)} total instances for all tasks and scenarios")

    # Get cuda visible devices, then get the gpus per model to use.
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_visible_devices = cuda_visible_devices.split(",") if cuda_visible_devices else []
    print(f"Using CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

    if num_gpus > 1:
        results = data
        # for model_class in models:
        results = run_prompts_parallel(results, models, videos_dir, num_gpus=num_gpus, models_per_gpu=models_per_gpu)
            # # clear models from memory
            # torch.cuda.empty_cache()
    else:
        results = []
        for model_class in models:
            progress_queue = None  # Manager().Queue()  # None
            current_model_status = None  # Manager().dict()  # None
            gpu_ids = [g for g in range(MODEL_TO_GPUS.get(model_class, 1))]
            results.extend(run_prompt_on_gpu_for_model(data, gpu_ids, model_class, videos_dir, progress_queue, current_model_status))

    # Combine duplicate entries so that each unique identifier appears once.
    results = combine_results(results)
    # Save the results to a file
    save_results(results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, help="The file with the prompts to run.")
    parser.add_argument("--models", type=str, nargs="+", help="List of model names to run. Must be one or more of: " + ", ".join(MODELS.keys()))
    # parser.add_argument("--parallel", action="store_true", help="Run the models in parallel on multiple GPUs.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--models_per_gpu", type=int, default=2, help="Number of models to run on each GPU.")
    parser.add_argument("--num_instances_per_task", type=int, default=-1, help="Max number of instances to run for each task.")
    parser.add_argument("--num_instances_per_task_scenario", type=int, default=-1, help="Max number of instances to run for each scenario per task.")
    args = parser.parse_args()
    # Example usage: python V2T_inference_non_surrogate.py --prompt_file ../data/v2t_final/V2T_final_prompts_pre_manual.json --models qwen_vl2 --num_gpus 1 --models_per_gpu 1 --num_instances_per_task 1 --num_instances_per_task_scenario -1
    # Ensure the models are valid
    for model in args.models:
        if model not in MODELS:
            raise ValueError(f"Model {model} not found. Must be one of: {', '.join(MODELS.keys())}")
    print(f"Running models: {args.models}")
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(os.path.join("output_files", current_datetime), exist_ok=True)
    output_file = os.path.join("output_files", current_datetime, os.path.splitext(os.path.basename(args.prompt_file))[0] + "_final_results.json")
    print(f"Saving results to {output_file}")
    videos_dir = f"text/{current_datetime}"
    os.makedirs(videos_dir, exist_ok=True)
    main(args.prompt_file, output_file, videos_dir, args.models, args.num_gpus, args.models_per_gpu, args.num_instances_per_task, args.num_instances_per_task_scenario)
    print(f"Saved results to {output_file}")
