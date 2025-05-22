"""Given a json file with prompts, run the prompts on various T2V models and save the results to a json file.

JSON keys: "task_name": task_name,
                    "scenario_name": scenario_name,
                    "temporal": temporal,
                    "prompt": new_prompt,
                    "gt": gt

We will implement models in various files which we import. Each model has an inference function that takes a prompt and returns a response (i.e., a path to a video).

The videos will be saved as mp4 files, with a dir specified by a unique id. Each video will be saved as video.mp4, e.g., videos/2021-08-31_14-30-00/uuid1/video.mp4. Later, frames and other information can be saved in the same directory that are unique to the video.

"""

import json
import os
import argparse
import pandas as pd
from datetime import datetime
import uuid
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm
from pathlib import Path

import torch.distributed as dist
from torch.distributed import is_initialized as dist_is_initialized


from models.t2v_models import load_t2v_model

MODELS = {
    "CogVideoX-5B": partial(load_t2v_model, "CogVideoX-5B"),
    "VideoCrafter2": partial(load_t2v_model, "VideoCrafter2"),
    "Vchitect2": partial(load_t2v_model, "Vchitect2"),
    "OpenSora1.2": partial(load_t2v_model, "OpenSora1.2"),
    "Luma": partial(load_t2v_model, "Luma"),
    "Pika": partial(load_t2v_model, "Pika"),
    "NovaReel": partial(load_t2v_model, "NovaReel"),
}

 # Removed load_prompts; using HuggingFace dataset directly in main

## [ADDED]
def is_distributed():
    return dist.is_available() and dist.is_initialized()

## [ADDED]
def gather_data_across_processes(local_data):
    """
    Gather JSON-serializable local_data from each rank onto rank 0,
    return the combined list on rank 0, otherwise return None on other ranks.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Convert to string so we can gather as an object
    local_data_str = json.dumps(local_data, ensure_ascii=False)
    gathered = [None] * world_size

    dist.all_gather_object(gathered, local_data_str)

    if rank == 0:
        merged = []
        for gstr in gathered:
            merged.extend(json.loads(gstr))
        return merged
    else:
        return None

## [ADDED]
def gather_data_to_rank0(local_data):
    """
    Gathers JSON-serializable local_data from each rank onto rank 0.
    Returns the combined list on rank 0; returns None on other ranks.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Convert local_data (possibly empty) to a JSON string
    local_data_str = json.dumps(local_data, ensure_ascii=False)

    # Rank 0 preps a list to receive strings from all ranks; others pass []
    if rank == 0:
        gather_list = [None for _ in range(world_size)]
    else:
        gather_list = []

    # Everyone calls gather_object in the same order
    dist.gather_object(local_data_str, gather_list, dst=0)

    # Optional barrier so no rank exits gather_data before rank 0 merges
    dist.barrier()

    if rank == 0:
        merged = []
        for data_str in gather_list:
            # data_str might be "[]"
            merged.extend(json.loads(data_str))
        return merged
    else:
        return None

def run_prompts(data, model_class, videos_dir):
    # Initialize the model and move it to the assigned device
    model = MODELS[model_class]()

    # current_model_status['model_name'] = model_class
    model_name = model_class

    print(f"Running model {model_name} on {len(data)} prompts")
    video_path = Path(videos_dir)
    # if model_class is videocrafter2, we do not want to run any temporal data (d["temporal"])
    if model_name == "VideoCrafter2":
        original_data_len = len(data)
        data = [d for d in data if not d["temporal"]]
        print(f"Running VideoCrafter2 on {len(data)} prompts. Removed {original_data_len - len(data)} temporal prompts bc VideoCrafter2 produces videos that are too short.")
    if len(data) == 0:
        print(f"[No data, skipping model inference.")
        return {}

    if all("Distraction" in d["scenario_name"] for d in data) and model_name == "OpenSora1.2":
        print("Running OpenSora1.2 on distraction scenarios: Not cleaning prompts.")
        out = model.generate_videos([d["prompt"] for d in data], video_path, clean_prompts=False)
    elif model_name in ["Luma", "Pika", "NovaReel"]:
        indices = None
        if all(d.get("T2V_final_index") is not None for d in data):
            indices = [d["T2V_final_index"] for d in data]
            # prompt_indices = [{"prompt": d["prompt"], "index": d["T2V_final_index"]} for d in data]
        assert len(data) == len(indices), f"Must have the same number of prompts and indices. {len(data)=} {len(indices)=}"
        assert len(indices) == len(set(indices)), f"Must have unique indices. {len(indices)=} {len(set(indices))=}"
        out = model.generate_videos([d["prompt"] for d in data],
                                    video_path,
                                    indices=indices)
    else:
        out = model.generate_videos([d["prompt"] for d in data], video_path)
    print(f"{out=}")
    for o in out:
        for d in data:
            if d["prompt"] == o.text_input:
                d["results"] = {model_name: str(o.video_path)} if "results" not in d else {**d["results"], model_name: str(o.video_path)}
    return data

def run_prompt_on_gpu_for_model(data_part, gpu_id, model_class, videos_dir, progress_queue, current_model_status):
    """
    Run a single model on a GPU for a portion of the data.
    """
    # Set the device for this process
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    data_part = run_prompts(data_part, model_class, videos_dir)
    return data_part


def update_progress(progress_queue, total_length, current_model_status):
    """
    Continuously updates the progress bar by checking the progress queue and tracking current models.
    """
    overall_pbar = tqdm(total=total_length, desc="Total progress", unit="instance")
    completed = 0

    # Keep track of where we are
    while completed < total_length:
        # Update overall progress
        progress_queue.get()
        completed += 1
        overall_pbar.update(1)

        # Use a separate bar for the current model
        current_model_name = current_model_status.get('model_name', 'Unknown Model')
        current_model_pbar = tqdm(total=total_length, desc=f"Processing {current_model_name}", unit="instance", leave=True)
        current_model_pbar.update(completed)

    overall_pbar.close()

def run_prompts_parallel(data, model_classes, videos_dir, num_gpus=8, models_per_gpu=4):
    """
    Distribute the workload across GPUs and spawn parallel processes for each model.
    """
    mp.set_start_method('spawn', force=True)

    # Setup a manager for shared state
    manager = Manager()
    progress_queue = None  # manager.Queue()
    current_model_status = {}  # manager.dict()

    # Start the progress updating as an additional process
    # progress_process = mp.Process(target=update_progress, args=(progress_queue, len(data) * len(model_classes), current_model_status))
    # progress_process.start()

    combined_results = []
    for model_class in model_classes:
        # Split data into chunks for each GPU
        data_chunks = [data[i::num_gpus] for i in range(num_gpus)]

        # Create arguments for each GPU-model process
        args = []
        for gpu_id in range(num_gpus):
            data_partitions = [data_chunks[gpu_id][i::models_per_gpu] for i in range(models_per_gpu)]
            for i in range(models_per_gpu):
                args.append((data_partitions[i], gpu_id, model_class, videos_dir))

        # Run each process in parallel
        with Pool(processes=num_gpus * models_per_gpu) as pool:
            all_results = pool.starmap(partial(run_prompt_on_gpu_for_model, progress_queue=progress_queue, current_model_status=current_model_status), args)

        # Combine results from all processes
        for result in all_results:
            combined_results.extend(result)

    return combined_results

def create_composite_key(entry):
    """
    Generate a composite key for a dictionary by combining all key-value pairs except for 'results'.
    """
    return tuple((k, frozenset(entry[k])) if isinstance(entry[k], list) else (k, entry[k]) for k in entry if k in ["task_name", "scenario_name", "temporal", "prompt", "video_id"])
                 # not in ["results", "ground_truth", "ground_truth_not_shown", "spans", "map_to_other_properties", "distraction_map"])  # ignore ground_truth and ground_truth_not_shown bc they are dicts and not necessary for matching if we have the same prompt.

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
        # May not be there due to errors (OpenSora1.2)
        if "results" in entry:
            combined_results[composite_key]["results"].update(entry["results"])
    
    # Return combined results as a list
    return list(combined_results.values())

def save_results(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_file}")

def main(dataset_data, output_file, videos_dir, models, scenarios, num_gpus, models_per_gpu, num_instances_per_task, num_instances_per_task_scenario):
    # dataset_data: list of examples loaded from the HuggingFace dataset
    # Remove entries without a prompt
    original_data = [d for d in dataset_data if d.get("prompt")]
    # Filter by scenario names if provided
    if scenarios:
        original_data = [d for d in original_data if d.get("scenario_name") in scenarios]
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
    print(f"Running {len(data)} total instances for all tasks and scenarios")

    if num_gpus > 1:
        results = data
        results = run_prompts_parallel(results, models, videos_dir, num_gpus=num_gpus, models_per_gpu=models_per_gpu)
    else:
        # models = [MODELS[model]() for model in args.models]
        for model_class in models:
            results = run_prompts(data, model_class, videos_dir)

    # Combine duplicate entries so that each unique identifier appears once.
    results = combine_results(results)
    save_results(results, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios", type=str, nargs='+',
        help="List of scenarios to run. Must be one or more of: NaturalSelection, CoOccurrence, Distraction, RandomDistraction, AllDistraction, Counterfactual, Misleading"
    )
    parser.add_argument("--models", type=str, nargs="+", help="List of model names to run. Must be one or more of: " + ", ".join(MODELS.keys()))
    # parser.add_argument("--parallel", action="store_true", help="Run the models in parallel on multiple GPUs.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--models_per_gpu", type=int, default=2, help="Number of models to run on each GPU.")
    parser.add_argument("--num_instances_per_task", type=int, default=-1, help="Max number of instances to run for each task.")
    parser.add_argument("--num_instances_per_task_scenario", type=int, default=-1, help="Max number of instances to run for each scenario per task.")
    args = parser.parse_args()
    # Ensure the models are valid
    for model in args.models:
        if model not in MODELS:
            raise ValueError(f"Model {model} not found. Must be one of: {', '.join(MODELS.keys())}")
    print(f"\nRunning models: {args.models} on scenarios: {args.scenarios}")
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(os.path.join("output_files", current_datetime), exist_ok=True)
    # Define output file and video directory
    output_file = os.path.join("output_files", current_datetime, "final_results.json")
    print(f"Saving results to {output_file}")
    videos_dir = f"videos/{current_datetime}"
    os.makedirs(videos_dir, exist_ok=True)
    print(f"Saving videos to {videos_dir}")
    # Load HuggingFace dataset for T2V tasks
    dataset_dict = load_dataset("mmfm-trust/T2V", name="hallucination")
    # Build dataset list by scenario splits
    scenarios = args.scenarios if args.scenarios else list(dataset_dict.keys())
    dataset_list = []
    for sc in scenarios:
        if sc not in dataset_dict:
            raise ValueError(f"Scenario {sc} not found; available: {list(dataset_dict.keys())}")
        dataset_list.extend(list(dataset_dict[sc]))
    # Run main processing on the filtered dataset
    main(
        dataset_list,
        output_file,
        videos_dir,
        args.models,
        scenarios,
        args.num_gpus,
        args.models_per_gpu,
        args.num_instances_per_task,
        args.num_instances_per_task_scenario,
    )
