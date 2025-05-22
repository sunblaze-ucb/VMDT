"""
Main script for running the video-to-text hallucination evaluation pipeline.
"""
import argparse
import os
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download
from hallucination.video_text.V2T_inference_non_surrogate import main as _inference_main, MODELS as _INFERENCE_MODELS

def get_args():
    parser = argparse.ArgumentParser(
        description="Run video-to-text hallucination generation, evaluation, and scoring."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        choices=list(_INFERENCE_MODELS.keys()),
        help="Model ID to use for generation",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="",
        choices=[
            "NaturalSelection",
            "Misleading",
            "Counterfactual",
            "Distraction",
        ],
        help="Scenario to filter (if unset, runs all scenarios)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (only run one instance of each scenario)",
    )
    return parser.parse_args()

def download_dataset(vids_dir):
    # vids_dir = os.path.join(vids_dir, "hallucination/video_text/vids")
    # raw = load_dataset("mmfm-trust/V2T", "hallucination")
    # data = []
    # for task_name, ds in raw.items():
    #     # ds_with_task = ds.add_column("task", [task_name] * len(ds))
    #     data.append(ds)
    # data = concatenate_datasets(data)
    # data = [V2TInstance.parse_obj(d) for d in data][:10]  # small number for testing
    # data = [d for d in data][:10]  # small number for testing
    
    if not os.path.exists("hallucination/vids"):
        snapshot_download(
            repo_id="mmfm-trust/V2T",  
            repo_type="dataset",
            local_dir=vids_dir,              
            local_dir_use_symlinks=False,          
            allow_patterns="hallucination/vids/**",          
        )
    print(vids_dir)

def main():
    args = get_args()
    model = args.model_id
    scenario = args.scenario

    KNOWN_MODEL_MODALITY = list(_INFERENCE_MODELS.keys())
    SCENARIOS = [
        "NaturalSelection",
        "Misleading",
        "Counterfactual",
        "Distraction",
    ]
    if model not in KNOWN_MODEL_MODALITY:
        raise ValueError(f"Unknown model: {model}")
    if scenario and scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir,
            "results", "v2t_results", "hallucination"
        )
    )
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # videos_dir = os.path.join(base_dir, model, date)
    # videos_dir = "VMDT"
    # os.makedirs(videos_dir, exist_ok=True)

    dataset = load_dataset("mmfm-trust/V2T", name="hallucination")
    target_scenarios = [scenario] if scenario else SCENARIOS
    data = []
    for sc in target_scenarios:
        if sc not in dataset:
            raise ValueError(f"Scenario {sc} not found in dataset")
        data.extend(dataset[sc])

    # Download the dataset
    download_dataset("./")
    videos_dir = "hallucination/"  # because 'file_name' attribute is 'vids/{video_id}.mp4'

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if scenario:
        filename = f"{scenario}_{timestamp}.json"
    else:
        filename = f"{model}_{timestamp}.json"
    output_file = os.path.join(base_dir, filename)
    _inference_main(
        data,
        output_file,
        videos_dir,
        [model],
        target_scenarios,
        num_gpus=1,
        models_per_gpu=1,
        num_instances_per_task=-1 if not args.debug else 1,
        num_instances_per_task_scenario=-1 if not args.debug else 1,
    )


if __name__ == "__main__":
    main()
