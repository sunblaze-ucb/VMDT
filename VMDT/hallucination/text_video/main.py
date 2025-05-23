"""
Main script for running the text-to-video hallucination evaluation pipeline.
"""
import os
import time
import subprocess
import argparse
from datetime import datetime

from datasets import load_dataset
from VMDT.hallucination.text_video.T2V_inference_non_surrogate import main as _inference_main, MODELS as _INFERENCE_MODELS
from VMDT.hallucination.text_video.average import average

KNOWN_MODEL_MODALITY = list(_INFERENCE_MODELS.keys())
SCENARIOS = [
    "OCR",
    "Distraction",
    "Counterfactual",
    "Misleading",
    "CoOccurrence",
    "NaturalSelection",
    "Temporal",
]

def start_vllm_server(num_gpus):
    """Start vLLM server in a separate process."""
            # CUDA_VISIBLE_DEVICES=<gpu_ids> vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  # --port 8001 \
  # --host 0.0.0.0 \
  # --allowed-local-media-path $(realpath results/t2v_results/hallucination) \
  # --limit-mm-per-prompt image=5 \
  # --tensor-parallel-size 4 \
  # --max-model-len 8192
    import pdb; pdb.set_trace()
    test_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results", "t2v_results", "hallucination")
    )
    cmd = [
        "conda", "run", "-n", "vllm",
        "vllm", "serve", "Qwen/Qwen2.5-VL-72B-Instruct",
        "--host", "0.0.0.0",
        "--port", "8001",
        "--allowed-local-media-path", os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results", "t2v_results", "hallucination")
        ),
        "--limit-mm-per-prompt", "image=5",
        "--tensor-parallel-size", str(num_gpus),
        "--max-model-len", "8192",
    ]

    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stdout and stderr
        text=True
    )

    for line in process.stdout:
        print(line, end='')  # already includes newline
    process.wait()
    
    # Wait for server to start
    time.sleep(300)  # Give it some time to initialize
    
    return process

def cleanup_vllm(vllm_process):
    if vllm_process.poll() is None:  # If process is still running
        vllm_process.terminate()
        vllm_process.wait()

def get_args():
    parser = argparse.ArgumentParser(
        description="Run text-to-video hallucination generation, evaluation, and scoring."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        choices=KNOWN_MODEL_MODALITY,
        help="Model ID to use for generation",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="",
        choices=SCENARIOS,
        help="Scenario to filter (if unset, runs all scenarios)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (only run one instance of each task)",
    )
    parser.add_argument(
        "--do_not_evaluate",
        action="store_true",
        help="Enable evaluation mode (run evaluation after generation)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = args.model_id
    if model not in KNOWN_MODEL_MODALITY:
        raise ValueError(f"Unknown model: {model}")
    scenario = args.scenario
    if scenario and scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir,
            "results", "t2v_results", "hallucination"
        )
    )
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    videos_dir = os.path.join(base_dir, model, date)
    os.makedirs(videos_dir, exist_ok=True)

    dataset = load_dataset("mmfm-trust/T2V", name="hallucination")
    target_scenarios = [scenario] if scenario else SCENARIOS
    data = []
    for sc in target_scenarios:
        if sc not in dataset:
            raise ValueError(f"Scenario {sc} not found in dataset")
        data.extend(dataset[sc])

    # Fix OCR
    new_target_scenarios = []
    for sc in target_scenarios:
        if sc == "OCR":
            new_target_scenarios.extend(["ComplexBackgroundOCR", "ContradictoryOCR", "MisleadingOCR", "DistortionOCR"])
        else:
            new_target_scenarios.append(sc)
    target_scenarios = new_target_scenarios

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
    
    if not args.do_not_evaluate:
        # Start vLLM server
        vllm_process = start_vllm_server(num_gpus=4)
        import pdb; pdb.set_trace()

        try:
    # Need to run the below:
    # python -m hallucination.text_video.T2V_evaluation_newer --video_json=<output_files> --n_frames=5 --num_instances_per_task=-1 --include_image_in_classification --combine_step --direct_evaluation --use_qwen
            os.system(
                f"python -m VMDT.hallucination.text_video.T2V_evaluation_newer "
                f"--video_json={output_file} "
                f"--n_frames=5 "
                f"--num_instances_per_task=-1 "
                f"--include_image_in_classification "
                f"--combine_step "
                f"--direct_evaluation "
                f"--use_qwen"
            )
        finally:
            # Cleanup vLLM server
            cleanup_vllm(vllm_process)

        # Run the average script
        output_filename = f"{output_file.replace('.json', f'_evaluated.json')}"
        average([output_filename])
