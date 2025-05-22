"""
Main script for running the text-to-video hallucination evaluation pipeline.
"""
import os
import argparse
from datetime import datetime

from datasets import load_dataset
from hallucination.text_video.T2V_inference_non_surrogate import main as _inference_main, MODELS as _INFERENCE_MODELS
# from .average import average

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

    # average()
