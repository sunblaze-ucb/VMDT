from VMDT.fairness.video_text.model_responses import model_responses
from VMDT.fairness.video_text.fairness_score import fairness_score
from VMDT.fairness.video_text.average import average
import argparse

KNOWN_MODEL_MODALITY = [
    "Qwen2-VL-7B",
    "Qwen2.5-VL-3B",
    "Qwen2.5-VL-7B",
    "Qwen2.5-VL-72B",
    "InternVL2.5-1B",
    "InternVL2.5-2B",
    "InternVL2.5-4B",
    "InternVL2.5-8B",
    "InternVL2.5-26B",
    "InternVL2.5-38B",
    "InternVL2.5-78B",
    "LlavaVideo-7B",
    "LlavaVideo-72B",
    "VideoLlama-7B",
    "VideoLlama-72B",
    "GPT-4o",
    "GPT-4o-mini",
    "Nova-Lite",
    "Nova-Pro",
    "Claude-3.5-Sonnet"
]
SCENARIOS=['stereotype', 'decision_making', 'factual_accuracy']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for generation', choices=KNOWN_MODEL_MODALITY)
    parser.add_argument('--scenario', type=str, default="", help='Scenario type (stereotype, decision_making, factual_accuracy)', choices=['stereotype', 'decision_making', 'factual_accuracy'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    model=args.model_id
    if model not in KNOWN_MODEL_MODALITY:
        raise Exception("Unknown models")
    
    scenario=args.scenario
    if scenario!="" and scenario not in SCENARIOS:
        raise Exception("Unknown scenarios")
    
    if scenario=="":
        model_responses(model)
        fairness_score()
        average()
    else:
        scenario_list=[scenario]
        model_responses(model,scenario_list)
        fairness_score(scenario_list)
        average()