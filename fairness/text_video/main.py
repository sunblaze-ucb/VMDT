from VMDT.fairness.text_video.model_responses import model_responses
from VMDT.fairness.text_video.video_frame import extract_videoframe
from VMDT.fairness.text_video.FairFace.predict import main2
from VMDT.fairness.text_video.score_calculation import score_calculation
from VMDT.fairness.text_video.average import average
import argparse

KNOWN_MODEL_MODALITY = ['Nova','Pika','Luma','OpenSora1.2','Vchitect2','VideoCrafter2','CogVideoX-5B']
SCENARIOS=['stereotype', 'decision_making', 'factual_accuracy']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for generation', choices=['Nova','Pika','Luma','OpenSora1.2','Vchitect2','VideoCrafter2','CogVideoX-5B'])
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
        extract_videoframe(model)
        main2(model)
        score_calculation()
        average()
    elif "stereotype" == scenario or "factual_accuracy" == scenario:
        scenario_list=[scenario]
        model_responses(model,[f'{s}.csv' for s in scenario_list])
        extract_videoframe(model, scenario_list)
        main2(model, scenario_list)
        score_calculation(scenario_list)
        average()
    elif "decision_making" == scenario:
        scenario_list=['hiring', 'finance', 'education']
        model_responses(model,[f'{s}.csv' for s in scenario_list])
        extract_videoframe(model, scenario_list)
        main2(model, scenario_list)
        score_calculation(['decision_making'])
        average()