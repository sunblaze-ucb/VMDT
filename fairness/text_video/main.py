from VMDT.fairness.text_video.model_responses import model_responses
from VMDT.fairness.text_video.video_frame import extract_videoframe
from VMDT.fairness.text_video.score_calculation import score_calculation
from VMDT.fairness.text_video.average import average
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent

KNOWN_MODEL_MODALITY = ['Nova','Pika','Luma','OpenSora1.2','Vchitect2','VideoCrafter2','CogVideoX-5B']
SCENARIOS=['stereotype', 'decision_making', 'factual_accuracy']

import subprocess

ENV_NAME   = "dlib_env"
ENV_YAML   = Path(f"{HERE}/FairFace/environment2.yml")  # ← adjust if stored elsewhere

def ensure_env(name: str = ENV_NAME, yaml_file: Path = ENV_YAML):
    """
    Create the conda env if it doesn’t already exist.
    """
    # Ask conda for a JSON list of envs → easier to parse than text
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        check=True,
        text=True,
        capture_output=True
    )
    envs = json.loads(result.stdout)["envs"]
    if any(Path(p).name == name for p in envs):
        return  # already present

    print(f"[setup] creating env '{name}' from {yaml_file} …")
    subprocess.run(
        ["conda", "env", "create", "-n", name, "-f", str(yaml_file)],
        check=True
    )
    print(f"[setup] done.")


def run_main2_in_dlib(model, scenario_list=None):
    script = "-m"  # we’ll run the module as a script
    module = "VMDT.fairness.text_video.FairFace.predict"
    if scenario_list==None:
        args   = [model]
    else:
        args   = [model, scenario_list] 

    subprocess.run(
        ["conda", "run", "-n", "dlib_env", "python", script, module, *args],
        check=True
    )

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
    
    ensure_env()

    if scenario=="":
        model_responses(model)
        extract_videoframe(model)
        run_main2_in_dlib(model)
        score_calculation()
        average()
    elif "stereotype" == scenario or "factual_accuracy" == scenario:
        scenario_list=[scenario]
        model_responses(model,[f'{s}.csv' for s in scenario_list])
        extract_videoframe(model, scenario_list)
        run_main2_in_dlib(model, scenario)
        score_calculation(scenario_list)
        average()
    elif "decision_making" == scenario:
        scenario_list=['hiring', 'finance', 'education']
        model_responses(model,[f'{s}.csv' for s in scenario_list])
        extract_videoframe(model, scenario_list)
        run_main2_in_dlib(model, scenario)
        score_calculation(['decision_making'])
        average()