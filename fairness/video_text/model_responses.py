from pathlib import Path, PurePosixPath

# directory that this .py file lives in
HERE = Path(__file__).resolve().parent

from huggingface_hub import list_repo_files, hf_hub_download

from shutil import copy2
import random, sys
import os, json
import pandas as pd
from tqdm import tqdm
from ...models.v2t_models import load_v2t_model
from datasets import load_dataset


def video_download():
    repo_id   = "mmfm-trust/V2T"
    remote_root  = "fairness/vid"          # everything under here is what we’re mirroring
    local_root   = Path(f"{HERE}/vid")         # top-level folder you want on disk (can be ".")

    local_root.mkdir(exist_ok=True)

    video_files = [
        p for p in list_repo_files(repo_id, repo_type="dataset")
        if p.startswith(f"{remote_root}") 
    ]

    for remote in video_files:
        # Download (into the HF cache) and get its local-cache path
        src_path = hf_hub_download(repo_id, filename=remote, repo_type="dataset")

        # Drop the "fairness/" prefix but keep the rest
        rel_path = PurePosixPath(remote).relative_to(remote_root)   # → vid/decision/clip.mp4

        # Combine with your chosen root; create any needed sub-folders
        dest_path = local_root / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if not dest_path.exists():
            copy2(src_path, dest_path)

def stereotype(modelname):
    ds=load_dataset("mmfm-trust/V2T", "fairness")['stereotype']
    ds = [dict(row) for row in ds]
    model=load_v2t_model(modelname)
    results=pd.DataFrame({'videofile':[],
                          'response':[]})
    prompts=[]
    videopaths=[]
    videopaths_simple=[]
    for datapoint in ds:        
        videopaths.append(f'{HERE}/vid/stereotype/{datapoint['video']}')
        videopaths_simple.append(datapoint['video'])
        prompts.append(datapoint['prompt'])
    
    systemprompt=datapoint['system_prompt']
    text_outputs=model.generate_texts(video_inputs=videopaths,prompts=prompts, system_prompt=systemprompt)
    
    for i,text_output in enumerate(text_outputs):
        if text_output.error:
            results.loc[len(results)] = [videopaths_simple[i], 'error']
        else:
            results.loc[len(results)] = [videopaths_simple[i], text_output.text_output]
    
    result_dir = Path(f'{HERE}/model_responses/stereotype')
    result_dir.mkdir(exist_ok=True, parents=True)
    results.to_csv(result_dir / f"{modelname}.csv", index=False)

def decision_making(modelname):
    ds=load_dataset("mmfm-trust/V2T", "fairness")['decision_making']
    ds = [dict(row) for row in ds]
    model=load_v2t_model(modelname)

    results=pd.DataFrame({'videofile':[],
                          'response':[]})
    prompts=[]
    videopaths=[]
    videopaths_simple=[]
    for datapoint in ds:        
        videopaths.append(f'{HERE}/vid/decision_making/{datapoint['video']}')
        videopaths_simple.append(datapoint['video'])
        prompts.append(datapoint['prompt'])
        
    systemprompt=datapoint['system_prompt']
    text_outputs=model.generate_texts(video_inputs=videopaths,prompts=prompts, system_prompt=systemprompt)

    for i,text_output in enumerate(text_outputs):
        if text_output.error:
            results.loc[len(results)] = [videopaths_simple[i], 'error']
        else:
            results.loc[len(results)] = [videopaths_simple[i], text_output.text_output]
    
    result_dir = Path(f'{HERE}/model_responses/decision_making')
    result_dir.mkdir(exist_ok=True, parents=True)
    results.to_csv(result_dir / f"{modelname}.csv", index=False)

def factual_accuracy(modelname,*args):
    videofiles=pd.read_csv(f'{HERE}/factual_accuracy/factual_accuracy.csv')
    results=videofiles.copy()
    model=load_v2t_model(modelname)

    prompts=[]
    videopaths=[]
    videopaths_simple=[]
    for _ ,row in tqdm(videofiles.iterrows()):  
        videopaths.append(f'{HERE}/vid/overkill/{row['video']}')
        videopaths_simple.append(row['video'])
        prompts.append(row['prompt'])
    
    text_outputs=model.generate_texts(video_inputs=videopaths,prompts=prompts)
    responses = [
            'error' if text_output.error else text_output.text_output
            for text_output in text_outputs
        ]
    results['response']=responses 
    result_dir = Path(f'{HERE}/model_responses/factual_accuracy')
    result_dir.mkdir(exist_ok=True, parents=True)
    results.to_csv(result_dir / f"{modelname}.csv", index=False)

def model_responses(model_name, scenarios=['stereotype','decision_making','factual_accuracy']):
    folders = [
        Path(f"{HERE}/vid/decision_making"),
        Path(f"{HERE}/vid/stereotype"),
        Path(f"{HERE}/vid/overkill")
    ]

    if any(not folder.exists() or not any(folder.iterdir()) for folder in folders):
        video_download()

    if 'stereotype' in scenarios:
        stereotype(model_name)

    if 'decision_making' in scenarios:
        decision_making(model_name)

    if 'factual_accuracy' in scenarios:
        factual_accuracy(model_name)