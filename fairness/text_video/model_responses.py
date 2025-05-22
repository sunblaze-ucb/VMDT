from pathlib import Path

# directory that this .py file lives in
HERE = Path(__file__).resolve().parent
import torch
import pandas as pd
from ...models.t2v_models import load_t2v_model

def model_responses(model_name, promptfiles=['stereotype.csv','factual_accuracy.csv','hiring.csv','finance.csv','education.csv']):
    repeat=4
    for promptfile in promptfiles:
        print(f"[*] starting the file: {promptfile}...")
        
        #for model_name in ["CogVideoX-5B"]:
        torch.cuda.empty_cache()
        model_out_dir = Path(f"{HERE}/model_responses/{model_name}/{promptfile[:promptfile.index('.csv')]}")
        model_out_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"[*] loading model {model_name}...")
        model = load_t2v_model(model_name)
        for r in range(repeat):
            promptpd=pd.read_csv(f'{HERE}/{promptfile}')
            prompts = promptpd['prompt'].tolist()
            outputpd=pd.DataFrame({})
            promptpd['repeat']=r
            promptpd['output']=''
            video_outputs = model.generate_videos(prompts, model_out_dir)
            for i,video_output in enumerate(video_outputs):
                promptpd.loc[promptpd.index[i], 'output'] =str(video_output.video_path)[str(video_output.video_path).index(model_name):]
            outputpd = pd.concat([outputpd, promptpd.copy()], ignore_index=True)
            outputpd.to_csv(f"{HERE}/model_responses/{model_name}_{promptfile[:promptfile.index('.csv')]}_{r}.csv",index=False)
        
        
        finalpd=pd.DataFrame({})
        for r in range(repeat):
            outputpd=pd.read_csv(f"{HERE}/model_responses/{model_name}_{promptfile[:promptfile.index('.csv')]}_{r}.csv")
            finalpd = pd.concat([finalpd, outputpd.copy()], ignore_index=True)
        finalpd.to_csv(f"{HERE}/model_responses/{model_name}_{promptfile[:promptfile.index('.csv')]}.csv",index=False)

#model_responses("VideoCrafter2",['factual_accuracy.csv'])