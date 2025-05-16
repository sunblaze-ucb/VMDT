####pip install lumaai (before running this code)####

import time, requests
from pathlib import Path
from tqdm import tqdm
from typing import List
from lumaai import LumaAI
from .base import T2VBaseModel, T2VOutput
import json
import os
import dotenv
dotenv.load_dotenv()

class Luma(T2VBaseModel):
    def load_model(self):
        self.client = LumaAI(
            auth_token=os.getenv("LUMA_API_KEY"),
        )
    def generate_videos(self,prompts: list[str],output_dir: Path, indices: List[int] = None, dry_run=False):
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        outputs=[]
        for i, prompt in enumerate(tqdm(prompts)):
            print(f"Generating video for prompt: {prompt} with index {indices[i] if indices is not None else i}")
            if dry_run:
                video_file_end = generation.id if indices is None else indices[i]
                print(f"Saving video to {output_dir}/{video_file_end}.mp4")
                with open(f'{output_dir}/{video_file_end}.mp4', 'wb') as file:
                    file.write(b"")
                continue
            generation = self.client.generations.create(
                prompt=prompt,
            )
            cnt=0
            for _ in range(3):
                try:
                    
                    start_time = time.time()
                    completed=False
                    
                    while not completed:
                        if time.time()-start_time>600:
                            raise RuntimeError("Timeout")
                        
                        generation = self.client.generations.get(id=generation.id)
                        
                        if generation.state == "completed":
                            completed=True
                        elif generation.state == "failed":
                            raise RuntimeError(f"Generation failed: {generation.failure_reason}")
                        
                        time.sleep(3)

                    video_url = generation.assets.video

                    # download the video
                    response = requests.get(video_url, stream=True)
                    video_file_end = generation.id if indices is None else indices[i]
                    print(f"Saving video to {output_dir}/{video_file_end}.mp4")
                    with open(f'{output_dir}/{video_file_end}.mp4', 'wb') as file:
                        file.write(response.content)
                        
                    outputs.append(
                        T2VOutput(
                            text_input=prompt,
                            video_path=f"{output_dir}/{video_file_end}.mp4",
                        )
                    )
                
                    json_output = {
                        "text_input": prompt,
                        "video_path": f"{output_dir}/{video_file_end}.mp4",
                        "generation_id": generation.id,
                    }
                    
                    with open("luma_progress.jsonl", "a") as pf:
                        pf.write(json.dumps(json_output) + "\n")
        
                    break
                
                except Exception as e:
                    cnt+=1
                    print(e)
                    time.sleep(1)
                    
                    
            if cnt==3:
                outputs.append(
                    T2VOutput(
                        text_input=prompt,
                        video_path="error",
                    )
                )
                
                json_output = {
                    "text_input": prompt,
                    "video_path": "error",
                }
                    
                with open("luma_progress.jsonl", "a") as pf:
                    pf.write(json.dumps(json_output) + "\n")
                
        return outputs
