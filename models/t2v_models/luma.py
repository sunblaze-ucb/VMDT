import time, requests, os
from pathlib import Path
from tqdm import tqdm
from lumaai import LumaAI
from .base import T2VBaseModel, T2VOutput

class Luma(T2VBaseModel):
    def load_model(self):
        self.client = LumaAI(
            auth_token=os.environ['LUMA_API'],
        )
    def generate_videos(self,prompts: list[str],output_dir: Path):
        outputs=[]
        for prompt in tqdm(prompts):
            cnt=0
            flag=0
            for _ in range(3):
                try:
                    generation = self.client.generations.create(
                                    prompt=prompt,
                                )
                    completed=False
                    start = time.time()
                    while not completed:
                        generation = self.client.generations.get(id=generation.id)
                        if generation.state == "completed":
                            completed=True
                        elif generation.state == "failed":
                            raise RuntimeError(f"Generation failed: {generation.failure_reason}")
                        time.sleep(3)
                        end = time.time()
                        if end-start>=300:
                            print('Infinite loop...')
                            time.sleep(3)
                            break

                    video_url = generation.assets.video

                    # download the video
                    response = requests.get(video_url, stream=True)
                    with open(f'{output_dir}/{generation.id}.mp4', 'wb') as file:
                        file.write(response.content)
                    outputs.append(
                        T2VOutput(
                            text_input=prompt,
                            video_path=output_dir /f'{generation.id}.mp4',
                        )
                    )
                    break
                except Exception as e:
                    cnt+=1
                    if 'prompt not allowed' in str(e):
                        flag=1
                        outputs.append(
                            T2VOutput(
                                text_input=prompt,
                                video_path=output_dir / 'error',
                            )
                        )
                        break
                    print(e)
                    time.sleep(1)
            if cnt==3 and flag==0:
                outputs.append(
                            T2VOutput(
                                text_input=prompt,
                                video_path=output_dir / 'error',
                            )
                        )
        return outputs
