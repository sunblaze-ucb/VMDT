####pip install lumaai (before running this code)####

import time, requests
from pathlib import Path
from lumaai import LumaAI
from .base import T2VBaseModel, T2VOutput, gen_id

class Luma(T2VBaseModel): 
    def load_model(self):
        self.client = LumaAI(
            auth_token='',
        )
    def generate_videos(self,prompts: list[str],output_dir: Path): ##############batch size =1, so please send a prompt array with length of 1
        generation = self.client.generations.create(
            prompt=prompts[0],
        )
        for _ in range(3):
            outputs=[]
            try:
                completed=False
                while not completed:
                    generation = self.client.generations.get(id=generation.id)
                    if generation.state == "completed":
                        completed=True
                    elif generation.state == "failed":
                        raise RuntimeError(f"Generation failed: {generation.failure_reason}")
                    time.sleep(3)

                video_url = generation.assets.video

                # download the video
                response = requests.get(video_url, stream=True)
                save_path = output_dir / gen_id(prefix="1" + "_", suffix="_" + "".join(val[0].lower() if (len(val) > 0 and val[0].isalnum()) else "" for val in prompts[0].split(" ")) + ".mp4")
                # with open(f'{output_dir}/{generation.id}.mp4', 'wb') as file:
                with open(save_path, 'wb') as file:
                    file.write(response.content)
                outputs.append(
                    T2VOutput(
                        text_input=prompts[0],
                        # video_path=output_dir /generation.id,
                        video_path=save_path,
                    )
                )
                return outputs
            except Exception as e:
                print(e)
                time.sleep(1)
        return outputs.append(
                    T2VOutput(
                        text_input=prompts[0],
                        video_path='error',
                    )
                )