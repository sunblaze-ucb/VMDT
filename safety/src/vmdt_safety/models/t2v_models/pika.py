import os
import random
import time
from pathlib import Path

import requests
from tqdm import tqdm

from .base import T2VBaseModel, T2VError, T2VOutput


def download_video(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Video successfully downloaded to {save_path}")
    else:
        print(f"Failed to download video. Status code: {response.status_code}")


class Pika(T2VBaseModel):
    def load_model(self, **kwargs):
        api_key = os.environ.get("PIKA_API_KEY")
        if not api_key:
            raise ValueError("PIKA_API_KEY environment variable not set")
        self.api = api_key
        self.video_id = None

    def _generate_single_video(
        self, prompt: str, output_dir: Path, max_trials: int = 3
    ) -> T2VOutput:
        for trial in range(max_trials):
            try:
                self.video_id = None
                # Step 1: Request video generation
                url = "https://devapi.pika.art/generate/2.2/t2v"
                payload = {
                    "promptText": prompt,
                    "seed": random.randint(0, 100000),
                    "duration": 5,
                    "resolution": "720p",
                }
                headers = {
                    "X-API-KEY": self.api,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                }
                response = requests.post(url, data=payload, headers=headers)
                resp_json = response.json()
                if response.status_code == 200:
                    self.video_id = resp_json["video_id"]
                elif (
                    "message" in resp_json
                    and "Your API key has reached its generate limit"
                    in resp_json["message"]
                ):
                    return T2VOutput(text_input=prompt, error=resp_json["message"])
                elif (
                    "message" in resp_json
                    and "prompt not allowed" in resp_json["message"]
                ):
                    return T2VOutput(text_input=prompt, error=T2VError.SAFETY_REFUSAL)
                else:
                    raise RuntimeError(f"Generation failed: {resp_json}")

                # Step 2: Poll for completion
                completed = False
                time.sleep(30)
                url = f"https://devapi.pika.art/videos/{self.video_id}"
                headers = {"X-API-KEY": self.api, "Accept": "application/json"}
                max_timeout = 10 * 60
                while not completed:
                    response = requests.get(url, headers=headers)
                    resp_json = response.json()
                    if resp_json.get("status") == "finished":
                        completed = True
                    elif resp_json.get("status") == "failed":
                        if "prompt not allowed" in resp_json.get("message", ""):
                            return T2VOutput(
                                text_input=prompt, error=T2VError.SAFETY_REFUSAL
                            )
                        raise RuntimeError(f"Generation failed: {resp_json}")
                    else:
                        time.sleep(5)
                        max_timeout -= 5
                        if max_timeout <= 0:
                            raise TimeoutError("Generation timeout")

                # Step 3: Download video
                url_link = resp_json["url"]
                video_path = output_dir / f"{self.video_id}.mp4"
                download_video(url_link, video_path)

                return T2VOutput(
                    text_input=prompt,
                    video_path=video_path,
                )
            except Exception as e:
                if trial < max_trials - 1:
                    print(f"Attempt {trial + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    return T2VOutput(text_input=prompt, error=str(e))

    def generate_videos(
        self, text_inputs: list[str], output_dir: Path, max_trials: int = 3, **kwargs
    ) -> list[T2VOutput]:
        outputs = []
        for prompt in tqdm(text_inputs):
            output = self._generate_single_video(prompt, output_dir, max_trials)
            outputs.append(output)
        return outputs
