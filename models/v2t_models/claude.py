import os
from pathlib import Path
from typing import List

from anthropic import Anthropic

from .base import V2TBaseModel, V2TError, V2TOutput
from .utils import process_video


class Claude(V2TBaseModel):
    def load_model(self, **kwargs):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_id = kwargs.get("model_id")
        self.max_trials = 3

    def _generate_text(
        self, base64_frames: List[str], prompt: str, video_path: Path, system_prompt: str = "" 
    ) -> V2TOutput:
        for _ in range(self.max_trials):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "These are the frames from the video.",
                            },
                            *map(
                                lambda x: {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": x,
                                    },
                                },
                                base64_frames,
                            ),
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                if system_prompt:
                    response = self.client.messages.create(
                        model=self.model_id,
                        system=system_prompt,
                        max_tokens=1024,
                        temperature=0,
                        messages=messages,
                    )
                else:
                    response = self.client.messages.create(
                        model=self.model_id,
                        max_tokens=1024,
                        temperature=0,
                        messages=messages,
                    )

                text_output = response.content[0].text
                return V2TOutput(video_input=video_path, text_output=text_output)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

        return V2TOutput(video_input=video_path, error=V2TError.GENERATION_ERROR)

    def generate_text(self, video_path: Path, prompt: str, system_prompt: str = "") -> V2TOutput:
        base64_frames = process_video(video_path)
        return self._generate_text(base64_frames, prompt, video_path,system_prompt)

    def generate_texts(
        self, video_inputs: List[Path], prompts: List[str], system_prompt: str = ""
    ) -> List[V2TOutput]:
        return [
            self.generate_text(video, prompt, system_prompt)
            for video, prompt in zip(video_inputs, prompts)
        ]
