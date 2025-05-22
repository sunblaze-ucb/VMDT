import os
from pathlib import Path
from typing import List

from openai import OpenAI

from .base import V2TBaseModel, V2TError, V2TOutput
from .utils import process_video


class GPT(V2TBaseModel):
    def load_model(self, **kwargs):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_id = kwargs.get("model_id")
        self.max_trials = 3

    def _generate_text(
        self, base64_frames: List[str], prompt: str, video_path: Path, system_prompt: str = "" 
    ) -> V2TOutput:
        for _ in range(self.max_trials):
            try:
                if system_prompt:
                    messages = [
                        {   "role": "system", 
                            "content":system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "These are the frames from the video.",
                                },
                                *map(
                                    lambda x: {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpg;base64,{x}",
                                            "detail": "low",
                                        },
                                    },
                                    base64_frames,
                                ),
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                else:
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
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpg;base64,{x}",
                                            "detail": "low",
                                        },
                                    },
                                    base64_frames,
                                ),
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]

                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0,
                )
                text_output = (
                    response.choices[0].message.content if response.choices else None
                )
                return V2TOutput(video_input=video_path, text_output=text_output)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                if "rejected as a result of our safety system" in str(e):
                    return V2TOutput(
                        video_input=video_path, error=V2TError.SAFETY_REFUSAL
                    )

        return V2TOutput(video_input=video_path, error=V2TError.GENERATION_ERROR)

    def generate_text(self, video_path: Path, prompt: str, system_prompt: str = "" ) -> V2TOutput:
        base64_frames = process_video(video_path)
        return self._generate_text(base64_frames, prompt, video_path, system_prompt)

    def generate_texts(
        self, video_inputs: List[Path], prompts: List[str], system_prompt: str = "" 
    ) -> List[V2TOutput]:
        return [
            self.generate_text(video, prompt, system_prompt)
            for video, prompt in zip(video_inputs, prompts)
        ]
