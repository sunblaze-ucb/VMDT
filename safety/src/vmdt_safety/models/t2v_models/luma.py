import os
import time
import requests
from pathlib import Path

from lumaai import LumaAI
from .base import T2VBaseModel, T2VError, T2VOutput, gen_id


class Luma(T2VBaseModel):
    def load_model(self, **kwargs):
        api_key = os.environ.get("LUMA_API_KEY")
        if not api_key:
            raise ValueError("LUMA_API_KEY environment variable not set")

        self.client = LumaAI(
            auth_token=api_key,
        )

    def _generate_single_video(
        self, prompt: str, output_dir: Path, max_trials: int = 3
    ) -> T2VOutput:
        """Generate a single video from a prompt with retries."""
        for trial in range(max_trials):
            try:
                # Create generation
                generation = self.client.generations.create(
                    prompt=prompt,
                )

                # Wait for completion
                completed = False
                max_timeout = 5 * 60 # 5 minutes
                while not completed:
                    generation = self.client.generations.get(id=generation.id)
                    if generation.state == "completed":
                        completed = True
                    elif generation.state == "failed":
                        if (
                            "prompt not allowed because advanced moderation"
                            in generation.failure_reason
                        ):
                            return T2VOutput(
                                text_input=prompt, error=T2VError.SAFETY_REFUSAL
                            )
                        if (
                            "prompt not allowed because it contains blacklisted word"
                            in generation.failure_reason
                        ):
                            return T2VOutput(
                                text_input=prompt, error=T2VError.SAFETY_REFUSAL
                            )
                        raise RuntimeError(
                            f"Generation failed: {generation.failure_reason}"
                        )
                    print(f"Generation: {generation.id}, {generation.state}")
                    time.sleep(3)
                    max_timeout -= 3
                    if max_timeout <= 0:
                        raise TimeoutError("Generation timeout")

                # Download video
                video_url = generation.assets.video
                response = requests.get(video_url, stream=True)

                # Generate a unique filename
                video_filename = gen_id(suffix=".mp4")
                video_path = output_dir / video_filename

                with open(video_path, "wb") as file:
                    file.write(response.content)

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
        """Generate videos for multiple prompts."""
        outputs = []

        for prompt in text_inputs:
            output = self._generate_single_video(prompt, output_dir, max_trials)
            outputs.append(output)

        return outputs
