from pathlib import Path

import torch

from ..base import T2VBaseModel, T2VOutput, gen_id
from ..model_name import T2VModelName
from .repo.models.pipeline import VchitectXLPipeline
from .repo.utils import save_as_mp4
from tqdm import tqdm

MODEL_ROOT = Path(__file__).parent
MODEL_CKPT_PATH = MODEL_ROOT / "cache/vchitect2"


class Vchitect2(T2VBaseModel):
    def load_model(self):
        self.pipe = VchitectXLPipeline(str(MODEL_CKPT_PATH), device="cuda")

    def generate_videos(
        self, text_inputs: list[str], output_dir: Path, **kwargs
    ) -> list[T2VOutput]:
        outputs = []
        batch_size = 1
        # for i in range(0, len(text_inputs), batch_size):
        for i in tqdm(range(0, len(text_inputs), batch_size)):
            batch_text_inputs = text_inputs[i : i + batch_size]
            outputs += self._generate_videos_batch(
                batch_text_inputs, output_dir, **kwargs
            )
        return outputs

    def _generate_videos_batch(
        self,
        text_inputs_batch: list[str],
        output_dir: Path,
        guidance_scale: int = 7.5,
        frames: int = 40,
        num_inference_steps: int = 100,
        width: int = 768,
        height: int = 432,
    ) -> list[T2VOutput]:
        assert len(text_inputs_batch) == 1
        text_input = text_inputs_batch[0]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            video = self.pipe(
                text_input,
                negative_prompt="",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                frames=frames,
            )

        save_path = output_dir / gen_id(suffix=".mp4")
        save_as_mp4(video, str(save_path), duration=125)  # 8 fps

        return [T2VOutput(text_input=text_input, video_path=save_path)]
