from pathlib import Path

import torch
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video

from .base import T2VBaseModel, T2VOutput, gen_id
from .model_name import T2VModelName

MODEL_GEN_CONFIG = {
    T2VModelName.CogVideoX1_5_5B: {
        "batch_size": 1,
        "guidance_scale": 6,
        "num_frames": 81,
        "num_inference_steps": 50,
    },
    T2VModelName.CogVideoX_5B: {
        "batch_size": 5,
        "guidance_scale": 6,
        "num_frames": 49,
        "num_inference_steps": 50,
    },
    T2VModelName.CogVideoX_2B: {
        "batch_size": 5,
        "guidance_scale": 6,
        "num_frames": 49,
        "num_inference_steps": 50,
    },
}


class CogVideoX(T2VBaseModel):
    def load_model(self):
        model_ids = {
            T2VModelName.CogVideoX1_5_5B: "THUDM/CogVideoX1.5-5B",
            T2VModelName.CogVideoX_5B: "THUDM/CogVideoX-5b",
            T2VModelName.CogVideoX_2B: "THUDM/CogVideoX-2b",
        }
        self.pipe = CogVideoXPipeline.from_pretrained(
            model_ids[self.model_name], torch_dtype=torch.bfloat16
        ).to("cuda")
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

    def generate_videos(
        self, text_inputs: list[str], output_dir: Path, **kwargs
    ) -> list[T2VOutput]:
        gen_config = MODEL_GEN_CONFIG[self.model_name]
        batch_size = gen_config.pop("batch_size")

        outputs = []
        for i in range(0, len(text_inputs), batch_size):
            batch_text_inputs = text_inputs[i : i + batch_size]
            outputs += self._generate_videos_batch(
                batch_text_inputs, output_dir, **gen_config
            )
        return outputs

    def _generate_videos_batch(
        self,
        text_inputs_batch: list[str],
        output_dir: Path,
        guidance_scale: int,
        num_frames: int,
        num_inference_steps: int,
    ) -> list[T2VOutput]:
        videos = self.pipe(
            prompt=text_inputs_batch,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames
        outputs = []

        for idx, (text, video) in enumerate(zip(text_inputs_batch, videos)):
            video_path = output_dir / gen_id(prefix=str(idx) + "_", suffix="_" + "".join(val[0].lower() if (len(val) > 0 and val[0].isalnum()) else "" for val in text.split(" ")) + ".mp4")
            # video_path = output_dir / gen_id(suffix=".mp4")
            export_to_video(video, video_path, fps=8)
            outputs.append(
                T2VOutput(
                    text_input=text,
                    video_path=video_path,
                )
            )
        return outputs
