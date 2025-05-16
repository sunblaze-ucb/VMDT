"""From https://huggingface.co/THUDM/CogVideoX-5b."""
from typing import List, Dict, Tuple, Optional, Any
import argparse
import time
import gc
from pathlib import Path
from tqdm import tqdm

import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

from adversarial.t2v.t2v_utils import T2VOutput
from adversarial.common.utils import gen_id

class Mochi:
    def __init__(self, disable_tqdm: bool = True, compile: bool = True, generation_kwargs: Dict[str, Any] = {"num_frames": 49, "num_videos_per_prompt": 1, "num_inference_steps": 50, "guidance_scale": 4.5}, seed: int = 42, dtype: str = "torch.bfloat16", device: str = "cuda", enable_memory_savings: bool = True) -> None:
        
        self.model_name = self.get_model_name()
        
        # self.pipe = MochiPipeline.from_pretrained(f"genmo/{self.model_name}", variant="bf16", torch_dtype=eval(dtype)).to(device)
        self.pipe = MochiPipeline.from_pretrained(f"genmo/{self.model_name}", torch_dtype=eval(dtype)).to(device)

        if enable_memory_savings:
            # self.pipe.enable_model_cpu_offload()  # gives an error with our current T2V_inference.py setup on A6000s.
            self.pipe.enable_vae_tiling()
        
        if disable_tqdm:
            self.pipe.set_progress_bar_config(disable=True)
        if compile:
            self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune", fullgraph=True)
        
        self.generation_kwargs = generation_kwargs
        self.seed = seed
        self.max_length = 256 # we need this property set for greedy and genetic attacks
        self.dtype = eval(dtype)
        self.device = device
        
    @staticmethod
    def get_model_name():
        return f"mochi-1-preview" # we need this property to save the prompts under the right path

    @torch.inference_mode()
    def generate(self, prompt, output_file):
        video = self.pipe(
            prompt=prompt,
            **self.generation_kwargs,
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
        ).frames[0]
        export_to_video(video, output_file, fps=8)
        
    def _del_transformer(self):
        del self.pipe.transformer
        torch.cuda.empty_cache()
        gc.collect()
        
    def _del_vae(self):
        del self.pipe.vae
        torch.cuda.empty_cache()
        gc.collect()
        
    def generate_videos(self, prompts, videos_dir):
        ret = []
        for prompt in tqdm(prompts):
            video_path = Path(videos_dir) / gen_id(".mp4")
            self.generate(prompt, video_path)
            ret.append(T2VOutput(video_path=video_path))
        return ret
        
