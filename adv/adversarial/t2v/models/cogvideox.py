"""From https://huggingface.co/THUDM/CogVideoX-5b."""
from typing import List, Dict, Tuple, Optional, Any
import argparse
import gc
from pathlib import Path
from tqdm import tqdm

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from adversarial.t2v.t2v_utils import T2VOutput
from adversarial.common.utils import gen_id

class CogVideoX:
    def __init__(self, model_size: str, disable_tqdm: bool = True, compile: bool = True, generation_kwargs: Dict[str, Any] = {"num_frames": 49, "num_videos_per_prompt": 1, "num_inference_steps": 50, "guidance_scale": 6}, seed: int = 42, dtype: str = "torch.bfloat16", device: str = "cuda") -> None:
        
        assert model_size in ["2b", "5b"], "CogVideoX model size must be one of ['2b', '5b']"
        self.model_name = self.get_model_name(model_size)
        
        self.pipe = CogVideoXPipeline.from_pretrained(f"THUDM/{self.model_name}", torch_dtype=eval(dtype)).to(device)
        
        if disable_tqdm:
            self.pipe.set_progress_bar_config(disable=True)
        if compile:
            self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune", fullgraph=True)
        
        self.generation_kwargs = generation_kwargs
        self.seed = seed
        self.max_length = 226 # we need this property set for greedy and genetic attacks
        self.dtype = eval(dtype)
        self.device = device

    @staticmethod
    def get_model_name(model_size):
        return f"CogVideoX-{model_size}" # we need this property to save the prompts under the right path

    @torch.inference_mode()
    def generate(self, prompt, output_file):
        video = self.pipe(
            prompt=prompt,
            **self.generation_kwargs,
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
        ).frames[0]
        export_to_video(video, output_file, fps=8)
    
    # callable interface for the ATM attacks (surrogate models only) -- TODO: deprecate this along with ATM
    def _prepare_inputs(self, prompt_embeds):
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        negative_prompt_embeds = self.pipe._get_t5_prompt_embeds(
            "", device=self.device, dtype=self.dtype
        )
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            extension = negative_prompt_embeds[:, -(negative_prompt_embeds.size(1)-prompt_embeds.size(1)):, :]
            prompt_embeds = torch.cat([prompt_embeds, extension], dim=1)
        return prompt_embeds, negative_prompt_embeds
    
    def __call__(self, prompt=None, prompt_embeds=None):
        if prompt is not None:
            video = self.pipe(
                prompt=prompt, 
                generator=torch.Generator(device=self.device).manual_seed(self.seed),
                output_type="pt",
                **self.generation_kwargs,
            ).frames[0].permute(0, 2, 3, 1)
        else:
            prompt_embeds, negative_prompt_embeds = self._prepare_inputs(prompt_embeds)
            video = self.pipe(
                prompt_embeds=prompt_embeds, 
                negative_prompt_embeds=negative_prompt_embeds, 
                generator=torch.Generator(device=self.device).manual_seed(self.seed),
                output_type="pt",
                **self.generation_kwargs,
            ).frames[0].permute(0, 2, 3, 1)
        video = (video * 255).clamp(0, 255).to(torch.uint8)
        return video
    
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