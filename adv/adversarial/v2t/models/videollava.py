from typing import Any, Dict, Tuple
import warnings
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_video
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

class VideoLlava:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = eval(dtype)
        
        self.model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", torch_dtype=self.dtype).to(device)
        self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", do_rescale=False, do_convert_rgb=False)
        self.model_name = self.get_model_name()
        
        self.prompt_format = "USER: <video>{prompt} ASSISTANT:"

    @staticmethod
    def get_model_name() -> str:
        return "VideoLLaVA"
    
    def load_video(self, video_path, start_time=0, end_time=None) -> torch.Tensor:
        video, _, _ = read_video(video_path, start_pts=start_time, end_pts=end_time, pts_unit='sec', output_format="TCHW")
        video = video.float().div(255)
        return video
    
    def select_frames(self, video, num_frames_to_select=8):
        T = video.shape[0]
        indices = torch.linspace(0, T - 1, steps=num_frames_to_select).round().long()
        return video[indices]
    
    def transform_frames(self, frames):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean, std)
        ])
        
        return transform(frames)
        
    def get_visual_features(self, frames, transform=True):
        if transform:
            frames = self.transform_frames(frames)
            
        frames = frames.unsqueeze(0)
        video_features, _ = self.model.get_video_features(frames, vision_feature_layer=-2)
        
        return video_features
    
    def get_last_hidden_state(self, prompt, visual_features):
        formatted_prompt = self.prompt_format.format(prompt=prompt)
        
        prompt_segs = formatted_prompt.split("<video>")
        if len(prompt_segs) != 2:
            raise ValueError("Formatted prompt must contain exactly one '<video>' placeholder.")
        
        seg0_tokens = self.processor.tokenizer(
            prompt_segs[0], return_tensors="pt", add_special_tokens=True
        ).to(self.device).input_ids
        seg1_tokens = self.processor.tokenizer(
            prompt_segs[1], return_tensors="pt", add_special_tokens=False
        ).to(self.device).input_ids
        
        seg0_embeds = self.model.get_input_embeddings()(seg0_tokens)
        seg1_embeds = self.model.get_input_embeddings()(seg1_tokens)
        
        print(seg0_embeds.shape, visual_features.shape, seg1_embeds.shape)
        
        inputs_embeds = torch.cat([seg0_embeds, visual_features, seg1_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=self.device)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        
        return outputs.hidden_states[-1]       
    
    def get_last_hidden_state(self, prompt, visual_features):
        
        height, width = 224, 224
        patch_size = 14
        
        num_image_tokens = (height // patch_size) * (width // patch_size) + 1
        num_video_tokens = num_image_tokens * 8

        formatted_prompt = self.prompt_format.format(prompt=prompt)
        formatted_prompt = formatted_prompt.replace("<video>", "<video>" * num_video_tokens)

        inputs = self.processor.tokenizer([formatted_prompt], return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        special_image_mask = (input_ids == 32001).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(self.device)
        assert inputs_embeds[special_image_mask].numel() == visual_features.numel()
        
        input_embeds = inputs_embeds.masked_scatter(special_image_mask, visual_features)
        attention_mask = torch.ones(input_embeds.shape[:-1], dtype=torch.long, device=self.device)
        
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        
        return outputs.hidden_states[-1]
    
    def generate(self, prompt, frames=None, video_path=None, start_time=0, end_time=None, num_frames_to_select=8):
        if video_path is not None:
            video = self.load_video(video_path, start_time, end_time)
            assert video is not None, "Video load failed"
            frames = self.select_frames(video, num_frames_to_select)
        else:
            assert frames is not None, "Either video_path or frames must be provided."
        
        prompt = self.prompt_format.format(prompt=prompt)
        
        inputs = self.processor(text=prompt, videos=frames, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        output_text = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].split("ASSISTANT:")[1].strip()
        
        return output_text