from typing import Any, Dict, Tuple
import os
import warnings
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_video
from transformers import AutoTokenizer, AutoModel

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

def construct_conversation(user_prompt, msg="", instruction=None, media_type="video", chat_history=[]):
    conversation = ""
    if instruction:
        conversation += instruction
    conversation += (
                "[INST]" + " "
            )

    if media_type == 'image':
        conversation +=( "<Image>" + IMG_TOKEN + "</Image>")
    else:
        conversation += ("<Video>" + VID_TOKEN + "</Video>")


    conversation += (
                msg.rstrip() + "[/INST]"
            )

    for q,a in chat_history:
        conversation += (" [INST] " + q + " [/INST]")
        conversation += (a + "</s>")

    conversation += (" [INST] " + user_prompt + " [/INST]")
    conversation += ("")
    
    return conversation

class InternVideo2Chat8B:
    
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = eval(dtype)
        
        self.tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, torch_dtype=self.dtype).to(device)
        self.model_name = self.get_model_name()
        
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def get_model_name():
        return "InternVideo2Chat8B"
        
    def load_video(self, video_path, start_time=0, end_time=None) -> torch.Tensor:
        video, _, _ = read_video(video_path, start_pts=start_time, end_pts=end_time, pts_unit='sec', output_format="TCHW")
        video = video.float().div(255)
        return video

    
    def select_frames(self, video, num_frames_to_select=8):
        total_frames = video.size(0)
        
        seg_size = float(total_frames - 1) / num_frames_to_select
        start = int(seg_size / 2)
        
        indices = torch.tensor([
            start + int(np.round(seg_size * idx)) for idx in range(num_frames_to_select)
        ], dtype=torch.long)
        
        return video[indices]

    def transform_frames(self, frames):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean, std)
        ])
        
        return transform(frames)
    
    def get_visual_features(self, frames, transform=True):
        if transform:
            frames = self.transform_frames(frames)
            
        frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
        return self.model.encode_vision(frames, instruction=None)
    
    def get_last_hidden_state(self, prompt, visual_features):
        # 1. Tokenize
        tokenized = self.model.build_input_ids(
            self.tokenizer,
            construct_conversation(prompt),
            max_length=248,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        input_ids = tokenized["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = tokenized["attention_mask"].unsqueeze(0).to(self.device)
        video_idx = tokenized["index"].unsqueeze(0).to(self.device)

        # 2. Get text embeddings
        text_embeds = self.model.lm.get_input_embeddings()(input_ids.long())  
        # text_embeds.shape: (B, T, D)

        # 3. Project visual features -> visual_embeds
        visual_embeds = self.model.project_up(visual_features)
        # Suppose shape = (N, D), where N is the number of "1" in video_idx
        visual_embeds = visual_embeds.view(-1, visual_embeds.shape[-1])

        B, T, D = text_embeds.shape
        # Flatten
        flat_text_embeds = text_embeds.view(B * T, D)
        flat_idx = video_idx.view(B * T)  # shape: (B*T,)

        # 4. Build a full-size "visual" tensor that is zero everywhere except
        #    the positions where video_idx == 1
        flat_visual = torch.zeros_like(flat_text_embeds)  # (B*T, D)
        one_positions = (flat_idx == 1).nonzero(as_tuple=False).squeeze(-1)
        flat_visual[one_positions] = visual_embeds  # Insert each row

        # Reshape back to (B, T, D)
        merged_visual = flat_visual.view(B, T, D)

        # 5. Use `torch.where` with a broadcastable mask
        mask = (video_idx == 1).unsqueeze(-1).expand(B, T, D)
        text_embeds_new = torch.where(mask, merged_visual, text_embeds)

        # 6. Forward through the language model
        outputs = self.model.lm(
            inputs_embeds=text_embeds_new,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )

        return outputs.hidden_states[-1]
    
    def generate(self, prompt, frames=None, video_path=None, start_time=0, end_time=None, num_frames_to_select=8):
        if video_path is not None:
            print(f"Loading video from {video_path}")
            video = self.load_video(video_path, start_time, end_time)
            assert video is not None, "Video load failed"
            frames = self.select_frames(video, num_frames_to_select)
        else:
            assert frames is not None, "Either video_path or frames must be provided."
            
        frames = self.transform_frames(frames)
        frames = frames.to(self.device, dtype=self.dtype)
        
        return self.model.chat(
            self.tokenizer, 
            "",
            prompt,
            media_type="video",
            media_tensor=frames,
            instruction=None,
            chat_history=[],
            return_history=False,
            generation_config={"do_sample": False},
        )