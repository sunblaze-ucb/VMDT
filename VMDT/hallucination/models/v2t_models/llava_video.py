# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")
def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]

    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 2, sample_fps, dtype=int)  # NICK: took me a lot of time to solve this problem, but originally was total_frame_num - 1 which caused the process to hang. -2 fixed it.
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time


class LlavaVideoClient:
    def __init__(self, model_id, device="cuda"):
        self.device = device
        self.pretrained = model_id # 
        model_name = "llava_qwen"
        device_map = None if model_id in ["lmms-lab/LLaVA-Video-7B-Qwen2"] else "auto"
        self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(self.pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args

        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

    def generate(self, video_path, question, frames=32, max_new_tokens=4096, do_sample=False, temperature = 0): # text, image_path, **kwargs):
        # Load the video as an np.arrau, sampling uniformly 8 frames
        #video_path = './test_dis.mp4'

        max_frames_num = frames
        video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        prompt = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{question}."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        cont = self.model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs


if __name__ == "__main__":
    client = LlavaVideoClient("lmms-lab/LLaVA-Video-7B-Qwen2")
    output = client.generate('./test_dis.mp4', "What is the video about?", frames=32)
    print(output)
