import torch

import av
import numpy as np
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])





class VideoLLavaClient:
    def __init__(self, model_id, device="cuda"):
        self.device = device
        self.model_id = model_id
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(model_id,device_map = 'auto').eval().to(device)
        self.processor = VideoLlavaProcessor.from_pretrained(model_id)

    def generate(self, video_path, question,num_frames, max_new_tokens=4096, do_sample=False, temperature = 0): # text, image_path, **kwargs):
        # Load the video as an np.arrau, sampling uniformly num_frames frames


        prompt = f"USER: <video>{question} ASSISTANT:"
        container = av.open(video_path)

        # sample uniformly 8 frames from the video
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
        clip = read_video_pyav(container, indices)

        inputs = self.processor(text=prompt, videos=clip, return_tensors="pt")

        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].cuda()
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        # Generate
        generate_ids = self.model.generate(**inputs, max_length=max_new_tokens)

        print(self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])



if __name__ == "__main__":
    client = VideoLLavaClient(f"LanguageBind/Video-LLaVA-7B-hf")
    client.generate('./test_dis.mp4', "What is the video about?", 8)
