import copy
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

from .base import V2TBaseModel, V2TOutput

sys.path.append(str(Path(__file__).parent.absolute()))

from .llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from .llava.conversation import conv_templates
from .llava.mm_utils import tokenizer_image_token
from .llava.model.builder import load_pretrained_model

# Suppress warnings if desired
# warnings.filterwarnings("ignore")


def load_video(
    video_path: Path, max_frames_num: int, fps: int = 1, force_sample: bool = False
):
    """
    Loads a video using decord and returns:
      - the sampled frames as a NumPy array,
      - a string listing the (sampled) frame timestamps,
      - and the total video duration in seconds.
    """
    if max_frames_num == 0:
        # Return a dummy frame if no frames are requested
        return np.zeros((1, 336, 336, 3)), "0.00s", 0.0

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    # Compute a sampling step (here fps is used as a divisor)
    sampling_rate = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, total_frame_num, sampling_rate)]
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]

    # If too many frames or force_sample is set, sample uniformly
    if len(frame_idx) > max_frames_num or force_sample:
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]

    # Create a string with frame timestamps (e.g., "0.00s,0.50s,...")
    frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time_str, video_time


class LlavaVideo(V2TBaseModel):
    def load_model(self, **kwargs):
        """
        Load the Llava-Video model using Llava's load_pretrained_model API.
        The caller may supply a `model_id` via load_kwargs; otherwise, self.model_name is used.
        """
        model_id = kwargs.get("model_id", self.model_name)
        # For Llava-Video we use a fixed conversation template identifier (e.g., "llava_qwen")
        model_name = "llava_qwen"
        device_map = "auto"
        # load_pretrained_model returns (tokenizer, model, image_processor, max_length)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_id, None, model_name, torch_dtype="bfloat16", device_map=device_map
        )
        self.model.eval()

    def generate_texts(
        self, video_inputs: list[Path], prompts: list[str], **gen_kwargs
    ) -> list[V2TOutput]:
        """
        For each video file (Path) and its corresponding prompt, generate a text description.
        Additional generation parameters (e.g. frames, max_new_tokens, do_sample, temperature)
        are passed via gen_kwargs.
        """
        outputs = []
        frames = gen_kwargs.get("frames", 32)
        max_new_tokens = gen_kwargs.get("max_new_tokens", 4096)
        do_sample = gen_kwargs.get("do_sample", False)
        temperature = gen_kwargs.get("temperature", 0)
        conv_template = gen_kwargs.get("conv_template", "qwen_1_5")

        for video_path, prompt in zip(video_inputs, prompts):
            try:
                # Load and sample video frames
                video_np, frame_time, video_time = load_video(
                    video_path, frames, fps=1, force_sample=True
                )
                # Process video frames via the Llava image_processor
                video_tensor = self.image_processor.preprocess(
                    video_np, return_tensors="pt"
                )["pixel_values"]
                video_tensor = video_tensor.cuda().bfloat16()
                # Wrap in a list to match the expected format
                video_tensor = [video_tensor]

                # Build an instruction that tells the model about the video duration and sampled frames
                time_instruction = (
                    f"The video lasts for {video_time:.2f} seconds, and {len(video_tensor[0])} frames are uniformly sampled from it. "
                    f"These frames are located at {frame_time}. Please answer the following questions related to this video."
                )
                # Construct the full prompt
                prompt_full = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt}."

                # Use a deep copy of the selected conversation template
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], prompt_full)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()

                # Tokenize the prompt using Llava's helper
                input_ids = (
                    tokenizer_image_token(
                        prompt_question,
                        self.tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    )
                    .unsqueeze(0)
                    .cuda()
                )

                # Generate model output
                cont = self.model.generate(
                    input_ids,
                    images=video_tensor,
                    modalities=["video"],
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                # Decode the generated tokens into text
                text_outputs = self.tokenizer.batch_decode(
                    cont, skip_special_tokens=True
                )[0].strip()
                outputs.append(
                    V2TOutput(video_input=video_path, text_output=text_outputs)
                )
            except Exception as e:
                outputs.append(V2TOutput(video_input=video_path, error=str(e)))
        return outputs


# Optional testing block:
# if __name__ == "__main__":
#     test_video = Path("./test_dis.mp4")
#     test_prompt = "What is the video about?"
#     client = LlavaVideo("lmms-lab/LLaVA-Video-7B-Qwen2")
#     results = client.generate_texts([test_video], [test_prompt], frames=32)
#     for res in results:
#         if res.error:
#             print(f"Error processing {res.video_input}: {res.error}")
#         else:
#             print(f"Video: {res.video_input}\nResponse: {res.text_output}")
