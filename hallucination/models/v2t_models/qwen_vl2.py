from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os

class QwenVLClient:
    """NOTE: Qwen2 has 2B, 7B, 72B while Qwen2.5 has 3B, 7B, 72B."""
    def __init__(self, model_id, device="cuda", use_less_tokens=False):
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        if "72B" in model_id:
            kwargs["device_map"] = "auto"

        # See https://github.com/QwenLM/Qwen2.5-VL/blob/f151060e2bdf4058c246d8bc9b9923eee909099a/qwen-vl-utils/src/qwen_vl_utils/vision_process.py. We need to set VIDEO_MAX_PIXELS (an environment variable) to ctx_length * 28 * 28 * 0.9, e.g., 128000 * 28 * 28 * 0.9 (default).
        generation_to_total_pixels = {
                "2": 32768,
                "2.5": 128000,
        }

        if "2.5-VL" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                **kwargs
            )
            # No need to use bc this is the default.
            # os.environ["VIDEO_MAX_PIXELS"] = str(generation_to_total_pixels["2.5"] * 28 * 28 * 0.9)
        elif "2-VL" in model_id:
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                **kwargs
            )
            # NOTE: Untested so far.
            os.environ["VIDEO_MAX_PIXELS"] = str(generation_to_total_pixels["2"] * 28 * 28 * 0.9)

        if "device_map" not in kwargs:
            self.model.to(device)


        if not use_less_tokens:
            # default processor
            self.processor = AutoProcessor.from_pretrained(model_id)

        else:
            # The default range for the number of visual tokens per image in the model is 4-16384.
            # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
            min_pixels = 256*28*28
            max_pixels = 1280*28*28
            self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)

    def generate(self, video_path, question,num_frames = None, max_new_tokens=4096, do_sample=False, temperature = 0): # text, image_path, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{video_path}"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # import pdb;pdb.set_trace()
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)
        return output_text

# class Qwen2VLClient:
#     def __init__(self, model_id, device="cuda"):
#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             model_id,
#             torch_dtype=torch.bfloat16,
#             attn_implementation="flash_attention_2",
#             # device_map="auto",
#         ).eval().to(device)

#         # default processer
#         self.processor = AutoProcessor.from_pretrained(model_id)


# OLD VERSION:
# from __future__ import annotations

# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# import torch 
# import base64
# import logging
# import math
# import os
# import sys
# import time
# import warnings
# from functools import lru_cache
# from io import BytesIO

# import requests
# import torch
# import torchvision
# from packaging import version
# from PIL import Image
# from torchvision import io, transforms
# from torchvision.transforms import InterpolationMode


# logger = logging.getLogger(__name__)

# IMAGE_FACTOR = 28
# MIN_PIXELS = 4 * 28 * 28
# MAX_PIXELS = 16384 * 28 * 28
# MAX_RATIO = 200

# VIDEO_MIN_PIXELS = 128 * 28 * 28
# VIDEO_MAX_PIXELS = 768 * 28 * 28
# VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
# FRAME_FACTOR = 2
# FPS = 2.0
# FPS_MIN_FRAMES = 32
# FPS_MAX_FRAMES = 768


# def round_by_factor(number: int, factor: int) -> int:
#     """Returns the closest integer to 'number' that is divisible by 'factor'."""
#     return round(number / factor) * factor


# def ceil_by_factor(number: int, factor: int) -> int:
#     """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
#     return math.ceil(number / factor) * factor


# def floor_by_factor(number: int, factor: int) -> int:
#     """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
#     return math.floor(number / factor) * factor


# def smart_resize(
#     height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
# ) -> tuple[int, int]:
#     """
#     Rescales the image so that the following conditions are met:

#     1. Both dimensions (height and width) are divisible by 'factor'.

#     2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

#     3. The aspect ratio of the image is maintained as closely as possible.
#     """
#     if max(height, width) / min(height, width) > MAX_RATIO:
#         raise ValueError(
#             f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
#         )
#     h_bar = max(factor, round_by_factor(height, factor))
#     w_bar = max(factor, round_by_factor(width, factor))
#     if h_bar * w_bar > max_pixels:
#         beta = math.sqrt((height * width) / max_pixels)
#         h_bar = floor_by_factor(height / beta, factor)
#         w_bar = floor_by_factor(width / beta, factor)
#     elif h_bar * w_bar < min_pixels:
#         beta = math.sqrt(min_pixels / (height * width))
#         h_bar = ceil_by_factor(height * beta, factor)
#         w_bar = ceil_by_factor(width * beta, factor)
#     return h_bar, w_bar


# def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
#     if "image" in ele:
#         image = ele["image"]
#     else:
#         image = ele["image_url"]
#     image_obj = None
#     if isinstance(image, Image.Image):
#         image_obj = image
#     elif image.startswith("http://") or image.startswith("https://"):
#         image_obj = Image.open(requests.get(image, stream=True).raw)
#     elif image.startswith("file://"):
#         image_obj = Image.open(image[7:])
#     elif image.startswith("data:image"):
#         if "base64," in image:
#             _, base64_data = image.split("base64,", 1)
#             data = base64.b64decode(base64_data)
#             image_obj = Image.open(BytesIO(data))
#     else:
#         image_obj = Image.open(image)
#     if image_obj is None:
#         raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
#     image = image_obj.convert("RGB")
#     ## resize
#     if "resized_height" in ele and "resized_width" in ele:
#         resized_height, resized_width = smart_resize(
#             ele["resized_height"],
#             ele["resized_width"],
#             factor=size_factor,
#         )
#     else:
#         width, height = image.size
#         min_pixels = ele.get("min_pixels", MIN_PIXELS)
#         max_pixels = ele.get("max_pixels", MAX_PIXELS)
#         resized_height, resized_width = smart_resize(
#             height,
#             width,
#             factor=size_factor,
#             min_pixels=min_pixels,
#             max_pixels=max_pixels,
#         )
#     image = image.resize((resized_width, resized_height))

#     return image


# def smart_nframes(
#     ele: dict,
#     total_frames: int,
#     video_fps: int | float,
# ) -> int:
#     """calculate the number of frames for video used for model inputs.

#     Args:
#         ele (dict): a dict contains the configuration of video.
#             support either `fps` or `nframes`:
#                 - nframes: the number of frames to extract for model inputs.
#                 - fps: the fps to extract frames for model inputs.
#                     - min_frames: the minimum number of frames of the video, only used when fps is provided.
#                     - max_frames: the maximum number of frames of the video, only used when fps is provided.
#         total_frames (int): the original total number of frames of the video.
#         video_fps (int | float): the original fps of the video.

#     Raises:
#         ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

#     Returns:
#         int: the number of frames for video used for model inputs.
#     """
#     assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
#     if "nframes" in ele:
#         nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
#     else:
#         fps = ele.get("fps", FPS)
#         min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
#         max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
#         nframes = total_frames / video_fps * fps
#         nframes = min(max(nframes, min_frames), max_frames)
#         nframes = round_by_factor(nframes, FRAME_FACTOR)
#     if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
#         raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
#     return nframes


# def _read_video_torchvision(
#     ele: dict,
# ) -> torch.Tensor:
#     """read video using torchvision.io.read_video

#     Args:
#         ele (dict): a dict contains the configuration of video.
#         support keys:
#             - video: the path of video. support "file://", "http://", "https://" and local path.
#             - video_start: the start time of video.
#             - video_end: the end time of video.
#     Returns:
#         torch.Tensor: the video tensor with shape (T, C, H, W).
#     """
#     video_path = ele["video"]
#     if version.parse(torchvision.__version__) < version.parse("0.19.0"):
#         if "http://" in video_path or "https://" in video_path:
#             warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
#         if "file://" in video_path:
#             video_path = video_path[7:]
#     st = time.time()
#     video, audio, info = io.read_video(
#         video_path,
#         start_pts=ele.get("video_start", 0.0),
#         end_pts=ele.get("video_end", None),
#         pts_unit="sec",
#         output_format="TCHW",
#     )
#     total_frames, video_fps = video.size(0), info["video_fps"]
#     logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
#     nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
#     idx = torch.linspace(0, total_frames - 1, nframes).round().long()
#     video = video[idx]
#     return video


# def is_decord_available() -> bool:
#     import importlib.util

#     return importlib.util.find_spec("decord") is not None

# # Timeout for Decord:
# # import multiprocessing
# # class TimeoutException(Exception):
# #     pass
# # def run_with_timeout(timeout, func, *args, **kwargs):
# #     """Run a function with a timeout using multiprocessing."""
# #     print(f"Running {func.__name__} with timeout {timeout} seconds with args={args} and kwargs={kwargs}")
# #     with multiprocessing.Pool(processes=1) as pool:
# #         result = pool.apply_async(func, args, kwargs)
# #         try:
# #             return result.get(timeout=timeout)  # Enforce timeout
# #         except multiprocessing.TimeoutError:
# #             pool.terminate()  # Kill the process
# #             raise TimeoutException(f"Function {func.__name__} timed out after {timeout} seconds")

# class TimeoutException(Exception):
#     pass

# import threading
# def run_with_timeout(timeout, func, *args, **kwargs):
#     result = []

#     def target():
#         try:
#             result.append(func(*args, **kwargs))
#         except Exception as e:
#             result.append(e)

#     thread = threading.Thread(target=target)
#     thread.start()
#     thread.join(timeout=timeout)

#     if thread.is_alive():
#         raise TimeoutException(f"Function did not complete within {timeout} seconds.")
#     if isinstance(result[0], Exception):
#         raise result[0]

#     return result[0]

# import decord

# # def process_video(video_path, total_frames, nframes, minus_num):
# #     print(f"Trying with {minus_num=}")
# #     vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
# #     idx = torch.linspace(0, total_frames - minus_num, nframes).round().long().tolist()
# #     video = vr.get_batch(idx).asnumpy()
# #     video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
# #     return video

# def _read_video_decord(
#     ele: dict,
# ) -> torch.Tensor:
#     """read video using decord.VideoReader

#     Args:
#         ele (dict): a dict contains the configuration of video.
#         support keys:
#             - video: the path of video. support "file://", "http://", "https://" and local path.
#             - video_start: the start time of video.
#             - video_end: the end time of video.
#     Returns:
#         torch.Tensor: the video tensor with shape (T, C, H, W).
#     """
#     video_path = ele["video"]
#     st = time.time()
#     # try:
#     # import pdb;pdb.set_trace()
#     vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)  # NOTE: Need to say num_threads=1 to avoid deadlock.
#         # vr = run_with_timeout(5, decord.VideoReader, video_path, ctx=decord.cpu(0), num_threads=1)
#     # except TimeoutException as e:
#         # print(f"Timeout when reading video with decord: {video_path}")
#         # return None
#     # TODO: support start_pts and end_pts
#     if 'video_start' in ele or 'video_end' in ele:
#         raise NotImplementedError("not support start_pts and end_pts in decord for now.")
#     total_frames, video_fps = len(vr), vr.get_avg_fps()
#     # logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
#     print(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
#     nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
#     idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
#     video = vr.get_batch(idx).asnumpy()
#     video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
#     # try:    
#     #     print(f"{total_frames=}, {nframes=}")
#     #     video = run_with_timeout(5, process_video, video_path, total_frames, nframes, 1)
#     # except TimeoutException as e:
#     #     video = _read_video_torchvision(ele)
#         # try:
#         #     video = run_with_timeout(5, process_video, video_path, total_frames, nframes, 2)  # - 2 easy fix for decord when it hangs.
#         # except TimeoutException as e:
#         #     # run with torchvision backend
#         #     video = _read_video_torchvision(ele)
#     return video


# VIDEO_READER_BACKENDS = {
#     "decord": _read_video_decord,
#     "torchvision": _read_video_torchvision,
# }

# FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


# @lru_cache(maxsize=1)
# def get_video_reader_backend() -> str:
#     if FORCE_QWENVL_VIDEO_READER is not None:
#         video_reader_backend = FORCE_QWENVL_VIDEO_READER
#     elif is_decord_available():
#         video_reader_backend = "decord"
#     else:
#         video_reader_backend = "torchvision"
#     print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
#     return video_reader_backend


# def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor | list[Image.Image]:
#     if isinstance(ele["video"], str):
#         video_reader_backend = get_video_reader_backend()
#         video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
#         nframes, _, height, width = video.shape

#         min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
#         total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
#         max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
#         max_pixels = ele.get("max_pixels", max_pixels)
#         if "resized_height" in ele and "resized_width" in ele:
#             resized_height, resized_width = smart_resize(
#                 ele["resized_height"],
#                 ele["resized_width"],
#                 factor=image_factor,
#             )
#         else:
#             resized_height, resized_width = smart_resize(
#                 height,
#                 width,
#                 factor=image_factor,
#                 min_pixels=min_pixels,
#                 max_pixels=max_pixels,
#             )
#         video = transforms.functional.resize(
#             video,
#             [resized_height, resized_width],
#             interpolation=InterpolationMode.BICUBIC,
#             antialias=True,
#         ).float()
#         return video
#     else:
#         assert isinstance(ele["video"], (list, tuple))
#         process_info = ele.copy()
#         process_info.pop("type", None)
#         process_info.pop("video", None)
#         images = [
#             fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
#             for video_element in ele["video"]
#         ]
#         nframes = ceil_by_factor(len(images), FRAME_FACTOR)
#         if len(images) < nframes:
#             images.extend([images[-1]] * (nframes - len(images)))
#         return images


# def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
#     vision_infos = []
#     if isinstance(conversations[0], dict):
#         conversations = [conversations]
#     for conversation in conversations:
#         for message in conversation:
#             if isinstance(message["content"], list):
#                 for ele in message["content"]:
#                     if (
#                         "image" in ele
#                         or "image_url" in ele
#                         or "video" in ele
#                         or ele["type"] in ("image", "image_url", "video")
#                     ):
#                         vision_infos.append(ele)
#     return vision_infos


# def process_vision_info(
#     conversations: list[dict] | list[list[dict]],
# ) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
#     vision_infos = extract_vision_info(conversations)
#     ## Read images or videos
#     image_inputs = []
#     video_inputs = []
#     for vision_info in vision_infos:
#         if "image" in vision_info or "image_url" in vision_info:
#             image_inputs.append(fetch_image(vision_info))
#         elif "video" in vision_info:
#             video_inputs.append(fetch_video(vision_info))
#         else:
#             raise ValueError("image, image_url or video should in content.")
#     if len(image_inputs) == 0:
#         image_inputs = None
#     if len(video_inputs) == 0:
#         video_inputs = None
#     return image_inputs, video_inputs



# class Qwen2VLClient:
#     def __init__(self, model_id, device="cuda"):
#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             model_id,
#             torch_dtype=torch.bfloat16,
#             attn_implementation="flash_attention_2",
#             # device_map="auto",
#         ).eval().to(device)

#         # default processer
#         self.processor = AutoProcessor.from_pretrained(model_id)

#     def generate(self, video_path, question,num_frames = None, max_new_tokens=4096, do_sample=False, temperature = 0): # text, image_path, **kwargs):

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "video", "video": f"file://{video_path}"},
#                     {"type": "text", "text": question},
#                 ],
#             }
#         ]

#         # Preparation for inference
#         text = self.processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = self.processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#         inputs = inputs.to("cuda")

#         # Inference: Generation of the output
#         generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature = 0)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = self.processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         # print(output_text)
#         return output_text


# if __name__ == "__main__":
#     client = Qwen2VLClient("Qwen/Qwen2-VL-7B-Instruct")
#     output = client.generate("/ib-scratch/chenguang02/scratch1/cnicholas/auxiliary-mmdt-video/video-safety-benchmark/models/v2t_models/test_dis.mp4", "What is the video about?")
#     print(output)
