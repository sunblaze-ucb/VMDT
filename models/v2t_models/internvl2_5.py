import math
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .base import V2TBaseModel, V2TOutput

### Helper Functions ###


def split_model(model_name: str) -> dict:
    device_map = {}
    world_size = torch.cuda.device_count()
    # Get number of layers from the model name.
    num_layers = {
        "InternVL2_5-1B": 24,
        "InternVL2_5-2B": 24,
        "InternVL2_5-4B": 36,
        "InternVL2_5-8B": 32,
        "InternVL2_5-26B": 48,
        "InternVL2_5-38B": 64,
        "InternVL2_5-78B": 80,
    }[model_name.replace("OpenGVLab/", "")]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios, width: int, height: int, image_size: int
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Construct possible (grid) ratios.
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = int(image_size * target_aspect_ratio[0])
    target_height = int(image_size * target_aspect_ratio[1])
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    num_cols = target_width // image_size
    for i in range(blocks):
        box = (
            (i % num_cols) * image_size,
            (i // num_cols) * image_size,
            ((i % num_cols) + 1) * image_size,
            ((i // num_cols) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_index(
    bound, fps: float, max_frame: int, first_idx: int = 0, num_segments: int = 32
):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices


def load_video(
    video_path: str,
    bound=None,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: int = 32,
):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments
    )
    for frame_index in frame_indices:
        frame = vr[frame_index]
        img = Image.fromarray(frame.asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        tiles_tensor = [transform(tile) for tile in tiles]
        tiles_tensor = torch.stack(tiles_tensor)
        num_patches_list.append(tiles_tensor.shape[0])
        pixel_values_list.append(tiles_tensor)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


### V2T Model Implementation ###


class InternVL2_5(V2TBaseModel):
    def load_model(self, **kwargs):
        """
        Load the InternVL2.5 model.
        Expects an optional 'model_id' kwarg; if not provided, uses self.model_name.
        """
        self.device = "cuda"
        self.model_id = kwargs.get("model_id", self.model_name)
        valid_models = [
            "OpenGVLab/InternVL2_5-1B",
            "OpenGVLab/InternVL2_5-2B",
            "OpenGVLab/InternVL2_5-4B",
            "OpenGVLab/InternVL2_5-8B",
            "OpenGVLab/InternVL2_5-26B",
            "OpenGVLab/InternVL2_5-38B",
            "OpenGVLab/InternVL2_5-78B",
        ]
        if self.model_id not in valid_models:
            raise ValueError(f"Invalid model id: {self.model_id}")
        # Optionally compute a device map; here we simply use auto device placement.
        device_map = split_model(self.model_id)

        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",  # or pass device_map=device_map if needed
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )

    def generate_texts(
        self, video_inputs: list[Path], prompts: list[str], **gen_kwargs
    ) -> list[V2TOutput]:
        """
        For each video (a Path) and its corresponding prompt, perform generation.
        Additional generation parameters (e.g. num_frames, max_new_tokens) are passed via gen_kwargs.
        """
        outputs = []
        num_frames = gen_kwargs.get("num_frames", 32)
        max_new_tokens = gen_kwargs.get("max_new_tokens", 512)
        do_sample = gen_kwargs.get("do_sample", False)
        temperature = gen_kwargs.get("temperature", 0)

        # Process each video-prompt pair
        for video_path, prompt in zip(video_inputs, prompts):
            try:
                # load_video expects a string path
                pixel_values, num_patches_list = load_video(
                    str(video_path), num_segments=num_frames, max_num=1
                )
                pixel_values = pixel_values.to(torch.bfloat16).cuda()

                # Create a prompt prefix from the number of frames
                video_prefix = "".join(
                    [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
                )
                full_prompt = video_prefix + prompt

                generation_config = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "temperature": temperature,
                }

                # Call the model's chat method. (Note: your model must expose a `chat` method as in the provided code.)
                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    full_prompt,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
                outputs.append(V2TOutput(video_input=video_path, text_output=response))
            except Exception as e:
                outputs.append(V2TOutput(video_input=video_path, error=str(e)))
        return outputs


# For testing locally (optional):
if __name__ == "__main__":
    # Make sure to have a valid video file at the specified path.
    test_video = Path("./test_dis.mp4")
    test_prompt = "What is the video about?"

    # Initialize the model (you can override model_id via load_kwargs if needed)
    model = InternVL2_5("OpenGVLab/InternVL2_5-8B")

    # Generate text for one video and prompt; you can pass additional kwargs (e.g. num_frames)
    results = model.generate_texts([test_video], [test_prompt], num_frames=32)
    for result in results:
        if result.error:
            print(f"Error processing {result.video_input}: {result.error}")
        else:
            print(f"Video: {result.video_input}\nResponse: {result.text_output}")
