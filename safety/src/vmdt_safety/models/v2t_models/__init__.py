from .base import V2TBaseModel
from .gpt import GPT
from .internvl2_5 import InternVL2_5
from .llava_video import LlavaVideo
from .model_name import V2TModelName
from .nova import Nova
from .qwen2vl import Qwen2VL
from .video_llama import VideoLlama
from .claude import Claude

providers: dict[V2TModelName, type[V2TBaseModel]] = {
    V2TModelName.GPT4o: GPT,
    V2TModelName.GPT4o_mini: GPT,
    V2TModelName.Qwen2_VL_7B: Qwen2VL,
    V2TModelName.Qwen2_5_VL_3B: Qwen2VL,
    V2TModelName.Qwen2_5_VL_7B: Qwen2VL,
    V2TModelName.Qwen2_5_VL_72B: Qwen2VL,
    V2TModelName.LlavaVideo_7B: LlavaVideo,
    V2TModelName.LlavaVideo_72B: LlavaVideo,
    V2TModelName.VideoLlama_7B: VideoLlama,
    V2TModelName.VideoLlama_72B: VideoLlama,
    V2TModelName.InternVL2_5_1B: InternVL2_5,
    V2TModelName.InternVL2_5_2B: InternVL2_5,
    V2TModelName.InternVL2_5_4B: InternVL2_5,
    V2TModelName.InternVL2_5_8B: InternVL2_5,
    V2TModelName.InternVL2_5_26B: InternVL2_5,
    V2TModelName.InternVL2_5_38B: InternVL2_5,
    V2TModelName.InternVL2_5_78B: InternVL2_5,
    V2TModelName.NovaLite: Nova,
    V2TModelName.NovaPro: Nova,
    V2TModelName.Claude_35_Sonnet: Claude,
}

model_ids: dict[V2TModelName, str] = {
    V2TModelName.GPT4o: "gpt-4o-2024-11-20",
    V2TModelName.GPT4o_mini: "gpt-4o-mini-2024-07-18",
    V2TModelName.Qwen2_VL_7B: "Qwen/Qwen2-VL-7B-Instruct",
    V2TModelName.Qwen2_5_VL_3B: "Qwen/Qwen2.5-VL-3B-Instruct",
    V2TModelName.Qwen2_5_VL_7B: "Qwen/Qwen2.5-VL-7B-Instruct",
    V2TModelName.Qwen2_5_VL_72B: "Qwen/Qwen2.5-VL-72B-Instruct",
    V2TModelName.LlavaVideo_7B: "lmms-lab/LLaVA-Video-7B-Qwen2",
    V2TModelName.LlavaVideo_72B: "lmms-lab/LLaVA-Video-72B-Qwen2",
    V2TModelName.VideoLlama_7B: "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
    V2TModelName.VideoLlama_72B: "DAMO-NLP-SG/VideoLLaMA2-72B",
    V2TModelName.InternVL2_5_1B: "OpenGVLab/InternVL2_5-1B",
    V2TModelName.InternVL2_5_2B: "OpenGVLab/InternVL2_5-2B",
    V2TModelName.InternVL2_5_4B: "OpenGVLab/InternVL2_5-4B",
    V2TModelName.InternVL2_5_8B: "OpenGVLab/InternVL2_5-8B",
    V2TModelName.InternVL2_5_26B: "OpenGVLab/InternVL2_5-26B",
    V2TModelName.InternVL2_5_38B: "OpenGVLab/InternVL2_5-38B",
    V2TModelName.InternVL2_5_78B: "OpenGVLab/InternVL2_5-78B",
    V2TModelName.NovaLite: "amazon.nova-lite-v1:0",
    V2TModelName.NovaPro: "amazon.nova-pro-v1:0",
    V2TModelName.Claude_35_Sonnet: "claude-3-5-sonnet-20241022",
}

v2t_model_list = list(providers.keys())


def load_v2t_model(model_name: V2TModelName | str, **load_kwargs) -> V2TBaseModel:
    model_name = V2TModelName(model_name)
    return providers[model_name](
        model_name, model_id=model_ids.get(model_name, str(model_name)), **load_kwargs
    )
