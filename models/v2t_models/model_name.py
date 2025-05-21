from enum import StrEnum


class V2TModelName(StrEnum):
    Qwen2_VL_7B = "Qwen2-VL-7B"

    Qwen2_5_VL_3B = "Qwen2.5-VL-3B"
    Qwen2_5_VL_7B = "Qwen2.5-VL-7B"
    Qwen2_5_VL_72B = "Qwen2.5-VL-72B"

    InternVL2_5_1B = "InternVL2.5-1B"
    InternVL2_5_2B = "InternVL2.5-2B"
    InternVL2_5_4B = "InternVL2.5-4B"
    InternVL2_5_8B = "InternVL2.5-8B"
    InternVL2_5_26B = "InternVL2.5-26B"
    InternVL2_5_38B = "InternVL2.5-38B"
    InternVL2_5_78B = "InternVL2.5-78B"

    LlavaVideo_7B = "LlavaVideo-7B"
    LlavaVideo_72B = "LlavaVideo-72B"

    VideoLlama_7B = "VideoLlama-7B"
    VideoLlama_72B = "VideoLlama-72B"

    GPT4o = "GPT-4o"
    GPT4o_mini = "GPT-4o-mini"

    NovaLite = "Nova-Lite"
    NovaPro = "Nova-Pro"

    Claude_35_Sonnet = "Claude-3.5-Sonnet"
