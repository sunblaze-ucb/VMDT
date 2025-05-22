from .base import T2VBaseModel
from .model_name import T2VModelName
from .video_crafter import VideoCrafter
from .cogvideox import CogVideoX
from .vchitect2 import Vchitect2
from .luma import Luma
from .pika import Pika
from .nova_reel import Nova_reel
from .opensora import OpenSora

providers: dict[T2VModelName, type[T2VBaseModel]] = {
    T2VModelName.VideoCrafter2: VideoCrafter,
    T2VModelName.CogVideoX_5B: CogVideoX,
    T2VModelName.Vchitect2: Vchitect2,
    T2VModelName.Luma: Luma,
    T2VModelName.Pika: Pika,
    T2VModelName.Nova: Nova_reel,
    T2VModelName.OpenSora1_2: OpenSora,
}

t2v_model_list = list(providers.keys())


def load_t2v_model(model_name: T2VModelName | str, **load_kwargs) -> T2VBaseModel:
    model_name = T2VModelName(model_name)
    return providers[model_name](model_name, **load_kwargs)
