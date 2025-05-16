from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel


def gen_id(prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}{uuid4().hex[:8]}{suffix}"


class T2VError(str, Enum):
    SAFETY_REFUSAL = "safety_refusal"
    GENERATION_ERROR = "generation_error"


class T2VOutput(BaseModel):
    text_input: str
    video_path: Path | None = None
    error: str | None = None


class T2VBaseModel(ABC):
    def __init__(self, model_name: str, **load_kwargs):
        self.model_name = model_name
        self.load_model(**load_kwargs)

    @abstractmethod
    def load_model(self, **kwargs):
        pass

    @abstractmethod
    def generate_videos(
        self, text_inputs: list[str], output_dir: Path, **gen_kwargs
    ) -> list[T2VOutput]:
        pass
