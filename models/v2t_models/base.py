from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from uuid import uuid4

def gen_id(prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}{uuid4().hex[:8]}{suffix}"

class V2TError(str, Enum):
    SAFETY_REFUSAL = "safety_refusal"
    GENERATION_ERROR = "generation_error"

class V2TOutput(BaseModel):
    video_input: Path
    # The generated text description (or caption) for the input video.
    text_output: str | None = None
    error: str | None = None

class V2TBaseModel(ABC):
    def __init__(self, model_name: str, **load_kwargs):
        self.model_name = model_name
        self.load_model(**load_kwargs)

    @abstractmethod
    def load_model(self, **kwargs):
        """
        Load the underlying V2T model.
        """
        pass

    @abstractmethod
    def generate_texts(
        self, video_inputs: list[Path], prompts: list[str], **gen_kwargs
    ) -> list[V2TOutput]:
        """
        Given a list of video file paths, generate a corresponding list
        of text outputs (e.g. captions or descriptions).

        :param video_inputs: List of video file paths.
        :param gen_kwargs: Additional keyword arguments to control generation.
        :return: List of V2TOutput objects.
        """
        pass
