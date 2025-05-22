from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from .utils import gen_id


class T2VData(BaseModel):
    id: str = Field(default_factory=gen_id)
    prompt: str
    redteaming: str
    category: str
    base_id: str


T2VDataList = TypeAdapter(list[T2VData])


class T2VLLMJudgeResult(BaseModel):
    judge_model: str
    answer: bool = False
    reason: str = ""


class T2VResult(BaseModel):
    id: str = Field(default_factory=gen_id)
    input_id: str
    video_path: Path | None = None
    by_pass: bool = True
    llm_judge: T2VLLMJudgeResult | None = None
