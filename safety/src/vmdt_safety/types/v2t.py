from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from .utils import gen_id


class V2TData(BaseModel):
    id: str = Field(default_factory=gen_id)
    video_path: Path
    video_content: str
    violate_reason: str | None = None
    prompt: str
    redteaming: str
    category: str
    base_id: str | None = None


V2TDataList = TypeAdapter(list[V2TData])


class V2TLLMJudgeResult(BaseModel):
    judge_model: str
    reason_harm: str = ""
    harm: bool = False
    reason_refusal: str = ""
    refusal: bool = False


class V2TResult(BaseModel):
    id: str = Field(default_factory=gen_id)
    text: str
    input_id: str
    by_pass: bool = True
    llm_judge: V2TLLMJudgeResult | None = None
