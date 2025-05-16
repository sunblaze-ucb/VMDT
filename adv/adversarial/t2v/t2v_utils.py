from typing import Union, List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, root_validator
from adversarial.common.properties import BaseProperty

from adversarial.common.properties import (
    ActionRecognitionProperty,
    AttributeRecognitionProperty,
    CountingProperty,
    ObjectRecognitionProperty,
    SpatialUnderstandingProperty,
)

task_to_property = {
    "ActionRecognition": ActionRecognitionProperty,
    "AttributeRecognition": AttributeRecognitionProperty,
    "Counting": CountingProperty,
    "ObjectRecognition": ObjectRecognitionProperty,
    "SpatialUnderstanding": SpatialUnderstandingProperty,
} 

class EvaluationResult(BaseModel):
    clean_vid_id: Optional[str] = None
    clean_score: Optional[float] = None
    clean_xclip: Optional[float] = None
    clean_vbench: Optional[Dict[str, float]] = {}
    adv_vid_id: Optional[str] = None
    adv_score: Optional[float] = None
    adv_xclip: Optional[float] = None
    adv_vbench: Optional[Dict[str, float]] = {}
    
class T2VInstance(BaseModel):
    id: str
    source: str
    video_id: Union[str, int]
    video_description: str
    task: str
    clean_prompt: str
    clean_property: BaseProperty
    target_prompt: str
    target_property: BaseProperty
    evaluation_question: str
    surrogate_model: Optional[str] = None
    attack: Optional[str] = None
    adversarial_prompt: Optional[str] = None
    eval_results: Optional[Dict[str, EvaluationResult]] = {}
    
    @root_validator(pre=True)
    def parse_properties(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        task = values.get('task')
        if task in task_to_property:
            prop_cls = task_to_property[task]
            for field_name in ['clean_property', 'target_property']:
                prop_value = values.get(field_name)
                if isinstance(prop_value, dict):
                    values[field_name] = prop_cls.parse_obj(prop_value)
        else:
            raise ValueError(f"No property mapping found for task '{task}'")
        return values
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "video_id": self.video_id,
            "video_description": self.video_description,
            "task": self.task,
            "clean_prompt": self.clean_prompt,
            "clean_property": self.clean_property.dict(),
            "target_prompt": self.target_prompt,
            "target_property": self.target_property.dict(),
            "evaluation_question": self.evaluation_question,
            "surrogate_model": self.surrogate_model,
            "attack": self.attack,
            "adversarial_prompt": self.adversarial_prompt,
            "eval_results": {k: v.dict() for k, v in self.eval_results.items()},
        }
        
class T2VOutput(BaseModel):
    video_path: Union[str, Path]