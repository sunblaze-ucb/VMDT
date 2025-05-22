from typing import Union, List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel

class EvaluationResult(BaseModel):
    benign_vid_id: Optional[str] = None
    benign_score: Optional[float] = None
    adv_vid_id: Optional[str] = None
    adv_score: Optional[float] = None
    
class T2VInstance(BaseModel):
    id: str
    surrogate: str
    attack: str
    benign: str
    adversarial: str
    evaluation: str
    eval_results: Optional[Dict[str, EvaluationResult]] = {}
        
class T2VOutput(BaseModel):
    video_path: Union[str, Path]