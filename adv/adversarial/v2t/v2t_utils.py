import re
from typing import Union, List, Optional, Dict, Any
from pydantic import BaseModel, root_validator

VATEX_VIDEO_DIR = "/ib-scratch/chenguang02/scratch1/cnicholas/mmdt-video/data/vatex/videos"
CLEVRER_VIDEO_DIR = "/ib-scratch/chenguang02/scratch1/cnicholas/mmdt-video/data/clevrer/video_validation"
NEPTUNE_VIDEO_DIR = "/ib-scratch/chenguang02/scratch1/cnicholas/mmdt-video/data/neptune/videos"

TASK_TO_DATASET = {
    "ObjectRecognition": "vatex",
    "AttributeRecognition": "vatex",
    "ActionRecognition": "vatex",
    "Counting": "clevrer",
    "SpatialUnderstanding": "clevrer",
    "SceneUnderstanding": "neptune"
}

def get_video_path(video_id, task_name):
    dataset = TASK_TO_DATASET[task_name]
    if dataset == "vatex":
        return f"{VATEX_VIDEO_DIR}/{video_id}.mp4"
    elif dataset == "clevrer":
        # There are 5 directories: video_10000-11000, video_11000-12000, video_12000-13000, video_13000-14000, video_14000-15000. Then inside the dir the video is named video_video_id.mp4
        return f"{CLEVRER_VIDEO_DIR}/video_{int(video_id/1000)*1000}-{int(video_id/1000)*1000+1000}/video_{video_id}.mp4"
    elif dataset == "neptune":
        return f"{NEPTUNE_VIDEO_DIR}/{video_id}.mp4"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
def get_mc_format(question, answer_choices, gt):
    """Returns the full question formatted with multiple choice answers, the answer choices as letters, and the ground truth answer choices as letters."""
    formatted_question = f"Question: {question}\n"
    # use letters for answer choices
    all_choices, gt_choices = [], []
    for i, choice in enumerate(answer_choices):
        letter = chr(65 + i)
        if i < len(answer_choices) - 1:
            formatted_question += f"{letter}. {choice}\n"
        else:
            formatted_question += f"{letter}. {choice}"
        all_choices.append(letter)
        if choice in gt:
            gt_choices.append(letter)
    return formatted_question, all_choices, gt_choices

def parse_mc_answer(model_output, answer_choices, choice_letters, one_correct_answer=True):
    """Parse the answer to the multiple choice question. First try to find the full answer of {letter}. {answer}. If that is not possible, find the letter. Then, if neither is possible, find the first instance of any answer choice without the letter, both lower and upper case. Use regex. If cannot find anything return the empty string."""
    if not one_correct_answer:
        raise NotImplementedError("Not implemented for multiple correct answers.")
    # Remove 'Answer:' or 'ASSISTANT:' from the model output in case it is there
    model_output = re.sub(r"Answer:|ASSISTANT:", "", model_output)
    # first try to find the full answer of {letter}. {answer}
    for letter, choice in zip(choice_letters, answer_choices):
        if f"{letter}. {choice}" in model_output:
            return letter
    # if not possible, use regex to find first instance of any answer choice without the letter, both lower and upper case
    for choice in answer_choices:
        match = re.search(f"{choice}", model_output, re.IGNORECASE)
        if match:
            return choice_letters[answer_choices.index(choice)]
    # if that is not possible, find the letter
    for letter in choice_letters:
        if f"{letter}" in model_output:
            return letter
    return ""

class EvaluationResult(BaseModel):
    clean_response: Optional[str] = None
    clean_score: Optional[float] = None
    adv_response: Optional[str] = None
    adv_score: Optional[float] = None

class V2TInstance(BaseModel):
    id: str
    source: str
    video_id: Union[str, int]
    task: str
    question: str
    answer_choices: List[str]
    gt: List[str]
    surrogate_model: Optional[str] = None
    attack: Optional[str] = None
    clean_frames_id: Optional[str] = None
    adv_frames_id: Optional[str] = None
    eval_results: Optional[Dict[str, EvaluationResult]] = {}