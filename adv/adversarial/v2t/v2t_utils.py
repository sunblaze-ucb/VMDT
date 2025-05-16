import re
from typing import Union, List, Optional, Dict, Any
from pydantic import BaseModel, root_validator
    
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
    benign_response: Optional[str] = None
    benign_score: Optional[float] = None
    adv_response: Optional[str] = None
    adv_score: Optional[float] = None

class V2TInstance(BaseModel):
    id: str
    surrogate: str
    attack: str
    benign: str
    adversarial: str
    question: str
    choices: List[str]
    answer: str
    task: Optional[str] = None
    eval_results: Optional[Dict[str, EvaluationResult]] = {}