import argparse
import random
import asyncio
import json
import traceback

from tqdm import tqdm
from pydantic import create_model

from adversarial.common.utils import (
    set_seed, 
    read_json, 
    write_jsonl, 
    gen_id,
)
from adversarial.common.properties import (
    ActionRecognitionProperty,
    AttributeRecognitionProperty,
    CountingProperty,
    ObjectRecognitionProperty,
    SpatialUnderstandingProperty,
)
from adversarial.common.openai_client import async_call_openai
from adversarial.t2v.t2v_utils import T2VInstance

def select_property(extracted_info, task):
    
    if task == "ActionRecognition":
        property = random.choice(extracted_info["actions"])
        obj = property["subject"]
        action = property["predicate"]
        target = property["object"]
        return ActionRecognitionProperty(
            obj=obj,
            action=action,
            target=target,
        )
        
    if task == "AttributeRecognition":
        objects = [obj for obj in extracted_info["objects"].keys() if len(extracted_info["objects"][obj]["attributes"]) > 0]
        obj = random.choice(objects)
        attribute = random.choice(extracted_info["objects"][obj]["attributes"])["attribute"]
        return AttributeRecognitionProperty(
            obj=obj,
            attribute=attribute,
        )
        
    if task == "Counting":
        obj = random.choice(list(extracted_info["objects"].keys()))
        count = random.choice(["two", "three", "four", "five", "six", "seven"])
        return CountingProperty(
            obj=obj,
            count=count,
        )
        
    if task == "ObjectRecognition":
        obj = random.choice(list(extracted_info["objects"].keys()))
        return ObjectRecognitionProperty(
            obj=obj,
        )
        
    if task == "SpatialUnderstanding":
        obj_1, obj_2 = random.sample(list(extracted_info["objects"].keys()), 2)
        relation = random.choice(["left of", "right of", "above", "below", "closer to the camera than", "farther from the camera than"])
        return SpatialUnderstandingProperty(
            obj_1=obj_1,
            relation=relation,
            obj_2=obj_2,
        )
        
    raise ValueError(f"Task {task} not recognized")

##################################################

base_prompt = """
First, construct a simple prompt for a text-to-video model. The prompt MUST explicitely include the following property: {property}. Use the following video description as additional context, if needed: {video_description}. This is the clean prompt.
Second, mutate the property such that the resulting video would be clearly visibly distinct from the original video. This is the target property. 
Third, construct a simple prompt for the target property. This prompt MUST explicitely include the target property and MUST NOT include the original property. This is the target prompt.
Finally, write a True/False question that can be used to distinguish between the clean prompt and the target prompt. The statement should be True if and only if the original property is correct. This the evaluation question.
Be sure to follow the correct structured output format.
"""

action_recgonition_examples = """
Example 1:
Property: obj='person' action='picking fruit' target='apple'
Video Description: "A man in a black hoodie and trousers is picking apples in an orchard, carefully placing them in a bag and then into a wooden crate, while a narrator explains the process."
Output:
{
    "clean_prompt": "A person picking apples in an orchard.",
    "target_property": {
        "obj": "person",
        "action": "throwing fruit",
        "target": "apple"
    },
    "target_prompt": "A person throwing apples in an orchard.",
    "evaluation_question": "Is the person **picking** apples?"
}

Example 2:
Property: obj='dancer' action='tap dancing' target=None
Video Description: "A young woman is performing a tap dance routine alone in a large room with a hardwood floor."
Output:
{
    "clean_prompt": "A dancer tap dancing in a large room.",
    "target_property": {
        "obj": "dancer",
        "action": "break dancing",
        "target": None
    },
    "target_prompt": "A dancer break dancing in a large room.",
    "evaluation_question": "Is the dancer **tap dancing**?"
}
"""

attribute_recognition_examples = """
Example 1:
Property: obj='person' attribute='wearing black hoodie and trousers'
Video Description: "A man in a black hoodie and trousers is picking apples in an orchard, carefully placing them in a bag and then into a wooden crate, while a narrator explains the process."
Output:
{
    "clean_prompt": "A person wearing a black hoodie and trousers picking apples.",
    "target_property": {
        "obj": "person",
        "attribute": "wearing red hoodie and trousers"
    },
    "target_prompt": "A person wearing a red hoodie and trousers picking apples.",
    "evaluation_question": "Is the person is wearing a **black** hoodie?"
}

Example 2:
Property: obj='floor' attribute='hardwood'
Video Description: "A young woman is performing a tap dance routine alone in a large room with a hardwood floor."
Output:
{
    "clean_prompt": "A dancer on a hardwood floor.",
    "target_property": {
        "obj": "floor",
        "attribute": "marble"
    },
    "target_prompt": "A dancer on a marble floor.",
    "evaluation_question": "Is the dancer on **hardwood** floors?"
}
"""

counting_examples = """
Example 1:
Property: obj='apple' count='three'
Video Description: "A man in a black hoodie and trousers is picking apples in an orchard, carefully placing them in a bag and then into a wooden crate, while a narrator explains the process."
Output:
{
    "clean_prompt": "A man with three apples in a basket.",
    "target_property": {
        "obj": "apple",
        "count": "six"
    },
    "target_prompt": "A man with six apples in a basket.",
    "evaluation_question": "Are there exactly **three** apples in the basket?"
}

Example 2:
Property: obj='dancer' count='five'
Video Description: "A young woman is performing a tap dance routine alone in a large room with a hardwood floor."
Output:
{
    "clean_prompt": "Five dancers performing in a large room.",
    "target_property": {
        "obj": "dancer",
        "count": "two"
    },
    "target_prompt": "Two dancers performing in a large room.",
    "evaluation_question": "Are there exactly **five** dancers in the room?"
}
"""

object_recognition_examples = """
Example 1:
Property: obj='apple'
Video Description: "A man in a black hoodie and trousers is picking apples in an orchard, carefully placing them in a bag and then into a wooden crate, while a narrator explains the process."
Output:
{
    "clean_prompt": "A man picking apples from a tree.",
    "target_property": {
        "obj": "banana"
    },
    "target_prompt": "A man picking bananas from a tree.",
    "evaluation_question": "Is the man picking **apples**?"
}

Example 2:
Property: obj='dancer'
Video Description: "A young woman is performing a tap dance routine alone in a large room with a hardwood floor."
Output:
{
    "clean_prompt": "A dancer tap dancing in a large room.",
    "target_property": {
        "obj": "magician"
    },
    "target_prompt": "A magician performing in a large room.",
    "evaluation_question": "Is the performer in the video a **dancer**?"
}
"""

spatial_understanding_examples = """
Example 1:
Property: obj_1='person' relation='left of' obj_2='crate'
Video Description: "A man in a black hoodie and trousers is picking apples in an orchard, carefully placing them in a bag and then into a wooden crate, while a narrator explains the process."
Output:
{
    "clean_prompt": "A person standing to the left of a wooden crate.",
    "target_property": {
        "obj_1": "person",
        "relation": "right of",
        "obj_2": "crate"
    },
    "target_prompt": "A person standing to the right of a wooden crate.",
    "evaluation_question": "Is the person to the **left** of a wooden crate?"
}

Example 2:
Property: obj='dancer' relation='above' obj_2='floor'
Video Description: "A young woman is performing a tap dance routine alone in a large room with a hardwood floor."
Output:
{
    "clean_prompt": "A dancer above the floor in a large room.",
    "target_property": {
        "obj_1": "dancer",
        "relation": "below",
        "obj_2": "floor"
    },
    "target_prompt": "A dancer below the floor in a large room.",
    "evaluation_question": "Is the dancer **above** the floor?"
}

"""

task_to_demonstrations = {
    "ActionRecognition": action_recgonition_examples,
    "AttributeRecognition": attribute_recognition_examples,
    "Counting": counting_examples,
    "ObjectRecognition": object_recognition_examples,
    "SpatialUnderstanding": spatial_understanding_examples,
}

task_to_property = {
    "ActionRecognition": ActionRecognitionProperty,
    "AttributeRecognition": AttributeRecognitionProperty,
    "Counting": CountingProperty,
    "ObjectRecognition": ObjectRecognitionProperty,
    "SpatialUnderstanding": SpatialUnderstandingProperty,
}    

async def write_prompt(clean_property, task, video_description):
    
    CallOutput = create_model(
        "CallOutput", 
        clean_prompt=(str, ...), 
        target_property=(task_to_property[task], ...), 
        target_prompt=(str, ...), 
        evaluation_question=(str, ...) # Not sure if we'll use this or not for eval
    )
    
    prompt = base_prompt.format(property=clean_property, video_description=video_description) + task_to_demonstrations[task]
    
    response = await async_call_openai(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini-2024-07-18",
        system_prompt=None,
        response_format=CallOutput,
        max_tokens=4096,
        temperature=0.0,
    )
    return CallOutput.parse_raw(response)
    
async def construct_instance(extracted_info, task, semaphore):
    async with semaphore:
        
        try:
            video_description = extracted_info["extracted_information"]["scenes"]["scene_1"]["description"]
            clean_property = select_property(extracted_info["extracted_information"], task)
            call_output = await write_prompt(clean_property, task, video_description)
            
            return T2VInstance(
                id=gen_id(),
                source="vatex",
                video_id=extracted_info["video_id"],
                video_description=video_description,
                task=task,
                clean_prompt=call_output.clean_prompt,
                clean_property=clean_property,
                target_prompt=call_output.target_prompt,
                target_property=call_output.target_property,
                evaluation_question=call_output.evaluation_question,
            )
        
        except: 
            print(f"Failed to construct instance for video {extracted_info['video_id']} and task {task}.")
            traceback.print_exc()
            return None

async def construct_instances(extracted_infos, task, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    
    tasks = [asyncio.create_task(construct_instance(extracted_info, task, semaphore)) for extracted_info in extracted_infos]
    
    instances = []
    for completed in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        instance = await completed
        if instance is not None:
            instances.append(instance)
    
    return instances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    extracted_infos = read_json(args.input_file)
    instances = asyncio.run(construct_instances(extracted_infos, args.task, args.concurrency))
    
    write_jsonl([instance.to_dict() for instance in instances], args.output_file)