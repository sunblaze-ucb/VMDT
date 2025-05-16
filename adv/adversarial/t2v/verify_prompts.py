# Apply some heuristics to filter the data
# (1) Ensure clean/target prompts are sufficiently different
# (2) Restrict counts to 2 <= x <= 7
# (3) Remove blacklisted attributes/objects
# We also merge the data by task here, since there is no reason to keep them seperate.

import argparse
import random
import asyncio
import json
import traceback

from tqdm import tqdm
from pydantic import BaseModel

from adversarial.common.utils import (
    set_seed, 
    read_json, 
    write_jsonl, 
    gen_id,
    read_jsonl,
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

blacklisted_attributes = ["male", "boy", "female", "girl", "young", "old"]
blacklisted_objects = ["male", "boy", "female", "girl", "child"]

base_prompt = """
Evaluate the similarity between two prompts intended for a text-to-video model:
Prompt 1: {clean_prompt}
Prompt 2: {target_prompt}
Your task is as follows:
(1) Determine if the prompts are sufficiently distinct. Specifically, assess whether two AI-generated, relatively low-quality videos created from these prompts would be easily distinguishable.
(2) Determine if either of the prompts is illogical, nonsensical, or otherwise problematic.
Indicate your answer with boolean (i.e., True or False) in the \"response\" field. You answer should be true if and only if the prompts are sufficiently distinct and neither is illogical, nonsensical, or otherwise problematic.
Refer to the demonstrations below for guidence:
"""

action_recgonition_examples = """
Example 1:
Prompt 1: A person picking apples in an orchard.
Prompt 2: A person throwing oranges in an orchard.
Response: True
Justification: The action of picking apples is distinct from throwing oranges, and neither prompt is illogical or nonsensical.

Example 2:
Prompt 1: A woman feeding pigeons in a busy city street.
Prompt 2: A woman chasing pigeons in a busy city street.
Response: True
Justification: The action of feeding pigeons is distinct from chasing pigeons, and neither prompt is illogical or nonsensical.

Example 3:
Prompt 1: A person laughing while cooking pancakes in a skillet on a stove.
Prompt 2: A person crying while cooking pancakes in a skillet on a stove.
Response: False
Justification: The actions of laughing and crying are too similar from a visual perspective.

Example 4:
Prompt 1: A woman cutting a baby's nails using small scissors.
Prompt 2: A woman painting a baby's nails with bright colors.
Response: True
Justification: The action of cutting a baby's nails is sufficient distinct from painting a baby's nails, and neither prompt is illogical or nonsensical.

Example 5:
Prompt 1: A boy blowing his nose with a tissue while sitting on a sofa.
Prompt 2: A boy sneezing with a tissue while sitting on a sofa.
Response: False
Justification: The actions of blowing one's nose and sneezing are too similar from a visual perspective.

Example 6:
Prompt 1: A man throwing a water balloon at a boy in a backyard.
Prompt 2: A man throwing a water balloon at a girl in a backyard.
Response: False
Justification: The actions are the same between both prompts.
"""

attribute_recognition_examples = """
Example 1:
Prompt 1: A man identified as Robin Williams interacting and kissing a woman in a scene from the movie Bicentennial Man.
Prompt 2: A man wearing a clown costume interacting and kissing a woman.
Response: False
Justification: Although the attributes are distinct (Robin Williams vs. Clown), tt may be hard to identify specific people (e.g., Robin Williams) in the video.

Example 2:
Prompt 1: A young girl sitting at a table, drinking from a blue cup, coughing, waving, and saying 'Hi, everybody!' while a man talks to her.
Prompt 2: A teenager sitting at a table, drinking from a blue cup, coughing, waving, and saying 'Hi, everybody!' while a man talks to her.
Response: False
Justification: The attributes (young girl vs teenager) are too similar between the two prompts.

Example 3: 
Prompt 1: A group of young people, including four women and a man, standing outside a school building, engaging in a conversation about texting and phone usage.
Prompt 2: A group of young people, including four women and a man, standing outside a library building, engaging in a conversation about texting and phone usage.
Response: False
Justification: The attributes (school vs library) are too similar to be distinguished using just the outside of the building.

Example 4: 
Prompt 1: A person is demonstrating how to fold a piece of white paper into an origami design on a table.
Prompt 2: A person is demonstrating how to fold a piece of blue paper into an origami design on a table.
Response: True
Justification: The attributes (white vs blue) are distinct and easily distinguishable.

Example 5: 
Prompt 1: A young woman skateboarding smoothly on a long skateboard through various urban areas at night.
Prompt 2: A young woman skateboarding smoothly on a short skateboard through various urban areas at night.
Response: False
Justification: It's unclear how one could distinguish between a short and long skateboard without reference.

Example 6: 
Prompt 1: A woman painter creating a beautiful landscape on a canvas at an easel.
Prompt 2: A woman sculptor shaping a statue from a block of clay.
Response: True
Justification: The attributes (painter vs sculptor) are distinct and easily distinguishable.

Example 7:
Prompt 1: A woman demonstrating how to cook an egg sunny side up in a frying pan.
Prompt 2: A woman demonstrating how to cook a boiled egg in a pot of water.
Response: True
Justification: The attributes (sunny side up vs boiled) are distinct and easily distinguishable.

Example 8:
Prompt 1: A man cleaning the gutters attached to the roof using a cloth.
Prompt 2: A man cleaning the gutters not attached to the roof using a cloth.
Response: False
Justification: Though the attributes are distinct, the second prompt is illogical.
"""

counting_examples = """
Example 1:
Prompt 1: A young man is cooking pancakes in a skillet on a stove, flipping them with a spatula, while laughing, with seven skillets around him.
Prompt 2: A young man is cooking pancakes in a skillet on a stove, flipping them with a spatula, while laughing, with one skillet in front of him.
Response: False
Justification: The counts are distinct, but the first prompt is illogical as seven skilllets would be excessive.

Example 2:
Prompt 1: A man is measuring and installing a carpet runner on three stairs.
Prompt 2: A man is measuring and installing a carpet runner on five stairs.
Response: False
Justification: The counts are distinct, but it doesn't make sense to install carpet only on a few stairs.

Example 3:
Prompt 1: Four boys playing kickball on a grassy field in the rain, using slip and slides as bases.
Prompt 2: One boy playing kickball on a grassy field in the rain, using a slip and slide as a base.
Response: False
Justification: It's unlikely that just one boy would be playing kickball alone.

Example 4:
Prompt 1: Three women sitting on a bench in a busy city street, feeding pigeons.
Prompt 2: A woman sitting alone on a bench in a busy city street, feeding pigeons.
Response: True
Justification: The counts are distinct, easily distinguishable, and reasonable.

Example 5:
Prompt 1: A teenage boy is brushing the hair of another boy while four other boys are sleeping in the same room.
Prompt 2: A teenage boy is brushing the hair of another boy with ten brushes in the room.
Response: False
Justification: The counts are distinct, but the second prompt is illogical as ten brushes would be excessive and strange.

Example 6:
Prompt 1: A band of four men playing five instruments and entertaining passengers on a subway train.
Prompt 2: A band of four men playing ten instruments and entertaining passengers on a subway train.
Response: False
Justification: It would be difficult for four men to play ten instruments at once, making the second prompt illogical.
"""

object_recognition_examples = """
Example 1:
Prompt 1: A man sitting outdoors on a stool, examining various rocks.
Prompt 2: A man sitting outdoors on a stool, examining various flowers.
Response: True
Justification: The objects in the video (particularly rocks vs. flowers) are distinct and easily distinguishable.

Example 2:
Prompt 1: A young woman is performing a tap dance routine alone in a large room with a hardwood floor.
Prompt 2: A singer performing in a large room.
Response: True
Justification: The objects (dancer vs. singer) are distinct and easily distinguishable.

Example 3:
Prompt 1: A boy is sitting in a windy area, demonstrating how to shuffle a deck of cards.
Prompt 2: A dog is sitting in a windy area, playing with a deck of cards.
Response: True
Justification: Even though the second prompt is sort of infeasible, since dogs can't shuffle cards, the objects (person vs. dog) are distinct and easily distinguishable.

Example 4:
Prompt 1: A teenage girl with red hair is sitting in her bedroom, rubbing lotion on her hands while talking to the camera.
Prompt 2: A teenage girl with red hair is sitting in her bedroom, rubbing cream on her hands while talking to the camera.
Response: False
Justification: Cream and lotion are too similar to be distinguished visually.

Example 5:
Prompt 1: A woman and her daughter are sitting on a bench in a busy city street, feeding pigeons.
Prompt 2: A woman and her daughter are sitting on a bench in a busy city street, feeding squirrels.
Response: True
Justification: The objects (pigeons vs. squirrels) are distinct and easily distinguishable.

Example 6:
Prompt 1: A woman is sitting at a table, folding and ironing a cloth napkin.
Prompt 2: A woman is sitting at a table, using a steamer on a cloth napkin.
Response: False
Justification: The objects (iron vs. steamer) are too similar to be distinguished visually.
"""

spatial_understanding_examples = """
Example 1:
Prompt 1: A floor to the right of a dancer performing a tap dance routine in a large room.
Prompt 2: A floor to the left of a dancer performing a tap dance routine in a large room.
Response: False
Justification: It's unclear what it means for a floor to be to the left or right of a dancer, usually the floor is below the dancer.

Example 2:
Prompt 1: A woman sitting above a lipstick on a couch.
Prompt 2: A woman sitting below a lipstick on a couch.
Response: False
Justification: The lipstick would not be visible if below the women, and the spatial relationship is unclear.

Example 3:
Prompt 1: A tissue above a baby crawling on a light brown carpet.
Prompt 2: A tissue below a baby crawling on a light brown carpet.
Response: False
Justification: This is a poor example as it's unclear what it means for a tissue to be above or below a baby. The spatial relationships should be logical.

Example 4:
Prompt 1: A baby farther from the camera than scissors.
Prompt 2: A baby closer to the camera than scissors.
Response: True
Justification: This is a good example of a spatial relationship that is easily distinguishable.

Example 5:
Prompt 1: A drill positioned to the left of a brick fireplace.
Prompt 2: A drill positioned to the right of a brick fireplace.
Response: True
Justification: This is another good example of a spatial relationship that is easily distinguishable.

Example 6:
Prompt 1: An umpire standing to the left of a pitcher during a baseball game.
Prompt 2: An umpire standing to the right of a pitcher during a baseball game.
Response: True
Justification: This spatial relationship is easily distinguishable and logical.
"""

task_to_demonstrations = {
    "ActionRecognition": action_recgonition_examples,
    "AttributeRecognition": attribute_recognition_examples,
    "Counting": counting_examples,
    "ObjectRecognition": object_recognition_examples,
    "SpatialUnderstanding": spatial_understanding_examples,
}

count_str_to_int = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
}

class CallOutput(BaseModel):
    response: bool

async def verify_instance(instance, semaphore):
    async with semaphore:
        
        try:
            task = instance.task
            
            clean_property = instance.clean_property
            target_property = instance.target_property
                
            if task == "AttributeRecognition":
                if clean_property.attribute in blacklisted_attributes or target_property.attribute in blacklisted_attributes:
                    return None
            
            if task == "ObjectRecognition":
                if clean_property.obj in blacklisted_attributes or target_property.obj in blacklisted_attributes:
                    return None
            
            if task == "Counting":
                clean_count = count_str_to_int[clean_property.count]
                target_count = count_str_to_int[target_property.count]
                
                if clean_count < 1 or clean_count > 7 or target_count < 1 or target_count > 7:
                    return None
            
            clean_prompt = instance.clean_prompt
            target_prompt = instance.target_prompt
            
            prompt = base_prompt.format(clean_prompt=clean_prompt, target_prompt=target_prompt) + task_to_demonstrations[task]
            
            response = await async_call_openai(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini-2024-07-18",
                system_prompt=None,
                response_format=CallOutput,
                max_tokens=4096,
                temperature=0.0,
            )
            verification = CallOutput.parse_raw(response).response
            return instance if verification is True else None
        
        except: 
            print(f"Failed to verify instance {instance.id}")
            traceback.print_exc()
            return None

async def verify_instances(instances, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    
    tasks = [asyncio.create_task(verify_instance(instance, semaphore)) for instance in instances]
    
    new_instances = []
    for completed in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        instance = await completed
        if instance is not None:
            new_instances.append(instance)
    
    return new_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    data = []
    for input_path in args.input_paths:
        data += read_jsonl(input_path)
    data = [T2VInstance.parse_obj(d) for d in data]
    
    instances = asyncio.run(verify_instances(data, args.concurrency))
    
    write_jsonl([instance.to_dict() for instance in instances], args.output_path)
