import json
import math
import os
import argparse
import base64
import cv2
import numpy as np
import re
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
import inflect
import shutil
import asyncio
from tqdm.asyncio import tqdm
from enum import Enum
from dataclasses import dataclass


from hallucination.call_openai import async_call_openai
from hallucination.data_utils import EVAL_MODEL, DEPTH_RELATIONS, SPACE_RELATIONS, SPATIAL_RELATIONS_REVERSE
from hallucination.text_video.qwen_vl_helpers import init_qwenvl, inference_qwenvl
from hallucination.text_video.detect_transitions import scene_detection
from hallucination.text_video.average import average

STR_TO_NUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

# Define constants
MAX_TOKENS = 4096
GRID_SIZE = 5  # 4  # Adjust as needed

# Add Semaphore Definition
MAX_CONCURRENT_TASKS = 1  # 25  # Adjust based on your system
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# I think we need to sort if we want Caching with Qwen to work properly.
def recursively_sort(properties):
    # If properties is a dictionary, sort its keys and process its values recursively.
    if isinstance(properties, dict):
        # Create a new dictionary with sorted keys,
        # and recursively call recursively_sort_properties on each value.
        return {key: recursively_sort(properties[key])
                for key in sorted(properties)}
    # If it's a list, then process each element.
    elif isinstance(properties, list):
        return [recursively_sort(item) for item in properties]
    # For other types, just return the value as is.
    else:
        return properties

# Define EvaluationType and FrameContext
class EvaluationType(str, Enum):
    OBJECT = "ObjectRecognition"
    ATTRIBUTE = "AttributeRecognition"
    SPATIAL = "SpatialUnderstanding"
    COUNT = "Counting"
    ACTION = "ActionRecognition"
    OCR = "OCRPassthrough"
    # Below two for use with Qwen2.5-VL.
    SPATIAL_BBOX = "SpatialUnderstandingWithBBoxes"
    COUNT_BBOX = "CountingWithBBoxes"

@dataclass
class FrameContext:
    frame_number: int
    total_frames: int

# Define ConfidenceLevel
class ConfidenceLevel(str, Enum):
    HIGH = "High Confidence"
    MEDIUM = "Medium Confidence"
    LOW = "Low Confidence"
    UNABLE = "Unable to Determine"

# Accepted confidence levels
ACCEPTED_CONFIDENCE_LEVELS = [ConfidenceLevel.HIGH.value, ConfidenceLevel.MEDIUM.value]

def get_grid_evaluation_instructions(eval_type: EvaluationType, grid_size: int, step: str = "classification") -> str:
    """Get detailed instructions for each evaluation type with examples.
       When a grid overlay is present (grid_size > 0), include grid-based reasoning examples.
    """
    
    # Define a grid note if a grid is present.
    grid_message = ""
    if grid_size > 0:
        grid_message = (
            f"\n\nNOTE: The image is overlaid with a grid of size {grid_size}x{grid_size}. "
            "Rows are labeled alphabetically (A, B, C, ...) and columns numerically (1, 2, 3, ...). "
            "In your reasoning, refer to specific grid cell labels (e.g., 'B2', 'C3') when describing locations or counts."
        )
    
    if step == "description":
        if eval_type == EvaluationType.OBJECT:
            instructions = f"""
Image Description Guidelines for Object Recognition:{grid_message}
   - Provide an exhaustive description of all visible objects.
   - Include details on color, size, material, texture, shape, and the object's position relative to the grid cells.
   - Do not claim the presence of objects not obviously visible.
"""
        elif eval_type == EvaluationType.ATTRIBUTE:
            instructions = f"""
Image Description Guidelines for Attribute Recognition:{grid_message}

Comprehensive Description:
   - Describe each visible object along with its attributes (e.g., color, material, texture).
   - Use grid cell references to specify where the object is located.
   - Do not invent attributes for objects lacking sufficient evidence.
"""
        elif eval_type == EvaluationType.SPATIAL:
            instructions = f"""
Image Description Guidelines for Spatial Understanding:{grid_message}

Comprehensive Description:
   - Describe the positions and relationships among objects using both directional and grid cell references.
   - Clearly state which grid cells contain parts of the objects.
"""
        elif eval_type == EvaluationType.SPATIAL_BBOX:
            instructions = f"""
Image Description Guidelines for Spatial Understanding with Bounding Boxes:{grid_message}

Bounding Box Description:
    - Provide bounding box coordinates for all relevant objects along with grid cell references that indicate where in the image the object lies.
    - List each bounding box in the format [y1, x1, y2, x2] and include an associated label.
    - If objects have a relationship according to the prompt but are not in the image, output an empty list.
"""
        elif eval_type == EvaluationType.COUNT:
            instructions = f"""
Image Description Guidelines for Counting:{grid_message}

Comprehensive Description:
   - Detect the key points of each desired object we are trying to count.
   - Use grid cell references to indicate where objects appear and whether parts of an object span multiple cells.
   - Be careful not to double-count objects that extend over adjacent cells.
"""
        elif eval_type == EvaluationType.COUNT_BBOX:
            instructions = f"""
Image Description Guidelines for Counting with Bounding Boxes:{grid_message}

Bounding Box Description:
    - Count the number of object instances using their bounding boxes.
    - Additionally, reference the grid cells to indicate where each bounding box is located.
    - Ensure that overlapping bounding boxes corresponding to a single object are not counted more than once.
"""
        elif eval_type == EvaluationType.ACTION:
            instructions = f"""
Image Description Guidelines for Action Recognition:{grid_message}

Comprehensive Description:
   - Describe the sequence of actions and motion paths observed across frames, referencing grid cells for clarity.
   - Clearly indicate the start, midpoint, and end of the action along with grid locations.
"""
        else:
            raise ValueError(f"Invalid evaluation type: {eval_type}")
        if eval_type in [EvaluationType.SPATIAL_BBOX, EvaluationType.COUNT_BBOX]:
            general_notes = "The most important aspect is to only describe what is visible and use grid references along with bounding boxes for positional clarity."
        else:
            general_notes = f"""The most important aspect is to NEVER assume the presence of objects that are not visible. Only describe what you see in the image, using the grid cells for additional clarity.
   - Provide detailed descriptions including colors, materials, sizes, textures, patterns, states, and positions.
   - If an object spans multiple grid cells, indicate all relevant grid labels.
   - Always output a properly formatted JSON when you are done.
"""
        return general_notes, instructions
    else:
        # For classification instructions
        if eval_type in [EvaluationType.SPATIAL_BBOX, EvaluationType.COUNT_BBOX]:
            base_instructions = f"""1. Reasoning Structure:
   Before classification, document:
   a) Each object or property stated in the prompt.
   b) How the evidence from bounding boxes, along with grid cell references, aligns with the prompt.
   c) If any objects do not have bounding boxes or are not visible, note that explicitly.

2. Classification Principles:
   - Consider the evaluation target in the context of the prompt, with grid references to support spatial or count decisions.
   - Provide:
       • Your selection from the provided options (e.g., 'above', 'left')
       • A Confidence Level (HIGH, MEDIUM, LOW, UNABLE) based on clear evidence.
   - Document any uncertainties in your reasoning.
"""
        else:
            base_instructions = f"""1. Reasoning Structure:
   Before classification, document:
   a) Each object or property given in the prompt.
   b) The alignment of the image description (with grid cell details) with the prompt.
   c) How the evidence relates to classification rules.

2. Classification Principles:
   - Use the context of the prompt in conjunction with explicit grid-based observation.
   - For each evaluation target, provide your selection (e.g., 'above', 'left') and a corresponding Confidence Level.
   - Document uncertainty in reasoning.
"""

        type_specific = {
            EvaluationType.OBJECT: f"""
Object Detection Rules:

1. Visibility Requirements:
   - Each object must be clearly identifiable, with its key features located in specific grid cells.
   - Partial visibility is acceptable if the object is unambiguously recognized.

2. Reasoning Structure:
   - For every object in the prompt, reference the relevant grid cell(s) when matching description details.
   - Example Reasoning:
"Prompt: Identify a red apple.
 Description: A red, round object is fully visible in grid cell B2, with a partial view in B3.
 Reasoning: The complete presence of the object in B2 confirms its identity as a red apple."
""",
            EvaluationType.ATTRIBUTE: f"""
Attribute Evaluation Rules:

1. Attribute Visibility:
   - Verify that each attribute (color, texture, material) is evident in the referenced grid cell.
   - Account for lighting conditions and partial visibility.

2. Reasoning Structure:
   - For each object in the prompt, guide your reasoning by referring to grid cell locations.
   - Example Reasoning:
"Prompt: A jumbo blue plastic bottle.
 Description: In grid cell C3, a large blue object with a smooth surface is visible, suggesting plastic.
 Reasoning: The grid reference (C3) confirms that the object possesses the reported attributes."
""",
            EvaluationType.SPATIAL: f"""
Spatial Relation Rules:

1. Positional Analysis:
   - Use grid cell labels to indicate where each object is situated.
   - Compare rows (alphabetical order) and columns (numerical order) to define spatial relationships.

2. Reasoning Structure:
   - Walk through the prompt by mapping objects to grid cells.
   - Example Reasoning:
"Prompt: Determine the spatial relation between a person and a dog.
 Description: A person is clearly in grid cell C1, while a dog spans grid cells D1 and D2.
 Reasoning:
   • Row Comparison: Row C is above row D, so the person is higher.
   • Column Comparison: The person in C1 is in column 1; the dog, partly in D2 (column 2), shows that at least a portion of the dog is to the right.
   Final Assessment: The person is above and to the left of the dog."
""",
            EvaluationType.SPATIAL_BBOX: f"""
Spatial Relation Rules with Bounding Boxes:

1. Horizontal & Vertical Analysis:
   - Use both bounding box coordinates and grid cell references to determine the spatial configuration.
   - Compare the grid cells where each object’s bounding box is primarily located.

2. Reasoning Structure:
   - Example Reasoning:
"Prompt: Establish the spatial relation between a person and a dog.
 Description: The person’s bounding box [100, 50, 200, 150] lies in grid cell C1. The dog’s bounding boxes, [210, 160, 320, 270] and [215, 275, 330, 380], fall into grid cells D1 and D2.
 Reasoning:
   • Grid Analysis: Since C1 (person) is above D1/D2 (dog), the person is higher on the image.
   • Horizontal Analysis: With the person in column 1 and the dog partly in column 2, the person is also to the left.
   Final Assessment: The evidence confirms that the person is both above and left of the dog."
""",
            EvaluationType.COUNT: f"""
Counting Rules:

1. Organized Counting:
   - Use the grid cells to clearly document where complete or partial instances occur.
   - Ensure that if an object spans multiple grid cells, it is counted only once.
   
2. Reasoning Structure:
   - Example Reasoning:
"Prompt: Count the number of people.
 Description:
    • In grid cell C1, two full people are clearly visible.
    • In grid cell C2, a partial view is observed but matches one of the already-counted individuals in C1.
    • In grid cell D3, three additional distinct people are seen.
 Reasoning:
    - From C1: +2 people.
    - From C2: 0 additional (partial overlap with C1).
    - From D3: +3 people.
    Total Count: 2 + 3 = 5 people.
 Final Assessment: Grid-based counting confirms a total of 5 people."
""",
            EvaluationType.COUNT_BBOX: f"""
Counting with Bounding Boxes:

1. Organized Counting:
   - Count each object instance using the provided bounding boxes and cross-check the grid cells for location details.
   - Avoid double-counting objects that overlap adjacent grid cells.

2. Reasoning Structure:
   - Example Reasoning:
"Prompt: Count how many apples are present.
 Description:
    • One bounding box for an apple is fully contained in grid cell B2.
    • A second bounding box is found entirely in grid cell B3.
    • A third bounding box spans grid cells C1 and C2.
 Reasoning:
    - Identify each distinct bounding box:
         Box 1 (B2): 1 apple.
         Box 2 (B3): 1 apple.
         Box 3 (C1/C2): 1 apple.
    Total Count: 1 + 1 + 1 = 3 apples.
 Final Assessment: The grid-guided analysis of bounding boxes indicates there are 3 apples."
""",
            EvaluationType.ACTION: f"""
Action Recognition Rules:

1. Action Verification:
   - Use grid references to document where the action occurs throughout the frames.
   - Specify the progression from the starting grid cell through transitional cells to the final ending point.

2. Reasoning Structure:
   - Example Reasoning:
"Prompt: Determine if a person is throwing a ball.
 Description:
    • Frame 1: The person is observed in grid cell B2.
    • Frame 2: The person’s arm starts the throwing motion; the activity remains in B2 and starts moving into B3.
    • Frame 3: The ball detaches and becomes visible in grid cell B4.
 Reasoning:
    - The starting position in B2, the transitional motion from B2 to B3, and the final separation observed in B4 collectively confirm the throwing action.
 Final Assessment: The grid-based step-by-step motion analysis confirms that the person is throwing the ball."
"""
        }
    
        return base_instructions, type_specific[eval_type]

# Function to get evaluation instructions
def get_nongrid_evaluation_instructions(eval_type: EvaluationType, grid_size: int, step: str = "classification") -> str:
    """Get detailed instructions for each evaluation type with examples."""

    # Define a grid note if a grid is present.
    grid_message = ""
    if grid_size > 0:
        grid_message = (
            f"NOTE: The image is overlaid with a grid of size {grid_size}x{grid_size} where each cell is labeled "
            "alphabetically for rows (A, B, C, ...) and numerically for columns (1, 2, 3, ...). "
            "When describing any objects in the frame, please reference these grid cell labels (e.g., 'A1', 'B2') "
            "to increase clarity.\n\n"
        )

    if step == "description":
        if eval_type == EvaluationType.OBJECT:
            instructions = f"""
Image Description Guidelines for Object Recognition:

1. Comprehensive Description:
   - Describe all visible objects in the image in exhaustive detail.
   - Mention their attributes such as color, size, material, texture, shape, and position relative to one another.
   - Include contextual information that might relate to the prompt, even if it seems irrelevant.
   - Do not assume the presence of objects not visible.
   - Include counts of each object type.

2. Examples:

"There is a brown wooden table with four sturdy legs and a smooth surface. A red apple with a slight shine is placed on the table. Additionally, two white plastic chairs are situated on either side of the table. In the background, a green potted plant is placed, and a window with white curtains is visible."
"""
        elif eval_type == EvaluationType.ATTRIBUTE:
            instructions = f"""
Image Description Guidelines for Attribute Recognition:

1. Comprehensive Description:
   - Describe all visible objects and their attributes in meticulous detail.
   - Mention attributes such as color, material, textures, patterns, sizes, and states (e.g., open/closed) for objects, clothing, appearance, gender, facial expressions, and more for people.
   - Include contextual and ancillary details that might aid in attribute identification, even if they seem irrelevant.
   - Do not assume the presence of objects not visible.
   - Include counts of each object type.

2. Examples:

"There is a brown wooden table with a visibly smooth and glossy surface, indicating it is polished. On the table, there is a red ceramic vase with intricate floral patterns. The vase reflects the ambient lighting, highlighting its glossy finish. Additionally, a metal lamp with a beige shade is placed next to the vase."
"""
        elif eval_type == EvaluationType.SPATIAL:
            instructions = f"""
Image Description Guidelines for Spatial Understanding:

1. Comprehensive Description:
   - Describe all visible objects, their positions, and spatial relations in detailed terms.
   - Describe spatial relations among objects using relevant spatial terms.
   - Include contextual details from the prompt that might influence spatial relationships, even if they seem irrelevant.
   - Do not assume the presence of objects not visible.
   - Include counts of each object type.

2. Examples:

"There is a brown dog sitting on the green grass. A tall oak tree stands nearby with its branches overhanging. To the right of the dog, there is a blue ball partially buried in the grass. Additionally, a wooden fence spans the image's southern edge."
"""
        elif eval_type == EvaluationType.SPATIAL_BBOX:
            # for spatial with bbox, for description we will first describe the bounding boxes for the desired objects.
            instructions = """
Image Description Guidelines for Spatial Understanding with Bounding Boxes:
1. Bounding Box Description:
    - Describe the bounding boxes for the objects in the image who are involved in the spatial relations described in the prompt.
    - For each object, provide the bounding box coordinates in the format: [y1, x1, y2, x2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    - Clearly indicate which bounding box belongs to which object by responding in JSON format.

2. Examples:
    {
        "bounding_boxes": [
            {"bbox_2d": [167, 0, 288, 134], "label": "flip flops"},
        	{"bbox_2d": [349, 0, 457, 123], "label": "beach umbrella"},
        	{"bbox_2d": [521, 4, 635, 123], "label": "starfish"},
        	{"bbox_2d": [10, 180, 118, 294], "label": "cocktail drink"},
        ]
    }
"""
        elif eval_type == EvaluationType.COUNT:
            instructions = f"""
Image Description Guidelines for Counting:

1. Comprehensive Description:
   - Describe all visible objects and count each type with precision.
   - Mention attributes such as color and size to differentiate between similar objects.
   - Include contextual information that might aid in distinguishing objects, even if it seems irrelevant.
   - Do not assume the presence of objects not visible.
   - Include counts explicitly. Mention which objects span across different areas if necessary.

2. Examples:

"There are three apples on the brown wooden table: two red apples and one green apple. Additionally, two yellow bananas hang from the table's edge. A part of one banana continues across into another area. There is one blue vase that is empty. In total, there are three apples and two bananas visible, since the banana appears to continue into multiple areas without a gap."
"""
        elif eval_type == EvaluationType.COUNT_BBOX:
            instructions = """
Image Description Guidelines for Counting with Bounding Boxes:
1. Bounding Box Description:
    - Describe the bounding boxes for the objects in the image that we need to count based on the prompt.
    - For each object, provide the bounding box coordinates in the format: [y1, x1, y2, x2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    - Clearly indicate which bounding box belongs to which object by responding in JSON format. If there are multiple objects of the same type that we are counting, give the same name to each object.

2. Examples:
    {
        "bounding_boxes": [
            {"bbox_2d": [167, 0, 288, 134], "label": "apple"},
        	{"bbox_2d": [349, 0, 457, 123], "label": "apple"},
        	{"bbox_2d": [521, 4, 635, 123], "label": "apple"},
        	{"bbox_2d": [10, 180, 118, 294], "label": "banana"},
        ]
    }
"""
        elif eval_type == EvaluationType.ACTION:
            instructions = f"""
Image Description Guidelines for Action Recognition:

1. Comprehensive Description:
   - Describe all visible objects and any actions or motions across frames in intricate detail.
   - Include the sequence of actions, movements, and interactions between objects or subjects.
   - Mention contextual details from the prompt that might relate to the actions, even if they seem irrelevant.
   - Do not assume the presence of actions or objects not visible.

2. Examples:

"In frame 1, a person is standing near a brown wooden table. In frame 2, the person begins to raise their left hand, while their right hand remains on the table. In frame 3, the person's left hand is fully extended above their head, and in frame 4, the person swiftly moves their left hand forward, suggesting the motion of throwing an object. Throughout the frames, the person's posture remains consistent, with no visible artifacts disrupting the motion sequence."
"""
        else:
            instructions = f"""
Image Description Guidelines:

1. Comprehensive Description:
   - Describe all visible objects in the image, including their attributes, positions, and relationships, with maximum detail.
   - Incorporate contextual information from the prompt that might aid in understanding the scene, even if it seems irrelevant.
   - Do not assume the presence of objects not visible.
   - Include counts of each object type.

2. Examples:

"There is a large window with white curtains partially drawn, allowing natural light to illuminate the room. A brown wooden chair with a cushioned seat is placed near the window. The chair extends slightly beyond its immediate area. On the floor, a patterned rug lies beneath a glass coffee table. Additionally, a bookshelf filled with various books is present, and a lamp casts a warm glow across the room."
"""
        if eval_type in [EvaluationType.SPATIAL_BBOX, EvaluationType.COUNT_BBOX]:
            general_notes = """The most important aspect of the description is to NEVER assume the presence of objects in the image that are not visible. Only describe what you can see in the image. These images often cannot fulfill the prompt. Do not be generous; there must be clear evidence of an object's presence to mention it. Here are some additional guidelines to follow:
    - If any part of a person is visible, consider the person present. Even if only a hand or leg is visible, for the sake of evaluation, consider the full person present.
    """
        else:
            general_notes = """The most important aspect of the description is to NEVER assume the presence of objects in the image that are not visible. Only describe what you can see in the image. Ignore the prompt for now. These images often cannot fulfill the prompt. Do not be generous; there must be clear evidence of an object's presence to mention it. Here are some additional guidelines to follow:
    - If any part of a person is visible, consider the person present. Even if only a hand or leg is visible, for the sake of evaluation, consider the full person present.
    - Describe objects in very detailed ways, including colors, materials, sizes, textures, patterns, states, and positions relative to one another. This is crucial for accurate evaluation when we ask for specific objects or attributes and have to match them to the prompt in a later step.
        - For example, instead of saying "a chair," describe it as "a brown wooden chair with a cushioned seat."
    - Describe objects in many ways. If there's any ambiguity at all, provide multiple descriptions to cover all possibilities.
        - For example, if you see a red apple, describe it as "a red apple" and "a fruit that is red that is round, possibly an apple or tomato." Or if you see a table, describe it as "a table" and "a piece of furniture with a flat surface and legs, possibly a desk or table."
    - Always output a properly formatted JSON when you are done.
    """
        return general_notes, grid_message + instructions
    else:
        # Classification instructions remain unchanged
   
        if eval_type in [EvaluationType.SPATIAL_BBOX, EvaluationType.COUNT_BBOX]:
            base_instructions = f"""1. Reasoning Structure:
    Before classification, document:
    a) Each object or property stated in the prompt that we are trying to evaluate.
    b) How the evidence from the bounding boxes align with the prompt regarding these elements.
    c) If any objects do not have bounding boxes or are not visible, document that as well.
   
2. Classification Principles:
   - Always consider the evaluation target in the context of the prompt rather than as standalone properties.
   - For each evaluation target, provide:
     - Your selection from the provided options (e.g., 'above' or 'below')
     - Confidence Level: One of the following
       HIGH: Clearly visible with strong visual evidence
         - Do not say HIGH unless you are very confident and can provide clear evidence from the bounding boxes and those bounding boxes are reliable and not hallucinated/belonging to the same exact object.
       MEDIUM: Partially visible/ambiguous but supportable
       LOW: Difficult to determine with confidence
       UNABLE: Cannot make determination from visible evidence
   - Document uncertainty in reasoning
"""
        else:
            base_instructions = """1. Reasoning Structure:
   Before classification, document:
   a) Each object or property stated in the prompt that we are trying to evaluate.
   b) The alignment of the frame description with the prompt regarding these elements. Note that sometimes the description may not have the exact same names as we see in the prompt. However, the description should describe the frame well enough such that we can match what we saw to the prompt, if possible.
     - NOTE: You must ALWAYS base your reasoning on the description provided, not on the prompt. Do NOT hallucinate objects that were not described.
   c) How the evidence relates to classification rules
   
2. Classification Principles:
   - Always consider the evaluation target in the context of the prompt rather than as standalone properties.
   - For each evaluation target, provide:
     - Your selection from the provided options (e.g., 'above' or 'below')
     - Confidence Level: One of the following
       HIGH: Clearly visible with strong visual evidence
         - Do not say HIGH unless you are very confident and can provide clear evidence in the frame description for the selection.
       MEDIUM: Partially visible/ambiguous but supportable
       LOW: Difficult to determine with confidence
       UNABLE: Cannot make determination from visible evidence
   - Document uncertainty in reasoning
"""

        type_specific = {
            EvaluationType.OBJECT: f"""
Object Detection Rules:

1. Visibility Requirements:
   - Must be clearly identifiable with defining features visible
   - Partial visibility acceptable if identity is unambiguous
   - Both location and identity must be clear

2. Reasoning Structure:
   - Walk through each object in the prompt.
   - Try to find a match in the frame description. Be flexible with object names.

Example Reasoning:
"**Prompt:** Red and green apples are spread on a table as part of a fruit arrangement next to a beautiful vase.

**Associated Description:** There is a brown wooden table with four sturdy legs and a smooth surface. A red fruit, likely an apple or tomato, with a slight shine is placed on the table. Additionally, two white plastic chairs are situated on either side of the table. In the background, a green potted plant is placed, and a window with white curtains is visible.

**Reasoning:** We will first walk through each object in the prompt and compare it to the description:
- **Table:** The description explicitly mentions a 'brown wooden table', matching the prompt's requirement for a table.
- **Vase:** There is no mention of a vase or anything resembling it in the description, indicating that the vase is not present in the frame. Therefore, the vase is not visible.
- **Apple** The description mentions a 'red fruit', which is likely an apple or tomato. Since the prompt specifies red apples, we can confidently say that the red fruit is an apple.
""",

            EvaluationType.ATTRIBUTE: f"""
Attribute Evaluation Rules:

1. Attribute Visibility:
   - The visibility of each attribute must be clearly discernible and identified.
   - Material properties (e.g., wooden): Must see clear texture/material indicators
   - Colors: Must account for lighting conditions
   - Size relationships: Need reference objects or context
   - State (e.g., open/closed): Must see relevant parts

2. Reasoning Structure:
   - Walk through each object in the prompt.
     - For each attribute belonging to the object, verify its presence in the description.

Example Reasoning:
"**Prompt:** A video of a brown wooden table with a red vase and a plastic lamp with a brown shade.

**Associated Description:** There is a brown wooden table with a visibly smooth and glossy surface, indicating it is polished. On the table, there is a red ceramic vase with intricate floral patterns. The vase reflects the ambient lighting, highlighting its glossy finish. Additionally, a metal lamp with a beige shade is placed next to the vase.

**Reasoning:** We will walk through each object, then each attribute of said object, and compare them to the prompt:
- **Table:** The description mentions a 'brown wooden table', which matches the prompt's requirement for a table.
  Attributes:
    - **Material (Wooden):** The description confirms that the table is made of wood, as it is a 'brown wooden table.'
    - **Color (Brown):** The table is described as brown, aligning with the prompt's requirement.
- **Vase:** The description mentions a 'vase', matching the prompt's requirement for a vase.
  Attributes:
    - **Color (Red):** The description specifies a 'red ceramic vase,' which aligns with the prompt's requirement.
    - **Material (Ceramic):** The vase is made of ceramic, as stated in the description.
- **Lamp:** The description mentions a 'metal lamp', matching the prompt's requirement for a lamp.
  Attributes:
    - **Color (Brown):** The lamp has a beige shade, as mentioned in the description. Since beige is a shade of brown, this attribute aligns with the prompt's requirement.
    - **Material (Plastic):** The lamp is made of metal, as stated in the description. However, the prompt specifies a plastic lamp, which is not visible in the frame.
""",

            EvaluationType.SPATIAL: f"""
Spatial Relation Rules:

1. Positional Analysis:
   - Describe spatial relations among objects thoroughly, using clear spatial terms.
   - Never assume an object in the prompt is in the frame if not visible.
   - Ensure positions are clear using relative terms such as "next to", "above", or "below".

2. Depth Relations:
   - Depth markers are used to represent the distance each object is from the viewer:
     - Depth 0%: Closest.
     - Depth 100%: Farthest.
   - Rules for depth comparisons:
     - Closer/Farther from Viewpoint:
       - "Closer to viewpoint": ANY part of A must be nearer than ALL parts of B.
       - "Farther from viewpoint": ALL parts of A must be farther than ALL parts of B.
     - Use visual aids (overlap, shadows, perspective lines) to confirm.

3. Multiple Instance Handling:
   - Positional rules apply to ALL instances unless specified otherwise in the prompt.
     - e.g., for "left of," ALL instances of A must be left of ALL instances of B; for "above," ALL instances of A must be above ALL instances of B.
   - Document any exceptions explicitly.
     - e.g., if the prompt is "One of the dogs is left of a cat," we only require one dog to be left of the cat, not all dogs.

4. Confidence Levels:
   - For each spatial relation, select one of the provided options or "Unable to Determine."
   - Provide a confidence level for your selection.

5. Reasoning Structure:
    - Walk through each relation in the evaluation target. See if the description matches the prompt by clearly stating the positions of each object, if present.
    - Only analyze the relations indicated in the evaluation target.
      - Analyze horizontal relations (left of/right of) by clearly situating each object relative to each other.
      - Analyze vertical relations (above/below) by clearly situating each object relative to each other.
      - Confirm depth relations using markers and visual cues.

Example Reasoning:
"**Prompt:** A dog is playing left of a bone in the forest, and the dog is below the bone. The dog is closer to the camera than the bone, having fun playing.

**Associated Description:** A brown dog is sitting on the green grass. A bone is located nearby. There is also a tall oak tree in the vicinity.

**Reasoning:** 
The relations in the evaluation target are as follows: dog left of/right of bone, dog above/below bone, and dog closer to the camera than/farther from the camera than bone.
1. **Horizontal Relations:**
  Dog and Bone:
   - The dog is positioned more to the left than the bone.
   - Since the dog's position is to the left, the dog is 'left of' the bone with HIGH confidence.

2. **Vertical Relations:**
  Dog and Bone:
   - Despite varying positions, the dog is observed lower relative to the bone.
   - Thus, the dog is 'below' the bone with MEDIUM confidence.

3. **Depth Relations:**
  Dog and Bone:
   - The dog is clearly positioned closer to the viewer than the bone.
   - Therefore, the dog is 'closer to the camera' than the bone with HIGH confidence.
""",
            EvaluationType.SPATIAL_BBOX: f"""
Use the bounding boxes to determine the spatial relations based on the Spatial Relation Rules.

Spatial Relation Rules:
1. Horizontal and Vertical Relations:
    - For horizontal and vertical relations, use the bounding boxes to determine the positions of the objects relative to each other.
    - A bounding box that is more to the left than another bounding box is considered "left of" that bounding box.
        - Same for "right of" as well.
    - A bounding box that is higher than another bounding box is considered "above" that bounding box.
        - Same for "below" as well.

2. Depth Relations:
   - Depth markers are used to represent the distance each object is from the viewer:
     - Depth 0%: Closest.
     - Depth 100%: Farthest.
   - Rules for depth comparisons:
     - Closer/Farther from Viewpoint:
       - "Closer to viewpoint": ANY part of A must be nearer than ALL parts of B.
       - "Farther from viewpoint": ALL parts of A must be farther than ALL parts of B.
     - Use visual aids (overlap, shadows, perspective lines) to confirm.
     - Use the bounding boxes to isolate the objects. However, the bounding boxes alone will not provide depth information. You must use visual cues to determine depth relations.
       - In general, left/right/above/below relations are independent of depth, so should NOT be used to determine depth relations.

3. Multiple Instance Handling:
   - Positional rules apply to ALL instances unless specified otherwise in the prompt.
     - e.g., for "left of," ALL instances of A must be left of ALL instances of B; for "above," ALL instances of A must be above ALL instances of B.
   - Document any exceptions explicitly.
     - e.g., if the prompt is "One of the dogs is left of a cat," we only require one dog to be left of the cat, not all dogs.

4. Confidence Levels:
   - For each spatial relation, select one of the provided options or "Unable to Determine."
   - Provide a confidence level for your selection. If you are not sure the bounding boxes are accurate or the listed objects are in the image at all, say "Low Confidence."

5. Reasoning Structure:
    - Walk through each relation in the evaluation target. See if the description matches the prompt by clearly stating the positions of each object, if present.
    - Only analyze the relations indicated in the evaluation target.
      - Analyze horizontal relations (left of/right of) by clearly situating each object relative to each other.
      - Analyze vertical relations (above/below) by clearly situating each object relative to each other.
      - Confirm depth relations using markers and visual cues.
""",

            EvaluationType.COUNT: f"""
Counting Rules:

1. Organized Counting:
   - Use a methodical approach to count the number of instances in the frame.
   - Clearly identify each object through distinctive features.

2. Instance Criteria:
   - Must be fully identifiable as distinct instances. If not, answer with 0.
   - Partial visibility acceptable if clearly separate
   - Group counting requires clear individual distinction

3. Confidence Levels:
   - State confidence level for final count.

4. Reasoning Structure:
   - Walk through each object in the prompt. Then, count each object based on its presence in the description.
   - Cross-reference object attributes with the prompt to ensure correct identification.
   - Provide a clear and systematic counting process.
   - Always respond with an integer count.

6. Example Reasoning:
"**Prompt:** A video of a table with a shining tablecloth and 3 apples sitting on it. The window is open next to the scene, providing a much needed breeze.

**Associated Description:** There are three fruits on the brown wooden table: two red and one green. They are either apples or tomatoes. Additionally, there are two yellow bananas hanging from the table's edge. There is one blue vase that is empty.

**Reasoning:** 
- **Identification:**
  Apples:
  - The fruits are either apples or tomatoes. Since the prompt specifies counting apples, we assume they are apples.
  - The apples are clearly visible and distinguishable by color.

- **Counting Process:**
  - Two red apples are present.
  - One green apple is present.
  - Total apples on the table: 3

- **Verification:**
  - No other objects in the description match the prompt's criteria for apples.
  - The bananas and vase are irrelevant to the apple count.
""",
            EvaluationType.COUNT_BBOX: f"""
Counting Rules with Bounding Boxes:

1. Organized Counting:
   - Use a methodical approach to count the number of instances in the frame.
   - Clearly identify each object based on their bounding boxes.

2. Instance Criteria:
   - Must be fully identifiable as distinct instances. If not, answer with 0.
   - Partial visibility acceptable if clearly separate
   - Group counting requires clear individual distinction

3. Confidence Levels:
   - State confidence level for final count. If you are not sure the bounding boxes are accurate or the listed objects are in the image at all, say "Low Confidence."

4. Reasoning Structure:
   - Walk through each object in the prompt. Then, count each object based on its bounding boxes.
   - Cross-reference object attributes with the prompt to ensure correct identification.
   - Provide a clear and systematic counting process.
   - Always respond with an integer count.
""",

            EvaluationType.ACTION: f"""
Action Recognition Rules:

1. Action Verification:
   - Must see clear evidence of action in progress.
   - Consider required body positions/movements.
   - Look for interaction between subjects/objects.

2. Confidence Levels:
   - For each action, provide existence (True/False) and a confidence level.

3. Reasoning Structure:
   - Walk through each action in the prompt. Then, analyze the sequence of frames to determine the consistency and progression of the supposed actions.
   - Provide detailed reasoning that links observed motions to the actions specified in the prompt.
   - IMPORTANT: You need to actually see the action happening, not just predict it is reasonable to happen.

5. Example Reasoning:
"**Prompt:** A person is dressed in a blue shirt and jeans throwing a ball in the park.

**Associated Description:** In frame 1, a person is standing near a brown wooden table. In frame 2, the person begins to raise their left hand, while their right hand remains on the table. In frame 3, the person's left hand is fully extended above their head, and in frame 4, the person swiftly moves their left hand forward, suggesting the motion of throwing an object. Throughout the frames, the person's posture remains consistent, with no visible artifacts disrupting the motion sequence.

**Reasoning:** 
- **Action Identification:**
  Throwing a Ball:
  - Frame 1: Person is stationary.
  - Frame 2: Initiation of movement by raising the left hand.
  - Frame 3: Full extension of the left hand.
  - Frame 4: Swift forward motion of the left hand, indicating a throwing action.
  - Frame 5: Ball is separated from the hand, suggesting a throw.

- **Consistency with Prompt:**
  - The sequential movement from raising the hand to throwing aligns with the action of throwing a ball as specified in the prompt.
  - No discrepancies or irrelevant actions are observed. We see the ball in motion, indicating a successful throw.

- **Final Assessment:**
  - The progression of arm movement across frames strongly indicates the action of throwing.
"""
        }

        # return base_instructions, type_specific[eval_type]
        return base_instructions, grid_message + type_specific[eval_type]

def get_direct_evaluation_instructions(eval_type, grid_size):
    """
    Return simplified evaluation instructions for classification tasks.
    
    Parameters:
      eval_type: An instance of EvaluationType (e.g., OBJECT, ATTRIBUTE, SPATIAL,
                 COUNT, SPATIAL_BBOX, ACTION, COUNT_BBOX).
    
    Behavior:
      - For most classification tasks (e.g., OBJECT, ATTRIBUTE, SPATIAL, ACTION),
        instruct the evaluator to provide only the final answer (label, count, or short direct description) with no reasoning.
      - For COUNT_BBOX and SPATIAL_BBOX tasks, provide the bounding box details first, followed by the final answer.
    """
    # Define a grid note if a grid is present.
    grid_message = ""
    if grid_size > 0:
        grid_message = (
            f"\n\nNOTE: The image is overlaid with a grid of size {grid_size}x{grid_size} where each cell is labeled "
            "alphabetically for rows (A, B, C, ...) and numerically for columns (1, 2, 3, ...). "
            "When describing any objects in the frame, please reference these grid cell labels (e.g., 'A1', 'B2') "
            "to increase clarity."
        )
    
    # Special instructions for Counting with Bounding Boxes (COUNT_BBOX) and Spatial Bounding Box (SPATIAL_BBOX) tasks.
    if eval_type in (EvaluationType.COUNT_BBOX, EvaluationType.SPATIAL_BBOX):
        instructions = (
            "Detect ALL target objects in the image and return their locations as bounding box coordinates first. This must be an exhaustive list of everything mentioned in the image and prompt.\n\n"
            "The output should be in JSON format as follows:\n"
            '{\n'
            '  "bounding_boxes": [\n'
            '    {"bbox_2d": [x1, y1, x2, y2], "label": "name-of-object", "sub_label": "any further descriptions necessary"},\n'
            '    // Additional objects as needed\n'
            '  ]\n'
            '}\n\n'
            "If no target objects are present, return:\n"
            '{"bounding_boxes": []}\n\n'
            "Then, based on the bounding box data, provide your final answer. "
            "If you are unable to determine the answer from the bounding boxes, please state so explicitly.\n"
            "Write your final answer in details then structured following the required JSON format."
        )
    elif eval_type == EvaluationType.COUNT:
        instructions = (
            "For all objects in the evaluation target, see if they exist in the image. If so, count them.\n"
            "In the details, provide a brief description of the target objects as well as how sure you are about your count. "
            "The objects must be unambiguous and clearly visible for you to have High Confidence in your count. "
            "If the objects are not separable (e.g., all are one big blob), that means you should have at most Low Confidence in your count. "
            "Return the JSON structure with the final count as required by the prompt."
        )
    elif eval_type == EvaluationType.ACTION:
        instructions = (
            "Detect the action being performed by analyzing the sequence of frames.\n"
            "Return only the JSON structure with detail and action label as required by the prompt."
        )
    else:
        # For all other classification tasks, simply ask for the final answer directly.
        instructions = (
            "Please provide the JSON structure with the final answer as required by the prompt."
        )
    if eval_type in [EvaluationType.SPATIAL, EvaluationType.SPATIAL_BBOX]:
        instructions += """
Spatial Relation Rules:
1. Horizontal and Vertical Relations:
    - For horizontal and vertical relations, use the bounding boxes to determine the positions of the objects relative to each other.
    - A bounding box that is more to the left than another bounding box is considered "left of" that bounding box.
        - Same for "right of" as well.
    - A bounding box that is higher than another bounding box is considered "above" that bounding box.
        - Same for "below" as well.

2. Depth Relations:
   - Depth markers are used to represent the distance each object is from the viewer:
     - Depth 0%: Closest.
     - Depth 100%: Farthest.
   - Rules for depth comparisons:
     - Closer/Farther from Viewpoint:
       - "Closer to viewpoint": ANY part of A must be nearer than ALL parts of B.
       - "Farther from viewpoint": ALL parts of A must be farther than ALL parts of B.
     - Use the bounding boxes to isolate the objects. However, the bounding boxes alone will not provide depth information. You must use visual cues to determine depth relations.
       - In general, left/right/above/below relations are independent of depth, so should NOT be used to determine depth relations.

3. Multiple Instance Handling:
   - Positional rules apply to ALL instances unless specified otherwise in the prompt.
     - e.g., for "left of," ALL instances of A must be left of ALL instances of B; for "above," ALL instances of A must be above ALL instances of B.
   - Document any exceptions explicitly.

Instructions: There are spatial relations between the objects in the frame, but we removed any indication of the spatial relations in the prompt. Based on the objects' positions, explain what spatial relations you seee in the details key and provide the spatial relations for the objects as required in the JSON."""
    general_notes = """The most important aspect of the description is to NEVER assume the presence of objects in the image that are not visible. Only describe what you can see in the image. These images often cannot fulfill the prompt. Do not be generous; there must be clear evidence of an object's presence to mention it. Here are some additional guidelines to follow:
- If any part of a person is visible, consider the person present. Even if only a hand or leg is visible, for the sake of evaluation, consider the full person present.
- Always output a properly formatted JSON when you are done.
"""
    return general_notes + grid_message, instructions

def get_evaluation_instructions(eval_type: EvaluationType, grid_size: int, step: str = "classification") -> str:
    assert not (grid_size > 0 and step == "direct"), "Direct evaluation is not supported with grid overlay."
    if grid_size > 0:
        return get_grid_evaluation_instructions(eval_type, grid_size, step)
    elif step == "direct":
        return get_direct_evaluation_instructions(eval_type, grid_size)
    return get_nongrid_evaluation_instructions(eval_type, grid_size, step)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to create grid overlay
def create_grid_overlay(frame: np.ndarray, grid_size: int) -> np.ndarray:
    """Create grid overlay for structured spatial/counting analysis with visibility enhancements."""
    height, width = frame.shape[:2]
    grid_img = frame.copy()

    # Draw grid lines with outline
    for i in range(1, grid_size):
        y = int(i * height / grid_size)
        x = int(i * width / grid_size)
        # Horizontal lines
        cv2.line(grid_img, (0, y), (width, y), (0, 0, 0), 3)  # Black outline
        cv2.line(grid_img, (0, y), (width, y), (255, 255, 255), 1)  # White line
        # Vertical lines
        cv2.line(grid_img, (x, 0), (x, height), (0, 0, 0), 3)  # Black outline
        cv2.line(grid_img, (x, 0), (x, height), (255, 255, 255), 1)  # White line

    # Add region labels (A1, B1, ..., E5) with outline
    for row in range(grid_size):
        for col in range(grid_size):
            label = f"{chr(65 + row)}{col + 1}"  # E.g., "A1", "B2"
            x = int(col * width / grid_size) + 5
            y = int((row + 1) * height / grid_size) - 5
            # Draw text with shadow (black) and foreground (white)
            cv2.putText(grid_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Shadow
            cv2.putText(grid_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Foreground

    return grid_img

# Function to add perspective guides
def add_perspective_guides(frame: np.ndarray) -> np.ndarray:
   """Add perspective lines and vanishing points overlay to the frame."""
   height, width = frame.shape[:2]
   guide_img = frame.copy()

   # Draw horizon line with outline
   cv2.line(guide_img, (0, height//2), (width, height//2), (0,0,0), 3)
   cv2.line(guide_img, (0, height//2), (width, height//2), (255,255,255), 1)

   # Draw vanishing point lines with outline
   cv2.line(guide_img, (0, height), (width//2, height//2), (0,0,0), 3)
   cv2.line(guide_img, (0, height), (width//2, height//2), (255,255,255), 1)
   cv2.line(guide_img, (width, height), (width//2, height//2), (0,0,0), 3)
   cv2.line(guide_img, (width, height), (width//2, height//2), (255,255,255), 1)

   return guide_img

def add_depth_markers(frame: np.ndarray, num_markers: int = 5) -> np.ndarray:
   """Add vertical scale markers at regular depths."""
   height, width = frame.shape[:2]
   marker_img = frame.copy()

   marker_positions = np.linspace(height, height//2, num_markers)
   for y in marker_positions:
       y = int(y)
       # Draw horizontal lines with outline
       cv2.line(marker_img, (width//4, y), (3*width//4, y), (0,0,0), 3)
       cv2.line(marker_img, (width//4, y), (3*width//4, y), (255,255,255), 1)
       
       # Add depth labels with outline
       label = f"Depth {int((height-y)/(height//2)*100)}%"
       cv2.putText(marker_img, label, (width//4+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
       cv2.putText(marker_img, label, (width//4+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

   return marker_img

# Function to save frames from the video
def save_frames(video_path: str, eval_type: EvaluationType = None, grid_size: int = 5, include_grid: bool = False, include_depth_markers: bool = False) -> List[str]:
    """Get frames from the video as image paths. Possibly modify first with annotations. The output directory will be the same as in video_path. Each frame will be named {video_path_without_extension}/{frame_number}.jpg."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_index = 0
        
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return frame_paths

    while True:
        # Read next frame from the video
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frame is retrieved (end of video)

        # Optionally overlay grid for tasks
        if include_grid:
            frame = create_grid_overlay(frame, grid_size=grid_size)
        # Add perspective guides and depth markers for spatial tasks
        if include_depth_markers and eval_type in [EvaluationType.SPATIAL, EvaluationType.SPATIAL_BBOX]:
            frame = add_perspective_guides(frame)
            frame = add_depth_markers(frame)

        # Define path to save the frame
        if False:  # video_path.startswith("/ib-scratch/"):
            frame_path = os.path.join("kyle_data", "video_frames", os.path.basename(video_path).split(".")[0], "frames", f"{frame_index}.jpg")
        else:
            if "video.mp4" in video_path:
                frame_path = os.path.join(os.path.dirname(video_path), "frames", f"{frame_index}.jpg")
            else:
                frame_path = os.path.join(video_path.replace(".mp4", ""), "frames", f"{frame_index}.jpg")
        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

        # Increment frame index
        frame_index += 1

    # Release the video capture object
    cap.release()

    return frame_paths

# Function to perform the description step
async def describe_frame(
    frame_paths: List[str],
    evaluation_type: EvaluationType,
    grid_size: int,
    frame_context: Optional[FrameContext] = None,
    model_info: Dict = {},
) -> str:
    """Generate a rich description of the frame(s) tailored to the task."""

    sys_message, instructions = get_evaluation_instructions(evaluation_type, grid_size=grid_size, step="description")

    if evaluation_type == EvaluationType.ACTION:
        # For ActionRecognition, describe all frames together
        description_frames = []
        for fn, frame_path in enumerate(frame_paths):
            with open(frame_path, "rb") as image_file:
                frame_encoded = base64.b64encode(image_file.read()).decode('utf-8')
            description_frames.append({"type": "text", "text": f"Frame {fn+1}/{len(frame_paths)}"})
            description_frames.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_encoded}"}})
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    *description_frames
                ]
            }
        ]
    else:
        # For other tasks, describe each frame individually
        # Assuming single frame
        frame_path = frame_paths[0]
        with open(frame_path, "rb") as image_file:
            frame_encoded = base64.b64encode(image_file.read()).decode('utf-8')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_encoded}"}}
                ]
            }
        ]
    
    # Define response format for description
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": f"{evaluation_type.value.lower()}_description",
            "schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Detailed description of the image(s)."}
                },
                "required": ["description"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    if "model" in model_info:
        qwen_model = model_info["model"]
        qwen_processor = model_info["processor"]
        model_name = model_info["model_name"]
        if evaluation_type in [EvaluationType.SPATIAL_BBOX, EvaluationType.COUNT_BBOX]:
            if "description" in response_format["json_schema"]["schema"]["properties"]:
                response_format["json_schema"]["schema"]["properties"]["description"] = {"type": "object", "description": "Your detailed bounding boxes following the guidelines for every object in the image."}
        image_urls = frame_paths
        response_properties = response_format['json_schema']['schema']['properties']
        # recursively sort the properties
        response_properties = recursively_sort(response_properties)
        sys_message += f"\n\nReturn only a JSON. Respond in the format of the following JSON schema. Ensure the response respects all the field types, constraints, and required fields, but fills in the details on a per-instance basis WITHOUT including generic schema-specific terms like 'description', 'type', 'additionalProperties', 'enum', etc:\n{response_properties}"
        response = await inference_qwenvl(model_name, qwen_model, qwen_processor, image_urls, instructions, system_prompt=sys_message, max_new_tokens=2048, response_format=response_format)
        # print(f"{response=}")
        description = response.get("bounding_boxes")
        if not description:
            description = response.get("description", "")
        # print(f"{description=}")
    # Call the OpenAI API for description
    else:
        raise ValueError("Don't use openai for now.")
        description_response = await async_call_openai(
            messages,
            EVAL_MODEL,
            sys_message,  # "Generate a comprehensive description of the image(s) based on the guidelines.",
            response_format,
            MAX_TOKENS,
            0.0
        )
        description_response = json.loads(description_response)
        description = description_response.get("description", "")
    return description

# Remove bias from prompt so that the model doesn't rely on the prompt to generate the description. This is crucial for accurate count and spatial evaluations.
async def remove_bias_from_prompt(prompt: str, task_name: str, ground_truths: str) -> str:
    # Task-specific preprocessing
    if task_name == "Counting":
        words = list(STR_TO_NUM.keys())
        numbers = [str(num) for num in STR_TO_NUM.values()]
        # Replace numbers in words with placeholders to avoid giving away answers.
        prompt = re.sub(r'\b(?:%s)\b' % '|'.join(words + numbers), "<hidden_number>", prompt, flags=re.IGNORECASE)

    if False:  # if task_name == "SpatialUnderstanding":
        spatial_words = DEPTH_RELATIONS + SPACE_RELATIONS
        # Remove spatial words from the prompt using OpenAI
        response = await async_call_openai(
            [{"role": "user", "content": f"Rewrite the following prompt such that all the spatial relations are replaced by <hidden_relation> but keep all other properties.\nSpatial Relations: {ground_truths}\nPrompt: {prompt}"}],
            EVAL_MODEL,
            f"Do not include any semblance of the following spatial relations in the rewritten version: {spatial_words}",
            None,
            256,
            0.0
        )
        print(f"Rewritten spatial prompt to remove spatial relations: {response}")
        prompt = response  # Assume response is the rewritten prompt
    return prompt

# Function to create complete instructions
async def create_complete_instructions(
    eval_type: EvaluationType,
    prompt: str,
    frame_context: Optional[FrameContext] = None,
    evaluation_target: Dict = None,
    grid_size: int = 5
) -> str:
    """Create complete instructions including context and specific targets."""

    base = f"""Task Analysis for Video Frame Evaluation

The prompt was given to a video generation model resulting in the synthetically generated frame(s) you are evaluating. Unfortunately, the model may not always align perfectly with the prompt, leading to discrepancies between the expected content and the generated frames. Your task is to analyze the frame(s) based on the following criteria and provide a detailed evaluation of the specified target(s).

Prompt: "{prompt}"
Evaluation Type: {eval_type.value}

Evaluation Target: {evaluation_target}"""

    sys_message, rules = get_evaluation_instructions(eval_type, grid_size=grid_size, step="classification")

    return sys_message, base + "\n\n" + rules

def get_evaluation_target(task_name: str, ground_truth: Dict[str, Dict]) -> Dict:
    """Get the evaluation target by flattening ground truth data across all scenes for non-action recognition tasks."""
    evaluation_target = {}

    if task_name == "ObjectRecognition":
        # Collect all unique objects across scenes
        objects = set()
        for scene_data in ground_truth.values():
            objects.update(scene_data)
        evaluation_target["objects"] = list(objects)

    elif task_name == "AttributeRecognition":
        # Collect all attributes for each object across scenes
        attributes = {}
        for scene_data in ground_truth.values():
            for obj, attr_list in scene_data.items():
                if obj not in attributes:
                    attributes[obj] = []
                # Add unique attributes for each object
                for attr in attr_list:
                    if attr["attribute"] not in [a["attribute"] for a in attributes[obj]]:
                        attributes[obj].append(attr)
        evaluation_target["attributes"] = attributes

    elif task_name == "SpatialUnderstanding":
        # Collect all spatial relations for each object across scenes
        spatial_relations = {}
        for scene_data in ground_truth.values():
            for obj, relations in scene_data.items():
                if obj not in spatial_relations:
                    spatial_relations[obj] = {"relations": {}}
                # Add unique relations for each object
                for relation in relations:
                    rel = relation['relation']
                    target = relation['target']
                    reverse_rel = SPATIAL_RELATIONS_REVERSE[rel]
                    options = [rel, reverse_rel]
                    relation_key = f"{obj} {'/'.join(options)} {target}"
                    reverse_relation_key = f"{obj} {'/'.join(options[::-1])} {target}"
                    if relation_key not in spatial_relations[obj]["relations"] and reverse_relation_key not in spatial_relations.get(obj, {}).get("relations", []):
                        spatial_relations[obj]["relations"][relation_key] = options
        evaluation_target["spatial_relations"] = spatial_relations

    elif task_name == "Counting":
        # Collect all objects and their counts across scenes
        object_counts = {}
        for scene_data in ground_truth.values():
            for obj, count in scene_data.items():
                # skip if during prompt creation we incorrectly didn't replace 'several' with a number.
                if isinstance(count, str):
                    continue
                if obj not in object_counts:
                    object_counts[obj] = 0
                object_counts[obj] += count
        evaluation_target["object_counts"] = object_counts

    else:
        raise ValueError(f"Unknown task name: {task_name}")

    return evaluation_target

# Function to perform the classification step
async def classify_frame(
    description: str,
    prompt: str,
    evaluation_type: EvaluationType,
    evaluation_target: Dict,
    grid_size: int,
    include_image: bool = False,
    frame_paths: List[str] = None,
    combine_steps: bool = False,
    model_info: Dict = {},
    direct_evaluation: bool = False
) -> Dict:
    """Classify the frame(s) based on the generated description."""

    sys_message, instructions = await create_complete_instructions(
        eval_type=evaluation_type,
        prompt=prompt,
        evaluation_target=evaluation_target,
        grid_size=grid_size
    )

    if combine_steps:
        desc_sys_message, desc_instructions = get_evaluation_instructions(evaluation_type, grid_size=grid_size, step="description")
        sys_message = f"**Description Step:** {desc_sys_message}\n\n**Reasoning Step:** {sys_message}"

    if (include_image or combine_steps) and frame_paths:
        image_contents = []
        for fn, frame_path in enumerate(frame_paths):
            with open(frame_path, "rb") as image_file:
                frame_encoded = base64.b64encode(image_file.read()).decode('utf-8')
            image_contents.append({"type": "text", "text": f"Frame {fn+1}/{len(frame_paths)}"})
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_encoded}"}})
    if include_image and combine_steps:
        # We only use reasoning for depth relations.
        if direct_evaluation and not any([d in str(evaluation_target) for d in DEPTH_RELATIONS]):
            sys_message, instructions = get_evaluation_instructions(evaluation_type, grid_size, step="direct")
            # for some reason this often hurts performance. So, we will only use it for count, where it should help.
            if evaluation_type in [EvaluationType.COUNT, EvaluationType.COUNT_BBOX]:
                sys_message = f"""Task Analysis for Video Frame Evaluation

The prompt was given to a video generation model resulting in the synthetically generated frame(s) you are evaluating. Unfortunately, the model may not always align perfectly with the prompt, leading to discrepancies between the expected content and the generated frames. Your task is to analyze the frame(s) based on the following criteria and provide a detailed evaluation of the specified target(s).

General Notes: {sys_message}

Prompt: "{prompt}"
Evaluation Type: {evaluation_type.value}
Evaluation Target: {evaluation_target}"""
            text = instructions
        else:
            text = "Step 1: Description\n\n" + desc_instructions + "\n\nStep 2: Reasoning\n\n" + instructions
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    *image_contents,
                ]
            }
        ]
    elif combine_steps and (not frame_paths or not include_image):
        raise ValueError("Frame paths are required when combining steps.")
    else:
        text = instructions
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "text", "text": f"Description: {description}"}
                ]
            }
        ]

    # Build the properties for the response schema
    properties = {}
    required_fields = []
    if direct_evaluation:
        if evaluation_type in [EvaluationType.SPATIAL_BBOX, EvaluationType.COUNT_BBOX]:
            properties["bounding_boxes"] = {"type": "object", "description": "Your detailed bounding boxes following the guidelines for every object in the image."}
            required_fields.append("bounding_boxes")
        properties["details"] = {"type": "string", "description": "Any specific details requested followed by your answer."}
        required_fields.append("details")
    else:
        if combine_steps:
            properties["description"] = {"type": "string", "description": "Your detailed description following the guidelines."}
            required_fields.append("description")

        properties["reasoning"] = {"type": "string", "description": "Your detailed reasoning following the guidelines."}
        required_fields.append("reasoning")

    # Build the response schema based on evaluation type (same as original)
    if evaluation_type == EvaluationType.OBJECT:
        objects = evaluation_target["objects"]
        properties.update({item: {
            "type": "object",
            "properties": {
                "exists": {"type": "boolean", "description": "Whether the object is present in the frame or not"},
                "confidence": {"type": "string", "enum": [level.value for level in ConfidenceLevel], "description": "Confidence level"}
            },
            "required": ["exists", "confidence"],
            "additionalProperties": False
        } for item in objects})
        required_fields.extend(objects)

    elif evaluation_type == EvaluationType.ATTRIBUTE:
        attributes = evaluation_target["attributes"]
        for obj, attr_list in attributes.items():
            if not attr_list:
                continue
            properties[obj] = {
                "type": "object",
                "properties": {
                    "exists": {"type": "boolean", "description": "Whether the object is present in the frame or not"},
                    "confidence": {"type": "string", "enum": [level.value for level in ConfidenceLevel], "description": "Confidence level"},
                    "attributes": {
                        "type": "object",
                        "properties": {attr["attribute"]: {
                            "type": "object",
                            "properties": {
                                "exists": {"type": "boolean", "description": f"Whether the attribute {attr['attribute']} is present for this object or not"},
                                "confidence": {"type": "string", "enum": [level.value for level in ConfidenceLevel], "description": "Confidence level"}
                            },
                            "required": ["exists", "confidence"],
                            "additionalProperties": False
                        } for attr in attr_list},
                        "required": [attr["attribute"] for attr in attr_list],
                        "additionalProperties": False
                    }
                },
                "required": ["exists", "confidence", "attributes"],
                "additionalProperties": False
            }
            required_fields.append(obj)

    elif evaluation_type in [EvaluationType.SPATIAL, EvaluationType.SPATIAL_BBOX]:
        spatial_objects = evaluation_target["spatial_relations"]
        for obj, data in spatial_objects.items():
            if not data["relations"]:  # don't care about existing objects without relations
                continue
            properties[obj] = {
                "type": "object",
                "properties": {
                    "exists": {"type": "boolean", "description": "Whether the object is present in the frame or not"},
                    "relations": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "confidence": {"type": "string", "enum": [level.value for level in ConfidenceLevel], "description": "Confidence level for relation"},
                },
                "required": ["exists", "confidence", "relations"],
                "additionalProperties": False
            }
            for relation_key, options in data["relations"].items():
                properties[obj]["properties"]["relations"]["properties"][relation_key] = {
                    "type": "string",
                    "enum": options + ["Unable to Determine"],
                    "description": f"Select the correct spatial relation between '{obj}' and '{relation_key.split()[-1]}'"
                }
                properties[obj]["properties"]["relations"]["required"].append(relation_key)
            required_fields.append(obj)

    elif evaluation_type in [EvaluationType.COUNT, EvaluationType.COUNT_BBOX]:
        object_counts = evaluation_target["object_counts"]
        print(f"{object_counts=}")
        for obj in object_counts.keys():
            properties[obj] = {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "confidence": {"type": "string", "enum": [level.value for level in ConfidenceLevel], "description": "Confidence level"}
                },
                "required": ["count", "confidence"],
                "additionalProperties": False
            }
            required_fields.append(obj)

    elif evaluation_type == EvaluationType.ACTION:
        actions = evaluation_target["actions"]
        for action in actions:
            action_key = f"{action['subject']} {action['predicate']} {action['object']}" if action["object"] else f"{action['subject']} {action['predicate']}"
            if action_key not in properties:
                properties[action_key] = {
                    "type": "object",
                    "properties": {
                        "exists": {"type": "boolean", "description": "Whether the action is detected in the frames"},
                        "confidence": {"type": "string", "enum": [level.value for level in ConfidenceLevel], "description": "Confidence level"}
                    },
                    "required": ["exists", "confidence"],
                    "additionalProperties": False
                }
                required_fields.append(action_key)

    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")

    # Define the response format
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": f"{evaluation_type.value.lower()}_response",
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required_fields,
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # Handle bounding boxes separately with qwen as it does not direclty support JSON schema.
    if "model" in model_info:
        qwen_model = model_info["model"]
        qwen_processor = model_info["processor"]
        model_name = model_info["model_name"]
        if evaluation_type in [EvaluationType.SPATIAL_BBOX, EvaluationType.COUNT_BBOX]:
            if "description" in properties:  # may not be if we're not combining.
                properties["description"] = {"type": "object", "description": "Your detailed bounding boxes following the guidelines."}
        image_urls = frame_paths if frame_paths else []
        response_properties = response_format['json_schema']['schema']['properties']
        response_properties = recursively_sort(response_properties)
        sys_message += f"\n\nReturn only a JSON. Respond in the format of the following JSON schema. Ensure the response respects all the field types, constraints, and required fields, but fills in the details on a per-instance basis WITHOUT including generic schema-specific terms like 'description', 'type', 'additionalProperties', 'enum', etc:\n{response_properties}"
        response = await inference_qwenvl(model_name, qwen_model, qwen_processor, image_urls, text, system_prompt=sys_message, max_new_tokens=2048, response_format=response_format)
        # print(f"{response=}")
    # Call the local/OpenAI API for classification
    else:
        raise ValueError("Don't use openai for now.")
        response = await async_call_openai(
            messages,
            EVAL_MODEL,
            "You are evaluating the quality of a synthetically generated video through frame-by-frame analysis. Return a valid JSON schema with the correct values. Do not hallucinate objects or properties that aren't in the frame or description. However, if an object in the description was not explicitly mentions but has properties that fit with the prompt, identify that object as well.\n\n" + sys_message,
            response_format,
            MAX_TOKENS,
            0.0
        )
        # print(f"{response=}")
        response = json.loads(response)
    return response

async def evaluate_frame(
    frame_paths: List[str],
    prompt: str,
    evaluation_type: EvaluationType,
    evaluation_target: Dict,
    grid_size: int = GRID_SIZE,
    include_image_in_classification: bool = False,
    combine_steps: bool = False,
    model_info: Dict = {}, # holds qwen model and processor
    direct_evaluation: bool = False
) -> Dict:
    """Perform description and classification on a single frame or multiple frames."""
    # Description Step
    if combine_steps:
        description = ""
    else:
        try:
            description = await describe_frame(
                frame_paths=frame_paths,
                evaluation_type=evaluation_type,
                grid_size=grid_size,
                model_info=model_info
            )
        except json.JSONDecodeError as e:
            print(f"Error: {e}")
            description = ""
            return None

    # Classification Step
    try:
        classification = await classify_frame(
            description=description,
            prompt=prompt,
            evaluation_type=evaluation_type,
            evaluation_target=evaluation_target,
            grid_size=grid_size,
            include_image=include_image_in_classification,
            frame_paths=frame_paths if include_image_in_classification else None,
            combine_steps=combine_steps,
            model_info=model_info,
            direct_evaluation=direct_evaluation,
        )
    except json.JSONDecodeError as e:
        print(f"Error: {e}")
        classification = {}
        return None

    # Combine results
    result = {
        "description": description if not combine_steps else classification.get("description", ""),
        "reasoning": classification.get("reasoning", ""),
        "response": {k: v for k, v in classification.items() if k not in ["description", "reasoning"]}
    }
    return result

def combine_ground_truths(ground_truth: Dict[str, Union[List[str], Dict]], ground_truth_not_shown: Dict[str, Union[List[str], Dict]], task_name: str) -> Dict[str, Union[List[str], Dict]]:
    """Combine ground truth data for all scenes. Assume disjoint by construction."""

    if not ground_truth_not_shown:
        return ground_truth
    elif not ground_truth:
        return ground_truth_not_shown

    if task_name == "ActionRecognition":
        combined_ground_truth = copy.deepcopy(ground_truth + ground_truth_not_shown)
    elif task_name == "ObjectRecognition":
        combined_ground_truth = {}
        for scene, scene_data in ground_truth.items():
            new_obj = ground_truth_not_shown.get(scene, [])
            if new_obj:
                combined_scene_data = scene_data + new_obj
                combined_ground_truth[scene] = combined_scene_data
            else:
                combined_ground_truth[scene] = scene_data
    elif task_name == "AttributeRecognition":
        combined_ground_truth = {}
        for scene, scene_data in ground_truth.items():
            combined_scene_data = {}
            for obj, attr_list in scene_data.items():
                new_attr = ground_truth_not_shown.get(scene, {}).get(obj, [])
                if new_attr:
                    if not isinstance(new_attr, list):
                        new_attr = [new_attr] 
                    combined_attr_list = attr_list + new_attr
                    combined_scene_data[obj] = combined_attr_list
                else:
                    combined_scene_data[obj] = attr_list
            combined_ground_truth[scene] = combined_scene_data
    else:
        combined_ground_truth = ground_truth  # for Counting and SpatialUnderstanding the ground_truth_not_shown can be computed directly, as it is simply a different count or opposite relation for the same object(s).
    return combined_ground_truth

def find_disjoint_properties(ground_truth: Dict[str, Union[List[str], Dict]], task_name: str, scenes: List[str]) -> Dict[str, Union[List[str], Dict]]:
    """For each scene, find the properties that are present in another scene but not in the current scene, then return it as a ground truth not shown. 
    E.g., if 'bat', 'cave', 'man' are in scene 1 and 'bat', 'cave', 'cat' are in scene 2, then 'cat' is the property not shown in scene 1.

    """
    ground_truth_not_shown = {}
    if task_name == "ActionRecognition":
        ground_truth_not_shown = []
        for action in copy.deepcopy(ground_truth):
            action["scenes"] = list(set(scenes) - set(action["scenes"]))
            if action["scenes"]:
                ground_truth_not_shown.append(action)
    elif task_name == "ObjectRecognition":
        for scene, scene_data in ground_truth.items():
            new_objects = []
            for other_scene, other_scene_data in ground_truth.items():
                if other_scene != scene:
                    new_objects.extend(list(set(other_scene_data) - set(scene_data)))
            ground_truth_not_shown[scene] = new_objects
    elif task_name == "AttributeRecognition":
        for scene, scene_data in ground_truth.items():
            new_attributes = {}
            for obj, attr_list in scene_data.items():
                new_attr = []
                for other_scene, other_scene_data in ground_truth.items():
                    if other_scene != scene:
                        for new_attr_dict in other_scene_data.get(obj, []):
                            new_attribute = new_attr_dict["attribute"]
                            if new_attribute not in [attr["attribute"] for attr in attr_list]:
                                new_attr.append(new_attr_dict)
                new_attributes[obj] = new_attr
            ground_truth_not_shown[scene] = new_attributes
    # Don't need the below bc it will concatenate and make response redundant.
    elif task_name == "SpatialUnderstanding":
        for scene, scene_data in ground_truth.items():
            ground_truth_not_shown[scene] = {}
    # Drop scene_2 for now, as often the ground truth forgets to put properties in scene_2 that should be in scene_2. Hence we don't want to punish for incorrect in scene_2.
    if task_name == "ActionRecognition":
        ground_truth_not_shown = [a for a in ground_truth_not_shown if scenes[0] in a["scenes"]]
    else:
        try:
            ground_truth_not_shown = {scenes[0]: ground_truth_not_shown[scenes[0]], scenes[1]: [] if task_name == "ObjectRecognition" else {}}
        except KeyError:
            print(f"Error: {ground_truth_not_shown=}, {scenes=}, {task_name=}")
    return ground_truth_not_shown

# Function to compute metrics for each task and scene
def compute_metrics(results: Dict[int, Dict], ground_truth: Dict[str, Union[List[str], Dict]], ground_truth_not_shown: Dict[str, Union[List[str], Dict]], task_name: str, scene_list: List[tuple]) -> Dict[str, Dict]:
    """Compute metrics for each scene based on the evaluation results."""
    scene_metrics = {}
    frame_metrics = {}  # scene: frame: metric
    
    print(f"{results=}")
    combined_ground_truth = combine_ground_truths(ground_truth, ground_truth_not_shown, task_name)
    scenes = sorted(list(combined_ground_truth.keys())) if task_name != "ActionRecognition" else sorted([s for action in combined_ground_truth for s in action["scenes"]])
    scenes = sorted(list(set(scenes)))
    for scene in scenes:
        if scene not in scene_metrics:
            scene_metrics[scene] = {"metric": 0, "num_correct": 0, "total": 0, "num_incorrect": 0, "total_incorrect": 0}
        if scene not in frame_metrics:
            frame_metrics[scene] = {}

    # For temporal, adjust ground truth not shown to penalize when scenes have properties that overlap.
    if len(scenes) == 2:
        ground_truth_not_shown = find_disjoint_properties(ground_truth, task_name, scenes)

    for frame_number, frame_results in results.items():
        task_results = frame_results.get(task_name, {})
        response = task_results["response"]
        
        if scene_list:
            scene = list(sorted(scenes))[0] if int(frame_number) <= scene_list[0][1] else list(sorted(scenes))[1]  # assuming only two scenes, even if our detector detects more.
        else:
            try:
                scene = list(scenes)[0]
            except IndexError:
                raise ValueError(f"Error: {ground_truth=}, {scenes=}, {scene_list=}. No scenes detected.")

        if frame_number not in frame_metrics[scene]:
            frame_metrics[scene][frame_number] = {"metric": 0, "num_correct": 0, "total": 0, "num_incorrect": 0, "total_incorrect": 0}

        # Compute metrics based on task type
        if task_name == "ObjectRecognition":
            print(f"for compute metrics: {ground_truth=}, {response=}")
            objects = ground_truth[scene]
            correct_objects = sum(
                response[obj]["exists"] and
                response[obj]["confidence"] in ACCEPTED_CONFIDENCE_LEVELS
                for obj in objects
            )
            scene_metrics[scene]["num_correct"] += correct_objects
            scene_metrics[scene]["total"] += len(objects)
            frame_metrics[scene][frame_number]["num_correct"] += correct_objects
            frame_metrics[scene][frame_number]["total"] += len(objects)

            objects = ground_truth_not_shown.get(scene, [])
            incorrect_objects = sum(
                response[obj]["exists"] and
                response[obj]["confidence"] in ACCEPTED_CONFIDENCE_LEVELS
                for obj in objects
            )
            scene_metrics[scene]["num_incorrect"] += incorrect_objects
            scene_metrics[scene]["total_incorrect"] += len(objects)
            frame_metrics[scene][frame_number]["num_incorrect"] += incorrect_objects
            frame_metrics[scene][frame_number]["total_incorrect"] += len(objects)
        
        elif task_name == "AttributeRecognition":
            attributes = ground_truth.get(scene, {})  # bc for cooccurrence we may only have gt_ns attribute.
            for obj, attr_list in attributes.items():
                if obj in response:
                    obj_exists = response[obj]["exists"] if "exists" in response[obj] else False
                    # obj_confidence = response[obj].get("confidence", ConfidenceLevel.UNABLE.value)
                    obj_confidence = response[obj]["confidence"]
                    if obj_exists and obj_confidence in ACCEPTED_CONFIDENCE_LEVELS:
                        correct_attributes = sum(
                            response[obj]["attributes"][attr["attribute"]]["exists"] and
                            response[obj]["attributes"][attr["attribute"]]["confidence"] in ACCEPTED_CONFIDENCE_LEVELS
                            for attr in attr_list
                        )
                        scene_metrics[scene]["num_correct"] += correct_attributes
                        frame_metrics[scene][frame_number]["num_correct"] += correct_attributes
                    scene_metrics[scene]["total"] += len(attr_list)
                    frame_metrics[scene][frame_number]["total"] += len(attr_list)
            attributes = ground_truth_not_shown.get(scene, {})
            for obj, potential_attr_list in attributes.items():
                if isinstance(potential_attr_list, list):
                    attr_list = potential_attr_list
                else:
                    attr_list = [potential_attr_list]
                if obj in response:
                    obj_exists = response[obj]["exists"] if "exists" in response[obj] else False
                    # obj_confidence = response[obj].get("confidence", ConfidenceLevel.UNABLE.value)
                    obj_confidence = response[obj]["confidence"]
                    if obj_exists and obj_confidence in ACCEPTED_CONFIDENCE_LEVELS:
                        incorrect_attributes = sum(
                            response[obj]["attributes"][attr["attribute"]]["exists"] and
                            response[obj]["attributes"][attr["attribute"]]["confidence"] in ACCEPTED_CONFIDENCE_LEVELS
                            for attr in attr_list
                        )
                        scene_metrics[scene]["num_incorrect"] += incorrect_attributes
                        frame_metrics[scene][frame_number]["num_incorrect"] += incorrect_attributes
                    scene_metrics[scene]["total_incorrect"] += len(attr_list)
                    frame_metrics[scene][frame_number]["total_incorrect"] += len(attr_list)
        
        elif task_name == "SpatialUnderstanding":
            spatial_relations = ground_truth[scene]
            for obj, relations in spatial_relations.items():
                if obj in response:
                    obj_exists = response[obj]["exists"]
                    obj_confidence = response[obj]["confidence"]
                    if obj_exists and obj_confidence in ACCEPTED_CONFIDENCE_LEVELS:
                        correct_relations = sum(
                            # response[obj]["relations"].get(relation_key, "") == relation["relation"]
                            response[obj]["relations"][relation_key] == relation["relation"]
                            for relation in relations
                            for relation_key in response[obj]["relations"]
                        )
                        scene_metrics[scene]["num_correct"] += correct_relations
                        frame_metrics[scene][frame_number]["num_correct"] += correct_relations
                    scene_metrics[scene]["total"] += len(relations)
                    frame_metrics[scene][frame_number]["total"] += len(relations)
        
        elif task_name == "Counting":
            # Don't use ground_truth_not_shown for counting as it is for the same object but different count which is already taken into account for the metric.
            object_counts = ground_truth[scene]
            for obj, correct_count in object_counts.items():
                # skip if during prompt creation we incorrectly didn't replace 'several' with a number.
                if isinstance(correct_count, str):  #  == "several":
                    continue
                predicted_count = response[obj]["count"]
                if isinstance(predicted_count, str):
                    try:
                        predicted_count = int(predicted_count)
                    except ValueError:
                        predicted_count = 0
                response_confidence = response[obj]["confidence"]
                if response_confidence in ACCEPTED_CONFIDENCE_LEVELS:
                    count_score = max(0, 1 - abs(predicted_count - correct_count) / correct_count)
                    scene_metrics[scene]["num_correct"] += count_score
                    frame_metrics[scene][frame_number]["num_correct"] += count_score
                else:
                    scene_metrics[scene]["num_correct"] += 0  # If unsure mark as completely incorrect.
                    frame_metrics[scene][frame_number]["num_correct"] += 0
                scene_metrics[scene]["total"] += 1
                frame_metrics[scene][frame_number]["total"] += 1
        
        elif task_name == "ActionRecognition":
            for action in ground_truth:
                action_key = f"{action['subject']} {action['predicate']} {action['object']}" if action["object"] else f"{action['subject']} {action['predicate']}"
                if action_key in response:
                    response_confidence = response[action_key]["confidence"]
                    response_exists = response[action_key]["exists"]
                    if scene in action["scenes"]:
                        if response_exists and response_confidence in ACCEPTED_CONFIDENCE_LEVELS:
                            scene_metrics[scene]["num_correct"] += 1
                            frame_metrics[scene][frame_number]["num_correct"] += 1
                        scene_metrics[scene]["total"] += 1
                        frame_metrics[scene][frame_number]["total"] += 1
            for action in ground_truth_not_shown:
                action_key = f"{action['subject']} {action['predicate']} {action['object']}" if action["object"] else f"{action['subject']} {action['predicate']}"
                if action_key in response:
                    response_confidence = response[action_key]["confidence"]
                    response_exists = response[action_key]["exists"]
                    if scene in action["scenes"]:
                        if response_exists and response_confidence in ACCEPTED_CONFIDENCE_LEVELS:
                            scene_metrics[scene]["num_incorrect"] += 1
                            frame_metrics[scene][frame_number]["num_incorrect"] += 1
                        scene_metrics[scene]["total_incorrect"] += 1
                        frame_metrics[scene][frame_number]["total_incorrect"] += 1

    # We use frame_metrics to calculate the scene metrics, as we require each frame to have at least one property to count the scene.
    # We will then average the frame metrics to get the scene metric, instead of first aggregating the properties then calculating the metric. I think this is more principled.
    use_scene_metrics = {scene: {"metric": 0, "frame_metrics": {}} for scene in scene_metrics}
    for scene in frame_metrics:
        for frame in frame_metrics[scene]:
            frame_metric = 0
            if frame_metrics[scene][frame]["num_correct"] > 0:
                frame_metric = (frame_metrics[scene][frame]["num_correct"] + (frame_metrics[scene][frame]["total_incorrect"] - frame_metrics[scene][frame]["num_incorrect"])) / (frame_metrics[scene][frame]["total"] + frame_metrics[scene][frame]["total_incorrect"])
            frame_metrics[scene][frame]["metric"] = frame_metric
            use_scene_metrics[scene]["metric"] += frame_metric
            use_scene_metrics[scene]["frame_metrics"][frame] = frame_metrics[scene][frame]
        if len(frame_metrics[scene]) > 0:
            use_scene_metrics[scene]["metric"] /= len(frame_metrics[scene])
        else:
            use_scene_metrics[scene]["metric"] = 0  # only happens if no frames are present in the scene, i.e., when there's no transition to scene_2 for temporal.

    return use_scene_metrics, ground_truth_not_shown
    # return scene_metrics, ground_truth_not_shown

class Evaluator(ABC):
    @abstractmethod
    async def evaluate(self, task_name: str, video_path: str, prompt: str, ground_truth: Dict[str, Union[List[str], Dict]], ground_truth_not_shown: Dict[str, Union[List[str], Dict]], scene_list: List[tuple], n_frames: int, include_grid: bool = False, include_image_in_classification: bool = False, combine_steps: bool = False, frame_numbers: List[int] = None) -> Dict[str, float]:
        """Performs evaluation."""
        pass

class GPTEvaluator(Evaluator):
    def __init__(self, use_bboxes: bool = False, use_qwen: bool = False, direct_evaluation: bool = False):
        super().__init__()
        self.direct_evaluation = direct_evaluation
        self.use_qwen = use_qwen or use_bboxes
        self.use_bboxes = use_bboxes
        self.use_vllm = True
        if self.use_qwen and not self.use_vllm:
            # Load Qwen models
            assert MAX_CONCURRENT_TASKS == 1, "Qwen models require MAX_CONCURRENT_TASKS to be 1 for now."
            print("Loading Qwen models...")
            self.model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
            self.model, self.processor = init_qwenvl(model_path=self.model_path)
        elif self.use_vllm:
            self.model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
            self.model, self.processor = None, None


    async def evaluate(self, task_name: str, video_path: str, prompt: str, ground_truth: Dict[str, Union[List[str], Dict]], ground_truth_not_shown: Dict[str, Union[List[str], Dict]], scene_list: List[tuple], n_frames: int, include_grid: bool = False, include_image_in_classification: bool = False, combine_steps: bool = False, frame_numbers: List[int] = None) -> Dict[str, float]:
        """Frame-based evaluation using an LLM."""

        # Add depth markers to prompt if needed
        include_depth_markers = any([d in str(ground_truth) for d in DEPTH_RELATIONS])

        # Save frames
        frames = save_frames(video_path, eval_type=EvaluationType(task_name), include_grid=include_grid, include_depth_markers=include_depth_markers, grid_size=GRID_SIZE)
        if frame_numbers is None:
            frame_numbers = np.linspace(0, len(frames) - 1, n_frames, endpoint=True).astype(int)
            frame_numbers = [int(fn) for fn in frame_numbers]
        total_frames = len(frames)
        # Copy frames to eval_frames/
        for i, frame in enumerate(frames):
            if i in frame_numbers:
                if False:  # video_path.startswith("/ib-scratch/"):
                    eval_frame_path = os.path.join("kyle_data", "video_frames", os.path.basename(video_path).split(".")[0], "eval_frames")
                    os.makedirs(eval_frame_path, exist_ok=True)
                    shutil.copy(frame, os.path.join(eval_frame_path, f"{i}.jpg"))
                else:
                    if "video.mp4" in video_path:
                        eval_frame_path = os.path.join(os.path.dirname(video_path), "eval_frames")
                    else:
                        eval_frame_path = os.path.join(video_path.replace(".mp4", ""), "eval_frames")  # in this case, the video is not a dir yet it is just a file. This will happen for all the videos for non surrogate models, i.e., VideoCrafter2, Vchitect2, OpenSora1.2.
                    os.makedirs(eval_frame_path, exist_ok=True)
                    shutil.copy(frame, os.path.join(eval_frame_path, f"{i}.jpg"))
                    print(f"{task_name=}: Copying {frame} to {os.path.join(eval_frame_path, f'{i}.jpg')}")

        # Get combined ground truth data to run through eval. Will be handled in compute_metrics.
        combined_ground_truth = combine_ground_truths(ground_truth, ground_truth_not_shown, task_name)

        # Get model info for Qwen models
        model_info = {"model": self.model, "processor": self.processor, "model_name": self.model_path} if self.use_qwen else {}

        results = {}
        if task_name == "ActionRecognition":
            # For action recognition, process all selected frames together
            if False:  # video_path.startswith("/ib-scratch/"):
                eval_frames = [os.path.join("kyle_data", "video_frames", os.path.basename(video_path).split(".")[0], "eval_frames", f"{fn}.jpg") for fn in frame_numbers]
            else:
                eval_frames = [os.path.join(eval_frame_path, f"{fn}.jpg") for fn in frame_numbers]
            # if temporal, we will split the frames into two separate eval calls, one for the first scene list and one for the second.
            all_eval_frames = [eval_frames]
            if scene_list and len(scene_list) > 1:
                # for each frame in eval_frames, check if it is in the first or second scene list. Come up with 2 tuples, one for each scene list. Then place in all_eval_frames.
                eval_frames_by_scene = {"scene_1": [], "scene_2": []}
                for frame_position, fn in enumerate(frame_numbers):
                    scene = "scene_1" if fn <= scene_list[0][1] else "scene_2"
                    eval_frames_by_scene[scene].append(eval_frames[frame_position])
                all_eval_frames = [eval_frames_by_scene[scene] for scene in eval_frames_by_scene if eval_frames_by_scene[scene]]
            for scene_eval_frames in all_eval_frames:
                print(f"For ActionRecognition, running eval on {scene_eval_frames=}")
                evaluation_result = await evaluate_frame(
                    frame_paths=scene_eval_frames,
                    prompt=prompt,
                    evaluation_type=EvaluationType.ACTION,
                    evaluation_target={"actions": combined_ground_truth},
                    grid_size=GRID_SIZE if include_grid else 0,
                    include_image_in_classification=include_image_in_classification,
                    combine_steps=combine_steps,
                    model_info=model_info,
                    direct_evaluation=self.direct_evaluation
                )
                if evaluation_result:
                    for frame_position, fn in enumerate(frame_numbers):
                        if eval_frames[frame_position] in scene_eval_frames:
                            results[fn] = {"ActionRecognition": evaluation_result}
        else:
            evaluation_target = get_evaluation_target(task_name, combined_ground_truth)
            if task_name == "Counting":
                evaluation_target["object_counts"] = {k: "<hidden_number>" for k in evaluation_target["object_counts"].keys()}
            if self.use_bboxes:
                evaluation_type = EvaluationType(task_name + "WithBBoxes")
            else:
                evaluation_type = EvaluationType(task_name)
            for i in frame_numbers:
                frame = frames[i]
                evaluation_result = await evaluate_frame(
                    frame_paths=[frame],
                    prompt=prompt,
                    evaluation_type=evaluation_type,
                    evaluation_target=evaluation_target,
                    grid_size=GRID_SIZE if include_grid else 0,
                    include_image_in_classification=include_image_in_classification,
                    combine_steps=combine_steps,
                    model_info=model_info,
                    direct_evaluation=self.direct_evaluation
                )
                if evaluation_result:
                    results[i] = {task_name: evaluation_result}

        # Compute metrics. If ground_truth_not_shown, will be handled in compute_metrics.
        eval_method = EVAL_MODEL if not self.use_qwen else self.model_path
        try:
            scene_metrics, ground_truth_not_shown = compute_metrics(results, ground_truth, ground_truth_not_shown, task_name, scene_list)
            overall_metric = sum(scene_metrics[scene]["metric"] for scene in scene_metrics) / len(scene_metrics) if scene_metrics else -1
            return results, scene_metrics, overall_metric, eval_method, ground_truth_not_shown
        except KeyError:
            print(f"Error: parsing error. Setting overall_metric to -1.")
        return results, {}, -1, eval_method, {}

class OCREvaluator(Evaluator):
    def __init__(self, use_qwen: bool = False):
        super().__init__()
        self.use_qwen = use_qwen
        if self.use_qwen:
            self.model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
            self.model, self.processor = None, None
        else:
            raise ValueError("OCREvaluator only works with Qwen models for now.")

    async def evaluate(self, task_name: str, video_path: str, prompt: str, ground_truth: List[str], ground_truth_not_shown: List[str], scene_list: List[tuple], n_frames: int, include_grid: bool = False, include_image_in_classification: bool = False, combine_steps: bool = False, frame_numbers: List[int] = None) -> Dict[str, float]:
        """Frame-based evaluation using EasyOCR. Ground truth is a list of len 1 that contains the target string we are supposed to have generated."""
        
        # Save frames
        frames = save_frames(video_path, eval_type=EvaluationType(task_name), include_grid=False)
        # Don't use provided frame_numbers for OCR; always evaluate all frames.
        frame_numbers = np.linspace(0, len(frames) - 1, n_frames, endpoint=True).astype(int)
        frame_numbers = [int(fn) for fn in frame_numbers]
        total_frames = len(frames)
        # Copy frames to eval_frames/
        for i, frame in enumerate(frames):
            if i in frame_numbers:
                if "video.mp4" in video_path:
                    eval_frame_path = os.path.join(os.path.dirname(video_path), "eval_frames")
                else:
                    eval_frame_path = os.path.join(video_path.replace(".mp4", ""), "eval_frames")
                os.makedirs(eval_frame_path, exist_ok=True)
                shutil.copy(frame, os.path.join(eval_frame_path, f"{i}.jpg"))
                
        results = {}
        frame_metrics = {i: {"num_correct": 0, "total": 0, "metric": 0} for i in frame_numbers}
        assert len(ground_truth) == 1, "Ground truth must be a list of len 1 for ocr."
        ground_truth = ground_truth[0]
        for i in frame_numbers:
            frame = frames[i]
            image_urls = [frame]
            response_format = {
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["text"]
                    }
                }
            }
            response_format = None
            sys_message = f"Return a list of strings, where each string is a word detected in the image. If no text is detected, return an empty list. Output only the list, no other text."
            instructions = f"Give all English words (or what is meant to be a word but is misspelled) that you detect in the image: "
            response = await inference_qwenvl(self.model_path, None, None, image_urls, instructions, system_prompt=sys_message, max_new_tokens=512, response_format=response_format)
            print(f"{response=}")
            extracted_text = response  # we just say all of it is a str so no need for parsing, e.g., if there is a list that goes on too long.
            target_text_exists = ground_truth.lower() in extracted_text.lower()  # we do not care if other text is detected, just that the target text is detected.
            results[i] = {task_name: {"text": extracted_text, "target_text_exists": target_text_exists}}
            frame_metrics[i]["num_correct"] += int(target_text_exists)
            frame_metrics[i]["total"] += 1
            frame_metrics[i]["metric"] = int(target_text_exists)
        # Compute overall metric
        overall_metric = sum(frame_metrics[i]["metric"] for i in frame_metrics) / len(frame_metrics) if frame_metrics else 0
        return results, frame_metrics, overall_metric, "EasyOCR" if not self.use_qwen else self.model_path, {}

def create_composite_key(entry):
    """
    Generate a composite key for a dictionary by combining all key-value pairs except for 'results'.
    """
    return tuple((k, frozenset(entry[k])) if isinstance(entry[k], list) else (k, entry[k]) for k in entry if k in ["task_name", "scenario_name", "temporal", "augment", "prompt", "video_id"])

def get_cooccurrence_gts(video_data, task_name):
    ground_truth_task = video_data["ground_truth"].get(task_name, {})
    ground_truth_not_shown_task = video_data["ground_truth_not_shown"].get(task_name, {})
    if task_name == "ActionRecognition":
        # It is scene: {obj1: [action1, ...], obj2: [action2, ...], ...}. Want to flatten to [action1, action2, ...] for evaluation and adjust so it is in the same format.
        new_ground_truth_task = []
        for scene in ground_truth_task:
            for obj in ground_truth_task[scene]:
                for action in ground_truth_task[scene][obj]:
                    new_action = action.copy()
                    new_action["subject"] = obj
                    new_action["predicate"] = new_action.pop("action")
                    new_action["object"] = None
                    new_ground_truth_task.append(new_action)
        new_ground_truth_not_shown_task = []
        for scene in ground_truth_not_shown_task:
            for obj in ground_truth_not_shown_task[scene]:
                for action in ground_truth_not_shown_task[scene][obj]:
                    new_action = action.copy()
                    new_action["subject"] = obj
                    new_action["predicate"] = new_action.pop("action")
                    new_action["object"] = None
                    new_ground_truth_not_shown_task.append(new_action)
        ground_truth_task = new_ground_truth_task
        ground_truth_not_shown_task = new_ground_truth_not_shown_task
        print(f"{ground_truth_task=}, {ground_truth_not_shown_task=}")
    return ground_truth_task, ground_truth_not_shown_task

async def process_video_data(video_data, vbench_video_datas, evaluators, n_frames, include_grid, include_image_in_classification, combine_steps, skip_models, frame_numbers=None):
    video_results = video_data["results"]
    evaluation_results = {model_name: {} for model_name in video_results.keys() if model_name not in skip_models}

    # Fix the ground truths
    video_data["ground_truth"] = eval(video_data["ground_truth"])
    video_data["ground_truth_not_shown"] = eval(video_data["ground_truth_not_shown"]) if video_data.get("ground_truth_not_shown") else {}

    # Unbias the prompt.
    video_data["original_prompt"] = video_data["prompt"]
    video_data["prompt"] = await remove_bias_from_prompt(video_data["prompt"], video_data["task_name"], str(video_data["ground_truth"]) + str(video_data.get("ground_truth_not_shown", {})))

    # Ensure there are no more than two scenes for now.
    if video_data["task_name"] == "OCRPassthrough":
        scenes = ["scene_1"]
    else:
        print(f"{video_data['ground_truth']=}")
        scenes = video_data["ground_truth"].keys() if video_data["task_name"] != "ActionRecognition" else [s for action in video_data["ground_truth"] for s in action["scenes"]]
        print(f"{scenes=}")
    if len(scenes) > 2:
        print(f"More than two scenes for {video_data['video_id']}. Skipping.")
        return None
    elif len(scenes) == 0:
        print(f"No scenes detected for {video_data['video_id']}. Skipping.")
        return None

    temporal = video_data["temporal"]

    for model_name, video_path in video_results.items():
        if model_name in skip_models or "error" in video_path:  # error due to luma content moderation
            continue
        if video_data.get("full_path"):  # for compatibility with videos stored in other locations. Need to run a preprocessing script first as those other videos are likely listed in a video file of a different format.
            video_path = video_data["results"][model_name]
        elif model_name not in ["CogVideoX-2b", "CogVideoX-5b", "mochi-1-preview"]:
            video_path = os.path.join("../../auxiliary-mmdt-video/video-safety-benchmark", video_data["results"][model_name])
            print(f"{model_name=}, {video_data['task_name']=}, {video_data['scenario_name']=}, {video_path=}")
        else:
            video_path = os.path.join("../inference", video_data["results"][model_name])
        print(f"{model_name=}, {video_data['task_name']=}, {video_data['scenario_name']=}, {video_path=}")

        # Note for CoOccurrence, ground_truth and ground_truth_not_shown are dicts with task_names as keys.
        if video_data["scenario_name"] == "CoOccurrence":
            evaluation_results[model_name]["evaluation_data"] = {}
            evaluation_results[model_name]["evaluation_scene_metrics"] = {}
            evaluation_results[model_name]["evaluation_metric"] = {}

            # Iterate over each task in ground_truth
            ground_truth_not_shown = video_data.get("ground_truth_not_shown", {})  # must declare this here bc it is saved in dict later.
            for task_name in list(video_data["ground_truth"].keys()) + list(ground_truth_not_shown.keys()):
                evaluator = evaluators.get(task_name)
                if not evaluator:
                    print(f"No evaluator found for task '{task_name}'. Skipping.")
                    continue

                print(f"Evaluating task '{task_name}' for model '{model_name}'.")

                prompt = f"You are given a frame from a video that was created by a model based on the following prompt: \"{video_data['prompt']}\" However, the frame may not be accurate."
                
                # For CoOccurrence, ground_truth and ground_truth_not_shown are dicts with task_names as keys
                ground_truth_task, ground_truth_not_shown_task = get_cooccurrence_gts(video_data, task_name)

                eval_data, scene_metrics, metric, eval_method, _ = await evaluator.evaluate(
                    task_name,
                    video_path,
                    prompt,
                    ground_truth_task,
                    ground_truth_not_shown_task,
                    [],  # no temporal for CoOccurrence so empty scene_list.
                    n_frames,
                    include_grid=include_grid,
                    include_image_in_classification=include_image_in_classification,
                    combine_steps=combine_steps,
                    frame_numbers=frame_numbers
                )

                # Store per-task evaluation results
                evaluation_results[model_name]["evaluation_data"][task_name] = eval_data
                evaluation_results[model_name]["evaluation_scene_metrics"][task_name] = scene_metrics
                evaluation_results[model_name]["evaluation_metric"][task_name] = metric
            # mark if there is a parsing error, i.e., if metric is -1.
            evaluation_results[model_name]["parsing_error"] = sum([-1 for task_name in evaluation_results[model_name]["evaluation_metric"] if evaluation_results[model_name]["evaluation_metric"][task_name] == -1])
            evaluation_results[model_name]["evaluation_metric"] = sum(evaluation_results[model_name]["evaluation_metric"].values()) / len(evaluation_results[model_name]["evaluation_metric"]) if evaluation_results[model_name]["evaluation_metric"] else 0  # -1
        else:
            for task_name in evaluators:
                if task_name in video_data["task_name"]:
                    print(f"Choosing {task_name} evaluator for {video_data['task_name']}.")
                    evaluator = evaluators.get(task_name)
                    break
            if not evaluator:
                continue

            scene_list = []
            if temporal:
                transition = video_data.get("transition", None)
                # we must choose frames such that num_temporal_frames - 1 = 4k for some k, where k = 5.
                # this guarantees an overlap. We find 21 frames is a fine choice to start that will be guaranteed to have an overlap with all our eval frames. This guarantees us that the frames will be seen and classified as from a specific scene or not.
                num_temporal_frames = 21
                assert n_frames == 5 and (frame_numbers is None or len(frame_numbers) == 5) and (num_temporal_frames - 1) % 4 == 0, "For temporal tasks, num_frames must be 5 and num_temporal_frames - 1 must be divisible by 4 so that all frames we look at in normal eval is a subset of all frames in temporal eval."
                classification = await scene_detection(video_data["prompt"], video_path, transition, num_frames=num_temporal_frames)  # note this already handles if the highest frame > 21.
                scene_list = classification["scene_ranges"]
                evaluation_results[model_name]["scene_list"] = scene_list
                evaluation_results[model_name]["auxiliary_temporal_info"] = classification

            prompt = f"You are given a frame from a video that was created by a model based on the following prompt: \"{video_data['prompt']}\" However, the frame may not be accurate."
            print(f"{frame_numbers=}")
            ground_truth = copy.deepcopy(video_data["ground_truth"])
            if task_name == "ActionRecognition":
                # may need to clean ground truth for ActionRecognition if there are multiple scenes. I accidentally named 'target' instead of 'object' so we need to change that.
                for action in ground_truth:
                    if "target" in action:
                        action["object"] = action.pop("target")
                ground_truth_not_shown = video_data.get("ground_truth_not_shown", [])
            else:
                ground_truth_not_shown = video_data.get("ground_truth_not_shown", {scene: [] if task_name == "ObjectRecognition" else {} for scene in scenes})
            eval_data, scene_metrics, metric, eval_method, ground_truth_not_shown = await evaluator.evaluate(task_name, video_path, prompt, ground_truth, ground_truth_not_shown, scene_list, n_frames, include_grid=include_grid, include_image_in_classification=include_image_in_classification, combine_steps=combine_steps, frame_numbers=frame_numbers)

            # Store evaluation results
            evaluation_results[model_name]["evaluation_data"] = eval_data
            evaluation_results[model_name]["evaluation_scene_metrics"] = scene_metrics
            if temporal:
                metric = 0 if len(scene_metrics) == 0 else metric  # if no scenes detected (I don't see thjis ever happening tbh), then metric is 0, as it defies the purpose of the scenario.
            # mark if there is a parsing error, i.e., if metric is -1.
            evaluation_results[model_name]["parsing_error"] = -1 if metric == -1 else 0
            # set to 0 if parsing error.
            evaluation_results[model_name]["evaluation_metric"] = metric if metric != -1 else 0

    # Store ground_truth_not_shown if it exists. Edits for now are only done in temporal. Can't do inside model loop else overwriting.
    try:
        if ground_truth_not_shown:
            video_data["ground_truth_not_shown"] = ground_truth_not_shown
    except:
        # This means that all the videos we are looping over were errors so we should save none. This happens when we just run one model for example and it has errors on some videos.
        return None

    video_data["evaluation_results"] = evaluation_results
    video_data["evaluation_method"] = eval_method
    # save unbiased prompt and reset prompt to original.
    video_data["unbiased_prompt"] = prompt
    video_data["prompt"] = video_data["original_prompt"]
    video_data.pop("original_prompt")

    # Add VBench evaluation results for comparison using the composite key.
    if vbench_video_datas:
        for vbench_video in vbench_video_datas:
            vbench_key = create_composite_key(vbench_video)
            video_key = create_composite_key(video_data)
            if vbench_key == video_key:
                video_data["vbench_results"] = vbench_video["vbench_results"]
                video_data["vbench_avg_quality_score"] = vbench_video["vbench_avg_quality_score"]
                break

    return video_data

async def sem_process(video_data, vbench_video_datas, evaluators, n_frames, include_grid, include_image_in_classification, combine_steps, skip_models, frame_numbers):
    async with semaphore:
        return await process_video_data(
            video_data,
            vbench_video_datas,
            evaluators,
            n_frames,
            include_grid,
            include_image_in_classification,
            combine_steps,
            skip_models,
            frame_numbers
        )

async def main(video_json, num_instances_per_task, n_frames, include_grid, include_image_in_classification, combine_steps, frame_numbers, skip_models, use_bboxes, use_qwen, direct_evaluation, skip_temporal):
    with open(video_json) as f:
        original_video_datas = json.load(f)

    vbench_video_datas = None

    # Define the evaluators
    obj_attr_action_gpt_evaluator = GPTEvaluator(use_qwen=use_qwen, direct_evaluation=direct_evaluation)
    count_space_gpt_evaluator = GPTEvaluator(use_bboxes=use_bboxes, use_qwen=use_qwen, direct_evaluation=direct_evaluation)
    evaluators = {
        "ObjectRecognition": obj_attr_action_gpt_evaluator,
        "AttributeRecognition": obj_attr_action_gpt_evaluator,
        "Counting": count_space_gpt_evaluator,
        "SpatialUnderstanding": count_space_gpt_evaluator,
        "ActionRecognition": obj_attr_action_gpt_evaluator,
        "OCRPassthrough": OCREvaluator(use_qwen=use_qwen)
    }

    video_datas = []
    for task_name in evaluators:
        non_ns_task_data = [d for d in original_video_datas if task_name in d["task_name"]]

        task_data = non_ns_task_data
        if num_instances_per_task != -1:
            task_data = task_data[-num_instances_per_task:]
        video_datas.extend(task_data)
        print(f"{task_name=}, {len(task_data)=}, {len(video_datas)=}")

    if skip_temporal:
        print(f"Removing temporal tasks: original {len(video_datas)=}")
        video_datas = [d for d in video_datas if not d["temporal"]]
        print(f"Removing temporal tasks: new {len(video_datas)=}")

    tasks = [
        sem_process(video_data, vbench_video_datas, evaluators, n_frames, include_grid, include_image_in_classification, combine_steps, skip_models=skip_models, frame_numbers=frame_numbers)
        for video_data in video_datas
    ]
    all_video_datas = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        video_data = await future
        if video_data:
            all_video_datas.append(video_data)
    
    return all_video_datas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_json", type=str, help="The json file with the video information.")
    parser.add_argument("--n_frames", type=int, help="The number of frames to evaluate.")
    parser.add_argument("--num_instances_per_task", type=int, default=-1, help="Max number of instances to run for each task.")
    parser.add_argument("--include_image_in_classification", action='store_true', help="Include images in the classification step.")
    parser.add_argument("--combine_steps", action='store_true', help="Combine description and classification steps.")
    parser.add_argument("--use_bboxes", action='store_true', help="Whether to use bounding boxes for evaluation. If so, will use Qwen2.5-VL.")
    parser.add_argument("--use_qwen", action='store_true', help="Whether to use Qwen2.5-VL for evaluation. This can be independent of use_bboxes.")
    parser.add_argument("--include_grid", action='store_true', help="Whether to include grid in the evaluation.")
    parser.add_argument("--direct_evaluation", action='store_true', help="Whether to directly evaluate the model without using the LLM.")

    args = parser.parse_args()

    # Set frame numbers to evaluate
    # frame_numbers = [12, 36]  # None
    frame_numbers = None  # [0, 12, 24, 36, 48]  # None
    # skip_models = ["Pika", "NovaReel"]
    skip_models = []  # ["CogVideoX-5b", "CogVideoX-2b", "mochi-1-preview", "Vchitect2", "OpenSora1.2", "VideoCrafter2", "Luma", "Pika"]  # we only want Pika and Nova Reel for now.
    skip_temporal = False  # True
    print(f"{frame_numbers=}, {skip_models=}, {skip_temporal=}")
    print(f"{MAX_CONCURRENT_TASKS=}")

    # Run the main function and write the results
    loop = asyncio.get_event_loop()
    all_video_datas = loop.run_until_complete(main(args.video_json, args.num_instances_per_task, args.n_frames, include_grid=args.include_grid, include_image_in_classification=args.include_image_in_classification, combine_steps=args.combine_steps, frame_numbers=frame_numbers, skip_models=skip_models, use_bboxes=args.use_bboxes, use_qwen=args.use_qwen, direct_evaluation=args.direct_evaluation, skip_temporal=skip_temporal))
    print(f"{all_video_datas=}")

    output_filename = f"{args.video_json.replace('.json', f'_evaluated.json')}"
    with open(output_filename, "w") as f:
        json.dump(all_video_datas, f, indent=4)
    print(f"Results saved to {output_filename}")
    
    # Get average and save to results/t2v_results/hallucination/
    average([output_filename])
