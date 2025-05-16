import os

MODEL = "gpt-4o-mini"
EXTRACT_INFO_MODEL = "gpt-4o-mini" 
V2T_MODEL = "gpt-4o"
V2T_ANSWER_QUESTION_MODEL = "gpt-4o"
V2T_SCENARIOS_MODEL = "gpt-4o-mini"
V2T_OBJ_EXTRACTION_MODEL = "gpt-4o"
EVAL_MODEL = "gpt-4o-mini"  # "gpt-4o"
TEMPORAL_MODEL = "gpt-4o"
DATASET_TYPE = {"vatex": "non-temporal"} 
ACCEPTED_DATASETS = list(DATASET_TYPE.keys()) 
DEPTH_RELATIONS = ["closer to the camera than", "farther from the camera than"]  # because 'in front of' and 'behind' are not always clear
SPACE_RELATIONS = ["left of", "right of", "above", "below"]
ACCEPTED_SPATIAL_RELATIONS = DEPTH_RELATIONS + SPACE_RELATIONS
SPATIAL_RELATIONS_REVERSE = {
    "closer to the camera than": "farther from the camera than",
    "farther from the camera than": "closer to the camera than",
    "left of": "right of",
    "right of": "left of",
    "above": "below",
    "below": "above"
}
STR_TO_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
BAD_ATTRS = ['male', 'man', 'men', 'female', 'woman', 'women', 'boy', 'girl', 'young', 'old', 'teenage', 'child', 'adult', 'elderly']
if os.path.exists("../utils/kinetics_600_classes.txt"):
    with open("../utils/kinetics_600_classes.txt") as f:
        kinetics_classes = f.read().splitlines()
else:
    kinetics_classes = []
DATASET_TO_ACTIONS = {
    "vatex": kinetics_classes
}
POSSIBLE_TRANSITIONS = ['pan', 'pull out', 'tilt', 'zoom'] 

# New where scene is first.
def get_response_format(action_list):
    predicate_dict = { "type": "string" , "enum": action_list} if action_list else { "type": "string" }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "video_caption_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "scenes": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": "Short description of the scene."
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Name of the scene."
                                },
                                "style": {
                                    "type": "string",
                                    "description": "Style of the scene, such as the way it is shot."
                                },
                                "transition": {
                                    "type": "string",
                                    "description": "The best way to transition from the previous scene to this scene. If the scene is the first scene, write null."
                                },
                                "background": {
                                    "type": "string",
                                    "description": "Background of the scene."
                                },
                                "locations": {
                                    "type": "object",
                                    "additionalProperties": {
                                        "type": "string",
                                        "description": "Each object's location in relation to the background. Must be a valid object that will also be placed in the objects dict."
                                    }
                                }
                            },
                            "required": ["description", "name", "background", "style", "transition", "locations"],
                            "additionalProperties": False
                        },
                    },
                    "objects": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "count": {
                                    "type": "object",
                                    "properties": {
                                        "total_count": { "type": ["integer", "string"] },
                                        "count_by_scene": {
                                            "type": "object",
                                            "properties": {
                                                "scene": { "type": ["integer", "string"] }
                                            },
                                            "description": "Number of objects in each scene. If the count is not specified, use 'several'."
                                        }
                                    },
                                },
                                "is_human": {
                                    "type": "boolean",
                                    "description": "Whether the object is a human."
                                },
                                "attributes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "attribute": { "type": "string" },
                                            "id": { "type": "array", "items": { "type": "integer" } },
                                            "scenes": { "type": "array", "items": { "type": "string" } }
                                        }
                                    },
                                    "description": "List of adjectives describing the object, in order of appearance. If the count is greater than 1 and the attributes apply to only some of the objects, specify the objects by their id. These ids range from 1 to the count of the object."
                                },
                                "general_relations": {
                                    "type": ["array", "null"],
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "relation": { "type": "string" },
                                            "id": { "type": "array", "items": { "type": "integer" } },
                                            "target": { "type": "string" },
                                            "target_id": { "type": "array", "items": { "type": "integer" } },
                                            "scenes": { "type": "array", "items": { "type": "string" } }
                                        }
                                    },
                                    "description": "Relations between this object and others (e.g., 'inside', 'beside') in temporal order. target must be a valid object that will also be placed in the objects dict. These relations are location/interaction based. To be specific, these should all be relative positions. Null if unspecified."
                                },
                                "spatial_relations": {
                                    "type": ["array", "null"],
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "relation": { "type": "string" , "enum": ACCEPTED_SPATIAL_RELATIONS},
                                            "id": { "type": "array", "items": { "type": "integer" } },
                                            "target": { "type": "string" },
                                            "target_id": { "type": "array", "items": { "type": "integer" } },
                                            "scenes": { "type": "array", "items": { "type": "string" } }
                                        }
                                    },
                                    "description": "Spatial relations such as 'left of', 'right of', 'above', 'below', etc. target must be a valid object that will also be placed in the objects dict. If the object has a count > 1, it can have spatial relations with itself. Null if unspecified."
                                },
                            },
                            "required": ["count", "is_human", "attributes", "general_relations", "spatial_relations"],
                            "additionalProperties": False
                        }
                    },
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": { "type": "string" },
                                "subject_id": { "type": "array", "items": { "type": "integer" } },
                                "predicate": predicate_dict,
                                "predicate_id": { "type": "array", "items": { "type": "integer" } },
                                "object": { "type": ["string", "null"] },
                                "descriptor": { "type": ["string", "null"] },
                                "scenes": { "type": "array", "items": { "type": "string" } }
                            },
                            "required": ["subject", "predicate", "object", "descriptor", "scenes"]
                        },
                        "description": "List of actions with subject, predicate, object, and descriptor. Actions should be verbs of what subjects are doing to themselves or other objects. If the subject is doing the action to itself, write null for the object. Each action should be placed in a separate tuple if it is done by different objects. The descriptor describes an action, e.g., 'the dog is quickly running to the end of the street' is (dog, running, None, quickly, ...). The subject and object MUST be objects in the objects dict. If something should be included but is not an object, include it in the descriptor. If the subject or object count > 1 and only some of the subjects are doing the action/objects are having the action done to them, specify the subjects/objects by their id. These ids range from 1 to the count of the subject/object."
                    },
                },
                "required": ["scenes", "objects", "actions"],
                "additionalProperties": False,
                "strict": True
            }
        }
    }
    return response_format
