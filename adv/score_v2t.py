import json
import os

with open("data/v2t/combined-selected.jsonl", "r") as f:
    v2t_data = [json.loads(line) for line in f]
    
models = ['InternVL2_5-1B', 'InternVL2_5-2B', 'InternVL2_5-4B', 'InternVL2_5-8B', 'InternVL2_5-26B', 'InternVL2_5-38B', 'InternVL2_5-78B', 'Qwen2.5-VL-3B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'Qwen2.5-VL-72B-Instruct', 'LLaVA-Video-7B-Qwen2', 'LLaVA-Video-72B-Qwen2', 'VideoLLaMA2.1-7B-AV', 'VideoLLaMA2-72B', 'NovaLite', 'NovaPro', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-11-20', 'claudesonnet']

surrogate_models = ["InternVideo2Chat8B", "VideoChat2", "VideoLLaVA"]

### by task
# print("################## By Task ##################")
# for task in ["ActionRecognition", "AttributeRecognition", "Counting", "ObjectRecognition", "SpatialUnderstanding"]:
#     print(f"Task: {task}")
#     task_data = [x for x in v2t_data if x["task"] == task]
#     d = {model: {"clean_score": 0, "adv_score": 0, "count": 0} for model in models}
#     for instance in task_data:
#         for model in instance["eval_results"].keys():
#             if model in surrogate_models:
#                 continue
#             d[model]["clean_score"] += instance["eval_results"][model]["clean_score"] or 0
#             d[model]["adv_score"] += instance["eval_results"][model]["adv_score"] or 0
#             d[model]["count"] += 1

#     for model in d.keys():
#         if d[model]["count"] > 0:
#             d[model]["clean_score"] /= d[model]["count"]
#             d[model]["adv_score"] /= d[model]["count"]

#         print(f"{model}:")
#         print(f"clean_score: {d[model]['clean_score']}")
#         print(f"adv_score: {d[model]['adv_score']}")
#         print(f"count: {d[model]['count']}")
#         print()
        
### by surrogate model
# print("################## By Surrogate Model ##################")
# for surrogate_model in surrogate_models:
#     print(f"Surrogate Model: {surrogate_model}")
#     surrogate_data = [x for x in v2t_data if x["surrogate_model"] == surrogate_model]
#     d = {model: {"clean_score": 0, "adv_score": 0, "count": 0} for model in models}
#     for instance in surrogate_data:
#         for model in instance["eval_results"].keys():
#             if model in surrogate_models:
#                 continue
#             d[model]["clean_score"] += instance["eval_results"][model]["clean_score"] or 0
#             d[model]["adv_score"] += instance["eval_results"][model]["adv_score"] or 0
#             d[model]["count"] += 1

#     for model in d.keys():
#         if d[model]["count"] > 0:
#             d[model]["clean_score"] /= d[model]["count"]
#             d[model]["adv_score"] /= d[model]["count"]

#         print(f"{model}:")
#         print(f"clean_score: {d[model]['clean_score']}")
#         print(f"adv_score: {d[model]['adv_score']}")
#         print(f"count: {d[model]['count']}")
#         print()
        
### overall
# print("################## Overall ##################")
d = {model: {"clean_score": 0, "adv_score": 0, "count": 0} for model in models}
for instance in v2t_data:
    for model in instance["eval_results"].keys():
        if model in surrogate_models:
            continue
        d[model]["clean_score"] += instance["eval_results"][model]["clean_score"] or 0
        d[model]["adv_score"] += instance["eval_results"][model]["adv_score"] or 0
        d[model]["count"] += 1

for model in d.keys():
    if d[model]["count"] > 0:
        d[model]["clean_score"] /= d[model]["count"]
        d[model]["adv_score"] /= d[model]["count"]

    print(f"{model}:")
    print(f"clean_score: {d[model]['clean_score']}")
    print(f"adv_score: {d[model]['adv_score']}")
    print(f"count: {d[model]['count']}")
    print()