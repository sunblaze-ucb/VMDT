import json
from tqdm import tqdm

with open("combined-selected_CogVideoX-5b.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
    id_to_cog5b = {d["id"]: d["eval_results"]["CogVideoX-5b"] for d in data}

with open("combined-selected_luma.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
    id_to_luma = {d["id"]: d["eval_results"]["Luma"] for d in data}

with open("combined-selected_OpenSora1.2.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
    id_to_opensora = {d["id"]: d["eval_results"]["OpenSora1.2"] for d in data}

with open("combined-selected_Vchitect2.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
    id_to_vchitect = {d["id"]: d["eval_results"]["Vchitect2"] for d in data}

with open("combined-selected_VideoCrafter2.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
    id_to_videocrafter = {d["id"]: d["eval_results"]["VideoCrafter2"] for d in data}

with open("../t2v/combined-selected.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

for instance in data:
    instance["eval_results"]["CogVideoX-5b"] = id_to_cog5b[instance["id"]]
    instance["eval_results"]["Luma"] = id_to_luma[instance["id"]]
    instance["eval_results"]["OpenSora1.2"] = id_to_opensora[instance["id"]]
    instance["eval_results"]["Vchitect2"] = id_to_vchitect[instance["id"]]
    instance["eval_results"]["VideoCrafter2"] = id_to_videocrafter[instance["id"]]

with open("combined-selected-qwen.jsonl", "w") as f:
    for instance in tqdm(data):
        f.write(json.dumps(instance) + "\n")
    print("Merge done!")