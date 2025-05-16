import json
import uuid
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def write_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
            
def append_jsonl(item, file_path):
    with open(file_path, "a") as f:
        f.write(json.dumps(item) + "\n")
        
def gen_id(suffix=""):
    return uuid.uuid4().hex + suffix