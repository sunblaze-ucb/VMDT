import random
from adversarial.common.utils import read_jsonl, write_jsonl
from adversarial.t2v.t2v_utils import T2VInstance

def by_task(x, task):
    return x.task == task

def by_attack(x, attack):
    return x.attack == attack

def by_surrogate(x, surrogate):
    return x.surrogate_model == surrogate

def merge_scores(instance1, instance2, method="avg"):
    """Merge scores using the specified aggregation method (avg, min, max)."""
    merged_instance = instance1.copy()
    
    for model in instance1.eval_results.keys():
        clean_score1 = instance1.eval_results[model].clean_score
        adv_score1 = instance1.eval_results[model].adv_score
        clean_score2 = instance2.eval_results[model].clean_score
        adv_score2 = instance2.eval_results[model].adv_score

        # Skip this model if any score is None
        if any(score is None for score in [clean_score1, adv_score1, clean_score2, adv_score2]):
            merged_instance.eval_results[model].clean_score = None
            merged_instance.eval_results[model].adv_score = None
            continue

        if method == "avg":
            merged_clean = (clean_score1 + clean_score2) / 2
            merged_adv = (adv_score1 + adv_score2) / 2
        elif method == "min":
            merged_clean = min(clean_score1, clean_score2)
            merged_adv = min(adv_score1, adv_score2)
        elif method == "max":
            merged_clean = max(clean_score1, clean_score2)
            merged_adv = max(adv_score1, adv_score2)
        else:
            raise ValueError("Invalid aggregation method. Use 'avg', 'min', or 'max'.")

        merged_instance.eval_results[model].clean_score = merged_clean
        merged_instance.eval_results[model].adv_score = merged_adv

    return merged_instance

def compute_scores(data):
    d = {
        "CogVideoX-2b": {"clean_score": 0, "adv_score": 0, "count": 0},
        "mochi-1-preview": {"clean_score": 0, "adv_score": 0, "count": 0},
        "VideoCrafter2": {"clean_score": 0, "adv_score": 0, "count": 0},
        "CogVideoX-5b": {"clean_score": 0, "adv_score": 0, "count": 0},
        "OpenSora1.2": {"clean_score": 0, "adv_score": 0, "count": 0},
        "Vchitect2": {"clean_score": 0, "adv_score": 0, "count": 0},
        "Luma": {"clean_score": 0, "adv_score": 0, "count": 0},
        "Nova_reel": {"clean_score": 0, "adv_score": 0, "count": 0},
        "Pika": {"clean_score": 0, "adv_score": 0, "count": 0},
    }

    for instance in data:
        for model in instance.eval_results.keys():
            clean_score = instance.eval_results[model].clean_score
            adv_score = instance.eval_results[model].adv_score
            
            # Only include scores if both clean and adv scores are present
            if clean_score is not None and adv_score is not None:
                d[model]["clean_score"] += clean_score
                d[model]["adv_score"] += adv_score
                d[model]["count"] += 1

    for model in d.keys():
        if model in ["CogVideoX-2b", "mochi-1-preview"]:
            continue
        
        if d[model]["count"] > 0:
            d[model]["clean_score"] /= d[model]["count"]
            d[model]["adv_score"] /= d[model]["count"]

        
        print(f"{model}:")
        print(f"clean_score: {d[model]['clean_score']}")
        print(f"adv_score: {d[model]['adv_score']}")
        print(f"count: {d[model]['count']}")
        print()
    
data_gpt = read_jsonl("data/t2v/combined-selected-new.jsonl")
data_gpt = [T2VInstance.parse_obj(d) for d in data_gpt]

data_qwen = read_jsonl("data/t2v_qwen/combined-selected-qwen-new.jsonl")
data_qwen = [T2VInstance.parse_obj(d) for d in data_qwen]

if not [x.id for x in data_gpt] == [x.id for x in data_qwen]:
    data_gpt = sorted(data_gpt, key=lambda x: x.id)
    data_qwen = sorted(data_qwen, key=lambda x: x.id)
    assert [x.id for x in data_gpt] == [x.id for x in data_qwen], "ID mismatch between files!"

data = [merge_scores(d1, d2, method="avg") for d1, d2 in zip(data_gpt, data_qwen)]

print(f"*********************By Task*********************")
for task in ["ActionRecognition", "AttributeRecognition", "Counting", "ObjectRecognition", "SpatialUnderstanding"]:
    print(f"Task: {task}")
    task_data = [x for x in data if by_task(x, task) and by_attack(x, "gcg")]
    compute_scores(task_data)
print("Overall")
compute_scores(data)

print(f"*********************By Attack*********************")
for attack in ["greedy", "genetic", "gcg"]:
    print(f"Attack: {attack}")
    attack_data = [x for x in data if by_attack(x, attack)]
    compute_scores(attack_data)

print(f"*********************Surrogate*********************")
for surrogate in ["CogVideoX-2b", "mochi-1-preview"]:
    print(f"Surrogate: {surrogate}")
    surrogate_data = [x for x in data if by_surrogate(x, surrogate)]
    compute_scores(surrogate_data)