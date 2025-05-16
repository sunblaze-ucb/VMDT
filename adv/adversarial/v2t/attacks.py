import argparse
import json
from pathlib import Path
import uuid
import warnings
import traceback

from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

from adversarial.v2t.v2t_utils import get_mc_format, get_video_path
from adversarial.v2t.models.internvideo2 import InternVideo2Chat8B
from adversarial.v2t.models.videochat2 import VideoChat2
from adversarial.v2t.models.videollava import VideoLlava

class FMMAttack:
    def __init__(self, model, step_size, iters, lambda1, lambda2, lambda3, delta_max, max_grad_norm, device, dtype):
        self.model = model

        self.step_size = step_size
        self.iters = iters
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        self.delta_max = delta_max
        self.max_grad_norm = max_grad_norm
        
        self.device = device
        self.dtype = eval(dtype)
    
    def compute_l12_norm(self, delta):
        return torch.norm(delta, p=2, dim=(1, 2, 3)).sum()
    
    def compute_video_loss(self, clean_features, adv_features):
        return F.mse_loss(clean_features, adv_features)
    
    def compute_llm_loss(self, clean_hidden_state, adv_hidden_state):
        return F.mse_loss(clean_hidden_state, adv_hidden_state)
        
    def transform(self, prompt, frames):
        # frames: (T, C, H, W)
        
        frames = frames.to(self.device, self.dtype)
        
        delta = 0.001 * torch.randn_like(frames, device=self.device, dtype=self.dtype)
        delta.requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=self.step_size)
        
        with torch.no_grad():
            clean_features = self.model.get_visual_features(frames)
            # print(f"{clean_features.shape=}")
            clean_hidden_state = self.model.get_last_hidden_state(prompt, clean_features)
            # print(f"{clean_hidden_state.shape=}")
            
        for step in range(self.iters):
            adv_frames = frames + delta
            adv_frames = torch.clamp(adv_frames, 0, 1)
            
            adv_features = self.model.get_visual_features(adv_frames)
            adv_hidden_state = self.model.get_last_hidden_state(prompt, adv_features)
            
            norm = self.compute_l12_norm(delta)
            video_loss = self.compute_video_loss(clean_features, adv_features)
            llm_loss = self.compute_llm_loss(clean_hidden_state, adv_hidden_state)
            
            # print(f"{(self.lambda1 * norm)=}")
            # print(f"{(self.lambda2 * video_loss)=}")
            # print(f"{(self.lambda3 * llm_loss)=}")
            
            loss = (self.lambda1 * norm) - (self.lambda2 * video_loss) - (self.lambda3 * llm_loss)
            # print(f"{loss=}\n")
            if torch.isnan(loss):
                print(f"NaN loss at step {step}.")
                break
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([delta], max_norm=self.max_grad_norm)
            optimizer.step()
            
            with torch.no_grad():
                delta.clamp_(-self.delta_max, self.delta_max)
        
        adv_frames = frames + delta
        adv_frames = torch.clamp(adv_frames, 0, 1)
        
        return adv_frames.detach().cpu().float()
    
def load_model(model_name, device, dtype):
    if model_name == "InternVideo2Chat8B":
        return InternVideo2Chat8B(device, dtype)
    elif model_name == "VideoChat2":
        return VideoChat2(device, dtype)
    elif model_name == "VideoLLaVA":
        return VideoLlava(device, dtype)
    else:
        raise ValueError(f"Model {model_name} not found.")
    
def get_lambdas(model_name):
    if model_name == "InternVideo2Chat8B":
        return 0.01, 1.0, 0.1 # locked
    elif model_name == "VideoChat2":
        return 0.01, 1.0, 0.05 # locked
    elif model_name == "VideoLLaVA":
        return 0.01, 0.5, 1.0 # locked
    else:
        raise ValueError(f"Model {model_name} not found.")
    
def get_attack(model, args):
    if args.attack == "FMMAttack":
        return FMMAttack(model, args.step_size, args.iters, args.lambda1, args.lambda2, args.lambda3, args.delta_max, args.max_grad_norm, args.device, args.dtype)
    elif args.attack == "VLMAttack":
        # TODO
        pass
    else:
        raise ValueError(f"Attack {args.attack} not found.")
    
def gen_id(suffix=""):
    return uuid.uuid4().hex + suffix

def save_frames(frames, output_dir):
    frames_id = gen_id()
    output_path = Path(output_dir) / frames_id
    output_path.mkdir(parents=True, exist_ok=True) 
    
    torch.save(frames, output_path / "frames.pt")

    frames = frames.permute(0, 2, 3, 1).numpy()
    for i, frame in enumerate(frames):
        img = Image.fromarray((frame * 255).astype('uint8'))
        img.save(output_path / f"{i}.png")

    return frames_id

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    # model arguments
    argparser.add_argument("--model_name", type=str, required=True)
    
    # data arguments
    argparser.add_argument("--prompt_file", type=str, required=True)
    argparser.add_argument("--task", type=str, required=True)
    argparser.add_argument("--max_prompts", type=int, default=1000)
    argparser.add_argument("--output_file", type=str, required=True)
    argparser.add_argument("--frames_output_dir", type=str, default="v2t_frames/")
    
    # general attack arguments
    argparser.add_argument("--attack", type=str, required=True, choices=["FMMAttack", "VLMAttack"])
    
    # fmm attack arguments
    argparser.add_argument("--iters", type=int, default=100)
    argparser.add_argument("--step_size", type=float, default=0.01)
    argparser.add_argument("--lambda1", type=float, default=None)
    argparser.add_argument("--lambda2", type=float, default=None)
    argparser.add_argument("--lambda3", type=float, default=None)
    argparser.add_argument("--delta_max", type=float, default=0.03)
    argparser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # vlm attack arguments
    # TODO, feel free to move any args shared between attacks to the general attack args
    
    # other arguments
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--dtype", type=str, default="torch.bfloat16")
    
    args = argparser.parse_args()
    
    if args.lambda1 is None or args.lambda2 is None or args.lambda3 is None:
        args.lambda1, args.lambda2, args.lambda3 = get_lambdas(args.model_name)
    
    model = load_model(args.model_name, args.device, args.dtype)

    with open(args.prompt_file, "r") as f:
        instances = json.load(f)
    instances = [instance for instance in instances if instance["task_name"] == args.task]
    instances = instances[:args.max_prompts]
        
    attack = get_attack(model, args)
    
    for instance in tqdm(instances):
        
        try:
            prompt, _, _ = get_mc_format(instance["question"], instance["answer_choices"], instance["gt"])

            video_path = get_video_path(instance["video_id"], instance["task_name"])
            video = model.load_video(video_path)
            frames = model.select_frames(video)
            clean_frames_id = save_frames(frames, args.frames_output_dir)
            
            adv_frames = attack.transform(prompt, frames)
            adv_frames_id = save_frames(adv_frames, args.frames_output_dir)
            
            with open(args.output_file, "a") as f:
                f.write(json.dumps({
                    "id": gen_id(),
                    "source": instance["dataset"],
                    "video_id": instance["video_id"],
                    "surrogate_model": args.model_name,
                    "task": instance["task_name"],
                    "attack": args.attack,
                    "question": instance["question"],
                    "answer_choices": instance["answer_choices"],
                    "gt": instance["gt"],
                    "clean_frames_id": clean_frames_id,
                    "adv_frames_id": adv_frames_id,
                }) + "\n")
                
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            continue