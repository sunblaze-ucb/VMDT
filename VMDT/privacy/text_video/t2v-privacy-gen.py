import argparse
from json import load
import time
from pathlib import Path
import pandas as pd
import torch

from VDMT.models.t2v_models import T2VModelName, t2v_model_list, load_t2v_model

def generate_privacy_videos(
    out_dir: Path,
    models: list[T2VModelName] = None,
    prompts: list[str] = None,
    dataset_url: str = "https://huggingface.co/datasets/mmfm-trust/T2V/tree/main/privacy"
) -> None:
    """
    Generate videos for privacy evaluation using specified text-to-video models.
    
    Args:
        out_dir (Path): Directory to save generated videos
        models (list[T2VModelName], optional): List of models to use. Defaults to all available models.
        prompts (list[str], optional): List of prompts to generate videos from. If None, will load from dataset.
        dataset_url (str, optional): URL to load prompts from if not provided. Defaults to privacy dataset.
    """
    if models is None:
        models = t2v_model_list
        
    out_dir.mkdir(exist_ok=True, parents=True)
    
    if prompts is None:
        df = pd.read_csv(dataset_url)
        prompts = df["video_id"]
    
    for model_name in models:
        if model_name == 'VideoCrafter2':
            torch.cuda.empty_cache()
            model_out_dir = out_dir / model_name
            model_out_dir.mkdir(exist_ok=True, parents=True)

            print(f"[*] loading model {model_name}...")
            model = load_t2v_model(model_name)
            if len(prompts) > 3:
                for prompts_chunk in [prompts[i:i + 3] for i in range(0, len(prompts), 3)]:
                    print(f"[*] Generating videos for {model_name}...")
                    tic = time.perf_counter()
                    video_outputs = model.generate_videos(prompts_chunk, model_out_dir)
                    toc = time.perf_counter()
                    print(f"[+] Finished in {toc - tic:0.2f} seconds")

                    print(f"[+] Videos saved to {model_out_dir}")
                    for video_output in video_outputs:
                        print(f"[+] \t{video_output.video_path}")
            else:
                print(f"[*] Generating videos for {model_name}...")
                tic = time.perf_counter()
                video_outputs = model.generate_videos(prompts, model_out_dir)
                toc = time.perf_counter()
                print(f"[+] Finished in {toc - tic:0.2f} seconds")

                print(f"[+] Videos saved to {model_out_dir}")
                for video_output in video_outputs:
                    print(f"[+] \t{video_output.video_path}")
        elif model_name == "Luma" or model_name == "Nova":
            torch.cuda.empty_cache()
            model_out_dir = out_dir / model_name
            model_out_dir.mkdir(exist_ok=True, parents=True)

            print(f"[*] loading model {model_name}...")
            model = load_t2v_model(model_name)

            for prompt in prompts:
                print(f"[*] Generating videos for {model_name}...")
                tic = time.perf_counter()
                video_outputs = model.generate_videos([prompt], model_out_dir)
                toc = time.perf_counter()
                print(f"[+] Finished in {toc - tic:0.2f} seconds")

                print(f"[+] Videos saved to {model_out_dir}")
                for video_output in video_outputs:
                    print(f"[+] \t{video_output.video_path}")
        else:
            model_out_dir = out_dir / model_name
            model_out_dir.mkdir(exist_ok=True, parents=True)

            print(f"[*] loading model {model_name}...")
            model = load_t2v_model(model_name)

            print(f"[*] Generating videos for {model_name}...")
            tic = time.perf_counter()
            video_outputs = model.generate_videos(prompts, model_out_dir)
            toc = time.perf_counter()
            print(f"[+] Finished in {toc - tic:0.2f} seconds")

            print(f"[+] Videos saved to {model_out_dir}")
            for video_output in video_outputs:
                print(f"[+] \t{video_output.video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("out"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=T2VModelName,
        choices=t2v_model_list,
        default=t2v_model_list,
    )
    args = parser.parse_args()
    
    generate_privacy_videos(out_dir=args.out_dir, models=args.models) 