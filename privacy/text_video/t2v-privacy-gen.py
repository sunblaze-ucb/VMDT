import argparse
from json import load
import time
from pathlib import Path

import torch

from models import T2VModelName, t2v_model_list, load_t2v_model

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

    args.out_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv("")  ### TODO: Replace with https://huggingface.co/datasets/mmfm-trust/T2V/tree/main/privacy
    prompts = df["video_id"]
    
    for model_name in args.models:
        if model_name == 'VideoCrafter2':
            torch.cuda.empty_cache()
            model_out_dir = args.out_dir / model_name
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
            model_out_dir = args.out_dir / model_name
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
            model_out_dir = args.out_dir / (model_name)
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
