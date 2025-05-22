import asyncio
import argparse
from pathlib import Path

from .v2t_eval import main

RESULT_DIR = Path(__file__).parents[3].absolute() / "results/v2t_results/safety"

if __name__ == "__main__":
    assert RESULT_DIR.exists(), f"RESULT_DIR {RESULT_DIR} does not exist"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        choices=[
            "Qwen2-VL-7B",
            "Qwen2.5-VL-3B",
            "Qwen2.5-VL-7B",
            "Qwen2.5-VL-72B",
            "InternVL2.5-1B",
            "InternVL2.5-2B",
            "InternVL2.5-4B",
            "InternVL2.5-8B",
            "InternVL2.5-26B",
            "InternVL2.5-38B",
            "InternVL2.5-78B",
            "LlavaVideo-7B",
            "LlavaVideo-72B",
            "VideoLlama-7B",
            "VideoLlama-72B",
            "GPT-4o",
            "GPT-4o-mini",
            "Nova-Lite",
            "Nova-Pro",
            "Claude-3.5-Sonnet",
        ],
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            raw_args=[
                "--out_dir",
                str(RESULT_DIR),
                "--models",
                args.model_id,
                "--model_batch_size",
                "1",
                "--judge_batch_size",
                "1",
                "--run_model",
                "--run_judge",
            ]
        )
    )
