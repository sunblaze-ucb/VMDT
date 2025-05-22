import asyncio
import argparse
from pathlib import Path

from .t2v_eval import main

RESULT_DIR = Path(__file__).parents[3].absolute() / "results"

if __name__ == "__main__":
    assert RESULT_DIR.exists(), f"RESULT_DIR {RESULT_DIR} does not exist"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        choices=[
            "Nova",
            "Pika",
            "Luma",
            "OpenSora1.2",
            "Vchitect2",
            "VideoCrafter2",
            "CogVideoX-5B",
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
