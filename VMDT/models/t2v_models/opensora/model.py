from pathlib import Path
from typing import Literal

from ..base import T2VBaseModel, T2VOutput, gen_id
from ..model_name import T2VModelName
from .wrapper import OpenSoraWrapper

MODEL_ROOT = Path(__file__).parent


class OpenSora(T2VBaseModel):
    def load_model(self, num_frames: Literal["4s", "8s"] = "4s"):
        self.opensora = OpenSoraWrapper(
            cmd_args=[
                str(MODEL_ROOT / "repo/v1_2_config_sample.py"),
                "--num-frames",
                num_frames,
                "--resolution",
                "720p",
                "--aspect-ratio",
                "9:16",
                "--prompt",
                "NULL",  # dummy prompt
            ]
        )

    def generate_videos(
        self, text_inputs: list[str], output_dir: Path, batch_size: int = 2, **kwargs
    ) -> list[T2VOutput]:
        str_paths = self.opensora.run_inference(
            prompts=text_inputs,
            save_dir=output_dir,
            batch_size=batch_size,
        )

        return [
            T2VOutput(
                text_input=text_input,
                video_path=Path(str_path),
            )
            for text_input, str_path in zip(text_inputs, str_paths)
        ]
