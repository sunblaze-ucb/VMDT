import os
import sys
from pathlib import Path

from ..base import T2VBaseModel, T2VOutput, gen_id

MODEL_ROOT = Path(__file__).parent
MODEL_CKPT_PATH = MODEL_ROOT / "cache/ckpts"
# workaround for importing from the repo
sys.path.append(str(MODEL_ROOT / "repo"))
os.environ["HUNYUAN_MODEL_BASE"] = str(MODEL_CKPT_PATH)

from .repo.hyvideo.config import parse_args
from .repo.hyvideo.inference import HunyuanVideoSampler
from .repo.hyvideo.utils.file_utils import save_videos_grid


class HunyuanVideo(T2VBaseModel):
    def load_model(
        self,
        save_fps: int = 24,
        ulysses_degree: int = 1,
        ring_degree: int = 1,
    ):
        self.use_parallel = "LOCAL_RANK" in os.environ

        args_lst = [
            "--flow-reverse",
            "--model-base",
            str(MODEL_CKPT_PATH),
            "--dit-weight",
            str(
                MODEL_CKPT_PATH
                / "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
            ),
            "--ulysses-degree",
            str(ulysses_degree),
            "--ring-degree",
            str(ring_degree),
        ]

        if not self.use_parallel:
            args_lst += ["--use-cpu-offload"]

        args = parse_args(args_lst)
        self.video_sampler = HunyuanVideoSampler.from_pretrained(
            str(MODEL_CKPT_PATH), args=args
        )
        self.args = self.video_sampler.args
        self.save_fps = save_fps

    def generate_videos(
        self, text_inputs: list[str], output_dir: Path, batch_size=1, **gen_kwargs
    ) -> list[T2VOutput]:
        outputs = []
        for i in range(0, len(text_inputs), batch_size):
            batch_text_inputs = text_inputs[i : i + batch_size]
            outputs += self._generate_videos_batch(
                batch_text_inputs, output_dir, **gen_kwargs
            )
        return outputs

    def _generate_videos_batch(
        self,
        batch_text_inputs: list[str],
        output_dir: Path,
        width: int = 1280,
        height: int = 720,
        n_frames=129,
        steps=50,
        seed=None,
    ) -> list[T2VOutput]:
        assert len(batch_text_inputs) == 1, "Only batch_size=1 is supported"
        outputs = self.video_sampler.predict(
            prompt=batch_text_inputs[0],
            height=height,
            width=width,
            video_length=n_frames,
            seed=seed,
            infer_steps=steps,
            guidance_scale=self.args.cfg_scale,
            num_videos_per_prompt=1,
            flow_shift=self.args.flow_shift,
            batch_size=1,
            embedded_guidance_scale=self.args.embedded_cfg_scale,
        )

        if not self.use_parallel or int(os.environ["LOCAL_RANK"]) == 0:
            sample = outputs["samples"][0].unsqueeze(0)
            out_path = output_dir / gen_id(suffix=".mp4")
            save_videos_grid(sample, str(out_path), fps=self.save_fps)

        return [
            T2VOutput(
                text_input=batch_text_inputs[0],
                video_path=out_path,
            )
        ]
