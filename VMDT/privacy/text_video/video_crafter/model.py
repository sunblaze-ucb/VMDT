from pathlib import Path

from omegaconf import OmegaConf
import torch

from ..base import T2VBaseModel, T2VOutput, gen_id
from .repo.scripts.evaluation.funcs import (
    batch_ddim_sampling,
    load_model_checkpoint,
    save_videos,
)
from .repo.utils.utils import instantiate_from_config

MODEL_ROOT = Path(__file__).parent
DEFAULT_MODEL_PATH = MODEL_ROOT / "cache/model.ckpt"
DEFAULT_CONFIG_PATH = MODEL_ROOT / "repo/configs/inference_t2v_512_v2.0.yaml"


class VideoCrafter(T2VBaseModel):
    def load_model(
        self,
        ckpt_path: Path = DEFAULT_MODEL_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
        save_fps: int = 10,
    ):
        config = OmegaConf.load(config_path)
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model = instantiate_from_config(model_config)
        model = load_model_checkpoint(model, ckpt_path)
        self.model = model.eval().cuda()
        self.save_fps = save_fps

    def generate_videos(
        self, text_inputs: list[str], output_dir: Path, batch_size=5
    ) -> list[T2VOutput]:
        outputs = []
        for i in range(0, len(text_inputs), batch_size):
            batch_text_inputs = text_inputs[i : i + batch_size]
            outputs += self._generate_videos_batch(batch_text_inputs, output_dir)
        return outputs

    def _generate_videos_batch(
        self,
        batch_text_inputs: list[str],
        output_dir: Path,
        steps=50,
        cfg_scale=12.0,
        eta=1.0,
        fps=28,
    ) -> list[T2VOutput]:
        batch_size = len(batch_text_inputs)
        channels = self.model.model.diffusion_model.in_channels
        frames = self.model.temporal_length
        h, w = 320 // 8, 512 // 8
        noise_shape = [batch_size, channels, frames, h, w]

        # text cond
        text_emb = self.model.get_learned_conditioning(batch_text_inputs)
        cond = {
            "c_crossattn": [text_emb],
            "fps": torch.tensor([fps] * batch_size).to(self.model.device).long(),
        }

        ## inference
        batch_samples = batch_ddim_sampling(
            self.model,
            cond,
            noise_shape,
            n_samples=1,
            ddim_steps=steps,
            ddim_eta=eta,
            cfg_scale=cfg_scale,
        )

        filenames = [gen_id(prefix=str(idx) + "", suffix="_" + "".join(val[0].lower() if (len(val) > 0 and val[0].isalnum()) else "" for val in input.split(" "))) for idx, input in enumerate(batch_text_inputs)]

        save_videos(batch_samples, output_dir, filenames=filenames, fps=self.save_fps)

        outputs = [
            T2VOutput(text_input=text, video_path=output_dir / f"{filename}.mp4")
            for text, filename in zip(batch_text_inputs, filenames)
        ]

        return outputs
