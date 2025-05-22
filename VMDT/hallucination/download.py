from pathlib import Path
import sys
from huggingface_hub import hf_hub_download, snapshot_download
import subprocess

from models.opensora.wrapper import OpenSoraWrapper


def download_video_crafter2(out_dir: Path):
    ckpt_out_path = out_dir / "model.ckpt"
    if not ckpt_out_path.exists():
        hf_hub_download(
            repo_id="VideoCrafter/VideoCrafter2",
            filename="model.ckpt",
            local_dir=ckpt_out_path.parent,
            local_dir_use_symlinks=False,
        )


def download_hunyuan(out_dir: Path):
    """
    huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
    huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava-llama-3-8b-v1_1-transformers
    python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ckpts/text_encoder

    huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2
    """
    ckpt_out_dir = out_dir / "ckpts"
    # video model
    if not (ckpt_out_dir / "hunyuan-video-t2v-720p").exists():
        snapshot_download(
            repo_id="tencent/HunyuanVideo",
            local_dir=ckpt_out_dir,
            local_dir_use_symlinks=False,
        )

    # MLLM
    if not (ckpt_out_dir / "llava-llama-3-8b-v1_1-transformers").exists():
        snapshot_download(
            repo_id="xtuner/llava-llama-3-8b-v1_1-transformers",
            local_dir=ckpt_out_dir / "llava-llama-3-8b-v1_1-transformers",
            local_dir_use_symlinks=False,
        )

    if not (ckpt_out_dir / "text_encoder").exists():
        subprocess.run(
            [
                sys.executable,
                "models/hunyuan/repo/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py",
                "--input_dir",
                str(ckpt_out_dir / "llava-llama-3-8b-v1_1-transformers"),
                "--output_dir",
                str(ckpt_out_dir / "text_encoder"),
            ]
        )

    # CLIP
    if not (ckpt_out_dir / "text_encoder_2").exists():
        snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=ckpt_out_dir / "text_encoder_2",
            local_dir_use_symlinks=False,
        )


def download_vchitect2(out_dir: Path):
    out_dir = out_dir / "vchitect2"
    if not out_dir.exists():
        snapshot_download(
            repo_id="Vchitect/Vchitect-2.0-2B",
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )


def download_opensora():
    # TODO: figure out which repos to cache
    OpenSoraWrapper(device="cpu")
    


if __name__ == "__main__":
    download_video_crafter2(Path("models/video_crafter/cache"))
    download_hunyuan(Path("models/hunyuan/cache"))
    download_vchitect2(Path("models/vchitect2/cache"))
    download_opensora()

