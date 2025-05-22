import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

SAFETY_ROOT = Path(__file__).parent.absolute()
DATA_DIR = SAFETY_ROOT / "data"

# download videos and prompts
snapshot_download(
    repo_id="Zhaorun/SafeWatch-Bench",
    repo_type="dataset",
    allow_patterns=["real/videos/**", "genai/videos/**"],
    local_dir=str(DATA_DIR / "safewatch")
)

snapshot_download(
    repo_id="mmfm-trust/V2T",
    repo_type="dataset",
    allow_patterns=["safety/**"],
    local_dir=str(DATA_DIR / "v2t_data")
)

with open(SAFETY_ROOT / "video_mapping.json") as f:
    video_mapping = json.load(f)

(DATA_DIR / "v2t_data/vid").mkdir(exist_ok=True)

for k, v in video_mapping.items():
    from_path = DATA_DIR / v
    to_path = DATA_DIR / "v2t_data/safety" / k
    shutil.move(from_path, to_path)

shutil.rmtree(DATA_DIR / "safewatch")