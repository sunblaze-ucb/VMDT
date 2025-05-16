import base64
import logging
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic_core import to_json as pydantic_to_json


def to_json(obj, indent=2) -> bytes:
    return pydantic_to_json(obj, indent=indent)


def save_json(obj, path: Path):
    with open(path, "wb") as f:
        f.write(to_json(obj))


def save_jsonl(objs, path: Path):
    with open(path, "wb") as f:
        for obj in objs:
            f.write(to_json(obj, indent=None) + b"\n")


def init_logger(
    logfile: Path | None = None,
    debug_loggers: list[str] | None = None,
    warning_loggers: list[str] | None = None,
    log_level: str = "INFO",
):
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers = []

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    handlers.append(sh)

    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        handlers.append(fh)

    logging.basicConfig(level=log_level, handlers=handlers)

    if debug_loggers:
        for logger_name in debug_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)

    if warning_loggers:
        for logger_name in warning_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)


def image_to_base64(img: Image.Image, format="jpeg") -> str:
    img_bytes = BytesIO()
    img.save(img_bytes, format=format)
    base64_str = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return base64_str


def sample_frames(video_path: Path, num_frames: int = 5):
    # Load the video
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError("Error opening video file")

    # Get total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate indices for equally spaced frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    sampled_frames = []

    for idx in frame_indices:
        # Set the video position to the specific frame
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)

        # Read the frame
        success, frame = video.read()
        if not success:
            print(f"Warning: Unable to read frame at index {idx}")
            continue

        # Convert frame (OpenCV image) to RGB and then to PIL.Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Append the PIL image to the list
        sampled_frames.append(pil_image)

    # Release the video capture object
    video.release()

    return sampled_frames
