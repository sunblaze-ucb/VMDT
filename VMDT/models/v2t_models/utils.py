import base64
import cv2
from pathlib import Path


def process_video(video_path: Path, seconds_per_frame: int = 1, max_frames: int = 50):
    base64_frames = []
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = max(
        int(fps * seconds_per_frame),
        total_frames // max_frames if max_frames < total_frames else 1,
    )
    curr_frame = 0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    return base64_frames
