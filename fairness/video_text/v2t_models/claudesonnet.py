import os, cv2
from moviepy.editor import *
import base64, time
from anthropic import Anthropic

def process_video(video_path, seconds_per_frame=1):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    #print(f"Extracted {len(base64Frames)} frames")
    #print(f"Extracted audio to {audio_path}")
    return base64Frames

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class claudesonnet:
    def __init__(self) -> None:
        self.model = Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.max_trial=3

    def generate_response(self,video_path,prompt,seconds_per_frame=1,**kwargs):
        #image_path=encode_image(f"paired_UTKFace/{image_path}")

        for _ in range(self.max_trial):
            try:
                base64Frames = process_video(video_path, seconds_per_frame=seconds_per_frame) 
                message = self.model.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    temperature= 0,
                    system= "You MUST follow the instruction, replying in JSON format!!!", #for decision making and stereotype 
                    messages=[
                                {"role": "user", "content": [
                                    {"type": "text", "text": "These are the frames from the video."},
                                    *map(lambda x: {"type": "image", 
                                                    "source": {"type": "base64","media_type": "image/jpeg","data": x}}, base64Frames),
                                    {"type": "text", "text": prompt}
                                    ],
                                }
                            ],
                )
                print(message.content[0].text)
                return message.content[0].text
            except Exception as e:
                print(e)
                time.sleep(3)
                if 'Request exceeds the maximum size' in str(e):
                    seconds_per_frame+=0.5
        return None