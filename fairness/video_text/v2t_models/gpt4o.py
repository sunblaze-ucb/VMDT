import cv2, os, time, base64
from moviepy.editor import *
from openai import OpenAI 

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

    
    # Extract audio from video
    clip = VideoFileClip(video_path)
    if clip.audio!=None:
        audio_path = f"{base_video_path}.mp3"
        clip.audio.write_audiofile(audio_path, bitrate="32k", logger=None)
        clip.audio.close()        
    else:
        audio_path=None
    clip.close()

    #print(f"Extracted {len(base64Frames)} frames")
    #print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path

def generate_audio_transcript(client,audio_path):
    if audio_path==None:
        return None
    #transcribe the audio
    with open(audio_path, 'rb') as audio_file:
        audio_content = base64.b64encode(audio_file.read()).decode('utf-8')

    for _ in range(3):
        try:
            response = client.chat.completions.create(
                        model='gpt-4o-audio-preview-2024-12-17',
                        modalities=["text"],
                        messages=[
                                {   "role": "system", 
                                    "content":"You are generating a transcript. Create a transcript of the provided audio."
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        { 
                                            "type": "text",
                                            "text": "this is the audio."
                                        },
                                        {
                                            "type": "input_audio",
                                            "input_audio": {
                                                "data": audio_content,
                                                "format": "mp3"
                                            }
                                        }
                                    ]
                                },
                            ],
                        temperature=0,
                    )

            # Extract and return the transcription
            transcription = response.choices[0].message.content
            if len(transcription)>0:
                os.remove(audio_path)
                return transcription
        except:
            time.sleep(1)
    return None

class gpt4o:
    def __init__(self) -> None:
        self.client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.max_trial=3
    def generate_response(self,video_path,prompt,seconds_per_frame=1,audio_support=False):
        base64Frames, audio_path = process_video(video_path, seconds_per_frame=seconds_per_frame)  # Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate
        transcription=None
        for _ in range(self.max_trial):
            try:
                if (audio_path!=None) and (transcription==None) and audio_support==True:
                    transcription=generate_audio_transcript(self.client,audio_path)
                    response = self.client.chat.completions.create(
                            model='gpt-4o-2024-11-20',
                            messages=[
                                    {"role": "user", "content": [
                                        {"type": "text", "text": "These are the frames from the video."},
                                        *map(lambda x: {"type": "image_url", 
                                                       "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
                                        {"type": "text", "text": f"The audio transcription is: {transcription}\n\n{prompt}"}
                                        ],
                                    }
                            ],
                                temperature=0,
                            )
                else:
                    transcription=None
                    response = self.client.chat.completions.create(
                            model='gpt-4o-2024-11-20',
                            messages=[
                                    {"role": "user", "content": [
                                        {"type": "text", "text": "These are the frames from the video."},
                                        *map(lambda x: {"type": "image_url", 
                                                        "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
                                        {"type": "text", "text": prompt}
                                        ],
                                    }
                            ],
                                temperature=0,
                            )
                
                output=response.choices[0].message.content
                if len(output)>0:
                    return output
            except Exception as e:
                print(e)
                if 'rejected as a result of our safety system' in str(e):
                    return 'error'
                time.sleep(1)
            
    def generate_response_textonly(self,prompt):
        for _ in range(self.max_trial):
            try:
                response = self.client.chat.completions.create(
                        model='gpt-4o-2024-11-20',
                        messages=[
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt}
                                    ],
                                }
                        ],
                            temperature=0,
                        )
                
                output=response.choices[0].message.content
                if len(output)>0:
                    return output
            except Exception as e:
                print(e)
                if 'rejected as a result of our safety system' in str(e):
                    return 'error'
                time.sleep(1)
