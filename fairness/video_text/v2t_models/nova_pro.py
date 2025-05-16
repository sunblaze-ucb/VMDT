# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import boto3, base64
import json, time, cv2, tempfile, os
from moviepy.editor import VideoFileClip

def cut_video(input_path, start_time, end_time):
    """
    Cuts a video between specified start and end times.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the cut video.
        start_time (int or float): Start time in seconds.
        end_time (int or float): End time in seconds.
    """    
    video_clip = VideoFileClip(input_path)
    cut_clip = video_clip.subclip(start_time, end_time)
    output_path='temp2.mp4'
    cut_clip.write_videofile(output_path, verbose=False, logger=None)
    return output_path

def get_video_base64(video_path, max_duration=15):
    # First check video duration
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        # Get duration in seconds
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        if duration > max_duration:
            # Video is too long, use trimmed version
            temp_video_path=cut_video(video_path, start_time=0, end_time=max_duration)
            with open(temp_video_path, "rb") as video_file:
                binary_data = video_file.read()
                base_64_encoded_data = base64.b64encode(binary_data)
                base64_string = base_64_encoded_data.decode("utf-8")
            os.remove(temp_video_path)
            return base64_string
        else:
            # Video is short enough, use original method
            with open(video_path, "rb") as video_file:
                binary_data = video_file.read()
                base_64_encoded_data = base64.b64encode(binary_data)
                base64_string = base_64_encoded_data.decode("utf-8")
            return base64_string
    else:
        raise ValueError("Could not open video file")
    
class NovaPro:
    def __init__(self) -> None:
        # Create a Bedrock Runtime client in the AWS Region of your choice.
        self.client = boto3.client(service_name='bedrock-runtime',
                    region_name='us-east-1',  # Replace with your desired region
                    aws_access_key_id=os.environ.get("AWS_KEY"),
                    aws_secret_access_key=os.environ.get("AWS_KEY2"))

        self.MODEL_ID = "us.amazon.nova-pro-v1:0"

    def generate_response(self,video_path,prompt,**kwargs):
        # Open the image you'd like to use and encode it as a Base64 string.
        base64_string = get_video_base64(video_path)
        # Define your system prompt(s).
        '''system_list= [
            {
                "text": "You are an expert media analyst. When the user provides you with a video, provide 3 potential video titles"
            }
        ]'''
        # Define a "user" message including both the image and a text prompt.
        message_list = [
            {
                "role": "user",
                "content": [
                    {
                        "video": {
                            "format": "mp4",
                            "source": {"bytes": base64_string},
                        }
                    },
                    {
                        "text": prompt
                    },
                ],
            }
        ]

        # Configure the inference parameters.
        inf_params = {"max_new_tokens": 500, "top_p": 0.9, "top_k": 20, "temperature": 0}

        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            #"system": system_list,
            "inferenceConfig": inf_params,
        }

        for _ in range(3):
            try:
                # Invoke the model and extract the response body.
                response = self.client.invoke_model(modelId=self.MODEL_ID, body=json.dumps(request_body))
                model_response = json.loads(response["body"].read())

                content_text = model_response["output"]["message"]["content"][0]["text"]
                return content_text
            except Exception as e:
                print(e)
                time.sleep(1)

    def generate_response_textonly(self,prompt):
        
        # Define a "user" message including both the image and a text prompt.
        message_list = [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    },
                ],
            }
        ]

        # Configure the inference parameters.
        inf_params = {"max_new_tokens": 500, "top_p": 0.9, "top_k": 20, "temperature": 0}

        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "inferenceConfig": inf_params,
        }

        for _ in range(3):
            try:
                # Invoke the model and extract the response body.
                response = self.client.invoke_model(modelId=self.MODEL_ID, body=json.dumps(request_body))
                model_response = json.loads(response["body"].read())

                content_text = model_response["output"]["message"]["content"][0]["text"]
                return content_text
            except Exception as e:
                print(e)
                time.sleep(1)

#model=NovaPro()
#print(model.generate_response('../decision_making/_0bg1TLPP-I.000.mp4', 'The video has audio information. Can you understand that?'))