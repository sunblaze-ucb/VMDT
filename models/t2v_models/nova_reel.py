# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json, random, boto3, time, os
from pathlib import Path
from tqdm import tqdm
import botocore
from botocore.credentials import CredentialProvider
from .base import T2VBaseModel, T2VOutput

class StaticCredentialProvider(CredentialProvider):
    def __init__(self, access_key, secret_key):
        self._access_key = access_key
        self._secret_key = secret_key

    def load(self):
        return boto3.Session().get_credentials().get_frozen_credentials()

class s3_download: 
    def __init__(self) -> None:  
        session = boto3.Session()
        credentials = StaticCredentialProvider(access_key=os.environ['ACCESS_KEY'], secret_key=os.environ['SECRET_KEY'])
        self.s3 = session.client('s3',aws_access_key_id=credentials._access_key,aws_secret_access_key=credentials._secret_key)
        self.BUCKET_NAME = 'bedrock-video-generation-us-east-1-1p02b2' # replace with your bucket name
        
    def download_files(self,s3_filename, output_filename):
        
        for _ in range(3):
            try:
                self.s3.download_file(self.BUCKET_NAME, s3_filename,output_filename)
                return True
            except botocore.exceptions.ClientError as e:
                print(e)
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist.")
                else:
                    time.sleep(3)
                    raise
        return False

class Nova_reel(T2VBaseModel):
    def load_model(self):
        # Create the Bedrock Runtime client.
        session = boto3.Session()
        credentials = StaticCredentialProvider(access_key=os.environ['ACCESS_KEY'], secret_key=os.environ['SECRET_KEY'])

        # Create the client with the custom session and credentials
        self.bedrock_runtime = session.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',  # Replace with your desired region
            aws_access_key_id=credentials._access_key,
            aws_secret_access_key=credentials._secret_key
        )
        self.s3_download=s3_download()

    def generate_videos(self,prompts: list[str],output_dir: Path):
        outputs=[]
        for prompt in tqdm(prompts):
            model_input = {
                "taskType": "TEXT_VIDEO",
                "textToVideoParams": {"text": prompt},
                "videoGenerationConfig": {
                    "durationSeconds": 6,
                    "fps": 24,
                    "dimension": "1280x720",
                    "seed": random.randint(0, 1000000)
                },
            }
            invocation_arn=None
            cnt=0
            flag=0
            for _ in range(3):
                try:
                    if invocation_arn==None:
                        # Start the asynchronous video generation job.
                        invocation = self.bedrock_runtime.start_async_invoke(
                            modelId="amazon.nova-reel-v1:0",
                            modelInput=model_input,
                            outputDataConfig={
                                "s3OutputDataConfig": {
                                    "s3Uri": "s3://bedrock-video-generation-us-east-1-1p02b2"
                                }
                            }
                        )
                        invocation_arn = invocation["invocationArn"]
                        job_id = invocation_arn.split("/")[-1]
                        s3_location = f"s3://bedrock-video-generation-us-east-1-1p02b2/{job_id}"
                    
                    time.sleep(60)
                    while True:
                        response = self.bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)
                        status = response["status"]
                        if status != "InProgress":
                            break
                        time.sleep(15)

                    if status == "Completed":
                        print(f"\nVideo is ready at {s3_location}/output.mp4, Prompt: {prompt}")
                        self.s3_download.download_files(f"{job_id}/output.mp4", f"{output_dir}/{job_id}.mp4")
                        outputs.append(
                            T2VOutput(
                                text_input=prompt,
                                video_path=output_dir /f'{job_id}.mp4',
                            )
                        )
                    else:
                        message=response["failureMessage"]
                        print(f"\nVideo generation status: {status}, Reason: {message}, Prompt: {prompt}")
                        
                        outputs.append(
                            T2VOutput(
                                text_input=prompt,
                                video_path=output_dir /'error',
                            )
                        )
                    break

                except Exception as e:
                    print(e)
                    cnt+=1
                    if 'blocked by our content filters' in str(e): 
                        print(f"\nVideo generation status: error, Prompt: {prompt}")
                        flag=1
                        outputs.append(
                            T2VOutput(
                                text_input=prompt,
                                video_path=output_dir / 'error',
                            )
                        )
                        break
                    time.sleep(3)
            if cnt==3 and flag==0:
                print(f"\nVideo generation status: error, Prompt: {prompt}")
                outputs.append(
                            T2VOutput(
                                text_input=prompt,
                                video_path=output_dir / 'error',
                            )
                        )
        return outputs