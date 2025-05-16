import os
import random
import time
from pathlib import Path

import boto3
import botocore
from botocore.credentials import CredentialProvider
from tqdm import tqdm

from .base import T2VBaseModel, T2VOutput, T2VError


class StaticCredentialProvider(CredentialProvider):
    def __init__(self, access_key, secret_key):
        self._access_key = access_key
        self._secret_key = secret_key

    def load(self):
        return boto3.Session().get_credentials().get_frozen_credentials()


class S3Download:
    def __init__(self, credentials) -> None:
        session = boto3.Session()
        self.s3 = session.client(
            "s3",
            aws_access_key_id=credentials._access_key,
            aws_secret_access_key=credentials._secret_key,
        )
        self.BUCKET_NAME = "{prefix}/{name}".format(
            prefix=os.getenv("AWS_S3_PREFIX"),
            name=os.getenv("AWS_S3_BUCKET"),
        )

    def download_files(self, s3_filename, output_filename):
        for _ in range(3):
            try:
                self.s3.download_file(self.BUCKET_NAME, s3_filename, output_filename)
                return True
            except botocore.exceptions.ClientError as e:
                print(e)
                if e.response["Error"]["Code"] == "404":
                    print("The object does not exist.")
                else:
                    time.sleep(3)
                    raise
        return False


class NovaReel(T2VBaseModel):
    def load_model(self, **kwargs):
        session = boto3.Session()
        credentials = StaticCredentialProvider(
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self.bedrock_runtime = session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=credentials._access_key,
            aws_secret_access_key=credentials._secret_key,
        )
        self.s3_download = S3Download(credentials)

    def _generate_single_video(
        self, prompt: str, output_dir: Path, max_trials: int = 3
    ) -> T2VOutput:
        for trial in range(max_trials):
            try:
                model_input = {
                    "taskType": "TEXT_VIDEO",
                    "textToVideoParams": {"text": prompt},
                    "videoGenerationConfig": {
                        "durationSeconds": 6,
                        "fps": 24,
                        "dimension": "1280x720",
                        "seed": random.randint(0, 1000000),
                    },
                }
                invocation_arn = None
                # Start the asynchronous video generation job.
                invocation = self.bedrock_runtime.start_async_invoke(
                    modelId="amazon.nova-reel-v1:1",
                    modelInput=model_input,
                    outputDataConfig={
                        "s3OutputDataConfig": {
                            "s3Uri": f"s3://{self.s3_download.BUCKET_NAME}/",
                        }
                    },
                )
                invocation_arn = invocation["invocationArn"]
                job_id = invocation_arn.split("/")[-1]
                s3_location = f"s3://{self.s3_download.BUCKET_NAME}/{job_id}"
                print(f"\nMonitoring job folder: {s3_location}")

                time.sleep(60)
                while True:
                    response = self.bedrock_runtime.get_async_invoke(
                        invocationArn=invocation_arn
                    )
                    status = response["status"]
                    if status != "InProgress":
                        break
                    time.sleep(15)

                if status == "Completed":
                    print(f"\nVideo is ready at {s3_location}/output.mp4")
                    local_path = output_dir / f"{job_id}.mp4"
                    self.s3_download.download_files(
                        f"{job_id}/output.mp4", str(local_path)
                    )
                    return T2VOutput(
                        text_input=prompt,
                        video_path=local_path,
                    )
                else:
                    message = response.get("failureMessage", "Unknown error")
                    print(f"\nVideo generation status: {status}\nReason: {message}")
                    # Check for safety refusal
                    if "blocked by our content filters" in message:
                        return T2VOutput(
                            text_input=prompt,
                            error=T2VError.SAFETY_REFUSAL,
                        )
                    return T2VOutput(
                        text_input=prompt,
                        error=message or T2VError.GENERATION_ERROR,
                    )
            except Exception as e:
                print(f"Attempt {trial + 1} failed: {e}")
                # Final failure
                if "blocked by our content filters" in str(e):
                    return T2VOutput(
                        text_input=prompt,
                        error=T2VError.SAFETY_REFUSAL,
                    )
                if trial < max_trials - 1:
                    time.sleep(3)
                    continue
                return T2VOutput(
                    text_input=prompt,
                    error=str(e) or T2VError.GENERATION_ERROR,
                )

    def generate_videos(
        self, text_inputs: list[str], output_dir: Path, max_trials: int = 3, **kwargs
    ) -> list[T2VOutput]:
        outputs = []
        for prompt in tqdm(text_inputs):
            output = self._generate_single_video(prompt, output_dir, max_trials)
            outputs.append(output)
        return outputs
