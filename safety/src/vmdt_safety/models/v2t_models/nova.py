import base64
import hashlib
import json
import os
import time
from pathlib import Path

import boto3
import botocore

from .base import (
    V2TBaseModel,
    V2TError,
    V2TOutput,
)

MAX_VIDEO_SIZE = 10 * 1024 * 1024  # 10 MB


class Nova(V2TBaseModel):
    def load_model(self, **kwargs):
        """Load the Bedrock Runtime client for AWS."""
        self.model_id = kwargs.get("model_id")
        region = kwargs.get("region", "us-east-1")
        self.bucket = os.getenv("AWS_S3_BUCKET")
        self.s3_prefix = os.getenv("AWS_S3_PREFIX")
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=botocore.config.Config(read_timeout=600),
        )
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    def generate_texts(
        self, video_inputs: list[Path], prompts: list[str], **gen_kwargs
    ) -> list[V2TOutput]:
        """Generate text descriptions for video inputs."""
        results = []
        for video_path, prompt in zip(video_inputs, prompts):
            try:
                text_output = self._process_video(video_path, prompt)
                results.append(
                    V2TOutput(video_input=video_path, text_output=text_output)
                )
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append(
                    V2TOutput(video_input=video_path, error=V2TError.GENERATION_ERROR)
                )
        return results

    def _process_video(self, video_path: Path, prompt: str) -> str:
        if video_path.stat().st_size > MAX_VIDEO_SIZE:
            return self._process_video_s3(video_path, prompt)
        else:
            return self._process_video_base64(video_path, prompt)

    def _process_video_s3(self, video_path: Path, prompt: str) -> str:
        """Handle video processing and request to AWS Bedrock."""
        bucket = self.bucket
        s3_file_path = self._get_s3_file_path(video_path, self.s3_prefix)

        if not self._check_file_exists(s3_file_path):
            self._send_to_s3(str(video_path), s3_file_path)

        source = {
            "s3Location": {
                "uri": f"s3://{bucket}/{s3_file_path}",
            }
        }

        message_list = [
            {
                "role": "user",
                "content": [
                    {"video": {"format": "mp4", "source": source}},
                    {"text": prompt},
                ],
            }
        ]

        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "inferenceConfig": {
                "maxTokens": 500,
                "topP": 0.9,
                "topK": 20,
                "temperature": 0,
            },
        }

        return self._invoke_model(request_body)

    def _process_video_base64(self, video_path: Path, prompt: str) -> str:
        """Handle video processing and request to AWS Bedrock using base64 encoding."""
        with open(video_path, "rb") as video_file:
            binary_data = video_file.read()
            base_64_encoded_data = base64.b64encode(binary_data)
            base64_string = base_64_encoded_data.decode("utf-8")

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
                        "text": prompt,
                    },
                ],
            }
        ]

        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "inferenceConfig": {
                "maxTokens": 500,
                "topP": 0.9,
                "topK": 20,
                "temperature": 0,
            },
        }

        return self._invoke_model(request_body)

    def _invoke_model(self, request_body: dict) -> str:
        """Invoke the AWS Bedrock model and handle response."""
        for _ in range(3):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                )
                model_response = json.loads(response["body"].read())
                return model_response["output"]["message"]["content"][0]["text"]
            except Exception as e:
                print(f"Model invocation error: {e}")
                time.sleep(1)
        raise RuntimeError("Model invocation failed after 3 attempts.")

    def _check_file_exists(self, s3_file_path: str) -> bool:
        """Check if a file exists in an S3 bucket."""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_file_path)
            return True
        except Exception:
            return False

    def _send_to_s3(self, file_path: str, s3_file_path: str):
        """Upload a file to an S3 bucket."""
        self.s3_client.upload_file(file_path, self.bucket, s3_file_path)

    @staticmethod
    def _get_s3_file_path(file_path: Path, s3_prefix: str) -> str:
        # use the file name and md5 hash to create a unique file path
        file_name = file_path.name
        file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
        return f"{s3_prefix}{file_name}.{file_hash}"
