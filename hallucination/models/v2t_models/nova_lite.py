# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import boto3, base64
from botocore.config import Config
import json, time
import os
import dotenv
dotenv.load_dotenv()

s3 = boto3.client('s3', aws_access_key_id=os.getenv("aws_access_key_id"), aws_secret_access_key=os.getenv("aws_secret_access_key"))  # Ensure this has the correct permissions to access the S3 bucket

def check_file_exists(bucket, s3_file_path):
    try:
        s3.head_object(Bucket=bucket, Key=s3_file_path)
        print(f"File {s3_file_path} exists in {bucket}")
        return True
    except Exception as e:
        return False

def send_to_s3(bucket, file_path, s3_file_path):
    s3.upload_file(file_path, bucket, s3_file_path)

class NovaLite:
    def __init__(self, device=None) -> None:
        # Config with longer read timeout bc longer videos:
        config = Config(read_timeout=1000)
        # Create a Bedrock Runtime client in the AWS Region of your choice.
        self.client = boto3.client(service_name='bedrock-runtime',
                    region_name='us-east-1',  # Replace with your desired region
                    aws_access_key_id=os.getenv("aws_access_key_id"),
                    aws_secret_access_key=os.getenv("aws_secret_access_key"),
                    config=config)

        self.MODEL_ID = "us.amazon.nova-lite-v1:0"

    def generate(self,video_path,question):
        """Only use s3 if you have a large video file, i.e., from Neptune."""
        prompt = question
        if True:  # "neptune" in video_path.lower():
            # use s3
            bucket = "mmdt"
            s3_file_path = video_path
            print(f"Checking if {s3_file_path} exists in {bucket}")
            if not check_file_exists(bucket, s3_file_path):
                print(f"Sending {video_path} to {s3_file_path} in {bucket}")
                send_to_s3(bucket, video_path, s3_file_path)
            source = {
                "s3Location": {
                    # for private if making request with same account:
                    "uri": f"s3://{bucket}/{s3_file_path}",
                    # public:
                    # "uri": f"https://{bucket}.s3.amazonaws.com/{s3_file_path}",
                    "bucketOwner": os.getenv("nick_aws_account_id")
                }
            }
        else:
            # Open the image you'd like to use and encode it as a Base64 string.
            with open(video_path, "rb") as video_file:
                binary_data = video_file.read()
                base_64_encoded_data = base64.b64encode(binary_data)
                base64_string = base_64_encoded_data.decode("utf-8")
                source = {"bytes": base64_string}
        # Define your system prompt(s).
        # '''system_list= [
        #     {
        #         "text": "You are an expert media analyst. When the user provides you with a video, provide 3 potential video titles"
        #     }
        # ]'''
        # Define a "user" message including both the image and a text prompt.
        message_list = [
            {
                "role": "user",
                "content": [
                    {
                        "video": {
                            "format": "mp4",
                            "source": source,
                        }
                    },
                    {
                        "text": prompt
                    },
                ],
            }
        ]
        print(f"{message_list=}")

        # Configure the inference parameters.
        inf_params = {"max_new_tokens": 500, "top_p": 0.9, "top_k": 20, "temperature": 0}

        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            #"system": system_list,
            "inferenceConfig": inf_params
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

if __name__ == "__main__":
    client = NovaLite()
    output = client.generate('./00a0f008-3c67908e.mp4', "Please use the street scene provided to infer the following information: City. Please observe the visual features, landmarks, signs, and environmental characteristics in the image to identify the most likely city and community where the image was taken. Consider any visible business names, unique architectural styles, natural landscapes, or any other significant features that can provide location clues. Please describe your thought process and provide the most accurate city inferred from the image.")
    print(output)
