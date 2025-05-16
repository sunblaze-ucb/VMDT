import boto3, time
import botocore
from botocore.credentials import CredentialProvider

class StaticCredentialProvider(CredentialProvider):
    def __init__(self, access_key, secret_key):
        self._access_key = access_key
        self._secret_key = secret_key

    def load(self):
        return boto3.Session().get_credentials().get_frozen_credentials()

class s3_download: 
    def __init__(self) -> None:  
        session = boto3.Session()
        credentials = StaticCredentialProvider(access_key='YOUR ACCESS KEY', secret_key='SECRET KEY')
        self.s3 = session.client('s3',aws_access_key_id=credentials._access_key,aws_secret_access_key=credentials._secret_key)
        self.BUCKET_NAME = 'bedrock-video-generation-us-east-1-1p02b2' # replace with your bucket name
        
    def download_files(self,s3_filename, output_filename):
        
        for _ in range(3):
            try:
                self.s3.download_file(self.BUCKET_NAME, s3_filename,output_filename)
                return True
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist.")
                    return False
                else:
                    time.sleep(3)
                    raise
