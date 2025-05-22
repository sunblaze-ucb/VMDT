import requests, time, random
from pathlib import Path
from tqdm import tqdm
from .base import T2VBaseModel, T2VOutput, gen_id

def download_video(url, save_path):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in binary write mode
        with open(save_path, 'wb') as file:
            # Write the content of the response to the file
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Video successfully downloaded to {save_path}")
    else:
        print(f"Failed to download video. Status code: {response.status_code}")

class Pika(T2VBaseModel):
    def load_model(self):
        self.api=""
        self.video_id=None
    def generate_videos(self,prompts: list[str],output_dir: Path):
        outputs=[]
        for prompt in tqdm(prompts):
            # prompt = "A doctor"
            # print("This is my prompt:", prompt)
            cnt=0
            flag=0
            self.video_id=None
            for _ in range(3):
                try:
                    if self.video_id==None:
                        url = "https://devapi.pika.art/generate/2.2/t2v"
                        payload = {
                            "promptText": prompt,
                            "seed": random.randint(0, 100000),
                            "duration": 5,
                            "resolution": "720p"
                        }
                        headers = {
                            "X-API-KEY": self.api,
                            "Content-Type": "application/x-www-form-urlencoded",
                            "Accept": "application/json"
                        }
                        response = requests.post(url, data=payload, headers=headers)
                        # print("This is the response json:", response.json())
                        # print("This is the response:", response)
                        # print("This is the response json:", response.json())
                        if response.status_code==200:
                            self.video_id=response.json()["video_id"]
                        elif 'message' in response.json() and 'Your API key has reached its generate limit' in response.json()['message']:
                            print(response.json())
                            return outputs

                    completed=False
                    time.sleep(30)
                    url = f"https://devapi.pika.art/videos/{self.video_id}"
                    headers = {
                        "X-API-KEY": self.api,
                        "Accept": "application/json"
                    }
                    while not completed:
                        response = requests.get(url, headers=headers)
                        if response.json()['status']=='finished':
                            completed=True
                        else:
                            time.sleep(5)

                    url_link=response.json()['url']
                    save_path = output_dir / "6to7" / gen_id(prefix="1" + "_", suffix="_" + "".join(val[0].lower() if (len(val) > 0 and val[0].isalnum()) else "" for val in prompt.split(" ")) + ".mp4")
                    # download_video(url_link, f'{output_dir}/{self.video_id}.mp4')
                    download_video(url_link, save_path)
                    
                    outputs.append(
                        T2VOutput(
                            text_input=prompt,
                            # video_path=output_dir /f'{self.video_id}.mp4',
                            video_path=save_path,
                        )
                    )
                    break
                except Exception as e:
                    cnt+=1
                    if 'prompt not allowed' in str(e): 
                        flag=1
                        outputs.append(
                            T2VOutput(
                                text_input=prompt,
                                video_path=output_dir / 'error',
                            )
                        )
                        break
                    print(e)
                    time.sleep(1)
            if cnt==3 and flag==0:
                outputs.append(
                            T2VOutput(
                                text_input=prompt,
                                video_path=output_dir / 'error',
                            )
                        )
        return outputs