import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import argparse


class VideoLlama2_1Client:
    def __init__(self, model_id):
        self.device = "cuda"
        self.model_id = model_id
        self.modal_type="v" #no audio for us
        self.model, self.processor, self.tokenizer = model_init(model_id)
        if self.modal_type == "a":
            self.model.model.vision_tower = None
        elif self.modal_type == "v":
            self.model.model.audio_tower = None
        elif self.modal_type == "av":
            pass
        else:
            raise NotImplementedError

        #self.model.to(self.device)

    def generate(self, video_path, question, do_sample=False ): # text, image_path, **kwargs):
        preprocess = self.processor['audio' if self.modal_type == "a" else "video"]
        if self.modal_type == "a":
            audio_video_tensor = preprocess(video_path)
        else:
            audio_video_tensor = preprocess(video_path) #va=True if modal_type == "av" else False)

        output = mm_infer(
            audio_video_tensor,
            question,
            model=self.model,
            tokenizer=self.tokenizer,
            modal='audio' if self.modal_type == "a" else "video",
            do_sample=do_sample,
        )

        return output


if __name__ == "__main__":
    client = VideoLlama2_1Client('DAMO-NLP-SG/VideoLLaMA2.1-7B-AV')
    output = client.generate('./test_dis.mp4', "What is the video about?")  # , 32)
    print(output)
