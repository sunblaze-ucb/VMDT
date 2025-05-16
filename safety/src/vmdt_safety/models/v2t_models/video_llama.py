import argparse
from pathlib import Path

import torch

from .base import V2TBaseModel, V2TOutput
from .videollama2 import mm_infer, model_init
from .videollama2.utils import disable_torch_init  # if needed


class VideoLlama(V2TBaseModel):
    def load_model(self, **kwargs):
        """
        Load the VideoLlama2 model. The caller may supply a 'model_id' via load_kwargs;
        otherwise, self.model_name is used.
        """
        model_id = kwargs.get("model_id", self.model_name)
        self.device = "cuda"
        self.model_id = model_id
        # For video-to-text, we set the modality to video ("v")
        self.modal_type = "v"  # no audio for video-to-text

        # Initialize the model, processor, and tokenizer
        self.model, self.processor, self.tokenizer = model_init(model_id)
        # Disable the audio tower since we are working with video only.
        if self.modal_type == "a":
            self.model.model.vision_tower = None
        elif self.modal_type == "v":
            self.model.model.audio_tower = None
        elif self.modal_type == "av":
            pass
        else:
            raise NotImplementedError("Modal type not supported")
        self.model.eval()

    def generate_texts(
        self, video_inputs: list[Path], prompts: list[str], **gen_kwargs
    ) -> list[V2TOutput]:
        """
        For each video file (a Path) and its corresponding prompt, perform inference
        to generate text using VideoLlama2.

        Additional generation parameters (e.g. do_sample) may be passed via gen_kwargs.
        """
        outputs = []
        do_sample = gen_kwargs.get("do_sample", False)

        for video_path, prompt in zip(video_inputs, prompts):
            try:
                # Select the proper preprocessor based on modality.
                # For video mode ("v"), use the "video" preprocessor.
                preprocess = self.processor["video"]
                # Preprocess the video. Note that preprocess typically expects a string path.
                video_tensor = preprocess(str(video_path))

                # Perform inference via mm_infer.
                output = mm_infer(
                    video_tensor,
                    prompt,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    modal="audio" if self.modal_type == "a" else "video",
                    do_sample=do_sample,
                )
                outputs.append(V2TOutput(video_input=video_path, text_output=output))
            except Exception as e:
                outputs.append(V2TOutput(video_input=video_path, error=str(e)))
        return outputs


# Optional testing block:
if __name__ == "__main__":
    # For testing, use a sample video and prompt.
    test_video = Path("./test_dis.mp4")
    test_prompt = "What is the video about?"

    # Initialize the model. The model ID is passed here or can be provided via load_kwargs.
    client = VideoLlama("DAMO-NLP-SG/VideoLLaMA2.1-7B-AV")

    # Generate text output for the test video and prompt.
    results = client.generate_texts([test_video], [test_prompt], do_sample=False)
    for res in results:
        if res.error:
            print(f"Error processing {res.video_input}: {res.error}")
        else:
            print(f"Video: {res.video_input}\nResponse: {res.text_output}")
