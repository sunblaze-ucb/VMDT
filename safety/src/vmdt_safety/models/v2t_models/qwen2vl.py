import logging
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from .base import V2TBaseModel, V2TOutput

logger = logging.getLogger(__name__)


class Qwen2VL(V2TBaseModel):
    def _batch_size(self):
        return 1

    def _max_frames(self):
        if "72B" in self.model_name:
            return 384
        return 768

    def load_model(self, **kwargs):
        """
        Load the Qwen2VL model.
        The caller may provide a 'model_id' via load_kwargs;
        otherwise, self.model_name is used.
        """
        model_id = kwargs.get("model_id", self.model_name)
        if "Qwen2.5" in model_id:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate_texts(
        self,
        video_inputs: list[Path],
        prompts: list[str],
        **gen_kwargs,
    ) -> list[V2TOutput]:
        """
        For each video file and its corresponding prompt, generate the text output.
        Additional generation kwargs (e.g. max_new_tokens, do_sample) can be passed via gen_kwargs.
        """
        outputs = []
        max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
        do_sample = gen_kwargs.get("do_sample", False)
        temperature = gen_kwargs.get("temperature", 0)

        batch_size = self._batch_size()

        total_examples = len(video_inputs)
        for i in range(0, total_examples, batch_size):
            # Prepare mini-batches
            batch_video_inputs = video_inputs[i : i + batch_size]
            batch_prompts = prompts[i : i + batch_size]

            # Build a list of message dictionaries for this batch
            messages_list = []
            for video_path, prompt in zip(batch_video_inputs, batch_prompts):
                message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{str(video_path.absolute())}",
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
                messages_list.append(message)

            # Generate chat prompts for each conversation in the mini-batch.
            chat_prompts = [
                self.processor.apply_chat_template(
                    [msg], tokenize=False, add_generation_prompt=True
                )
                for msg in messages_list
            ]

            # Process vision inputs from the mini-batch.
            image_inputs, video_inputs_proc = process_vision_info(messages_list)
            print(f"[*] {len(video_inputs_proc)} video inputs")
            print(f"[*] {video_inputs_proc[0].shape}")
            print(f"[*] {batch_video_inputs[0]}")

            # for idx, v in enumerate(video_inputs_proc):
            #     if v.shape[0] > self._max_frames():
            #         print(
            #             f"[*] Truncating video from {v.shape[0]} to {self._max_frames()} frames"
            #         )

            #         step = v.shape[0] / self._max_frames()
            #         video_inputs_proc[idx] = torch.stack(
            #             [v[int(i * step)] for i in range(self._max_frames())]
            #         )

            # Prepare the processor inputs for the mini-batch.
            inputs = self.processor(
                text=chat_prompts,
                images=image_inputs,
                videos=video_inputs_proc,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: generate output tokens in batch.
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

            # Remove the input prompt tokens from the generated output.
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Wrap the results in V2TOutput objects.
            for video_path, text in zip(batch_video_inputs, output_texts):
                outputs.append(V2TOutput(video_input=video_path, text_output=text))
        return outputs


# ----------------------------
# Optional testing block
# ----------------------------
if __name__ == "__main__":
    test_video = Path("/home/fpinto1/v2t_mmdt_video/models/test_dis.mp4")
    test_prompt = "What is the video about?"
    model = Qwen2VL("Qwen/Qwen2-VL-7B-Instruct")
    results = model.generate_texts([test_video], [test_prompt], max_new_tokens=4096)
    for res in results:
        if res.error:
            print(f"Error processing {res.video_input}: {res.error}")
        else:
            print(f"Video: {res.video_input}\nResponse: {res.text_output}")
