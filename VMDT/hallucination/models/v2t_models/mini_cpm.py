import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord


def encode_video(video_path, num_frames):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > num_frames:
        frame_idx = uniform_sample(frame_idx, num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames



class MiniCPMClient:
    def __init__(self, model_id):
        self.device = "cuda"
        self.model_id = model_id
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)



        #self.model.to(self.device)

    def generate(self, video_path, question,num_frames): # text, image_path, **kwargs):
        # Load the video as an np.arrau, sampling uniformly 8 frames
        #video_path = './test_dis.mp4'
        frames = encode_video(video_path, num_frames)
        msgs = [
            {'role': 'user', 'content': frames + [question]}, 
        ]

        # Set decode params for video
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution > 448*448

        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            **params
        )
        print(answer)



if __name__ == "__main__":
    client = MiniCPMClient(f'openbmb/MiniCPM-V-2_6')
    client.generate('./test_dis.mp4', "What is the video about?", 32)