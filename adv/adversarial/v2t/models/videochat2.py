from typing import Any, Dict, Tuple
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_video
from transformers import AutoModel, StoppingCriteria, StoppingCriteriaList
from easydict import EasyDict

### From https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/dataset/hd_utils.py.
def HD_transform_padding(frames, image_size=224, hd_num=6):
    def _padding_224(frames):
        _, _, H, W = frames.shape
        tar = int(np.ceil(H / 224) * 224)
        top_padding = (tar - H) // 2
        bottom_padding = tar - H - top_padding
        left_padding = 0
        right_padding = 0

        padded_frames = F.pad(
            frames, 
            pad=[left_padding, right_padding, top_padding, bottom_padding], 
            mode='constant', value=255
        )
        return padded_frames

    _, _, H, W = frames.shape
    trans = False
    if W < H:
        frames = frames.flip(-2, -1)
        trans = True
        width, height = H, W
    else:
        width, height = W, H

    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * image_size)
    new_h = int(new_w / ratio)

    resized_frames = F.interpolate(
        frames, size=(new_h, new_w), 
        mode='bicubic', 
        align_corners=False
    )
    padded_frames = _padding_224(resized_frames)

    if trans:
        padded_frames = padded_frames.flip(-2, -1)

    return padded_frames

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

def HD_transform_no_padding(frames, image_size=224, hd_num=6):
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    # calculate the existing video aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the frames
    resized_frame = F.interpolate(
        frames, size=(target_height, target_width),
        mode='bicubic', align_corners=False
    )
    return resized_frame

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    # print(f"n_position: {n_position}")
    # print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            # print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            # print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        # print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        # print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret

def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret

def get_context_emb(conv, model, img_list, device, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        # seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

class VideoChat2:
    
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = eval(dtype)
        
        self.model = AutoModel.from_pretrained("OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf", trust_remote_code=True, torch_dtype=self.dtype).to(device)
        self.model_name = self.get_model_name()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.num_frames_to_select = 8
        self.resolution = 224
        self.hd_num = 6
        self.padding = False
            
    @staticmethod
    def get_model_name():
        return "VideoChat2"
    
    def load_video(self, video_path, start_time=0, end_time=None) -> torch.Tensor:
        video, _, _ = read_video(video_path, start_pts=start_time, end_pts=end_time, pts_unit='sec', output_format="TCHW")
        video = video.float().div(255)
        return video
    
    def select_frames(self, video, num_frames_to_select=8):
        total_frames = video.size(0)
        
        seg_size = float(total_frames - 1) / num_frames_to_select
        start = int(seg_size / 2)
        
        indices = torch.tensor([
            start + int(np.round(seg_size * idx)) for idx in range(num_frames_to_select)
        ], dtype=torch.long)
        
        return video[indices]
    
    def transform_frames(self, frames):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        transform = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        
        frames = transform(frames)
        
        if self.padding:
            frames = HD_transform_padding(frames, image_size=self.resolution, hd_num=self.hd_num)
        else:
            frames = HD_transform_no_padding(frames, image_size=self.resolution, hd_num=self.hd_num)
            
        return frames
    
    def get_visual_features(self, frames, transform=True):
        if transform:
            frames = self.transform_frames(frames)
            
        new_pos_emb = get_sinusoid_encoding_table(n_position=(self.resolution//16)**2*self.num_frames_to_select, cur_frame=self.num_frames_to_select)
        self.model.vision_encoder.encoder.pos_embed = new_pos_emb
        frames = frames.unsqueeze(0)
        
        image_emb, _, _ = self.model.encode_img(frames, "Watch the video and answer the question.")
        visual_features = torch.cat(image_emb, dim=1)
        
        return visual_features
    
    def get_last_hidden_state(self, prompt, visual_features):
        
        chat = EasyDict({
            "system": "",
            "roles": ("[INST]", "[/INST]"),
            "messages": [],
            "sep": ""
        })
        
        chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
        chat.messages.append([chat.roles[0], prompt])
        chat.messages.append([chat.roles[1], None])
        
        formatted_prompt = get_prompt(chat)
        prompt_segs = formatted_prompt.split("<VideoHere>")
        if len(prompt_segs) != 2:
            raise ValueError("Formatted prompt must contain exactly one '<VideoHere>' placeholder.")
        
        seg0_tokens = self.model.mistral_tokenizer(
            prompt_segs[0], return_tensors="pt", add_special_tokens=True
        ).to(self.device).input_ids
        
        seg1_tokens = self.model.mistral_tokenizer(
            prompt_segs[1], return_tensors="pt", add_special_tokens=False
        ).to(self.device).input_ids
        
        seg0_embeds = self.model.mistral_model.model.embed_tokens(seg0_tokens)
        seg1_embeds = self.model.mistral_model.model.embed_tokens(seg1_tokens)
        
        inputs_embeds = torch.cat([seg0_embeds, visual_features, seg1_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=self.device)
        
        outputs = self.model.mistral_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        
        return outputs.hidden_states[-1]
    
    def generate(self, prompt, frames=None, video_path=None, start_time=0, end_time=None, num_frames_to_select=8):
        if video_path is not None:
            video = self.load_video(video_path, start_time, end_time)
            assert video is not None, "Video load failed"
            frames = self.select_frames(video, num_frames_to_select)
        else:
            assert frames is not None, "Either video_path or frames must be provided."
            
        frames = self.transform_frames(frames)
        frames = frames.to(self.device, dtype=self.dtype)
        
        new_pos_emb = get_sinusoid_encoding_table(n_position=(self.resolution//16)**2*num_frames_to_select, cur_frame=num_frames_to_select)
        self.model.vision_encoder.encoder.pos_embed = new_pos_emb
        frames = frames.unsqueeze(0)
        
        with torch.no_grad():
            image_emb, _, _ = self.model.encode_img(frames, "Watch the video and answer the question.")
            img_list = [image_emb[0]]
            
        stop_words_ids = [
            torch.tensor([2]).to(self.device),
            torch.tensor([29871, 2]).to(self.device)
        ]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        chat = EasyDict({
            "system": "",
            "roles": ("[INST]", "[/INST]"),
            "messages": [],
            "sep": ""
        })
        
        chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
        chat.messages.append([chat.roles[0], prompt])
        chat.messages.append([chat.roles[1], None])
        
        embs = get_context_emb(chat, self.model, img_list, self.device)
        
        with torch.no_grad():
            outputs = self.model.mistral_model.generate(
                inputs_embeds=embs,
                max_new_tokens=512,
                do_sample=False,
                stopping_criteria=stopping_criteria,
            )
        
        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        
        output_text = self.model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0]
        
        return output_text