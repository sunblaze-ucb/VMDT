import os
import torch
import clip
from PIL import Image
import copy
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
import cv2
import itertools
import pickle

from VDMT.models.t2v_models import T2VModelName, t2v_model_list

def show_image_plt(np_array):
	"""
	Display numpy array as image using matplotlib

	Args:
		np_array: numpy array of shape (H, W, 3)
	"""
	plt.figure(figsize=(10, 8))
	plt.imshow(np_array)
	plt.axis('off')  # Hide axes
	plt.show()

def cal_l2_distance(ori_img, gen_img):
	gen_img = np.resize(gen_img, (512, 512, 3))
	ori_img = np.resize(ori_img, (512, 512, 3))

	l2_distance= np.linalg.norm(np.array(ori_img) - np.array(gen_img))
	return l2_distance

def cal_clip_embedding_distance(model,  preprocess,  device, ori_img, gen_img):
	ori_pil = Image.fromarray(ori_img.astype('uint8'))
	gen_pil = Image.fromarray(gen_img.astype('uint8'))

	# Preprocess images
	ori_processed = preprocess(ori_pil).unsqueeze(0).to(device)
	gen_processed = preprocess(gen_pil).unsqueeze(0).to(device)

	with torch.no_grad():
		# Get image features
		ori_features = model.encode_image(ori_processed)
		gen_features = model.encode_image(gen_processed)

		# Normalize features
		ori_features = ori_features / ori_features.norm(dim=-1, keepdim=True)
		gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)

		# Calculate cosine distance
		similarity = torch.nn.functional.cosine_similarity(ori_features, gen_features)
		distance = 1 - similarity.item()

	return distance

def cal_clip_embedding_sim(model, preprocess, device, ori_img, gen_img):
	# Convert numpy arrays to PIL Images
	ori_pil = Image.fromarray(ori_img.astype('uint8'))
	gen_pil = Image.fromarray(gen_img.astype('uint8'))

	# Preprocess images
	ori_processed = preprocess(ori_pil).unsqueeze(0).to(device)
	gen_processed = preprocess(gen_pil).unsqueeze(0).to(device)

	with torch.no_grad():
		# Get image features
		ori_features = model.encode_image(ori_processed)
		gen_features = model.encode_image(gen_processed)

		# Normalize features
		ori_features = ori_features / ori_features.norm(dim=-1, keepdim=True)
		gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)

		# Calculate cosine similarity
		similarity = torch.nn.functional.cosine_similarity(ori_features, gen_features).item()

	return similarity

def get_frames(gif):
	im = Image.open(gif)
	num_frames = gif.n_frames
	gif = []
	try:
		while True:
		# Convert current frame to RGB
			current = im.convert("RGB")
			# Create a copy of the frame
			gif.append(current.copy())
			im.seek(len(gif))  # Move to next frame
	except EOFError:
		pass  # End of frames

	return gif

def extract_frames_to_list(video_path):
	# Store frames in a list
	frames = []

	# Open the video file
	video = cv2.VideoCapture(video_path)

	# Get video properties
	fps = video.get(cv2.CAP_PROP_FPS)
	frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

	print(f"FPS: {fps}")
	print(f"Total frames: {frame_count}")

	# Read frames
	while True:
		# Read a frame
		success, frame = video.read()

		# Break if video is finished
		if not success:
			break

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Append frame to list
		frames.append(frame_rgb)

	# Release video object
	video.release()
	print("Extraction complete!")

	return frames


def abr_name(input):
	return "".join(val[0].lower() if (len(val) > 0 and val[0].isalnum()) else "" for val in input.split(" "))

def process_frame_crossjoins(
    generated_vid_dir: Path,
    webvid_vid_dir: Path,
    out_dir: Path,
    dataset_url: str = "https://huggingface.co/datasets/mmfm-trust/T2V/tree/main/privacy",
    models: list[T2VModelName] = None
) -> None:
    """
    Process frame crossjoins between generated and original videos to calculate similarity metrics.
    
    Args:
        generated_vid_dir (Path): Directory containing generated videos
        webvid_vid_dir (Path): Directory containing original WebVid videos
        out_dir (Path): Directory to save processed results
        dataset_url (str): URL to load video metadata from
        models (list[T2VModelName]): List of models to process. If None, processes all models
    """
    if models is None:
        models = t2v_model_list
        
    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    
    # Load dataset
    df = pd.read_csv(dataset_url)
    name_url_dict = dict(zip(df["name"], df["video_id"]))
    
    model_names = [str(model) for model in models]
    
    for model_name in model_names:
        all_webvid_subset_frames = []
        all_generated_caption_subset_frames = []
        
        for caption in name_url_dict.keys():
            abriged_name = abr_name(caption)
            
            webvid_file = webvid_vid_dir / abriged_name
            generated_caption_file = generated_vid_dir / model_name / abriged_name
            
            webvid_frames = extract_frames_to_list(str(webvid_file))
            generated_caption_frames = extract_frames_to_list(str(generated_caption_file))
            
            print(f"Processing {caption}: {len(webvid_frames)} original frames, {len(generated_caption_frames)} generated frames")
            
            scale = int((len(webvid_frames) * len(generated_caption_frames) / 100) ** (0.5)) + 1
            print(f"Using scale factor: {scale}")
            
            webvid_subset_frames = webvid_frames[::scale]
            generated_caption_subset_frames = generated_caption_frames[::scale]
            
            all_webvid_subset_frames.append(webvid_subset_frames)
            all_generated_caption_subset_frames.append(generated_caption_subset_frames)
            
            print(f"Subset frames: {len(webvid_subset_frames)} original, {len(generated_caption_subset_frames)} generated")
        
        print(f"Processing {len(name_url_dict)} videos for model {model_name}")
        
        all_webvid_video_crafter_subset_frames = zip(
            list(name_url_dict.keys()),
            all_webvid_subset_frames,
            all_generated_caption_subset_frames
        )
        
        all_webvid_video_crafter_subset_frames_copy = copy.deepcopy(all_webvid_video_crafter_subset_frames)
        
        # Save subset frames
        subset_frames_path = out_dir / model_name / "all_webvid_subset_frames.pkl"
        subset_frames_path.parent.mkdir(exist_ok=True, parents=True)
        with open(subset_frames_path, "wb") as f:
            pickle.dump(all_webvid_video_crafter_subset_frames_copy, f)
        
        # Process frame pairs and calculate distances
        all_generated_caption_info = []
        for name, webvid_subset_frames, generated_caption_subset_frames in all_webvid_video_crafter_subset_frames:
            info = {
                'name': name,
                'num_webvid_subset_frames': len(webvid_subset_frames),
                'num_model_subset_frames': len(generated_caption_subset_frames)
            }
            
            cross_join_frames = itertools.product(webvid_subset_frames, generated_caption_subset_frames)
            frame_pairs_to_distance = {}
            
            for idx, (orig, generated) in enumerate(cross_join_frames):
                if idx % 10 == 0:
                    print(f"Processing pair {idx}")
                
                distances = {
                    "l2": cal_l2_distance(orig, generated),
                    "clip_dist": cal_clip_embedding_distance(clip_model, preprocess, device, orig, generated),
                    "clip_sim": cal_clip_embedding_sim(clip_model, preprocess, device, orig, generated)
                }
                
                frame_pairs_to_distance[idx] = distances
            
            info['frame_pairs_to_distance'] = frame_pairs_to_distance
            all_generated_caption_info.append(info)
        
        # Save results
        info_path = out_dir / model_name / "all_webvid_info.pkl"
        info_path.parent.mkdir(exist_ok=True, parents=True)
        with open(info_path, "wb") as f:
            pickle.dump(all_generated_caption_info, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_vid_dir", type=Path, required=True)
    parser.add_argument("--webvid_vid_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset_url", type=str, default="https://huggingface.co/datasets/mmfm-trust/T2V/tree/main/privacy")
    parser.add_argument("--models", nargs="+", type=T2VModelName, choices=t2v_model_list, default=None)
    args = parser.parse_args()
    
    process_frame_crossjoins(
        generated_vid_dir=args.generated_vid_dir,
        webvid_vid_dir=args.webvid_vid_dir,
        out_dir=args.out_dir,
        dataset_url=args.dataset_url,
        models=args.models,
        device=args.device
    ) 