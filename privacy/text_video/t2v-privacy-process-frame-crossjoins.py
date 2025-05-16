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

from models import T2VModelName, t2v_model_list

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

if __name__ == "__main__":
    ### Inputs
    generated_vid_dir = ""
    webvid_vid_dir = ""
    out_dir = ""
    
	clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

	df = pd.read_csv("")  ### TODO: Replace with https://huggingface.co/datasets/mmfm-trust/T2V/tree/main/privacy
	name_url_dict = dict(zip(df["name"], df["video_id"]))

	model_names = [str(t2v_model) for t2v_model in t2v_model_list]

	for model_name in models_names:
		all_webvid_subset_frames = []
		all_generated_caption_subset_frames = []
		for caption in name_url_dict.keys():

			### TODO: Replace abriged_name with the mapping from caption to the random generated file name
			abriged_name = abr_name(caption)

			webvid_file = os.path.join(webvid_vid_dir, abriged_name)
			generated_caption_file = os.path.join(generated_vid_dir, model_name, abriged_name)

			webvid_frames = extract_frames_to_list(webvid_file)
			generated_caption_frames = extract_frames_to_list(generated_caption_file)

			print(len(webvid_frames), len(generated_caption_frames))

			scale = int((len(webvid_frames) * len(generated_caption_frames) / 100 ) ** (0.5)) + 1
			print(scale)

			webvid_subset_frames = webvid_frames[::scale]
			generated_caption_subset_frames = generated_caption_frames[::scale]
			
			all_webvid_subset_frames.append(webvid_subset_frames)
			all_generated_caption_subset_frames.append(generated_caption_subset_frames)

			print(len(webvid_subset_frames), len(generated_caption_subset_frames))

		print("name_url_dict.keys(): ",len(list(name_url_dict.keys())))
		print("all_webvid_subset_frames: ",len(all_webvid_subset_frames))
		print("all_generated_caption_subset_frames): ",len(all_generated_caption_subset_frames))
		all_webvid_video_crafter_subset_frames = zip(list(name_url_dict.keys()), all_webvid_subset_frames, all_generated_caption_subset_frames)

		all_webvid_video_crafter_subset_frames_copy = copy.deepcopy(all_webvid_video_crafter_subset_frames)

		if not os.path.exists(os.path.join(out_dir, model_name, "all_webvid_subset_frames.pkl")):
			os.mkdir(os.path.join(out_dir, model_name, "all_webvid_subset_frames.pkl"))
		with open(os.path.join(out_dir, model_name, "all_webvid_subset_frames.pkl"), "wb") as f:
			pickle.dump(all_webvid_video_crafter_subset_frames_copy, f)

		all_generated_caption_info = []
		for name, webvid_subset_frames, generated_caption_subset_frames in all_webvid_video_crafter_subset_frames:
			info = {}
			info['name'] = name
			info['num_webvid_subset_frames'] = len(webvid_subset_frames)
			info['num_model_subset_frames'] = len(generated_caption_subset_frames)

			cross_join_frames = itertools.product(webvid_subset_frames, generated_caption_subset_frames)

			frame_pairs_to_distance = {}

			idx = 0

			for orig, generated in cross_join_frames:
				if (idx % 10 == 0):
					print(idx)

			distances = {}
			distances["l2"] = cal_l2_distance(orig, generated)
			distances["clip_dist"] = cal_clip_embedding_distance(clip_model, preprocess, device, orig, generated)
			distances["clip_sim"] = cal_clip_embedding_sim(clip_model, preprocess, device, orig, generated)

			print(distances)

			frame_pairs_to_distance[idx] = distances

			idx+=1

			info['frame_pairs_to_distance'] = frame_pairs_to_distance
			print(len(frame_pairs_to_distance))

			all_generated_caption_info.append(info)

		if not os.path.exists(os.path.join(out_dir, model_name, "all_webvid_info.pkl")):
			os.mkdir(os.path.join(out_dir, model_name, "all_webvid_info.pkl"))
		with open(os.path.join(out_dir, model_name, "all_webvid_info.pkl"), "wb") as f:
			pickle.dump(all_generated_caption_info, f)