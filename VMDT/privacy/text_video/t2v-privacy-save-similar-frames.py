import pickle
import heapq
import itertools
import sys
import os
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from VDMT.models.t2v_models import T2VModelName, t2v_model_list

def save_array_as_image(array: np.ndarray, filepath: Path, method: str = 'pil') -> None:
	"""
	Save a numpy array as an image file.
	
	Args:
		array: numpy array to save
		filepath: where to save the image (include extension like .jpg, .png)
		method: 'pil' or 'cv2' for saving method
	"""
	# Ensure array is uint8
	if array.dtype != np.uint8:
		if array.max() <= 1.0:
			array = (array * 255).astype(np.uint8)
		else:
			array = array.astype(np.uint8)
	
	if method == 'pil':
		Image.fromarray(array).save(filepath)
	elif method == 'cv2':
		# OpenCV expects BGR format for color images
		if len(array.shape) == 3 and array.shape[2] == 3:
			array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
		cv2.imwrite(str(filepath), array)

def save_similar_frames(
	all_webvid_subset_frames_dir: Path,
	out_dir: Path,
	num_examples: int = 5,
	models: List[T2VModelName] = None
) -> None:
	"""
	Save the most similar frames between original and generated videos.
	
	Args:
		all_webvid_subset_frames_dir (Path): Directory containing subset frames information
		out_dir (Path): Directory to save similar frames
		num_examples (int): Number of most similar examples to save
		models (List[T2VModelName]): List of models to process. If None, processes all models
	"""
	if models is None:
		models = t2v_model_list
		
	model_names = [str(model) for model in models]

	for model_name in model_names:
		subset_frames_file = all_webvid_subset_frames_dir / model_name / "all_webvid_subset_frames.pkl"
		info_file = all_webvid_subset_frames_dir / model_name / "all_webvid_info.pkl"

		# Open and read the pickle files
		with open(info_file, 'rb') as f:
			data = pickle.load(f)
		with open(subset_frames_file, 'rb') as f:
			subset_frames_data = pickle.load(f)
			
		name_to_subset_frames = {}
		for name, all_webvid_subset_frames, all_model_subset_frames in subset_frames_data:
			name_to_subset_frames[name] = (all_webvid_subset_frames, all_model_subset_frames)

		model = {}
		for info in data:
			name = info['name']
			num_webvid_subset_frames = info['num_webvid_subset_frames']
			num_model_subset_frames = info['num_model_subset_frames']
			frame_pairs_to_distance = info['frame_pairs_to_distance']
			
			# Calculate L2 distance metrics
			pair_idx_l2_dist = [(distances['l2'], idx) for idx, distances in frame_pairs_to_distance.items()]
			pair_idx_l2_dist.sort(key=lambda x: x[0])
			l2_top_n_average = pair_idx_l2_dist[0]
			
			# Calculate CLIP distance metrics
			pair_idx_clip_dist = [(distances['clip_dist'], idx) for idx, distances in frame_pairs_to_distance.items()]
			pair_idx_clip_dist.sort(key=lambda x: x[0])
			clip_dist_top_n_average = pair_idx_clip_dist[0]
			
			# Calculate CLIP similarity metrics
			pair_idx_clip_sim_dist = [(distances['clip_sim'], idx) for idx, distances in frame_pairs_to_distance.items()]
			pair_idx_clip_sim_dist.sort(key=lambda x: abs(x[0]))
			clip_sim_top_n_average = pair_idx_clip_sim_dist[0]
			
			top_n_average = {
				"l2_top_n_average": l2_top_n_average,
				"clip_dist_top_n_average": clip_dist_top_n_average,
				"clip_sim_top_n_average": clip_sim_top_n_average
			}
			
			model[name] = top_n_average
		
		# Find top examples for each metric
		l2_top_n_average_overall = []
		clip_dist_top_n_average_overall = []
		clip_sim_top_n_average_overall = []
		
		for name, val in model.items():
			heapq.heappush(l2_top_n_average_overall, (val["l2_top_n_average"][0], val["l2_top_n_average"][1], name))
			heapq.heappush(clip_dist_top_n_average_overall, (val["clip_dist_top_n_average"][0], val["clip_dist_top_n_average"][1], name))

		l2_num_examples_smallest = heapq.nsmallest(num_examples, l2_top_n_average_overall)
		clip_dist_num_examples_smallest = heapq.nsmallest(num_examples, clip_dist_top_n_average_overall)
		
		# Create output directories
		model_out_dir = out_dir / model_name
		l2_out_dir = model_out_dir / "l2"
		clip_dist_out_dir = model_out_dir / "clip_dist"
		
		model_out_dir.mkdir(exist_ok=True, parents=True)
		l2_out_dir.mkdir(exist_ok=True, parents=True)
		clip_dist_out_dir.mkdir(exist_ok=True, parents=True)
		
		# Save L2 similar frames
		for j, (dist, idx, caption) in enumerate(l2_num_examples_smallest):
			print(f"Processing L2 example {j+1}/{num_examples} for {model_name}")
			print(f"Caption: {caption}")
			print(f"Distance: {dist:.4f}")
			
			subset_frames = name_to_subset_frames[caption]
			cross_join_frames_iter = list(itertools.product(subset_frames[0], subset_frames[1]))[idx]
			orig, model_frame = cross_join_frames_iter

			ex_dir = l2_out_dir / f"ex_{j}"
			ex_dir.mkdir(exist_ok=True, parents=True)
			save_array_as_image(orig, ex_dir / "orig.png")
			save_array_as_image(model_frame, ex_dir / "model.png")
		
		# Save CLIP distance similar frames
		for j, (dist, idx, caption) in enumerate(clip_dist_num_examples_smallest):
			print(f"Processing CLIP distance example {j+1}/{num_examples} for {model_name}")
			print(f"Caption: {caption}")
			print(f"Distance: {dist:.4f}")
			
			subset_frames = name_to_subset_frames[caption]
			cross_join_frames_iter = list(itertools.product(subset_frames[0], subset_frames[1]))[idx]
			orig, model_frame = cross_join_frames_iter
			
			ex_dir = clip_dist_out_dir / f"ex_{j}"
			ex_dir.mkdir(exist_ok=True, parents=True)
			save_array_as_image(orig, ex_dir / "orig.png")
			save_array_as_image(model_frame, ex_dir / "model.png")

if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--subset_frames_dir", type=Path, required=True,
					  help="Directory containing subset frames information")
	parser.add_argument("--out_dir", type=Path, required=True,
					  help="Directory to save similar frames")
	parser.add_argument("--num_examples", type=int, default=5,
					  help="Number of most similar examples to save")
	parser.add_argument("--models", nargs="+", type=T2VModelName, choices=t2v_model_list, default=None,
					  help="List of models to process. If not provided, processes all models")
	args = parser.parse_args()
	
	save_similar_frames(
		all_webvid_subset_frames_dir=args.subset_frames_dir,
		all_webvid_info_dir=args.info_dir,
		out_dir=args.out_dir,
		num_examples=args.num_examples,
		models=args.models
	)