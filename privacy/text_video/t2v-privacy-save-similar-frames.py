import pickle
import heapq
import itertools
import sys
import os
import cv2  # for OpenCV method
from PIL import Image  # for PIL method
import numpy as np
from models import T2VModelName, t2v_model_list

def save_array_as_image(array, filepath, method='pil'):
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
		cv2.imwrite(filepath, array)

if __name__ == "__main__":
    ### Inputs:
    all_webvid_subset_frames_dir = ""
    all_webvid_info_dir = ""
    num_examples = 5
    out_dir = ""
    
    model_names = [str(t2v_model) for t2v_model in t2v_model_list]

    for model_name in models_names:

		subset_frames_file = os.path.join(all_webvid_subset_frames_dir, model_name, "all_webvid_subset_frames.pkl")

		file = os.path.join(all_webvid_info_dir, model_name, "all_webvid_info.pkl")

		# Open and read the pickle file
		with open(file, 'rb') as f:
			data = pickle.load(f)
		with open(subset_frames_file, 'rb') as f:
			subset_frames_file = pickle.load(f)
			
		name_to_subset_frames = {}
		for name, all_webvid_subset_frames, all_model_subset_frames in subset_frames_file:
			name_to_subset_frames[name] = (all_webvid_subset_frames, all_model_subset_frames)

		model = {}
		for info in data:
			
			name = info['name']
			num_webvid_subset_frames = info['num_webvid_subset_frames']
			num_model_subset_frames = info['num_model_subset_frames']
			frame_pairs_to_distance = info['frame_pairs_to_distance']
			
			num = 1
			
			pair_idx_l2_dist = [(distances['l2'], idx) for idx, distances in frame_pairs_to_distance.items()]
			pair_idx_l2_dist.sort(key=lambda x: x[0])

			l2_top_n_average = pair_idx_l2_dist[0]
			
			pair_idx_clip_dist = [(distances['clip_dist'], idx) for idx, distances in frame_pairs_to_distance.items()]
			pair_idx_clip_dist.sort(key=lambda x: x[0])

			clip_dist_top_n_average = pair_idx_clip_dist[0]
			
			pair_idx_clip_sim_dist = [(distances['clip_sim'], idx) for idx, distances in frame_pairs_to_distance.items()]
			pair_idx_clip_sim_dist.sort(key=lambda x: abs(x[0]))

			clip_sim_top_n_average = pair_idx_clip_sim_dist[0]
			
			top_n_average = {"l2_top_n_average": l2_top_n_average, "clip_dist_top_n_average": clip_dist_top_n_average, "clip_sim_top_n_average": clip_sim_top_n_average}
			
			model[name] = top_n_average
		
		l2_top_n_average_overall = []
		clip_dist_top_n_average_overall = []
		clip_sim_top_n_average_overall = []
		for name, val in model.items():
			heapq.heappush(l2_top_n_average_overall, (val["l2_top_n_average"][0], val["l2_top_n_average"][1], name))
			heapq.heappush(clip_dist_top_n_average_overall, (val["clip_dist_top_n_average"][0], val["clip_dist_top_n_average"][1], name))

		l2_num_examples_smallest = heapq.nsmallest(num_examples, l2_top_n_average_overall)
		clip_dist_num_examples_smallest = heapq.nsmallest(num_examples, clip_dist_top_n_average_overall)
		
		if not os.path.exists(os.path.join(out_dir, model_name)):
			os.mkdir(os.path.join(out_dir, model_name))
		if not os.path.exists(os.path.join(out_dir, model_name, "l2")):
			os.mkdir(os.path.join(out_dir, model_name, "l2"))
		if not os.path.exists(os.path.join(out_dir, model_name, "clip_dist")):
			os.mkdir(os.path.join(out_dir, model_name, "clip_dist"))
		
		j = 0
		for dist, idx, caption in l2_num_examples_smallest:
			print(caption)
			print(dist)
			subset_frames = name_to_subset_frames[caption]
			cross_join_frames_iter = list(itertools.product(subset_frames[0],subset_frames[1]))[idx]
			orig = cross_join_frames_iter[0]
			model = cross_join_frames_iter[1]

			if not os.path.exists(os.path.join(out_dir, model_name, "l2", "ex_" + str(j))):
				os.mkdir(os.path.join(out_dir, model_name, "l2", "ex_" + str(j)))
			save_array_as_image(orig, os.path.join(out_dir, model_name, "l2", "ex_" + str(j), "orig.png"))
			save_array_as_image(model, os.path.join(out_dir, model_name, "l2", "ex_" + str(j), "model.png"))
			j+=1
		
		j = 0
		for dist, idx, caption in clip_dist_num_examples_smallest:
			print(caption)
			print(dist)
			subset_frames = name_to_subset_frames[caption]
			cross_join_frames_iter = list(itertools.product(subset_frames[0],subset_frames[1]))[idx]
			orig = cross_join_frames_iter[0]
			model = cross_join_frames_iter[1]
			
			if not os.path.exists(os.path.join(out_dir, model_name, "clip_dist", "ex_" + str(j))):
				os.mkdir(os.path.join(out_dir, model_name, "clip_dist", "ex_" + str(j)))
			save_array_as_image(orig, os.path.join(out_dir, model_name, "clip_dist", "ex_" + str(j), "orig.png"))
			save_array_as_image(model, os.path.join(out_dir, model_name, "clip_dist", "ex_" + str(j), "model.png"))
			j+=1