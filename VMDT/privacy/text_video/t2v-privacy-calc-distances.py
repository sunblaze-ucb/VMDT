import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from VDMT.models.t2v_models import T2VModelName, t2v_model_list

def calculate_distances(
    all_webvid_info_dir: Path,
    top_n: int = 10,
    models: List[T2VModelName] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate average distances between original and generated video frames.
    
    Args:
        all_webvid_info_dir (Path): Directory containing the processed frame information
        top_n (int): Number of top similar frames to average over
        models (List[T2VModelName]): List of models to process. If None, processes all models
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping model names to their average distances
    """
    if models is None:
        models = t2v_model_list
        
    model_names = [str(model) for model in models]
    results = {}
    
    for model_name in model_names:
        file = all_webvid_info_dir / model_name / "all_webvid_info.pkl"
        # Open and read the pickle file
        with open(file, 'rb') as f:
            data = pickle.load(f)

        model = {}
        for info in data:
            name = info['name']
            num_webvid_subset_frames = info['num_webvid_subset_frames']
            num_model_subset_frames = info['num_model_subset_frames']
            frame_pairs_to_distance = info['frame_pairs_to_distance']
            
            # Calculate L2 distance averages
            pair_idx_l2_dist = [(idx, distances['l2']) for idx, distances in frame_pairs_to_distance.items()]
            pair_idx_l2_dist.sort(key=lambda x: x[1])
            l2_top_n_average = sum(n for _,n in pair_idx_l2_dist[:top_n])/top_n
            
            # Calculate CLIP distance averages
            pair_idx_clip_dist = [(idx, distances['clip_dist']) for idx, distances in frame_pairs_to_distance.items()]
            pair_idx_clip_dist.sort(key=lambda x: x[1])
            clip_dist_top_n_average = sum(n for _,n in pair_idx_clip_dist[:top_n])/top_n
            
            # Calculate CLIP similarity averages
            pair_idx_clip_sim_dist = [(idx, distances['clip_sim']) for idx, distances in frame_pairs_to_distance.items()]
            pair_idx_clip_sim_dist.sort(key=lambda x: abs(x[1]))
            clip_sim_top_n_average = sum(n for _,n in pair_idx_clip_sim_dist[:top_n])/top_n
            
            top_n_average = {
                "l2_top_n_average": l2_top_n_average,
                "clip_dist_top_n_average": clip_dist_top_n_average,
                "clip_sim_top_n_average": clip_sim_top_n_average
            }
            
            model[name] = top_n_average
        
        # Calculate overall averages for the model
        l2_top_n_average_overall = 0.0
        clip_dist_top_n_average_overall = 0.0
        clip_sim_top_n_average_overall = 0.0

        for name, val in model.items():
            l2_top_n_average_overall += val["l2_top_n_average"]
            clip_dist_top_n_average_overall += val["clip_dist_top_n_average"]
            clip_sim_top_n_average_overall += val["clip_sim_top_n_average"]

        num_videos = len(model.items())
        results[model_name] = {
            "l2_top_n_average_overall": l2_top_n_average_overall / num_videos,
            "clip_dist_top_n_average_overall": clip_dist_top_n_average_overall / num_videos,
            "clip_sim_top_n_average_overall": clip_sim_top_n_average_overall / num_videos,
            "per_video": model
        }
        
        print(f"\nResults for {model_name}:")
        print(f"L2 Distance Average: {results[model_name]['l2_top_n_average_overall']:.4f}")
        print(f"CLIP Distance Average: {results[model_name]['clip_dist_top_n_average_overall']:.4f}")
        print(f"CLIP Similarity Average: {results[model_name]['clip_sim_top_n_average_overall']:.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_dir", type=Path, required=True,
                      help="Directory containing the processed frame information")
    parser.add_argument("--top_n", type=int, default=10,
                      help="Number of top similar frames to average over")
    parser.add_argument("--models", nargs="+", type=T2VModelName, choices=t2v_model_list, default=None,
                      help="List of models to process. If not provided, processes all models")
    args = parser.parse_args()
    
    calculate_distances(
        all_webvid_info_dir=args.info_dir,
        top_n=args.top_n,
        models=args.models
    )
