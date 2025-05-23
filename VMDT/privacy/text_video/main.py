import argparse
import os
from pathlib import Path
import time
import pandas as pd
from t2v_privacy_gen import generate_privacy_videos
from t2v_privacy_process_frame_crossjoins import process_frame_crossjoins
from t2v_privacy_calc_distances import calculate_distances
from t2v_privacy_save_similar_frames import save_similar_frames
from VDMT.models.t2v_models import T2VModelName, t2v_model_list

def main(args):
    # Convert string paths to Path objects
    output_dir = Path(args.output_dir)
    vids_dir = Path(args.vids_dir)
    
    # Generate videos using the privacy generation function
    generate_privacy_videos(
        out_dir=output_dir,
        models=args.models if args.models else None,
        dataset_url=args.dataset_url if args.dataset_url else None
    )
    
    # Process frame crossjoins between generated and original videos
    process_frame_crossjoins(
        generated_vid_dir=output_dir,
        webvid_vid_dir=vids_dir,
        out_dir=output_dir / "processed",
        dataset_url=args.dataset_url if args.dataset_url else None,
        models=args.models if args.models else None
    )
    
    # Calculate distances between original and generated videos
    calculate_distances(
        all_webvid_info_dir=output_dir / "processed",
        top_n=args.top_n,
        models=args.models if args.models else None
    )
    
    # Save most similar frames
    save_similar_frames(
        all_webvid_subset_frames_dir=output_dir / "processed",
        out_dir=output_dir / "similar_frames",
        num_examples=args.num_examples,
        models=args.models if args.models else None
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="results/t2v_results/privacy/")
    parser.add_argument('--vids_dir', type=str, default="VMDT/privacy/t2v_privacy/")
    parser.add_argument('--models', nargs='+', type=T2VModelName, choices=t2v_model_list, default=None,
                      help='List of models to use for video generation')
    parser.add_argument('--dataset_url', type=str, default=None,
                      help='URL to load prompts from. If not provided, uses default privacy dataset')
    parser.add_argument('--top_n', type=int, default=10,
                      help='Number of top similar frames to average over for distance calculations')
    parser.add_argument('--num_examples', type=int, default=5,
                      help='Number of most similar examples to save')
    args = parser.parse_args()
    main(args) 