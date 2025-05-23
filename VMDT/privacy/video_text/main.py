import argparse
import os
from pathlib import Path
import time
import pandas as pd
from v2t_privacy_gen import process_videos
from VDMT.models.v2t_models import V2TModelName, v2t_model_list
from v2t_city_eval import evaluate_city_predictions
from v2t_state_eval import evaluate_state_predictions
from v2t_zipcode_eval import evaluate_zipcode_predictions

def main(args):
    # Process videos
    process_videos(
        folder_path=args.vids_dir,
        out_path=args.output_dir,
        models=args.models
    )
    
    # Run evaluations
    print("\n=== Running Location Evaluations ===")
    
    # City evaluation
    print("\nEvaluating city predictions...")
    city_results = evaluate_city_predictions(
        input_dir=args.output_dir,
        ground_truth_csv=args.ground_truth_csv
    )
    
    # State evaluation
    print("\nEvaluating state predictions...")
    state_results = evaluate_state_predictions(
        input_dir=args.output_dir,
        ground_truth_csv=args.ground_truth_csv
    )
    
    # ZIP code evaluation
    print("\nEvaluating ZIP code predictions...")
    zipcode_results = evaluate_zipcode_predictions(
        input_dir=args.output_dir,
        ground_truth_csv=args.ground_truth_csv
    )
    
    # Print summary of results
    print("\n=== Evaluation Summary ===")
    for model_dir in city_results.keys():
        print(f"\nModel: {model_dir}")
        print(f"City Accuracy: {city_results[model_dir]['accuracy']:.2%}")
        print(f"State Accuracy: {state_results[model_dir]['accuracy']:.2%}")
        print(f"ZIP Code Accuracy: {zipcode_results[model_dir]['accuracy']:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vids_dir', type=str, default="VMDT/privacy/v2t_privacy/videos/")
    parser.add_argument('--output_dir', type=str, default="results/v2t_results/privacy/")
    parser.add_argument('--models', nargs='+', type=V2TModelName, choices=v2t_model_list, default=None,
                      help='List of models to use for video generation')
    parser.add_argument('--dataset_url', type=str, default=None,
                      help='URL to load prompts from. If not provided, uses default privacy dataset')
    parser.add_argument('--ground_truth_csv', type=str, required=True,
                      help='Path to the ground truth CSV file containing location data')
    args = parser.parse_args()
    main(args) 

