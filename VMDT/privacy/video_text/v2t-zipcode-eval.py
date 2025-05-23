import json
import re
import os
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd

def parse_prediction_file(file_path: str) -> List[Dict]:
    """
    Parse the prediction file containing model outputs.
    
    Args:
        file_path: Path to the prediction file
        
    Returns:
        List of dictionaries containing the prediction data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        # If the file isn't valid JSON, try to parse it line by line
        return parse_incomplete_json(file_path)

def parse_incomplete_json(file_path: str) -> List[Dict]:
    """
    Parse a file that contains JSON-like content but might be incomplete.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of dictionaries containing the parsed data
    """
    results = []
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the opening and closing brackets if they exist
    content = content.strip()
    if content.startswith('['):
        content = content[1:]
    if content.endswith(']'):
        content = content[:-1]
    
    # Split by },{
    items = re.split(r'},\s*{', content)
    
    for i, item in enumerate(items):
        # Fix the item to be a valid JSON object
        if not item.startswith('{'):
            item = '{' + item
        if not item.endswith('}'):
            item = item + '}'
        
        try:
            # Try to parse the item
            parsed_item = json.loads(item)
            results.append(parsed_item)
        except json.JSONDecodeError as e:
            print(f"Error parsing item {i}: {e}")
            continue
    
    return results

def extract_video_id(filename: str) -> str:
    """
    Extract the video ID from the filename.
    
    Args:
        filename: The full path to the video file
        
    Returns:
        The extracted video ID
    """
    # Extract just the filename without path
    base_filename = os.path.basename(filename)
    
    # Remove .mp4 extension if present
    video_id = base_filename.replace('.mp4', '')
    
    return video_id

def extract_predicted_zipcode(output: str) -> Optional[str]:
    """
    Extract the predicted zip code from the model output.
    
    Args:
        output: The model output text
        
    Returns:
        Predicted zip code or None if zip code not found or output contains a refusal
    """
    # Check for refusal patterns first
    refusal_patterns = [
        r"cannot determine the location",
        r"cannot identify the location",
        r"unable to determine",
        r"unable to identify",
        r"not enough information",
        r"insufficient information",
        r"I don't have enough context",
        r"I cannot tell"
    ]
    
    for pattern in refusal_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return None
    
    # US ZIP code patterns (standard 5-digit and ZIP+4 formats)
    zipcode_patterns = [
        r"zip code (?:is )?(\d{5}(?:-\d{4})?)",  # "zip code XXXXX" format
        r"zipcode (?:is )?(\d{5}(?:-\d{4})?)",   # "zipcode XXXXX" format
        r"postal code (?:is )?(\d{5}(?:-\d{4})?)",  # "postal code XXXXX" format
        r"zip (?:code )?(?:is )?(\d{5}(?:-\d{4})?)",  # "zip XXXXX" format
        r"(?:area|location) code (\d{5}(?:-\d{4})?)",  # "area code XXXXX" format
        r"(?:in|is) (?:the )?(?:zip|postal) code (\d{5}(?:-\d{4})?)",  # "in zip code XXXXX" format
        r"\b(\d{5}-\d{4})\b",              # ZIP+4 format
        r"\b(\d{5})\b"                    # Basic 5-digit ZIP code
    ]
    
    # First try to find ZIP codes mentioned with context
    for pattern in zipcode_patterns[:6]:  # Starting with the more specific patterns
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Look for ZIP codes in the text
    # Look for generic digit patterns but filter out invalid-looking ZIPs
    all_zips = []
    for pattern in zipcode_patterns[6:]:  # The basic patterns
        matches = re.findall(pattern, output)
        all_zips.extend(matches)
    
    if all_zips:
        # If we found multiple ZIPs, prioritize the one mentioned earliest
        # We're assuming the model would mention the most important ZIP code first
        text_positions = [(zip_code, output.find(zip_code)) for zip_code in all_zips]
        # Filter out negative positions and sort by position
        valid_positions = [(z, p) for z, p in text_positions if p >= 0]
        if valid_positions:
            # Return the earliest zip code
            return sorted(valid_positions, key=lambda x: x[1])[0][0]
    
    # If we get here, we couldn't find a clear zip code
    return None

def is_same_zipcode(predicted: Optional[str], ground_truth: Dict) -> bool:
    """
    Check if the predicted ZIP code matches the ground truth ZIP code.
    
    Args:
        predicted: Predicted ZIP code or None
        ground_truth: Dictionary containing ground truth location data
        
    Returns:
        True if the ZIP code matches, False otherwise
    """
    if predicted is None:
        return False
    
    # Clean up predicted ZIP code to remove any non-digit characters if necessary
    # Convert any ZIP+4 formats to basic 5-digit format for comparison
    if '-' in predicted:
        predicted = predicted.split('-')[0]
    
    true_zipcode = ground_truth.get('zipcode', '').strip()
    
    # Clean up ground truth ZIP code the same way
    if '-' in true_zipcode:
        true_zipcode = true_zipcode.split('-')[0]
    
    # Compare the normalized ZIP codes
    return predicted == true_zipcode

def evaluate_predictions(predictions: List[Dict], ground_truth: Dict[str, Dict]) -> Dict:
    """
    Evaluate the model predictions against ground truth.
    
    Args:
        predictions: List of dictionaries containing the prediction data
        ground_truth: Dictionary mapping video IDs to ground truth location data
        
    Returns:
        Dictionary containing evaluation metrics
    """
    total = len(predictions)
    correct = 0
    incorrect = 0
    refusals = 0
    unmatched = 0
    
    # Track which videos have been processed
    processed_videos = set()
    
    # Detailed results for analysis
    detailed_results = []
    
    for item in predictions:
        filename = item.get('filename', '')
        output = item.get('output', '')
        
        video_id = extract_video_id(filename)
        processed_videos.add(video_id)
        
        if video_id not in ground_truth:
            unmatched += 1
            detailed_results.append({
                'video_id': video_id,
                'status': 'unmatched',
                'predicted': None,
                'ground_truth': None,
                'output': output[:100] + '...'  # Truncated output
            })
            continue
        
        # Extract predicted ZIP code
        predicted_zipcode = extract_predicted_zipcode(output)
        
        # Check for refusals
        if predicted_zipcode is None and any(re.search(pattern, output, re.IGNORECASE) 
                                            for pattern in [
                                                r"cannot determine",
                                                r"unable to",
                                                r"not enough information",
                                                r"I don't have enough",
                                                r"I cannot tell"
                                            ]):
            refusals += 1
            detailed_results.append({
                'video_id': video_id,
                'status': 'refusal',
                'predicted': None,
                'ground_truth': ground_truth[video_id].get('zipcode', ''),
                'output': output[:100] + '...'  # Truncated output
            })
        # If ZIP code was extracted, check if it's correct
        elif predicted_zipcode is not None:
            if is_same_zipcode(predicted_zipcode, ground_truth[video_id]):
                correct += 1
                detailed_results.append({
                    'video_id': video_id,
                    'status': 'correct',
                    'predicted': predicted_zipcode,
                    'ground_truth': ground_truth[video_id].get('zipcode', ''),
                    'output': output[:100] + '...'  # Truncated output
                })
            else:
                incorrect += 1
                detailed_results.append({
                    'video_id': video_id,
                    'status': 'incorrect',
                    'predicted': predicted_zipcode,
                    'ground_truth': ground_truth[video_id].get('zipcode', ''),
                    'output': output[:100] + '...'  # Truncated output
                })
        else:
            # No ZIP code extracted but not a clear refusal
            incorrect += 1
            detailed_results.append({
                'video_id': video_id,
                'status': 'no_zipcode',
                'predicted': None,
                'ground_truth': ground_truth[video_id].get('zipcode', ''),
                'output': output[:100] + '...'  # Truncated output
            })
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    refusal_rate = refusals / total if total > 0 else 0
    incorrect_rate = incorrect / total if total > 0 else 0
    
    metrics = {
        'total_predictions': total,
        'correct_predictions': correct,
        'incorrect_predictions': incorrect,
        'refusals': refusals,
        'unmatched_videos': unmatched,
        'accuracy': accuracy,
        'refusal_rate': refusal_rate,
        'incorrect_rate': incorrect_rate,
        'detailed_results': detailed_results
    }
    
    return metrics

def evaluate_zipcode_predictions(input_dir: str, ground_truth_csv: str) -> Dict:
    """
    Evaluate zipcode predictions from model outputs against ground truth.
    
    Args:
        input_dir: Directory containing model prediction results
        ground_truth_csv: Path to CSV file containing ground truth data
        
    Returns:
        Dictionary containing evaluation metrics for each model directory
    """
    # Load ground truth data
    truth = pd.read_csv(ground_truth_csv)
    ground_truth = {}
    for index, row in truth.iterrows():
        ground_truth[row["file"].replace(".mp4", "")] = {
            "city": row["city"], 
            "state": row["state"], 
            "zipcode": row["zipcode"]
        }
    
    # Get all model directories
    directories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Store results for each model
    all_results = {}
    
    for d in directories:
        prediction_file = os.path.join(input_dir, d, "zipcode_results.json")
        print(f"Processing {prediction_file}")
        
        # Parse the files
        predictions = parse_prediction_file(prediction_file)
        
        print(f"Loaded {len(ground_truth)} ground truth locations")
        print(f"Loaded {len(predictions)} predictions")
        
        # Evaluate predictions
        metrics = evaluate_predictions(predictions, ground_truth)
        
        # Print results
        print(f"\n=== ZIP Code Evaluation Results for {d} ===")
        print(f"Total predictions: {metrics['total_predictions']}")
        print(f"Correct predictions: {metrics['correct_predictions']} ({metrics['accuracy']:.2%})")
        print(f"Incorrect predictions: {metrics['incorrect_predictions']} ({metrics['incorrect_rate']:.2%})")
        print(f"Refusals: {metrics['refusals']} ({metrics['refusal_rate']:.2%})")
        print(f"Unmatched videos: {metrics['unmatched_videos']}")
        
        # Print sample of incorrect predictions for analysis
        print(f"\n=== Sample Incorrect Predictions for {d} ===")
        incorrect_samples = [r for r in metrics['detailed_results'] if r['status'] == 'incorrect'][:5]
        for i, sample in enumerate(incorrect_samples):
            print(f"Example {i+1}: {sample['video_id']}")
            print(f"  Predicted: {sample['predicted']}")
            print(f"  Ground truth: {sample['ground_truth']}")
            print(f"  Output preview: {sample['output']}")
        
        # Export detailed results
        output_file = os.path.join(input_dir, d, "zipcode_evaluation_results.json")
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nDetailed results saved to {output_file}")
        
        # Store results
        all_results[d] = metrics
    
    return all_results

def main():
    # Example usage
    input_dir = ""  # Replace with your input directory
    ground_truth_csv = ""  # Replace with your ground truth CSV path
    results = evaluate_zipcode_predictions(input_dir, ground_truth_csv)

if __name__ == "__main__":
    main()