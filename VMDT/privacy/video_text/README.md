# Video-to-Text Models for Privacy Evaluation

This repository contains code for evaluating various video-to-text models on privacy-related tasks, including location identification (city, state, and ZIP code prediction).

## Installation

### Environment Setup

1. Create and activate conda environment using the provided yml file:
```bash
conda env create -f v2t.yml
conda activate v2t
```

### Model-Specific Dependencies

For GPT-4o with audio support, install additional dependencies:
```bash
pip install opencv-python
pip install moviepy
pip install openai
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.2.post1
```

## Usage

### Prerequisites

Before running any model, ensure you have:
1. Set up the required API keys:
   - Claude API key
   - GPT API key
   - Nova Lite & Pro API keys
2. Configured AWS credentials
3. Updated TODOs in the execution files

### Pipeline Execution

The complete pipeline consists of two main steps:

1. **Generate Model Predictions**
```bash
python3 v2t-privacy-gen.py \
    --vids_dir /path/to/videos \
    --output_dir /path/to/output \
    --models model1 model2
```

This script:
- Processes videos using specified models
- Generates predictions for each model
- Saves results in the output directory

2. **Run Evaluations**

The evaluation pipeline assesses model performance on three privacy-related tasks:

a. City Prediction:
```bash
python3 v2t-city-eval.py
```

b. State Prediction:
```bash
python3 v2t-state-eval.py
```

c. ZIP Code Prediction:
```bash
python3 v2t-zipcode-eval.py
```

Or run all evaluations at once:
```bash
python main.py \
    --vids_dir /path/to/videos \
    --output_dir /path/to/output \
    --models model1 model2 \
    --ground_truth_csv /path/to/ground_truth.csv
```

## Output

Each evaluation generates:
1. Detailed results in JSON format:
   - `city_evaluation_results.json`
   - `state_evaluation_results.json`
   - `zipcode_evaluation_results.json`
2. Summary statistics including:
   - Accuracy rates
   - Refusal rates
   - Incorrect prediction rates

## Notes

- Ensure all API keys and credentials are properly configured before running the models
- Some models may require specific GPU configurations
- Check the TODOs in each execution file for model-specific requirements
- The evaluation pipeline requires a ground truth CSV file with the correct format
- The v2t-privacy-gen.py script must be run before running evaluations
