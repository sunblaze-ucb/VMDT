# Video Generation Models

## Installation
conda environment for reference:
```
conda create -n aws-video python==3.12
conda install nvidia/label/cuda-12.1.1::cuda-toolkit -c nvidia/label/cuda-12.1.1
```

pip setup
```
# --no-deps to ignore version conflict due to colossalai
pip3 install --no-deps -r requirements.txt

# flash-attn
pip3 install flash-attn --no-build-isolation

# apex
pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
```

model weight
```
python3 download.py
```

## Configuration

Before running the pipeline, you need to set up the following:

1. API Keys:
   - Nova Reel API key
   - Pika API key
   - Luma API key

2. AWS Credentials:
   - AWS Access Key ID
   - AWS Secret Access Key
   - AWS Region

3. Update configuration files:
   - Check and modify TODOs in the pipeline scripts
   - Set appropriate paths and parameters

## Usage

### Video Generation Pipeline

The pipeline consists of several steps:

1. Generate videos:
```bash
python3 t2v-privacy-gen.py
```

2. Process frames and create cross-joins:
```bash
python3 t2v-privacy-process-frame-crossjoins.py
```

3. Calculate distances:
```bash
python3 t2v-privacy-calc-distances.py
```

4. Save similar frames:
```bash
python3 t2v-privacy-save-similar-frames.py
```

### Evaluation Pipeline

To evaluate the generated videos:

```bash
python main.py \
    --vids_dir /path/to/videos \
    --output_dir /path/to/output \
    --models model1 model2 \
    --ground_truth_csv /path/to/ground_truth.csv
```

This will:
1. Process the videos using specified models
2. Run evaluations for:
   - City predictions
   - State predictions
   - ZIP code predictions
3. Generate detailed evaluation reports
4. Print a summary of results

## Output

The pipeline generates several types of output:

1. Generated videos in the specified output directory
2. Evaluation results for each model:
   - `city_evaluation_results.json`
   - `state_evaluation_results.json`
   - `zipcode_evaluation_results.json`
3. Summary statistics and metrics
