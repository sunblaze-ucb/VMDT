# Installation

```bash
conda env create -f environment.yml -n vmdt
conda activate vmdt
```

# Download Models

```bash
python text_video/video-safety-benchmark/download.py
```

# Download Dataset

```bash
huggingface-cli download mmfm-trust/V2T --include 'fairness/**' --local-dir video_text --repo-type dataset
```

# T2V Model Evaluation

```bash
python text_video/video-safety-benchmark/model_responses.py
python video_frame.py
python FairFace/predict.py
python score_calculation.py
python analyze_results/average.py
```

# V2T Model Evaluation

```bash
python video_text/v2t_models/model_responses.py
python fairness_score.py
python average.py
```