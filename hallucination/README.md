# VMDT - Hallucination
## Installation
```bash
conda env create -f environment.yml
conda activate vmdt-hallu
```

## Download Models
```bash
python3 download.py
```

## Evaluation
prepare the api keys in `.env`
```bash
cp .env.template .env
```

### T2V Models
For example, to run on all scenarios and these three models, run:
```bash
python T2V_inference_non_surrogate.py --scenarios OCR Distraction Counterfactual Misleading CoOccurrence NaturalSelection Temporal --models "VideoCrafter2" "Vchitect2" "OpenSora1.2" --num_gpus 8 --models_per_gpu 1 --num_instances_per_task -1 --num_instances_per_task_scenario 1
```
This will save an output file.

To evaluate, first start up vLLM, filling in or changing the parameters as needed:
```
CUDA_VISIBLE_DEVICES=<gpu_ids> vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --port <port> \
  --host 0.0.0.0 \
  --allowed-local-media-path <path_to_videos> \
  --limit-mm-per-prompt image=5 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 
```

Then, run the following evaluation script.
```bash
python T2V_evaluation_newer.py --video_json=<output_file> --n_frames=5 --num_instances_per_task=-1 --include_image_in_classification --combine_step --direct_evaluation --use_qwen
```

This will save your evaluation results in a new file in the same directory as <output_file>.

## V2T Models
This will both run and evaluate the script.
```bash
cd models/v2t_models
python V2T_inference_non_surrogate.py --scenarios NaturalSelection Misleading Distraction Counterfactual --models qwen_vl2_7b llava_video_7b_fixed internvl2_5_8b --num_gpus 1 --models_per_gpu 1 --num_instances_per_task 1 --num_instances_per_task_scenario -1
```