# Hallucination

## Environments
### T2V
To create the environment, we suggest using a miniconda container. Then install the following:
```bash
apptainer shell --nv --bind /ib-scratch /ib-scratch/chenguang02/scratch1/cnicholas/containers/miniconda.sif
source ~/.bashrc
conda env create -f use_environment.yml -n vmdt-hallu-new
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit
conda install -c conda-forge gxx_linux-64
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
pip install flash-attn --no-build-isolation
pip install open-clip-torch==2.24.0
```
Also, for vLLM please create:
```bash
conda create -n vllm python=3.12 -y
conda activate vllm
conda install -c conda-forge gcc
pip install vllm
```

### V2T
To create the environment, we suggest using a miniconda container. Then install the following:
```bash
apptainer shell --nv --bind /ib-scratch,/usr/local/cuda:/cuda /ib-scratch/chenguang02/scratch1/cnicholas/containers/miniconda.sif
source ~/.bashrc
conda env create -f v2t.yml -n vmdt-hallu-v2t
pip install datasets joblib matplotlib pydantic openai opencv-python moviepy boto3 anthropic qwen-vl-utils[decord] transformers==4.51.3
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  # May depend on your CUDA version. We use 12.3
export CUDA_HOME=/cuda
export LD_LIBRARY_PATH=/cuda/lib64:$LD_LIBRARY_PATH
```

## Prepare to Run
Download T2V models:
```bash
export HF_TOKEN=<hf_token>
python -m hallucination.download
```

### Text-to-Video pipeline

Run text-to-video generation, evaluation, and scoring in one step using the `text_video` package.

From the project root directory, invoke:
```bash
python -m hallucination/text_video.main --model_id <MODEL_ID> [--scenario <SCENARIO>] [--debug] [--do_not_evaluate]
```

- **MODEL_ID**: one of the supported model IDs (e.g., `VideoCrafter2`, `Vchitect2`, `OpenSora1.2`).
- **SCENARIO**: one of the scenarios (`OCR`, `Distraction`, `Counterfactual`, `Misleading`, `CoOccurrence`, `NaturalSelection`, `Temporal`). Omit to run all scenarios.
- **DEBUG**: If present, will only generate 1 video per task.

This command will save generated videos and evaluation results under `results/t2v_results/hallucination/` and print the output file path to the console. If you do not add the `--do_not_evaluate` flag, it will also run the evaluation script and save the results in `results/t2v_results/hallucination/average.csv`.

If you want to evaluate many files altogether, include the `--do_not_evaluate` flag, which will not run evaluation. Then, to evaluate, first activate the `vllm` environment, then start up vLLM, filling in or changing the parameters as needed:
```
CUDA_VISIBLE_DEVICES=<gpu_ids> vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --port 8001 \
  --host 0.0.0.0 \
  --allowed-local-media-path $(realpath results/t2v_results/hallucination) \
  --limit-mm-per-prompt image=5 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 
```

Then, run the following evaluation script, supplying all output files from the previous step:
```bash
python -m hallucination.text_video.T2V_evaluation_newer --video_json=<output_files> --n_frames=5 --num_instances_per_task=-1 --include_image_in_classification --combine_step --direct_evaluation --use_qwen
```

This will save your evaluation results in a new file in the same directory as <output_file> and save the overall score per model in `t2v_results/hallucination/average.csv`.

## V2T Models
This will both run and evaluate the script.
```bash
python -m hallucination.video_text.main --model_id <MODEL_ID> [--scenario <SCENARIO>] [--debug]
```
- **MODEL_ID**: one of the supported model IDs (e.g., `Qwen/Qwen2.5-VL-72B-Instruct`, `llava_video_7b`, `internvl2_5_8b`).
- **SCENARIO**: one of the scenarios (`OCR`, `Distraction`, `Counterfactual`, `Misleading`, `CoOccurrence`, `NaturalSelection`, `Temporal`). Omit to run all scenarios.
- **DEBUG**: If present, will only generate 1 video per task.

*Note: Using only keyword matching for the multiple choice questions may not yield completely accurate results. For the results in the paper we supplemented with LLM and manual checking.*
