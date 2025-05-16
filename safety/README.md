# VMDT - Safety
## Installation
```bash
pip3 install --no-deps -r requirements.txt
pip3 install flash-attn --no-build-isolation
pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

pip3 install -e .
```

### Download Models
download `Hunyuan`, `OpenSora`, `VideoCrafter`, `Vchitect`
```bash
python3 download.py
```

### Download Data
```bash
huggingface-cli download mmfm-trust/V2T --include 'safety/**' --local-dir data/v2t_data --repo-type dataset
```

## Evaluation
prepare the api keys in `.env`
```bash
cp .env.template .env
```

### T2V Models
```bash
# check model list
python3 scripts/t2v_eval.py --help

# run the model
python3 scripts/t2v_eval.py --out_dir out/rmit --models Pika --run_model --run_judge
```

### V2T Models
```bash
# check model list
python3 scripts/v2t_eval.py --help

# run the model
python3 scripts/v2t_eval.py --out_dir out/rmit --models GPT-4o --data_dir data/v2t_data --run_model --run_judge
```