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

## Execution
Make sure to modify the TODOs in the below files as well as include the Nova Reel, Pika, and Luma API keys and AWS credentials.
Run the following
```
python3 t2v-privacy-gen.py
python3 t2v-privacy-process-frame-crossjoins.py
python3 t2v-privacy-calc-distances.py
python3 t2v-privacy-save-similar-frames.py
```
