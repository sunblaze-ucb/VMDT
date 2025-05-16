# Video Generation Models
## Model List

Video parameters are from their official recommendations.

| Model                                                               | Duration   | FPS    | Res.     | Gen. time (H100) |
| ------------------------------------------------------------------- | ---------- | ------ | -------- | ---------------- |
| [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter)          | 16f (1.6s) | 10 fps | 512x320  | ~100s            |
| [HunyuanVideo (TBD, as of now, we can skip it) ](https://github.com/Tencent/HunyuanVideo)             | 129f (5s)  | 24 fps | 1280x720 | ~30min           |
|                                                                     | 129f (5s)  | 24 fps | 832x624  | ~10min           |
|                                                                     |            |        |          | ~6min (2 GPUs)   |
| [~~CogVideoX1.5-5B~~](https://huggingface.co/THUDM/CogVideoX1.5-5B) | 81f (5s)   | 16 fps | 1360x768 | ~3h????          |
| [~~CogVideoX-5B~~](https://huggingface.co/THUDM/CogVideoX-5b)           | 49f (6s)   | 8 fps  | 720x480  | ~5min            |
| [Vchitect-2.0-2B](https://github.com/Vchitect/Vchitect-2.0)         | 40f (5s)   | 8 fps  | 768x432  | ~2min            |
| [OpenSora-1.2](https://github.com/hpcaitech/Open-Sora)              | 102f (4s)  | 24 fps | 1280x720 | ~3min            |
|                                                                     | 204f (8s)  | 24 fps | 1280x720 | ~6min            |

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
