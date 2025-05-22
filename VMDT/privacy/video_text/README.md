# INSTALL

Use the v2t.yml file to initialize a coda environment. Then each file contains a main that tests the correct execution of the code on a test video.

# Model list

- Qwen2-VL (open source): https://github.com/QwenLM/Qwen2-VL 

- LLaVA-Video (open source): https://huggingface.co/lmms-lab/LLaVA-Video-72B-Qwen2 

- DAMO-NLP-SG/VideoLLaMA2.1-7B-AV (open source): https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-AV 

- InternVL2.5 (open source): https://internvl.github.io/blog/2024-12-05-InternVL-2.5/

- gpt-4o-2024-11-20 (closed source) + gpt-4o-audio-preview-2024-12-17 (for audio support) --> To run gpt4o.py code, please run the following

```bash
pip install opencv-python
pip install moviepy
pip install openai

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.2.post1
```

# Execution

## Running the models
Check the TODOs in each execution file and add the relevant AWS credentials and API keys for Claude, GPT, Nova Lite & Pro
```
python3 claudesonnet.py
python3 gpt4o.py
python3 internvl2_5.py
python3 llava_video.py
python3 mini_cpm.py
python3 nova_lite.py
python3 nova_pro.py
python3 qwen_vl2.py
python3 video_llava.py
python3 videollama2_1.py
```

## Running the evaluation
Check the TODOs in each execution file
```
python3 v2t-city-eval.py
python3 v2t-state-eval.py
python3 v2t-zipcode-eval.py
```
