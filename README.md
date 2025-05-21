# VMDT

This repo contains the source code of VMDT (Video-Modal DecodingTrust). This research endeavor is designed to help researchers and practitioners better understand the capabilities, limitations, and potential risks involved in deploying these state-of-the-art Video Foundation Models (VFMs). 

This benchmark is organized around the following five key perspectives of trustworthiness:
1. Safety
2. Hallucination
3. Fairness
4. Privacy
5. Adversarial robustness

This repo supports the evaluation of 7 T2V models and 19 V2T models. To evaluate an additional model, please add its code.

## Project Structure

This project is structured around subdirectories dedicated to each area of trustworthiness. Each subdir includes scripts, data, and a dedicated README for easy comprehension.

## Dataset 

Our dataset is available at https://huggingface.co/datasets/mmfm-trust/T2V, https://huggingface.co/datasets/mmfm-trust/V2T 

## Getting Started

### Clone the repository

```bash
git clone https://github.com/sunblaze-ucb/VMDT.git 
```

### How to run the code

```bash
python -m VMDT.[perspective].[modality].main --model_id [model_id]
```

For example, you can run the following command:

```bash
python -m VMDT.fairness.text_video.main --model_id Vchitect2
```

This will create a result file in the results folder.

Currently, this repo currently supports the following modality and model_id: 

```bash
text_video: ['Nova', 'Pika', 'Luma', 'OpenSora1.2', 'Vchitect2', 'VideoCrafter2', 'CogVideoX-5B']
video_text: []
```