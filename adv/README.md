### Adversarial

#### Installation
```bash
conda env create -f environment.yml
conda activate vmdt-adv
```

#### T2V
To run T2V inference, run:
```python adversarial/t2v/inference.py --model CogVideoX-2b --output_path CogVideoX-2b.jsonl --benign --adversarial --num_gpus 8```

Two evaluate:
Create a .env file with an `OPENAI_API_KEY`. Then, run: 
```python adversarial/t2v/evaluation.py --input_file CogVideoX-2b.jsonl --output_file CogVideoX-2b.jsonl --model CogVideoX-2b --benign --adversarial```

#### V2T
To run V2T inference/eval, run: 
```python adversarial/v2t/inference.py --model InternVideo2Chat8B --output_path InternVideo2Chat8B.jsonl --n_gpus 8```
