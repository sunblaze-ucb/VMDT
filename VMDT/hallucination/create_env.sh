# python3 -m venv mmenv
# source mmenv/bin/activate

# Todo: write an install_requirements.sh. If on cluster ensure I am in singularity container first!
pip install openai
pip install requests
pip install pandas
pip install --upgrade transformers accelerate imageio imageio-ffmpeg   # for cogvideo
pip install git+https://github.com/huggingface/diffusers
pip install sentencepiece
pip install opencv-python-headless
pip install inflect  # for number to words
pip install joblib  # for caching
pip install av  # for xclip
pip install pyarrow  # for parquet for pexels dataset for co-occurrence
pip install spacy  # for co-occurrence
pip install hydra-core bert-score  # for advsarial
python -m spacy download en_core_web_sm
pip install diskcache  # for caching async
pip install langfuse python-dotenv
pip install flask  # for human annotation pipeline
pip install decord easydict protobuf timm einops peft==0.5.0

# For local development
pip install -e .
