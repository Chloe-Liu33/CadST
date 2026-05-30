#export HF_HUB_DISABLE_CACHE=1
#export TRANSFORMERS_CACHE=/home/limeil/scratch/hf_cache
#export HF_HOME=/home/limeil/scratch/hf_cache
#export TRANSFORMERS_NO_TORCHVISION=1

export HF_HOME=/home/llm/multi-modal/stgcn-main/huggingface
export HF_HUB_ENABLE_HF_ENDPOINT=False
export TRANSFORMERS_OFFLINE=0
export HF_HUB_URL=https://huggingface.co
export HF_ENDPOINT=https://huggingface.tuna.tsinghua.edu.cn/models


python BERT.py
