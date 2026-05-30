import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

import os

# 选择 BERT 模型
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME,mirror="tuna")
model = BertModel.from_pretrained(MODEL_NAME,mirror="tuna")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 输入和输出目录
input_root = "../../BjTT/text_events"
output_root = "../../BjTT/BERT_embedding_events"

# 确保输出目录存在
# os.makedirs(output_root, exist_ok=True)

# 遍历子目录 (1, 2, 3)
for subdir in os.listdir(input_root):
    input_subdir = os.path.join(input_root, subdir)
    # output_subdir = os.path.join(output_root, subdir)
    # os.makedirs(output_subdir, exist_ok=True)

    if not os.path.isdir(input_subdir):
        continue
    # output_subdir = os.path.join(output_root, subdir)
    output_subdir = output_root
    os.makedirs(output_subdir, exist_ok=True)

    for filename in os.listdir(input_subdir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_subdir, filename)
            output_filename = os.path.splitext(filename)[0] + ".npy"  # 去掉 .txt 只保留 .npy
            output_path = os.path.join(output_subdir, output_filename)

            # 读取文本
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # BERT 编码
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy().astype(np.float32)   # [CLS] 向量

            # 保存为 .npy
            np.save(output_path, embedding)
            print(f"保存: {output_path}")
