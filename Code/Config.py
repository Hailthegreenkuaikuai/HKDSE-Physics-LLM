"""
Universal Config
"""

import torch
#Beware of the current directory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_MODEL = "Llama-3.2-3B-Instruct" #Pre-trained Model
FFT_MODEL = "FFT_Model"#Full Fine-tuned Model
HFT_MODEL = "HFT_Model"#Half Fine-tuned Model
QLORA_MODEL = "QLoRA_Model"#QLoRA Model
TEST_DATA = "Dataset/test.json" #Testing Data
TRAIN_DATA = "Dataset/train.json" #Training Data
EXPLANATION_DATA = "Dataset/explanation.json"
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"

with open("Code/System_Prompt.txt", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read() #System Prompt
f.close()