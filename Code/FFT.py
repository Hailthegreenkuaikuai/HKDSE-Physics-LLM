"""
1. Run "accelerate config"
2. Run "OMP_NUM_THREADS=$VALUE accelerate launch ./Code/FFT.py"
"""

from Config import *
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
import json
from datasets import Dataset
from functools import partial

def load_training_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = []
    for id, example in enumerate(data):
        input = example["instruction"]
        if example["input"].strip():
            input += "\n" + example["input"]
        prompt = {
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": input
                }
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": example["output"]
                }
            ]
        }
        dataset.append(prompt)
    formatted_dataset = Dataset.from_list(dataset)
    return formatted_dataset

def load_pretrained_model(model_path):
    model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    #model.config.pad_token_id = tokenizer.pad_token_id
    #model.config.eos_token_id = tokenizer.eos_token_id
    #model.config.bos_token_id = tokenizer.bos_token_id
    model.gradient_checkpointing_enable()
    return model, tokenizer

def full_fine_tune():
    dataset = load_training_dataset("Dataset/train.json")
    dataset = dataset.train_test_split(test_size=0.1)
    model, tokenizer = load_pretrained_model(PRETRAINED_MODEL)
    training_args = SFTConfig(
        per_device_train_batch_size=4,
        learning_rate = 3e-5,
        max_grad_norm = 1,
        lr_scheduler_type = "cosine",
        gradient_accumulation_steps=4,
        num_train_epochs = 3,
        bf16 = True,
        logging_steps = 5,
        completion_only_loss = True,
        packing = True,
        max_length = 2048,
        eval_strategy="epoch",
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset= dataset['train'],
        eval_dataset = dataset['test'],
        args = training_args,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model("FFT_Model")
    
full_fine_tune()