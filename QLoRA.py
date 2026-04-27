"""
1. Run "accelerate config"
2. Run "OMP_NUM_THREADS=$VALUE accelerate launch ./Code/QLoRA.py"
"""

from Config import *
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import json
from datasets import Dataset
from functools import partial
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
import os 

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
                    "content": example["output"][10:]
                }
            ]
        }
        dataset.append(prompt)
    formatted_dataset = Dataset.from_list(dataset)
    return formatted_dataset

def load_pretrained_model(model_path):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank} 
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True, quantization_config=quantization_config, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    loar_config = LoraConfig(
        r=16, # Rank: higher = more parameters. 16 is good for small datasets.
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, loar_config)

    return model, tokenizer

def QLoRA():
    dataset = load_training_dataset("Dataset/train.json")
    dataset = dataset.train_test_split(test_size=0.1)
    model, tokenizer = load_pretrained_model(PRETRAINED_MODEL)
    training_args = SFTConfig(
        per_device_train_batch_size=4,
        learning_rate = 3e-4,
        lr_scheduler_type = "cosine",
        gradient_accumulation_steps=4,
        num_train_epochs = 3,
        bf16 = True,
        logging_steps = 10,
        optim="paged_adamw_32bit",
        completion_only_loss = True,
        packing = False,
        max_length = 2048,
        eval_strategy="epoch",
    )
    trainer = SFTTrainer(
        model = model,
        train_dataset= dataset['train'],
        eval_dataset = dataset['test'],
        args = training_args,
        processing_class = tokenizer,
    )
    trainer.train()
    trainer.save_model("QLoRA_Model")
QLoRA()