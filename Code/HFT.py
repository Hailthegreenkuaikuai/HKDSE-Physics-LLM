"""
1. Run "accelerate config"
2. Run "OMP_NUM_THREADS=$VALUE accelerate launch ./Code/HFT.py"
"""

from Config import *
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
import random
import json
from datasets import Dataset
import random 

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

def parameter_selection(model):
    random.seed(114514) 
    FFNs = []
    SANs = []
    LNs = []

    transformer_layers = model.model.layers
    num_layers = len(transformer_layers)

    mark_layers = random.sample(range(num_layers), num_layers // 2)
    
    for(i, layer) in enumerate(transformer_layers):
        for name, parameter in layer.named_parameters():
            if "self_attn" in name and "weight" in name:
                SANs.append(parameter)
            elif "mlp" in name and "weight" in name:
                FFNs.append(parameter)
            elif "layernorm" in name:
                LNs.append(parameter)

    freeze_san = random.sample(SANs, len(SANs) // 2)
    for p in freeze_san:
        p.requires_grad = False
    
    if i in mark_layers:
        num_to_freeze = int(torch.ceil(torch.tensor(len(FFNs) / 2)).item())
    else:
        num_to_freeze = int(torch.floor(torch.tensor(len(FFNs) / 2)).item())
    freeze_ffn = random.sample(FFNs, num_to_freeze)
    for p in freeze_ffn:
        p.requires_grad = False

    freeze_ln = random.sample(SANs, len(LNs) // 2)
    for p in freeze_ln:
        p.requires_grad = False

    model.get_input_embeddings().weight.requires_grad = True
    model.get_output_embeddings().weight.requires_grad = True

    return model

def load_pretrained_model(model_path: str) -> {LlamaForCausalLM, AutoTokenizer}:
    model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    model = parameter_selection(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.gradient_checkpointing_enable()

    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"\nTotal Parameters: {total_params:,}")
    # print(f"Trainable Parameters: {trainable_params:,}")
    # print(f"Trainable Ratio: {100 * trainable_params / total_params:.2f}%")

    return model, tokenizer
def HFT():
    dataset = load_training_dataset("Dataset/train.json")
    dataset = dataset.train_test_split(test_size=0.1)
    model, tokenizer = load_pretrained_model(PRETRAINED_MODEL)
    training_args = SFTConfig(
        per_device_train_batch_size=4,
        learning_rate = 2.5e-5,
        lr_scheduler_type = "cosine",
        max_grad_norm = 1,
        gradient_accumulation_steps=4,
        num_train_epochs = 3,
        bf16 = True,
        logging_steps = 10,
        completion_only_loss = True,
        dataset_text_field="completion",
        packing = False,
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
    trainer.save_model("HFT_Model")
HFT()