"""
Load model, dataset etc. & Generate response
"""
from Config import *
from transformers import pipeline, LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import json
from pathlib import Path
import datetime
import ollama
from Tools import available_functions, tools_schema

def load_model_and_tokenizer(model_path):
    if(model_path == "QLoRA_Model"):
        adapter_path = model_path
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = LlamaForCausalLM.from_pretrained(
            PRETRAINED_MODEL,
            quantization_config = bnb_config,
            local_files_only = True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True, device_map="auto",)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, return_full_text = False, device_map = "auto")
    return model, generator, tokenizer

def load_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = []
    for id, example in enumerate(data):
        input = example["instruction"]
        if example["input"].strip():
            input += "\n" + example["input"]

        dataset.append({
            "prompt": input,
            "reference": example["output"]
        })
    print(f"Loaded {len(dataset)} examples for evaluation")
    return dataset

#Generate responses given prompt 
def responses_generation(generator, tokenizer, dataset, batch_size = 8):
    generated_responses = []
    references = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating responses"):
        batch = dataset[i:i+batch_size]
        prompts = []
        for ex in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["prompt"]}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True,
            )
            prompts.append(prompt)
        outputs = generator(
            prompts,
            do_sample = True,
            temperature = 0.2,
            top_p = 0.5,
            max_new_tokens = 1024,
            batch_size = batch_size,
            pad_token_id = generator.tokenizer.eos_token_id,
            eos_token_id = generator.tokenizer.eos_token_id,
            repetition_penalty = 1.02,
        )
        for j, output in enumerate(outputs):
            generated_text = output[0]['generated_text']
            #print(generated_text)
            generated_responses.append(generated_text)
            references.append(batch[j]["reference"])
    return generated_responses, references

def ollama_responses_generation(dataset, model):
    generated_responses = []
    references = []

    for ex in tqdm(dataset, desc="Processing with Ollama Tools"):
        messages = [{'role': 'user', 'content': ex["prompt"]}]

        response = ollama.chat(
            model=model,
            messages=messages,
            tools=tools_schema, 
        )
        print(response)
        if response.get('message', {}).get('tool_calls'):
            for tool in response['message']['tool_calls']:
                if tool['function']['name'] == 'calculate':
                    args = tool['function']['arguments']
                    result = calculate(args['expression'])

                    messages.append(response['message'])
                    messages.append({
                        'role': 'tool',
                        'content': result,
                    })

            final_response = ollama.chat(model=model, messages=messages)
            generated_responses.append(final_response['message']['content'])
        else:
            generated_responses.append(response['message']['content'])
            
        references.append(ex["reference"])

    return generated_responses, references

def strip_thought(responses):
    clean_responses = []
    for response in responses:
        index = response.find("Question: ")
        if index != -1:
            clean = response[index:]
            clean_responses.append(clean)
        else:
            clean_responses.append(response)
    return clean_responses
        

def output_result(BERTScore, BLEURT, SelfCheckGPT, Perplexity):
    result_str = "\n" + "=" * 50
    result_str = f"\nTotal samples: {BERTScore['total_examples']}"
    result_str = "\n" + "=" * 50
    result_str += "\nBERTScore Evaluation Results"
    result_str += f"\nF1 Score: {BERTScore['f1']['mean']:.4f} ± {BERTScore['f1']['std']:.4f}"
    result_str += f"\nPrecision: {BERTScore['precision']['mean']:.4f} ± {BERTScore['precision']['std']:.4f}"
    result_str += f"\nRecall: {BERTScore['recall']['mean']:.4f} ± {BERTScore['recall']['std']:.4f}"
    result_str += "\n"+"="*50
    result_str += f"\nBLEURT: {BLEURT:.4f}"
    result_str += "\n"+"="*50 
    result_str += f"\nHallucination Score: {SelfCheckGPT:.4f}"
    result_str += "\n"+"="*50
    result_str += f"\nPerplexity scores: {Perplexity:.4f}"
    result_str += "\n"+"="*50
    return result_str

def output_responses(generated_responses, references, dataset):
    Path("Generated_Responses").mkdir(parents=True, exist_ok=True)
    output = "GENERATED RESPONSES"
    output += "\n"+"="*50
    for i, (gen, ref) in enumerate(zip(generated_responses, references)):
        output += f"\nSample {i+1}:"
        output += f"\nPrompt: {dataset[i]['prompt']}"
        output += f"\nGenerated: {gen}"
        output += f"\nReference: {ref}"
        output += "\n"+"="*50
    return output
