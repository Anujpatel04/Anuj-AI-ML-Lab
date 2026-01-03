#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig
import yaml


EVALUATION_PROMPTS = [
    {
        "instruction": "Explain what this Python function does",
        "input": """def process_data(data, threshold=0.5):
    filtered = [x for x in data if x > threshold]
    return sum(filtered) / len(filtered) if filtered else 0""",
        "description": "Function explanation"
    },
    {
        "instruction": "Add error handling to this code",
        "input": """def divide(a, b):
    return a / b""",
        "description": "Error handling"
    },
    {
        "instruction": "Refactor this code for better readability",
        "input": """x=[1,2,3,4,5]
y=[]
for i in x:
    if i%2==0:
        y.append(i*2)""",
        "description": "Code refactoring"
    },
    {
        "instruction": "Optimize this function for performance",
        "input": """def find_duplicates(arr):
    result = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j]:
                result.append(arr[i])
    return result""",
        "description": "Performance optimization"
    },
    {
        "instruction": "Add comprehensive documentation to this function",
        "input": """def calculate_total(items, tax_rate):
    subtotal = sum(item['price'] for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax""",
        "description": "Documentation"
    }
]


def load_model(model_path: Path, base_model_name: str = None, load_adapters: bool = True):
    if base_model_name is None:
        config_path = model_path / "training_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                base_model_name = config['model']['name']
        else:
            base_model_name = "deepseek-ai/DeepSeek-Coder-1.3B-Instruct"
    
    print(f"Loading base model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    if load_adapters:
        print(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    
    model.eval()
    return model, tokenizer


def format_prompt(instruction: str, input_text: str = "") -> str:
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate(model, tokenizer, prompt: str, max_tokens: int = 512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    return response


def evaluate(model_path: Path, compare_base: bool = True):
    print("=" * 80)
    print("Code LLM Evaluation")
    print("=" * 80)
    
    print("\nLoading fine-tuned model...")
    ft_model, ft_tokenizer = load_model(model_path, load_adapters=True)
    
    base_model = None
    base_tokenizer = None
    if compare_base:
        print("\nLoading base model for comparison...")
        base_model, base_tokenizer = load_model(model_path, load_adapters=False)
    
    print("\n" + "=" * 80)
    print("Running Evaluation")
    print("=" * 80 + "\n")
    
    for i, prompt_data in enumerate(EVALUATION_PROMPTS, 1):
        instruction = prompt_data["instruction"]
        input_text = prompt_data["input"]
        description = prompt_data["description"]
        
        print(f"\n{'='*80}")
        print(f"Test {i}: {description}")
        print(f"{'='*80}")
        print(f"Instruction: {instruction}")
        print(f"\nInput Code:\n{input_text}\n")
        
        prompt = format_prompt(instruction, input_text)
        
        print("Fine-Tuned Model Response:")
        print("-" * 80)
        ft_response = generate(ft_model, ft_tokenizer, prompt)
        print(ft_response)
        
        if compare_base:
            print("\nBase Model Response:")
            print("-" * 80)
            base_response = generate(base_model, base_tokenizer, prompt)
            print(base_response)
        
        print("\n")
    
    print("=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned code LLM')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model directory')
    parser.add_argument('--no_compare', action='store_true',
                        help='Skip base model comparison')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return
    
    evaluate(model_path, compare_base=not args.no_compare)


if __name__ == '__main__':
    main()
