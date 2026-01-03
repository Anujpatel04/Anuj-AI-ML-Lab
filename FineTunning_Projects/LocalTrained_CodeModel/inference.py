#!/usr/bin/env python3

import argparse
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig


def load_model_and_tokenizer(model_path: Path, base_model_name: str = None):
    model_path = Path(model_path)
    
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if base_model_name is None:
        config_path = model_path / "training_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                base_model_name = config['model']['name']
        else:
            base_model_name = "deepseek-ai/DeepSeek-Coder-1.3B-Instruct"
    
    print(f"Loading base model: {base_model_name}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    return model, tokenizer


def format_prompt(instruction: str, input_text: str = "") -> str:
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
):
    prompt = format_prompt(instruction, input_text)
    
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def interactive_mode(model, tokenizer, config: dict = None):
    print("\n" + "=" * 80)
    print("Interactive Code LLM Inference")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 80 + "\n")
    
    inference_config = config.get('inference', {}) if config else {}
    
    while True:
        try:
            instruction = input("Instruction: ").strip()
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                break
            
            if not instruction:
                continue
            
            input_text = input("Input (optional, press Enter to skip): ").strip()
            if not input_text:
                input_text = ""
            
            print("\nGenerating response...")
            response = generate_response(
                model,
                tokenizer,
                instruction,
                input_text,
                max_new_tokens=inference_config.get('max_new_tokens', 512),
                temperature=inference_config.get('temperature', 0.7),
                top_p=inference_config.get('top_p', 0.9),
                do_sample=inference_config.get('do_sample', True)
            )
            
            print(f"\nResponse:\n{response}\n")
            print("-" * 80 + "\n")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned code LLM')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model directory')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Base model name (auto-detected from config if not provided)')
    parser.add_argument('--instruction', type=str, default=None,
                        help='Instruction prompt')
    parser.add_argument('--input', type=str, default="",
                        help='Input text (optional)')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return
    
    model, tokenizer = load_model_and_tokenizer(model_path, args.base_model)
    
    config = None
    config_path = model_path / "training_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    if args.interactive or args.instruction is None:
        interactive_mode(model, tokenizer, config)
    else:
        response = generate_response(
            model,
            tokenizer,
            args.instruction,
            args.input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(response)


if __name__ == '__main__':
    main()
