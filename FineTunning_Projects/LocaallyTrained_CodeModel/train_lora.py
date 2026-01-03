#!/usr/bin/env python3

import os
import json
import yaml
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset


def load_config(config_path: Path) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(dataset_path: Path) -> Dataset:
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    return Dataset.from_list(data)


def format_prompt(example: Dict) -> str:
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return prompt


def tokenize_function(examples: Dict, tokenizer, max_length: int = 2048) -> Dict:
    instructions = examples.get('instruction', [])
    inputs = examples.get('input', [])
    outputs = examples.get('output', [])
    
    prompts = [
        format_prompt({
            'instruction': inst,
            'input': inp,
            'output': out
        })
        for inst, inp, out in zip(instructions, inputs, outputs)
    ]
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None
    )
    
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized


def setup_model_and_tokenizer(config: Dict):
    model_name = config['model']['name']
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading model: {model_name}")
    
    use_quantization = config['model'].get('load_in_4bit', False)
    
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    
    if use_quantization:
        print("Using 4-bit quantization (CPU only - bitsandbytes doesn't support MPS)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        device_map = None
        torch_dtype = torch.float16
    else:
        quantization_config = None
        device_map = None
        if use_mps:
            print("Using MPS (Apple Silicon GPU) - no quantization")
            if hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported() or use_mps:
                torch_dtype = torch.bfloat16
                print("Using bfloat16 for MPS (more stable)")
            else:
                torch_dtype = torch.float16
                print("Using float16 for MPS")
        elif use_cuda:
            print("Using CUDA GPU - no quantization")
            torch_dtype = torch.float16
        else:
            print("Using CPU - no quantization")
            torch_dtype = torch.float16
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": device_map,
    }
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    if hasattr(model, 'hf_device_map'):
        try:
            delattr(model, 'hf_device_map')
        except:
            pass
    if hasattr(model, 'device_map'):
        try:
            delattr(model, 'device_map')
        except:
            pass
    
    if use_quantization:
        model = prepare_model_for_kbit_training(model)
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        elif hasattr(model, 'config'):
            model.config.use_cache = True
    
    if not use_quantization and use_mps:
        print("Moving model to MPS device...")
        try:
            model = model.to("mps")
            test_tensor = torch.zeros(1, device="mps")
            del test_tensor
            print("✓ MPS device verified")
        except Exception as e:
            print(f"Warning: MPS device issue: {e}")
            print("Falling back to CPU...")
            model = model.to("cpu")
            use_mps = False
    
    if hasattr(model, 'to'):
        pass
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    return model, tokenizer


def train(config: Dict):
    print("=" * 80)
    print("Code LLM Fine-Tuning with QLoRA")
    print("=" * 80)
    
    dataset_path = Path(config['data']['dataset_path'])
    print(f"\nLoading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"Dataset size: {len(dataset)} examples")
    
    model, tokenizer = setup_model_and_tokenizer(config)
    
    print("\nTokenizing dataset...")
    tokenize_fn = lambda examples: tokenize_function(examples, tokenizer, config['training']['max_length'])
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print(f"Tokenized dataset size: {len(tokenized_dataset)} examples")
    
    val_split = config['training'].get('val_split', 0.1)
    min_val_samples = 5
    
    if len(tokenized_dataset) < min_val_samples * 2:
        print(f"Warning: Dataset too small for validation split. Using all data for training.")
        train_dataset = tokenized_dataset
        val_dataset = tokenized_dataset
    else:
        test_size = max(val_split, min_val_samples / len(tokenized_dataset))
        if test_size >= 1.0:
            test_size = 0.1
        
        split_dataset = tokenized_dataset.train_test_split(
            test_size=test_size,
            seed=42
        )
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(train_dataset) == len(val_dataset):
        eval_strategy = "no"
        eval_steps = None
        load_best_model_at_end = False
        metric_for_best_model = None
    else:
        eval_strategy = "steps"
        eval_steps = config['training'].get('eval_steps', 100)
        load_best_model_at_end = True
        metric_for_best_model = "eval_loss"
    
    use_quantization = config['model'].get('load_in_4bit', False)
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    
    use_fp16 = False
    use_bf16 = False
    
    if use_quantization:
        print("Note: Using CPU for training (bitsandbytes quantization doesn't support MPS)")
    else:
        if use_mps:
            print("Using MPS (Apple Silicon GPU) for training")
            use_bf16 = True
        elif use_cuda:
            print("Using CUDA GPU for training")
            use_fp16 = True
        else:
            print("Using CPU for training")
    
    training_args_dict = {
        "output_dir": str(output_dir),
        "num_train_epochs": config['training']['num_epochs'],
        "per_device_train_batch_size": config['training']['batch_size'],
        "per_device_eval_batch_size": config['training']['batch_size'],
        "gradient_accumulation_steps": config['training'].get('gradient_accumulation_steps', 4),
        "learning_rate": config['training']['learning_rate'],
        "fp16": use_fp16,
        "bf16": use_bf16,
        "logging_steps": config['training'].get('logging_steps', 5),
        "save_steps": config['training'].get('save_steps', 100),
        "gradient_checkpointing": False,
        "save_total_limit": config['training'].get('save_total_limit', 3),
        "warmup_steps": config['training'].get('warmup_steps', 100),
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
    }
    
    if eval_strategy != "no":
        training_args_dict["eval_strategy"] = eval_strategy
        training_args_dict["eval_steps"] = eval_steps
        training_args_dict["load_best_model_at_end"] = load_best_model_at_end
        if metric_for_best_model:
            training_args_dict["metric_for_best_model"] = metric_for_best_model
            training_args_dict["greater_is_better"] = False
    
    training_args = TrainingArguments(**training_args_dict)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    if use_mps and not use_quantization:
        print("\nTesting MPS with a small forward pass...")
        try:
            test_input = tokenizer("Test", return_tensors="pt", padding=True, truncation=True)
            test_input = {k: v.to("mps") for k, v in test_input.items()}
            with torch.no_grad():
                _ = model(**test_input)
            print("✓ MPS forward pass successful")
        except Exception as e:
            print(f"⚠ MPS test failed: {e}")
            print("Falling back to CPU for training...")
            model = model.to("cpu")
            use_mps = False
            training_args_dict["bf16"] = False
            training_args_dict["fp16"] = False
            training_args = TrainingArguments(**training_args_dict)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    if hasattr(trainer.model, 'hf_device_map'):
        try:
            delattr(trainer.model, 'hf_device_map')
        except:
            pass
    if hasattr(trainer.model, 'device_map'):
        try:
            delattr(trainer.model, 'device_map')
        except:
            pass
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    if use_mps and not use_quantization:
        print("\n" + "=" * 80)
        print("⚠ MPS TRAINING WARNING")
        print("=" * 80)
        print("MPS (Apple Silicon GPU) training is known to hang with Transformers Trainer.")
        print("If training gets stuck, please:")
        print("1. Stop the training (Ctrl+C)")
        print("2. Set 'load_in_4bit: true' in config.yaml")
        print("3. Restart training (will use CPU with quantization - more reliable)")
        print("=" * 80 + "\n")
        print("Attempting MPS training anyway...")
        print("(If it hangs, use the CPU option above)\n")
    
    print("Training started... This may take a while on CPU.")
    print("First step can take 1-2 minutes to initialize...")
    print()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial model checkpoints are saved in:", output_dir)
        raise
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        print("Check the error above for details.")
        raise
    
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    print(f"\nTraining complete! Model saved to: {final_model_dir}")
    
    with open(final_model_dir / "training_config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return trainer, model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Fine-tune code LLM with QLoRA')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return
    
    config = load_config(config_path)
    
    if torch.backends.mps.is_available():
        print("✓ Apple Silicon (MPS) detected")
        device = "mps"
    elif torch.cuda.is_available():
        print("✓ CUDA detected")
        device = "cuda"
    else:
        print("⚠ Using CPU (slower)")
        device = "cpu"
    
    train(config)


if __name__ == '__main__':
    main()
