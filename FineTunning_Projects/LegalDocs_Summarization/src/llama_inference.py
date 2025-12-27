from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import os
import time

# Define paths
base_model_name = "meta-llama/Llama-2-7b-hf"  # Base model name
# Use absolute path for the fine-tuned LoRA model
lora_model_path = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/FineTunning_Projects/LegalDocs/fine_tuned_lora_model"

# Check for available devices (CUDA, MPS for Apple Silicon, or CPU)
use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()

if use_cuda:
    print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    device = torch.device("cuda")
    use_gpu = True
elif use_mps:
    print("Apple Silicon GPU (MPS) detected")
    device = torch.device("mps")
    use_gpu = True
else:
    print("No GPU detected, using CPU")
    device = torch.device("cpu")
    use_gpu = False

# Check if fine-tuned model exists, otherwise use base model
if os.path.exists(lora_model_path):
    print(f"Loading fine-tuned LoRA model from: {lora_model_path}")
    # Load base model first
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # MPS doesn't support device_map="auto", use regular loading
    if use_mps:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float32,  # MPS works better with float32
            device_map=None
        )
        base_model = base_model.to(device)
    elif use_cuda:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16,
            device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float32,
            device_map=None
        )
        base_model = base_model.to(device)
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model = model.merge_and_unload()  # Merge LoRA weights with base model
    if not use_gpu:
        model = model.to(device)
else:
    print(f"Fine-tuned model not found at {lora_model_path}, using base model")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # MPS doesn't support device_map="auto", use regular loading
    if use_mps:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float32,  # MPS works better with float32
            device_map=None
        )
        model = model.to(device)
    elif use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float32,
            device_map=None
        )
        model = model.to(device)

# Ensure model is on the correct device
if use_cuda and hasattr(model, 'hf_device_map'):
    # CUDA with device_map="auto" - model is already distributed
    try:
        actual_device = next(model.parameters()).device
        print(f"Model loaded on device: {actual_device}")
    except:
        print(f"Model loaded on device: {device}")
else:
    # MPS or CPU - ensure model is on the device
    model = model.to(device)
    print(f"Model loaded on device: {device}")

# Define your input prompt
# path = "/data/aryan/extras/LLM_project/legal-llm-project/dataset/IN-Ext/judgement/1953_L_1.txt"
# text = open(path, "r").read()
# input_text = "Summarize the following legal text: " + 

# Use the correct dataset path
# path = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/FineTunning_Projects/LegalDocs_Summarization/dataset 2/IN-Abs/test-data/judgement/232.txt"
path = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/FineTunning_Projects/LegalDocs_Summarization/dataset 2/IN-Ext/judgement/1953_L_1.txt"
if os.path.exists(path):
    print(f"Loading document from: {path}")
    text = open(path, "r", encoding="utf-8").read()
    print(f"Document length: {len(text)} characters")
else:
    print(f"Warning: Test file not found at {path}")
    # Use a sample text if file doesn't exist
    text = "This is a sample legal document for testing purposes."
input_text = f"### Instruction: Summarize the following legal text.\n\n### Input:\n{text.strip()[:10000]}\n\n### Response:\n".strip()
print(f"Input prompt length: {len(input_text)} characters")

# input_text = "What is life?"
# breakpoint()

# Tokenize the input text with truncation explicitly set to True
print("Tokenizing input...")
inputs = tokenizer(input_text, max_length=4096, truncation=True, return_tensors="pt", padding=True)
# Add attention mask to avoid the warning
if "attention_mask" not in inputs:
    inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()
print(f"Input tokens: {inputs['input_ids'].shape[1]}")

# Perform inference (generate text from the input)
if use_mps:
    print("Generating summary on Apple Silicon GPU (MPS)...")
    print("Note: MPS generation can take 1-3 minutes for 500 tokens. Please be patient...")
    # MPS works better with greedy decoding and fewer tokens initially
    generation_config = {
        "max_new_tokens": 500,  # Start with fewer tokens for MPS
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,  # Greedy decoding is more reliable on MPS
        "temperature": 1.0,
    }
    print("Using MPS-optimized settings (max_new_tokens=500, greedy decoding)")
elif use_cuda:
    print("Generating summary on CUDA GPU (this should be fast)...")
    generation_config = {
        "max_new_tokens": 1000,  # More tokens for CUDA
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,  # Can use sampling on CUDA
        "temperature": 0.7,
        "top_p": 0.9,
    }
    print("Using CUDA-optimized settings (max_new_tokens=1000)")
else:
    print("Generating summary on CPU (this may take a while)...")
    print("Note: On CPU, this can take several minutes. Please be patient...")
    generation_config = {
        "max_new_tokens": 300,  # Reduced for faster generation on CPU
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,  # Use greedy decoding for faster generation
        "temperature": 1.0,
    }
    print("Using CPU-optimized settings (max_new_tokens=300)")

# Ensure inputs are on the correct device
print(f"Moving inputs to device: {device}")
if use_cuda and hasattr(model, 'hf_device_map'):
    # CUDA with device_map - inputs should go to the first device
    actual_device = next(model.parameters()).device
    inputs = {k: v.to(actual_device) for k, v in inputs.items()}
    print(f"Inputs moved to: {actual_device}")
else:
    # MPS or CPU - use the device we set
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"Inputs moved to: {device}")

# Add a callback to show progress (for MPS, this helps confirm it's working)
if use_mps:
    print("Starting generation... (this will take 1-3 minutes on MPS)")
    start_time = time.time()

print("Generating tokens...")
try:
    generated_output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **generation_config
    )
    if use_mps:
        elapsed = time.time() - start_time
        print(f"Generation complete! Took {elapsed:.2f} seconds")
    else:
        print("Generation complete!")
except Exception as e:
    print(f"Error during generation: {e}")
    print("Trying with simpler settings...")
    # Fallback to simpler generation
    generation_config["do_sample"] = False
    generation_config["max_new_tokens"] = 200
    generated_output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **generation_config
    )
    print("Generation complete with fallback settings!")

# Decode and print the generated output
# generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
generated_text = tokenizer.decode(generated_output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

# print(len(generated_text))
print("Generated Output:")
print(generated_text)