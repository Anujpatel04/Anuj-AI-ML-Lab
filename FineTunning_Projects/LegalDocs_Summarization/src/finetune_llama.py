import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig  
from peft import LoraConfig, get_peft_model

preprocessed_data_dir = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/FineTunning_Projects/dataset/processed-IN-Ext/"
output_dir = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/FineTunning_Projects/LegalDocs/results_lora"
model_save_dir = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/FineTunning_Projects/LegalDocs/fine_tuned_lora_model"

model_name = "meta-llama/Llama-2-7b-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  

from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True  
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quantization_config,
    device_map="auto"
)

lora_config = LoraConfig(
    lora_alpha=8,          
    lora_dropout=0.1,      
    r=8,                   
    bias="none",           
    task_type="CAUSAL_LM"  
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

def load_dataset(jsonl_file):
    """
    Load preprocessed data and format it into a structured text field.
    """
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    system_prompt = "Summarize the following legal text."

    texts = []
    for item in data:
        text = f"""### Instruction: {system_prompt}

### Input:
{item['judgement'].strip()[:10000]}

### Response:
{item['summary'].strip()}
""".strip()
        texts.append(text)

    dataset = Dataset.from_dict({"text": texts})
    return dataset

train_file_A1 = os.path.join(preprocessed_data_dir, "full_summaries_A1.jsonl")
train_file_A2 = os.path.join(preprocessed_data_dir, "full_summaries_A2.jsonl")

train_dataset_A1 = load_dataset(train_file_A1)
train_dataset_A2 = load_dataset(train_file_A2)

train_data = concatenate_datasets([train_dataset_A1, train_dataset_A2])

train_params = SFTConfig(
    output_dir=output_dir,               # Output directory for model checkpoints
    num_train_epochs=3,                  # Number of epochs
    per_device_train_batch_size=1,       # Batch size per device
    gradient_accumulation_steps=1,       # Accumulate gradients before updating model
    optim="paged_adamw_32bit",           # Optimizer to use
    save_steps=50,                       # Save checkpoints every 50 steps
    logging_steps=50,                    # Log training progress every 50 steps
    learning_rate=5e-3,                  # Learning rate
    weight_decay=0.001,                  # Weight decay for regularization
    fp16=True,                           # Enable mixed precision for stability
    bf16=False,                          # Disable bfloat16
    max_grad_norm=0.3,                   # Gradient clipping norm
    warmup_ratio=0.03,                   # Warm-up ratio for learning rate scheduler
    group_by_length=True,                # Group samples by length to minimize padding
    lr_scheduler_type="constant",        # Use a constant learning rate
    report_to="tensorboard",             # Log to TensorBoard for visualization
    dataset_text_field="text",           # Column containing the text for training
    max_seq_length=4096                  # Maximum sequence length for input text
)

fine_tuning = SFTTrainer(
    model=model,
    train_dataset=train_data,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=train_params
)

print("Starting fine-tuning...")
fine_tuning.train()

print("Saving the fine-tuned model...")
os.makedirs(model_save_dir, exist_ok=True)
model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)
print(f"Fine-tuned model saved at '{model_save_dir}'")
