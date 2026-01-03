# Code LLM Fine-Tuning Pipeline

> **Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)** - A comprehensive collection of AI/ML projects, LLM applications, agents, RAG systems, and core machine learning implementations.

A production-ready pipeline for fine-tuning code-focused language models locally on Apple Silicon (MacBook M4) using QLoRA. Trains models on your GitHub repository to learn your coding style and project structure.

## Features

- Fine-tune code LLMs using QLoRA (memory-efficient)
- Train on your own codebase
- Generate code in your style, refactor, explain, and optimize
- Run entirely locally (no cloud required)

## Model

**DeepSeek-Coder-1.3B-Instruct** - 1.3B parameters, code-focused, optimized for Apple Silicon. Supports 4-bit quantization for memory efficiency.

## Quick Start

### Installation

```bash
cd FineTunning_Projects/CodeModel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

1. **Prepare dataset from repository:**
```bash
python prepare_dataset.py \
    --repo_path /path/to/your/repo \
    --output dataset.jsonl
```

2. **Train model:**
```bash
python train_lora.py --config config.yaml
```

3. **Use fine-tuned model:**
```bash
# Interactive mode
python inference.py --model_path outputs/final_model --interactive

# Single query
python inference.py \
    --model_path outputs/final_model \
    --instruction "Explain this code" \
    --input "your code here"
```

4. **Evaluate:**
```bash
python evaluate.py --model_path outputs/final_model
```

## Configuration

Edit `config.yaml` to customize:
- **Model**: Base model name, quantization settings
- **LoRA**: Rank (r: 16), alpha (32), dropout (0.1)
- **Training**: Epochs (3), batch size (1), learning rate (2e-4)

**Note**: Set `load_in_4bit: true` for CPU training (stable) or `false` for MPS GPU (faster but may hang).

## How It Works

1. **Dataset Creation**: Scans repository, filters code files, generates instruction-following examples
2. **Training**: Loads base model with 4-bit quantization, applies LoRA adapters (~50MB trainable), trains on dataset
3. **Inference**: Loads base model + LoRA adapters for code generation

## Troubleshooting

- **Out of memory**: Reduce `batch_size`, increase `gradient_accumulation_steps`
- **MPS hangs**: Use CPU training (`load_in_4bit: true`)
- **Slow training**: Limit dataset size, reduce epochs or max_length
- **Poor quality**: Increase LoRA rank, train longer, improve dataset

## Requirements

- Python 3.10+ (3.11 or 3.12 recommended)
- MacBook M4 or compatible Apple Silicon
- 16GB+ RAM recommended
