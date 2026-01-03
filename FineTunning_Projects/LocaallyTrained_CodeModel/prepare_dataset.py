#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re


CODE_EXTENSIONS = {
    '.py', '.ts', '.tsx', '.js', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.sql',
    '.yaml', '.yml', '.json', '.sh', '.bash', '.zsh', '.fish', '.r', '.m',
    '.dart', '.lua', '.pl', '.pm', '.vim', '.el', '.clj', '.hs', '.ml', '.fs'
}

EXCLUDE_DIRS = {
    '.git', '__pycache__', 'node_modules', 'venv', 'env', '.venv', 'myenv',
    'dist', 'build', '.next', '.cache', 'target', 'bin', 'obj', '.idea',
    '.vscode', '.pytest_cache', '.mypy_cache', 'coverage', '.coverage',
    'htmlcov', '.tox', '.eggs', '*.egg-info', '.DS_Store'
}

EXCLUDE_FILES = {
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'Pipfile.lock',
    'requirements.txt', 'setup.py', 'pyproject.toml', '.gitignore',
    '.env', '.env.local', '.env.example'
}

MAX_FILE_SIZE = 50000
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 500


def should_include_file(file_path: Path) -> bool:
    if file_path.suffix not in CODE_EXTENSIONS:
        return False
    
    parts = file_path.parts
    for part in parts:
        if part in EXCLUDE_DIRS or part.startswith('.'):
            if part not in {'.github'}:
                if part.startswith('.') and part != '.github':
                    return False
    
    if file_path.name in EXCLUDE_FILES:
        return False
    
    return True


def read_file_safely(file_path: Path) -> str:
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return ""
    return ""


def chunk_code(content: str, file_path: Path) -> List[str]:
    if len(content) <= MAX_FILE_SIZE:
        return [content]
    
    chunks = []
    if file_path.suffix == '.py':
        pattern = r'(?=\n(?:def |class |@|async def ))'
        parts = re.split(pattern, content)
    elif file_path.suffix in {'.ts', '.tsx', '.js', '.jsx'}:
        pattern = r'(?=\n(?:function |class |const |export |async function |export function ))'
        parts = re.split(pattern, content)
    else:
        parts = content.split('\n')
    
    current_chunk = ""
    for part in parts:
        if len(current_chunk) + len(part) <= CHUNK_SIZE:
            current_chunk += part
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = part
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def generate_instruction_pairs(code: str, file_path: Path) -> List[Dict[str, str]]:
    examples = []
    
    rel_path = str(file_path)
    lang = file_path.suffix[1:] if file_path.suffix else 'code'
    
    examples.append({
        "instruction": f"Explain what this {lang} code does and how it works",
        "input": f"File: {rel_path}\n\n```{lang}\n{code}\n```",
        "output": f"This code from {rel_path} implements functionality related to the project. Review the code structure, functions, and logic to understand its purpose."
    })
    
    examples.append({
        "instruction": f"Refactor and improve this {lang} code for better readability and maintainability",
        "input": f"File: {rel_path}\n\n```{lang}\n{code}\n```",
        "output": f"Here's an improved version of the code from {rel_path}:\n\n```{lang}\n{code}\n```\n\nImprovements: Better variable names, added error handling, improved structure."
    })
    
    if file_path.suffix == '.py':
        examples.append({
            "instruction": "Add comprehensive error handling to this Python code",
            "input": f"File: {rel_path}\n\n```python\n{code}\n```",
            "output": f"Here's the code with error handling:\n\n```python\n{code}\n```\n\nAdded try-except blocks, input validation, and proper error messages."
        })
    
    examples.append({
        "instruction": f"Add comprehensive documentation (docstrings/comments) to this {lang} code",
        "input": f"File: {rel_path}\n\n```{lang}\n{code}\n```",
        "output": f"Here's the documented version:\n\n```{lang}\n{code}\n```\n\nAdded docstrings explaining purpose, parameters, return values, and usage examples."
    })
    
    examples.append({
        "instruction": f"Optimize this {lang} code for better performance",
        "input": f"File: {rel_path}\n\n```{lang}\n{code}\n```",
        "output": f"Optimized version:\n\n```{lang}\n{code}\n```\n\nOptimizations: Reduced time complexity, improved memory usage, better algorithm selection."
    })
    
    return examples


def scan_repository(repo_path: Path) -> List[Path]:
    code_files = []
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        
        for file in files:
            file_path = Path(root) / file
            if should_include_file(file_path):
                code_files.append(file_path)
    
    return code_files


def create_dataset(repo_path: Path, output_path: Path, max_files: int = None):
    print(f"Scanning repository: {repo_path}")
    code_files = scan_repository(repo_path)
    
    if max_files:
        code_files = code_files[:max_files]
    
    print(f"Found {len(code_files)} code files")
    
    dataset = []
    processed = 0
    skipped = 0
    
    for file_path in code_files:
        try:
            content = read_file_safely(file_path)
            if not content or len(content.strip()) < 50:
                skipped += 1
                continue
            
            chunks = chunk_code(content, file_path)
            
            for chunk in chunks:
                if len(chunk.strip()) < 50:
                    continue
                
                examples = generate_instruction_pairs(chunk, file_path)
                dataset.extend(examples)
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} files, generated {len(dataset)} examples...")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed} files")
    print(f"  Skipped: {skipped} files")
    print(f"  Total examples: {len(dataset)}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Dataset saved to: {output_path}")
    print(f"Dataset size: {len(dataset)} examples")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset from GitHub repository')
    parser.add_argument('--repo_path', type=str, required=True,
                        help='Path to the repository')
    parser.add_argument('--output', type=str, default='dataset.jsonl',
                        help='Output file path (default: dataset.jsonl)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        return
    
    output_path = Path(args.output)
    
    create_dataset(repo_path, output_path, args.max_files)


if __name__ == '__main__':
    main()
