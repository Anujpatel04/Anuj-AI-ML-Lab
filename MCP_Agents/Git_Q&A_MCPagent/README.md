# Codebase Q&A MCP Agent

> **Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)**

A production-grade Codebase Q&A Agent using Model Context Protocol (MCP) that analyzes Git repositories and answers architecture and code-level questions using RAG (Retrieval-Augmented Generation).

## Features

- Repository scanning with `.gitignore` support
- AST-aware code parsing with Tree-sitter
- Semantic embeddings and intelligent chunking
- FAISS-based vector search
- Persistent MCP context for multi-turn reasoning
- Question answering with file paths and line numbers

## Usage

### Web Frontend

```bash
./scripts/run_frontend.sh
```

Open `http://localhost:8501`, connect a repository, index it, and ask questions.

### Command Line

**Index repository:**
```bash
python main.py /path/to/repo --index
```

**Ask questions:**
```bash
python main.py /path/to/repo
# Or single question:
python main.py /path/to/repo --query "Where is authentication implemented?"
```

## Example Queries

- "Where is user authentication implemented?"
- "Which files handle database connections?"
- "Explain the request flow from API to database"
- "What services depend on the auth module?"
- "How does the payment processing work?"

## Architecture

The system processes repositories through:

1. **Ingestion**: Scans repository, respects `.gitignore`, detects languages
2. **Parsing**: Extracts functions, classes, imports using Tree-sitter (regex fallback)
3. **Chunking**: Function-level chunking with overlap for context
4. **Embedding**: Generates 384-dimensional vectors using sentence-transformers
5. **Indexing**: FAISS vector store for fast similarity search
6. **Context**: MCP stores metadata, architecture, conversation history
7. **QA**: Intent detection, semantic retrieval, LLM-based answer generation

## Output

Each answer includes:
- Clear explanation with code references
- Source file paths with line numbers
- Similarity scores for each source
- Overall confidence metric

## Troubleshooting

**Index not found:** Run `python main.py /path/to/repo --index`

**API key error:** Ensure `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` is set in `.env`

**Memory issues:** Reduce `CHUNK_SIZE` in config or use `faiss-cpu`

**Tree-sitter warnings:** System falls back to regex parsing automatically

