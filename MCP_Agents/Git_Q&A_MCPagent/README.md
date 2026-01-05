# Codebase Q&A MCP Agent

> **Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)** - A comprehensive collection of AI/ML projects, LLM applications, agents, RAG systems, and core machine learning implementations.

A production-grade Codebase Q&A Agent using Model Context Protocol (MCP) that analyzes entire Git repositories and answers architecture and code-level questions accurately using RAG (Retrieval-Augmented Generation).

## 🎯 Features

- **Repository Ingestion**: Recursively scans Git repositories while respecting `.gitignore`
- **AST-Aware Parsing**: Extracts functions, classes, imports, and method signatures using Tree-sitter
- **Semantic Embeddings**: Generates embeddings for intelligent code chunking
- **Vector Search**: FAISS-based retrieval for fast similarity search
- **MCP Context**: Persistent shared context for multi-turn reasoning
- **Question Answering**: Answers developer questions with file paths and line numbers

## 🏗️ Architecture

```
Git_Q&A_MCPagent/
├── ingestion/          # Repository scanning
├── parsing/           # AST parsing with Tree-sitter
├── embeddings/        # Chunking and embedding generation
├── retrieval/         # FAISS vector store
├── mcp_context/       # MCP context management
├── qa/               # Question answering engine
├── config/           # Configuration settings
├── scripts/          # Utility scripts
│   ├── setup_env.sh
│   ├── run.sh
│   └── run_frontend.sh
├── app.py           # Streamlit frontend
├── main.py          # CLI entry point
├── requirements.txt # Dependencies
└── README.md        # Documentation
```

## 📋 Prerequisites

- Python 3.8+
- Git repository to analyze
- API key (OpenAI or DeepSeek)

## 🚀 Installation

```bash
cd MCP_AGENTS/Git_Q&A_MCPagent
./scripts/setup_env.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ⚙️ Configuration

Add to root `.env` file:

```env
# Use DeepSeek (recommended, cheaper)
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Or use OpenAI
OPENAI_API_KEY=your-openai-api-key

# Optional settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
QA_MODEL=deepseek-chat
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

## 📖 Usage

### Option 1: Web Frontend (Recommended) 🌐

Launch the Streamlit web interface:

```bash
./scripts/run_frontend.sh
```

Or manually:
```bash
source venv/bin/activate
streamlit run app.py
```

Then:
1. Open your browser to `http://localhost:8501`
2. Enter a GitHub repository URL or local path
3. Click "Connect" to clone/connect the repository
4. Click "Index Repository" to build the search index
5. Go to "Ask Questions" tab and start chatting!

**Features:**
- ✅ Connect to GitHub repos via URL
- ✅ Connect to local repositories
- ✅ Visual indexing progress
- ✅ Chat interface for questions
- ✅ Source citations with file paths
- ✅ Example questions to get started

### Option 2: Command Line Interface

**1. Index a Repository:**

```bash
python main.py /path/to/repo --index
```

This will:
- Scan all files in the repository
- Parse code using AST
- Generate embeddings
- Build a FAISS index
- Store MCP context

**2. Ask Questions:**

**Interactive Mode:**
```bash
python main.py /path/to/repo
```

**Single Question:**
```bash
python main.py /path/to/repo --query "Where is authentication implemented?"
```

**3. Force Reindex:**

```bash
python main.py /path/to/repo --index --force
```

## 🧪 Example Queries

The agent can answer questions like:

- **"Where is user authentication implemented?"**
- **"Which files handle database connections?"**
- **"Explain the request flow from API to database"**
- **"What services depend on the auth module?"**
- **"How does the payment processing work?"**
- **"Where are API endpoints defined?"**

## 🔍 How It Works

### 1. Repository Ingestion
- Recursively scans the repository
- Respects `.gitignore` patterns
- Detects file languages
- Extracts file metadata

### 2. AST Parsing
- Uses Tree-sitter for AST extraction
- Extracts functions, classes, imports
- Links symbols to file paths
- Falls back to regex parsing if Tree-sitter unavailable

### 3. Code Chunking
- Function-level chunking (preferred)
- Class-level chunking
- Line-based fallback for unstructured code
- Overlap between chunks for context

### 4. Embedding Generation
- Uses sentence-transformers (default: `all-MiniLM-L6-v2`)
- Optional OpenAI embeddings
- Generates 384-dimensional vectors

### 5. Vector Storage
- FAISS index for fast similarity search
- Persistent storage on disk
- Efficient retrieval of top-k chunks

### 6. MCP Context
- Stores repository metadata
- Architecture overview
- AST summaries
- Module relationships
- Conversation history

### 7. Question Answering
- Intent detection (architecture, location, explanation, etc.)
- Semantic retrieval of relevant chunks
- Context assembly
- LLM-based answer generation with citations

## 📊 Output Format

Answers include:
- **Answer**: Clear explanation with code references
- **Sources**: File paths with line numbers
- **Similarity Scores**: Relevance scores for each source
- **Confidence**: Overall confidence in the answer

## 🎨 Design Decisions

1. **Function-Level Chunking**: Prefers chunking at function boundaries for better semantic coherence
2. **FAISS for Speed**: Uses FAISS for efficient vector search on large codebases
3. **MCP for Context**: Persistent context enables multi-turn reasoning
4. **Fallback Parsing**: Regex-based parsing when Tree-sitter unavailable
5. **Modular Architecture**: Clean separation of concerns for maintainability

## 🔧 Troubleshooting

**Index not found:**
```bash
# Run indexing first
python main.py /path/to/repo --index
```

**API key error:**
- Ensure `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` is set in `.env`

**Memory issues:**
- Reduce `CHUNK_SIZE` in config
- Use `faiss-cpu` instead of `faiss-gpu` for lower memory

**Tree-sitter warnings:**
- Tree-sitter is optional; the system falls back to regex parsing
- For full AST support, install language grammars separately

## 📝 License

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab) - MIT License

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows the existing architecture
- All components are typed
- Error handling is comprehensive
- Logging is appropriate

