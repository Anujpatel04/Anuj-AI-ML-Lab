# Anuj AI/ML Lab

A comprehensive collection of AI agents, RAG applications, and machine learning algorithms implemented from scratch. This repository serves as a practical learning resource and development playground for building production-ready AI systems.

## Status

> Active Development: This repository is under active development and undergoes frequent changes.  
If you find this project useful, please consider **starring** and **forking** to stay updated with the latest features and experiment with your own modifications.

## Overview

This lab contains a curated set of projects organized into the following categories:

- **AI Agents**: Specialized agents for content generation, web scraping, meeting transcription, business intelligence, and LLM/RAG applications (chat with Gmail, YouTube, PDFs, Tarot cards)
- **RAG Applications**: Retrieval-augmented generation implementations for document processing and knowledge retrieval
- **Machine Learning Algorithms**: Implementations of supervised and unsupervised learning algorithms from scratch, suitable for educational purposes and experimentation
- **Fine-Tuning Projects**: LLM fine-tuning implementations using techniques like LoRA for domain-specific applications
- **Voice Agents**: Voice-powered AI agents for customer support and web interaction
- **MCP Agents**: Model Context Protocol agents that integrate with external tools and services for enhanced functionality
- **N8N Automation Workflows**: N8N workflow configurations for automated AI-powered processes



## Quick Start

Each project is self-contained with its own dependencies and documentation. To get started:

1. **Navigate to the project directory:**
   ```bash
   cd <project_folder>
   ```

2. **Install dependencies:**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. **Follow the project-specific README** for detailed setup and usage instructions.

### Running Streamlit Applications

Most interactive applications use Streamlit:

```bash
streamlit run <app_file>.py
```

### Running Script-Based Projects

For command-line scripts:

```bash
python3 <script_file>.py
```

## Configuration

### Environment Variables

Most projects use a centralized `.env` file in the root directory for API key management. Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key
QDRANT_URL=your_url
QDRANT_API_KEY=your_key
FIRECRAWL_API_KEY=your_key
MODELSLAB_API_KEY=your_key
```

### Project-Specific Configuration

- Each project includes its own `requirements.txt` for dependency management
- Browser-based projects (using Playwright) may require additional setup steps documented in their respective READMEs
- MCP agents may require additional configuration files (see individual project documentation)

## Contributing

This is a personal learning lab, but suggestions and improvements are welcome. Please feel free to fork the repository and adapt the code for your own projects.

## License

See [LICENSE](LICENSE) file for details.
