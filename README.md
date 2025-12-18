# Anuj AI/ML Lab

A comprehensive collection of AI agents, RAG applications, and machine learning algorithms implemented from scratch. This repository serves as a practical learning resource and development playground for building production-ready AI systems.

## Status

> ** Active Development:** This repository is under active development and changes frequently.  
> If you find this project useful, please consider **starring** and **forking** to stay updated with the latest features and experiment with your own modifications.

## Overview

This lab contains a curated set of projects organized into four main categories:

- **LLM/RAG Applications**: Interactive applications for chatting with your data sources (PDFs, YouTube videos, Gmail inbox) using retrieval-augmented generation
- **Single-Purpose AI Agents**: Specialized agents designed for specific tasks such as content generation, web scraping, meeting transcription, and business intelligence
- **Machine Learning Algorithms**: Implementations of supervised and unsupervised learning algorithms from scratch, suitable for educational purposes and experimentation
- **MCP Agents**: Model Context Protocol agents that integrate with external tools and services for enhanced functionality

## Repository Structure

```
Anuj-AI-ML-Lab/
├── All_LargeLangugage_Models/
│   ├── LocalLama_Agent/          # Local RAG/chat app (Ollama + optional Qdrant/Exa)
│   ├── PDF_RAG/                  # Chat with PDF documents
│   ├── chat_youtube/             # Chat with YouTube video content
│   └── chat_with_gmail/          # Chat with Gmail inbox content
│
├── Single_AI_Agents/
│   ├── AI_Meme_Generator/        # Automated meme generation agent
│   ├── Health_Fitness_Agent/     # Health and fitness assistant
│   ├── Home_Renovation_agent/    # ADK-based multi-agent renovation planner
│   ├── Journalist_Agent/         # Content writing and journalism assistant
│   ├── LINKEDIN_ROSTER/          # LinkedIn profile analysis agent
│   ├── Meeting_Agent/            # Meeting transcription and summarization
│   ├── Simple_ScrapingAgent/     # Web scraping utilities
│   └── Startup_Insight_Agent/    # Startup analysis and insights
│
├── ALL_MachineLearning_Algos/
│   ├── Supervised_Learning/      # Classification and regression algorithms
│   └── Unsupervised_Learning/    # Clustering and dimensionality reduction
│
├── MCP_Agents/
│   ├── ai_travel_planner_mcp_agent_team/  # Travel planning with MCP tools
│   ├── Browser_mcp_agent/        # Browser automation via MCP
│   └── github_mcp_agent/         # GitHub integration agent
│
└── Automation_WorkFlows/
    └── N8N_Workflows/            # Workflow automation templates
```

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

API keys and service configurations are managed through a root-level `.env` file (git-ignored). Each project's README documents the specific environment variables it requires.

**Example `.env` configuration:**

```env
# Vector Database (Qdrant)
QDRANT_URL="https://your-qdrant-host:6333"
QDRANT_API_KEY="your-api-key"

# Web Search (Exa)
EXA_API_KEY="your-exa-api-key"

# Embedding Model
OLLAMA_EMBED_MODEL="nomic-embed-text"

# LLM API Keys (as needed)
DEEPSEEK_API_KEY="your-deepseek-key"
OPENAI_API_KEY="your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key"
GOOGLE_API_KEY="your-google-key"
```

### Project-Specific Configuration

- Each project includes its own `requirements.txt` for dependency management
- Browser-based projects (using Playwright) may require additional setup steps documented in their respective READMEs
- MCP agents may require additional configuration files (see individual project documentation)

## Features

- **Modular Design**: Each project is independent and can be run standalone
- **Comprehensive Documentation**: Every project includes detailed README with setup instructions
- **Production-Ready Code**: Clean, well-structured code suitable for learning and adaptation
- **Multiple LLM Support**: Projects support various LLM providers (OpenAI, Anthropic, DeepSeek, Ollama)
- **RAG Implementations**: Multiple RAG patterns and vector database integrations
- **Educational Focus**: ML algorithms implemented from scratch for learning purposes

## Contributing

This is a personal learning lab, but suggestions and improvements are welcome. Please feel free to fork the repository and adapt the code for your own projects.

## License

See [LICENSE](LICENSE) file for details.
