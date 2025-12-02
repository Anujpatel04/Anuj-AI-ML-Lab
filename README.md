# Anuj AI/ML Lab

A comprehensive collection of AI/ML projects and experiments using Large Language Models (LLMs) and various AI frameworks. This repository contains specialized AI agents, RAG applications, and automation tools.

> **Note**: This repository is still in active development. Many exciting features are coming up including advanced Voice Agents, Multi-Agent Systems, and more. Stay tuned for updates!

## Project Structure

```
Anuj-AI-ML-Lab/
├── All_LLMs/                    # RAG-based chat applications
│   ├── chat_youtube/           # Chat with YouTube videos
│   ├── PDF_RAG/                # Chat with PDF documents
│   └── chat_with_gmail/        # Chat with Gmail inbox
│
├── Single_AI_Agents/            # Specialized AI agent applications
│   ├── AI_Meme_Generator/      # Browser-automated meme generation
│   ├── Health_Fitness_Agent/   # Personalized health & fitness planning
│   ├── Journalist_Agent/       # Automated article writing
│   ├── Meeting_Agent/          # Meeting notes and insights
│   ├── Simple_ScrapingAgent/   # Web scraping with DeepSeek/Ollama
│   └── Startup_Insight_Agent/  # Startup company analysis
│
└── MCP_Agents/                  # Model Context Protocol agents
    ├── Browser_mcp_agent/      # Browser automation agent
    └── github_mcp_agent/       # GitHub repository analysis
```

## Projects Overview

### All_LLMs
RAG-based applications for chatting with different data sources using DeepSeek API.

- **Chat with YouTube**: Extract and chat with YouTube video content
- **PDF RAG**: Query and analyze PDF documents
- **Chat with Gmail**: Interact with Gmail inbox using natural language

### Single_AI_Agents
Specialized AI agents for specific use cases using DeepSeek API and other LLMs.

- **AI Meme Generator**: Browser-automated meme creation using imgflip.com
- **Health & Fitness Agent**: Generate personalized dietary and fitness plans
- **Journalist Agent**: Automated research and article writing
- **Meeting Agent**: Extract insights and summaries from meeting notes
- **Simple Scraping Agent**: Web scraping with DeepSeek API or local Ollama models
- **Startup Insight Agent**: Analyze startup companies and extract key information

### MCP_Agents
Agents using Model Context Protocol for advanced integrations.

- **Browser MCP Agent**: Control web browsers using natural language
- **GitHub MCP Agent**: Analyze and explore GitHub repositories

## Quick Start

### Prerequisites
- Python 3.10+ (Python 3.11+ required for some agents)
- API keys (see Configuration section)

### Installation

Each project has its own `requirements.txt`. Navigate to the project directory and install:

```bash
cd <project_directory>
pip install -r requirements.txt
```

### Running Projects

Each project includes a README with specific usage instructions. Most projects use Streamlit:

```bash
streamlit run <main_file>.py
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# DeepSeek API (used by most agents)
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# Azure OpenAI (used by MCP_Agents)
AZURE_OPENAI_API_KEY=your-azure-api-key-here
AZURE_OPENAI_BASE_URL=your-azure-endpoint-url
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_MODEL=gpt-4o
```

### Shared Configuration

- **Azure OpenAI**: Configured in `config.py` at the root directory (if present)
- **DeepSeek API**: Loaded from `.env` file automatically
- **MCP Agents**: Can use `mcp_agent.secrets.yaml` files in their directories
- **Ollama**: Required for local model agents (install separately)

## Technologies

- **LLM Providers**: DeepSeek API, Azure OpenAI, Ollama (local)
- **Frameworks**: Streamlit, Agno AI, CrewAI, Embedchain, Browser-Use, ScrapeGraphAI
- **Tools**: Playwright, LangChain, Python-dotenv

## Project Details

Each project folder contains:
- Main Python script(s)
- `requirements.txt` with dependencies
- `README.md` with detailed setup and usage instructions

## Upcoming Features

This repository is actively being developed. Here's what's coming:

- **Advanced Voice Agents**: Enhanced voice-based AI applications with improved natural language understanding and multi-modal interactions
- **Multi-Agent Systems**: Collaborative agent frameworks where multiple AI agents work together to solve complex tasks
- **Enhanced RAG Systems**: Improved retrieval-augmented generation with better context understanding
- **Agent Orchestration**: Tools for managing and coordinating multiple agents
- **More Specialized Agents**: Additional domain-specific agents for various use cases
- **Performance Optimizations**: Improved efficiency and speed across all agents
- **Better Documentation**: Comprehensive guides and tutorials

## Notes

- Each project is self-contained with its own dependencies
- Projects can be run independently
- Configuration is centralized in the root `.env` file
- Some agents require additional setup (Playwright browsers, Ollama, etc.)
- Check individual README files for specific requirements
- This repository is in active development - expect regular updates and new features

## Contributing

This is a personal lab repository for AI/ML experiments and projects. The repository is continuously evolving with new agents and features being added regularly.
