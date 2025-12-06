# Anuj AI/ML Lab

A comprehensive collection of AI/ML projects and experiments using Large Language Models (LLMs) and various AI frameworks. This repository contains specialized AI agents, RAG applications, and automation tools.

> **Note**: This repository is still in active development. Many exciting features are coming up including advanced Voice Agents, Multi-Agent Systems, and more. Stay tuned for updates!

## Project Structure

```
Anuj-AI-ML-Lab/
├── All_LargeLangugage_Models/   # RAG-based chat applications
│   ├── chat_youtube/            # Chat with YouTube videos
│   ├── PDF_RAG/                 # Chat with PDF documents
│   └── chat_with_gmail/         # Chat with Gmail inbox
│
├── Single_AI_Agents/            # Specialized AI agent applications
│   ├── AI_Meme_Generator/      # Browser-automated meme generation
│   ├── Health_Fitness_Agent/   # Personalized health & fitness planning
│   ├── Journalist_Agent/       # Automated article writing
│   ├── Meeting_Agent/          # Meeting notes and insights
│   ├── Simple_ScrapingAgent/   # Web scraping agent
│   └── Startup_Insight_Agent/  # Startup company analysis
│
├── ALL_MachineLearning_Algos/  # Machine learning algorithms
│   ├── Supervised_Learning/    # Supervised learning implementations
│   └── Unsupervised_Learning/  # Unsupervised learning implementations
│
└── MCP_Agents/                  # Model Context Protocol agents
    ├── Browser_mcp_agent/      # Browser automation agent
    └── github_mcp_agent/       # GitHub repository analysis
```

## Projects Overview

### All_LargeLangugage_Models
RAG-based applications for chatting with different data sources. You can use any LLM provider of your choice (DeepSeek, OpenAI, Anthropic, Ollama, etc.).

- **Chat with YouTube**: Extract and chat with YouTube video content
- **PDF RAG**: Query and analyze PDF documents
- **Chat with Gmail**: Interact with Gmail inbox using natural language

### Single_AI_Agents
Specialized AI agents for specific use cases. Each agent can be configured to use your preferred LLM provider.

- **AI Meme Generator**: Browser-automated meme creation using imgflip.com
- **Health & Fitness Agent**: Generate personalized dietary and fitness plans
- **Journalist Agent**: Automated research and article writing
- **Meeting Agent**: Extract insights and summaries from meeting notes
- **Simple Scraping Agent**: Web scraping with configurable LLM backend
- **Startup Insight Agent**: Analyze startup companies and extract key information

### ALL_MachineLearning_Algos
Machine learning algorithm implementations from scratch.

- **Supervised Learning**: Linear Regression, Logistic Regression, Polynomial Regression, Decision Trees, K-Nearest Neighbors, and more
- **Unsupervised Learning**: K-Means Clustering and other clustering algorithms

### MCP_Agents
Agents using Model Context Protocol for advanced integrations. Compatible with various LLM providers.

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

Create a `.env` file in the root directory with your preferred LLM provider:

```env
# Example: DeepSeek API
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# Example: OpenAI
OPENAI_API_KEY=your-openai-api-key-here

# Example: Anthropic Claude
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Or use any other LLM provider of your choice
```

### LLM Provider Support

Most projects support multiple LLM providers. You can use:
- **DeepSeek API**: Cost-effective OpenAI-compatible API
- **OpenAI**: GPT-3.5, GPT-4, and other models
- **Anthropic**: Claude models
- **Ollama**: Local models (install separately)
- **Any OpenAI-compatible API**: Most agents support any provider with OpenAI-compatible endpoints

### Configuration Files

- **`.env` file**: Centralized environment variables in the root directory
- **`config.py`**: Some projects may use this for additional configuration
- **`mcp_agent.secrets.yaml`**: MCP agents can use their own secrets files
- **Project-specific configs**: Check individual README files for specific setup instructions

## Technologies

- **LLM Providers**: Use any LLM provider of your choice (DeepSeek, OpenAI, Anthropic, Ollama, etc.)
- **Frameworks**: Streamlit, Agno AI, CrewAI, Embedchain, Browser-Use, ScrapeGraphAI
- **Tools**: Playwright, LangChain, Python-dotenv, scikit-learn, numpy, pandas

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
- Most projects support multiple LLM providers - choose the one that works best for you
- Some agents require additional setup (Playwright browsers, Ollama, etc.)
- Check individual README files for specific requirements and LLM provider configuration
- This repository is in active development - expect regular updates and new features

## Contributing

This is a personal lab repository for AI/ML experiments and projects. The repository is continuously evolving with new agents and features being added regularly.
