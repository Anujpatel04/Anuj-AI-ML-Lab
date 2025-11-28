# Anuj AI/ML Lab

A collection of AI/ML projects and experiments using Large Language Models (LLMs) and various AI frameworks.

## Project Structure

```
Anuj-AI-ML-Lab/
├── All_LLMs/                    # RAG-based chat applications
│   ├── chat_youtube/           # Chat with YouTube videos
│   ├── PDF_RAG/                # Chat with PDF documents
│   └── chat_with_gmail/        # Chat with Gmail inbox
│
├── Single_AI_Agents/            # Specialized AI agent applications
│   ├── Health_Fitness_Agent/    # Personalized health & fitness planning
│   ├── Journalist_Agent/       # Automated article writing
│   ├── Meeting_Agent/          # Meeting notes and insights
│   └── Startup_Insight_Agent/  # Startup company analysis
│
├── MCP_Agents/                  # Model Context Protocol agents
│   ├── Browser_mcp_agent/      # Browser automation agent
│   └── github_mcp_agent/       # GitHub repository analysis
│
├── config.py                    # Shared Azure OpenAI configuration
└── .env                         # Environment variables (API keys)
```

## Projects

### All_LLMs
RAG-based applications for chatting with different data sources using DeepSeek API.

- **Chat with YouTube**: Extract and chat with YouTube video content
- **PDF RAG**: Query and analyze PDF documents
- **Chat with Gmail**: Interact with Gmail inbox using natural language

### Single_AI_Agents
Specialized AI agents for specific use cases using DeepSeek API.

- **Health & Fitness Agent**: Generate personalized dietary and fitness plans
- **Journalist Agent**: Automated research and article writing
- **Meeting Agent**: Extract insights and summaries from meeting notes
- **Startup Insight Agent**: Analyze startup companies and extract key information

### MCP_Agents
Agents using Model Context Protocol for advanced integrations.

- **Browser MCP Agent**: Control web browsers using natural language
- **GitHub MCP Agent**: Analyze and explore GitHub repositories

## Setup

### Prerequisites
- Python 3.10+
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
# DeepSeek API (used by All_LLMs and Single_AI_Agents)
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# Azure OpenAI (used by MCP_Agents)
AZURE_OPENAI_API_KEY=your-azure-api-key-here
AZURE_OPENAI_BASE_URL=your-azure-endpoint-url
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_MODEL=gpt-4o
```

### Shared Configuration

- **Azure OpenAI**: Configured in `config.py` at the root directory
- **DeepSeek API**: Loaded from `.env` file
- **MCP Agents**: Can use `mcp_agent.secrets.yaml` files in their directories

## Technologies

- **LLM Providers**: DeepSeek API, Azure OpenAI
- **Frameworks**: Streamlit, Agno AI, CrewAI, Embedchain
- **Tools**: Exa API, Firecrawl, SerpAPI, Serper API

## Notes

- Each project is self-contained with its own dependencies
- Projects can be run independently
- Configuration is centralized for easy management
- All projects include individual README files with detailed instructions
