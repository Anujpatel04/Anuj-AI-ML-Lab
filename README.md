# Anuj AI/ML Lab

A collection of AI/ML projects and experiments using Large Language Models (LLMs).

## Structure

```
Anuj-AI-ML-Lab/
├── All_LLMs/
│   ├── chat_youtube/      # Chat with YouTube videos using RAG
│   ├── PDF_RAG/           # Chat with PDF documents using RAG
│   └── chat_with_gmail/   # Chat with Gmail inbox using RAG
└── MCP_Agents/
    ├── Browser_mcp_agent/ # Browser automation using MCP Agent with Azure OpenAI
    └── github_mcp_agent/  # GitHub repository analysis using MCP Agent
```

## Projects

### All_LLMs

#### Chat with YouTube
RAG application to chat with YouTube videos using DeepSeek API.

#### PDF RAG
RAG application to chat with PDF documents using DeepSeek API.

#### Chat with Gmail
RAG application to chat with Gmail inbox using DeepSeek API.

### MCP_Agents

#### Browser MCP Agent
Browser automation application that allows you to control a web browser using natural language commands. Uses Azure OpenAI and Playwright for browser automation.

#### GitHub MCP Agent
GitHub repository analysis tool that allows you to explore and analyze GitHub repositories using natural language queries through the Model Context Protocol.

## Setup

Each project has its own `requirements.txt` and README with specific setup instructions.

## Environment Variables

Create a `.env` file in the root directory with:
```env
DEEPSEEK_API_KEY=your-api-key-here
```

For MCP Agents, configure Azure OpenAI credentials in the respective `mcp_agent.secrets.yaml` files.

## Notes

This repository is organized systematically for easy project management and future additions.
