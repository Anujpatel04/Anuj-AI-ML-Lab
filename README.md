# Anuj AI/ML Lab

A personal lab for AI agents, RAG apps, and ML algorithms. Everything here is meant to be runnable and easy to iterate on.

This repository is actively being developed. More work is coming around voice agents, multi-agent systems, and deeper RAG workflows.

## Status

This repo is under active development and changes frequently. If you find it useful, feel free to star it and fork it to keep track of updates and experiment in your own copy.

## What you’ll find here

- **LLM/RAG apps**: small, focused apps that let you chat with your own data (PDFs, YouTube, Gmail) and experiment with retrieval.
- **Single-purpose agents**: practical agents for common tasks (scraping, content writing, meeting notes, etc.).
- **ML algorithms**: supervised and unsupervised algorithms implemented as learning references and demos.
- **MCP agents**: agents that integrate external tools/services using the Model Context Protocol.

## Repository structure

```text
Anuj-AI-ML-Lab/
├── All_LargeLangugage_Models/
│   ├── LocalLama_Agent/          # Local RAG/chat app (Ollama + optional Qdrant/Exa)
│   ├── PDF_RAG/                  # Chat with PDFs
│   ├── chat_youtube/             # Chat with YouTube content
│   └── chat_with_gmail/           # Chat with Gmail content
│
├── Single_AI_Agents/
│   ├── AI_Meme_Generator/
│   ├── Health_Fitness_Agent/
│   ├── Home_Renovation_agent/    # ADK-based multi-agent app
│   ├── Journalist_Agent/
│   ├── LINKEDIN_ROSTER/
│   ├── Meeting_Agent/
│   ├── Simple_ScrapingAgent/
│   └── Startup_Insight_Agent/
│
├── ALL_MachineLearning_Algos/
│   ├── Supervised_Learning/
│   └── Unsupervised_Learning/
│
├── MCP_Agents/
│   ├── ai_travel_planner_mcp_agent_team/
│   ├── Browser_mcp_agent/
│   └── github_mcp_agent/
│
└── Automation_WorkFlows/
    └── N8N_Workflows/
```

## Quick start

Most folders are self-contained (install deps per project):

```bash
cd <project_folder>
python3 -m pip install -r requirements.txt
```

Then run what that folder’s README says. Many projects are Streamlit apps:

```bash
streamlit run <file>.py
```

For script-style projects, you’ll typically run:

```bash
python3 <file>.py
```

## Configuration

- **Root `.env`**: central place for API keys and service URLs used by multiple apps (this file is git-ignored).
- **Project READMEs**: each folder documents what keys (if any) it expects and how to run it.

Minimal example (add only what you actually use):

```env
# Qdrant (used by local RAG apps when enabled)
QDRANT_URL="https://your-qdrant-host:6333"
QDRANT_API_KEY="..."

# Exa (used only if web-search fallback is enabled)
EXA_API_KEY="..."
```

## Notes

- Each project is intentionally isolated with its own `requirements.txt`.
- If something uses a browser (Playwright), you may need an extra install step described in that project’s README.
