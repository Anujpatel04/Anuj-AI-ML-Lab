# AI-Powered Research Assistant Workflow

![Workflow](https://raw.githubusercontent.com/Anujpatel04/Anuj-AI-ML-Lab/main/N8N_Automation_WorkFlows/ChatResearch_agent/Workflow.png)

An n8n workflow that transforms user queries into detailed research reports by intelligently searching the web, analyzing content, and synthesizing findings using AI.

> **Note**: This project is part of the Anuj-AI-ML-Lab repository.

## Overview

Takes a natural language query, generates optimized search queries, retrieves relevant web content, and produces structured markdown research reports with citations. Maintains conversational context across 20 previous messages.

## Features

- Generates up to 4 optimized search queries from user input
- Web search via SerpAPI (Google)
- Content extraction and analysis via Jina AI
- AI-powered context filtering and report generation
- Wikipedia integration for supplementary data

## Prerequisites

- n8n (self-hosted or cloud)
- **OpenRouter API** key ([Get Key](https://openrouter.ai/settings/keys)) - Model: `google/gemini-2.0-flash-001`
- **SerpAPI** key ([Get Key](https://serpapi.com/manage-api-key))
- **Jina AI** API key ([Get Key](https://jina.ai/api-dashboard/key-manager))

## Setup

1. Import `Chat_ResearchAgent.json` into n8n
2. Configure API credentials:
   - **OpenRouter**: Add as "OpenRouter account" credential
   - **SerpAPI**: Configure in n8n credentials
   - **Jina AI**: Set up HTTP Header Auth (endpoint: `https://r.jina.ai/`)
3. Verify webhook ID for Chat Trigger
4. Test with a sample query

## How It Works

1. User submits query via chat interface
2. LLM generates 4 optimized search queries
3. SerpAPI retrieves Google search results
4. Jina AI extracts and converts webpage content to markdown
5. LLM filters content for relevance
6. Final markdown report generated with citations

## Configuration

- **Context Window**: 20 messages (customizable)
- **Session ID**: `my_test_session` (customizable)
- **Batch Processing**: Automatic for API efficiency

## Usage

Activate workflow and send queries via chat interface.

**Example Queries:**
- "What are the latest developments in quantum computing?"
- "Analyze the impact of renewable energy policies in Europe"
- "Compare different approaches to AI alignment"

## Output

Structured markdown report with key findings, detailed analysis, and conclusions, including source citations.

## Customization

- **Search Engine**: Modify SerpAPI node (Bing, DuckDuckGo, Yahoo)
- **LLM Model**: Change in OpenRouter node (GPT-4, Claude, Mixtral)
- **Report Format**: Modify system message in report generation node

## Troubleshooting

- **Empty Results**: Check SerpAPI credentials and quota
- **Content Extraction Fails**: Verify Jina AI authentication
- **LLM Errors**: Confirm OpenRouter API key and model availability
- **Memory Issues**: Reduce context window length

## Best Practices

- Use n8n credentials for API keys (never hardcode)
- Monitor API usage to stay within rate limits
- Frame questions clearly for best results
- Clear session periodically for unrelated topics

## Credits

n8n, OpenRouter, SerpAPI, Jina AI, Wikipedia API
