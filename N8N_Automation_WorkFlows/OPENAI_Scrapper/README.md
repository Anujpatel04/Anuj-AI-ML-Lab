# Essay Summarizer

![Workflow](https://raw.githubusercontent.com/Anujpatel04/Anuj-AI-ML-Lab/main/N8N_Automation_WorkFlows/OPENAI_Scrapper/Workflow.png)

An n8n workflow that automatically scrapes and summarizes Paul Graham's essays using DeepSeek AI.

> **Note**: This project is part of the AI-ML Lab repository.

## Overview

This workflow automates the process of fetching essays from paulgraham.com, extracting their content, and generating AI-powered summaries. It returns structured data containing the essay title, summary, and URL.

## Features

- **Automated Web Scraping**: Fetches essay list from paulgraham.com
- **Configurable Processing**: Extracts a specified number of essays (default: 3)
- **AI-Powered Summarization**: Generates summaries using DeepSeek AI
- **Structured Output**: Returns JSON data with title, summary, and URL

## Prerequisites

- n8n (self-hosted or cloud instance)
- DeepSeek API key

## Setup

1. Import the `OpenAI_Scrapper.json` workflow file into your n8n instance
2. Configure the DeepSeek API credentials:
   - Open the "DeepSeek Chat Model" node
   - Add your DeepSeek API key to the credentials
3. Execute the workflow manually or set up a schedule

## Configuration

Adjust the **"Limit to first 3"** node to process more or fewer essays as needed.

## Output Format

The workflow returns JSON data in the following format:

```json
{
  "title": "Essay Title",
  "summary": "AI-generated summary...",
  "url": "http://www.paulgraham.com/essay.html"
}
```

## Workflow Components

- **HTTP Request Nodes**: Web scraping functionality
- **HTML Parser**: Extracts content using CSS selectors
- **LangChain Summarization Chain**: Processes and summarizes content
- **DeepSeek Chat Model**: AI processing for summary generation

## License

See the main repository LICENSE file for details.
