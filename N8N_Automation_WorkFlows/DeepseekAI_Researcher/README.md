# AI Company Research Automation

![Workflow](https://raw.githubusercontent.com/Anujpatel04/Anuj-AI-ML-Lab/main/N8N_Automation_WorkFlows/DeepseekAI_Researcher/Workflow.png)

An n8n workflow that automatically enriches company data using AI-powered research agents and updates Google Sheets with findings.

> **Note**: This project is part of the Anuj-AI-ML-Lab repository.

## Overview

This workflow automates company research by reading company names from Google Sheets, conducting AI-powered research using web search capabilities, and writing structured data back to the spreadsheet. The system enriches company profiles with domain information, market type, pricing details, API availability, integration partners, and case studies.

## Features

- **Automated Research**: AI agents research companies using web search and content extraction
- **Google Sheets Integration**: Reads from and writes to Google Sheets automatically
- **Structured Data Output**: Returns standardized company information including:
  - Domain & LinkedIn URL
  - Market type (B2B/B2C)
  - Pricing information (cheapest plan, enterprise/free trial availability)
  - API availability
  - Integration partners
  - Latest case study link
- **Scheduled Processing**: Runs automatically on a configurable schedule (default: every 2 hours)
- **Batch Processing**: Processes companies one at a time with status tracking

## Prerequisites

- n8n (self-hosted or cloud instance)
- DeepSeek API key
- SerpAPI key (for Google search)
- Google Sheets API access

## Setup

1. Import the `AI_Researcher.json` workflow file into your n8n instance
2. Configure the required credentials:
   - **DeepSeek API**: Add credentials to the "DeepSeek Chat Model1" node
   - **SerpAPI**: Add credentials to the "SerpAPI - Search Google1" node
   - **Google Sheets**: Configure OAuth credentials in Google Sheets nodes
3. Update the Google Sheets document ID to your target spreadsheet
4. Adjust the schedule trigger as needed (default: every 2 hours)

## Google Sheets Format

### Required Columns

- `input` - Company name
- `row_number` - Row identifier
- `enrichment_status` - Leave empty for new entries (will be marked as "done" after processing)

### Output Columns

Output columns will be automatically populated with the enriched data.

## How It Works

1. **Scheduled Trigger**: Initiates the workflow on a configurable schedule (default: every 2 hours)
2. **Data Fetching**: Retrieves rows from Google Sheets where `enrichment_status` is empty
3. **Company Processing**: Loops through companies one at a time
4. **AI Research**: For each company, the AI agent:
   - Uses SerpAPI for Google searches
   - Extracts website content using content extraction tools
   - Analyzes data using DeepSeek LLM
5. **Data Structuring**: Formats output according to the defined schema
6. **Sheet Update**: Writes enriched data back to the spreadsheet and marks status as "done"

## Configuration

- **Schedule**: Modify the "Schedule Trigger1" node to change the execution frequency (default: 2 hours)
- **Batch Size**: Adjust the "Loop Over Items1" node settings for batch processing
- **AI Instructions**: Customize the research prompt in the "AI company researcher1" node
- **Output Schema**: Modify the "Structured Output Parser1" node to change output fields

## Workflow Components

- **AI Agent with Tool Calling**: Executes research tasks using available tools
- **Web Scraping**: Content extraction from company websites
- **Google Search Integration**: SerpAPI for web search capabilities
- **Structured Output Parsing**: Formats AI responses into structured data
- **Google Sheets Integration**: Read/write operations for data management
- **Batch Processing**: Loop control for processing multiple companies

