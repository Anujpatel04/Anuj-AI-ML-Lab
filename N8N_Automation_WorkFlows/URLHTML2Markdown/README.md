# Web Scraper Workflow

![Workflow](https://raw.githubusercontent.com/Anujpatel04/Anuj-AI-ML-Lab/main/N8N_Automation_WorkFlows/URLHTML2Markdown/Workflow.png)

A professional automation workflow for scraping and processing web content using the Firecrawl.dev API with intelligent rate limiting and batch processing.

> **Note**: This project is part of the Anuj-AI-ML-Lab repository.

## Overview

This workflow retrieves markdown content, titles, descriptions, links, and other metadata from multiple URLs through the Firecrawl API. It features automatic batching, rate limiting, and seamless integration with your data sources.

## Features

- **Batch Processing**: Processes URLs in configurable batches (default: 40 items at a time)
- **Rate Limiting**: Respects API limits with automatic wait periods (10 requests per minute)
- **Memory Optimization**: Handles up to 40 concurrent items based on server memory
- **Full Content Extraction**: Retrieves markdown, titles, descriptions, links, and structured content
- **Flexible Integration**: Connects to your database for input and output

## Prerequisites

- n8n (self-hosted or cloud instance)
- Firecrawl.dev API key
- Input data source (database/table containing URLs)
- Output data source (database/table for storing scraped content)

## Setup

1. Import the `ConvertURLHTML2Markdown_Format .json` workflow file into your n8n instance
2. Configure your input data source:
   - Connect to your database/table containing URLs
   - Ensure the column is named `Page` (or customize as needed)
   - Format: One URL per row (like `split_out_page_urls`)
3. Configure Firecrawl API:
   - Add your Firecrawl.dev API key to the authentication header
   - Endpoint: `POST https://api.firecrawl.dev`
   - Authentication Header: `Authorization`
4. Connect your output data source (e.g., Airtable, database)
5. Adjust batch size and rate limits if needed (default: 40 items per batch, 10 requests per minute)

## Workflow Steps

1. **Data Source Connection**: Pull URLs from your database
2. **Field Mapping**: Define URL structure from your data source
3. **URL Splitting**: Distribute URLs across processing batches
4. **Batch Processing**: Process 40 items at a time
5. **Rate Limiting**: Implement 10 requests per minute with automatic waiting
6. **Content Retrieval**: Fetch markdown and metadata via Firecrawl API
7. **Data Processing**: Extract and structure markdown data and links
8. **Data Export**: Connect output to your database

## Configuration

### Batch Size

- **Default**: 40 items per batch
- **Note**: Adjust based on server memory capacity
- **Recommendation**: Monitor server memory when adjusting batch sizes

### Rate Limiting

- **Default**: 10 requests per minute
- **Implementation**: Automatic wait periods between batches
- **Best Practice**: Respect API rate limits to avoid throttling

## Input Data Format

### Required Structure

- **Column Name**: `Page` (customizable)
- **Format**: One URL per row
- **Example**: `["https://example.com/", "https://another-site.com/"]`

## Output Data

The workflow extracts and stores the following data:

- Markdown content
- Page titles
- Meta descriptions
- Internal and external links
- Additional structured content from Firecrawl

## Best Practices

- Monitor server memory when adjusting batch sizes
- Respect API rate limits to avoid throttling
- Use appropriate batch sizes for optimal performance
- Ensure URL format consistency in input data
- Test with small batches before processing large datasets

