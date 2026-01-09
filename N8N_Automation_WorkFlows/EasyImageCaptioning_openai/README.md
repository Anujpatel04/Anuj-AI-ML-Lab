# AI-Driven Image Captioning Workflow (n8n)

![Workflow](https://raw.githubusercontent.com/Anujpatel04/Anuj-AI-ML-Lab/main/N8N_Automation_WorkFlows/EasyImageCaptioning_openai/Workflow.png)

An end-to-end automated image captioning pipeline that combines multimodal LLMs with n8n's image processing capabilities. The workflow generates structured captions using vision-enabled models and overlays them directly onto images.

## Overview

The workflow:
- Downloads and processes images
- Generates structured captions (title + description) via multimodal LLM
- Dynamically calculates layout parameters
- Renders captions with background overlay

## Architecture

**Stage 1: Image Ingestion**
- HTTP request to fetch image
- Extract metadata and resize for LLM processing

**Stage 2: Caption Generation**
- Send image to vision-capable LLM
- Generate structured JSON output (title + descriptive text)
- Enforce schema validation

**Stage 3: Caption Overlay**
- Compute dynamic layout (font size, padding, positioning)
- Render semi-transparent background
- Overlay text at image bottom

## Output Schema

```json
{
  "caption_title": "",
  "caption_text": ""
}
```

## Technologies

- n8n (workflow automation)
- LangChain (LLM orchestration)
- OpenAI Chat Model (multimodal)
- Native n8n image editing

## Setup

1. Import workflow JSON into n8n
2. Configure OpenAI credentials
3. Verify font path availability
4. Execute via manual trigger

## Use Cases

- Social media content automation
- Marketing asset generation
- Visual annotation workflows
- Multimodal AI prototyping

## Extensibility

- Webhook/file upload integration
- Cloud storage output
- Branding/watermark layers
- Custom caption formatting
- Approval workflows

## Author

**Anuj Patel**  
Focus: LLM Systems, Multimodal AI, Automation Pipelines
