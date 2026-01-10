# AI Video Narration Pipeline using Vision + TTS (n8n)

![Workflow](https://raw.githubusercontent.com/Anujpatel04/Anuj-AI-ML-Lab/main/N8N_Automation_WorkFlows/VideoNarrating/Workflow.png)

An end-to-end AI-powered video narration workflow that combines computer vision, large language models, and text-to-speech to automatically generate narrated voiceovers from videos.

## Overview

The pipeline:
- Downloads video from URL
- Extracts evenly distributed frames using OpenCV
- Generates coherent narration via vision-capable LLM
- Converts narration to speech (MP3) using TTS
- Uploads final audio to Google Drive

## Architecture

```
Video URL → Download → Extract Frames → Batch (15 frames) → 
Vision LLM → Combine Script → Chunk Text (≤4096 chars) → 
Text-to-Speech → Upload to Google Drive
```

### Stage 1: Frame Extraction
- OpenCV-based frame sampling
- Extracts up to 90 frames, evenly distributed
- Converts frames to Base64 JPEGs
- `step_size = max(1, total_frames // max_frames)`

### Stage 2: Vision-Based Narration
- Processes frames in batches of 15
- Uses vision-capable OpenAI model
- Generates cohesive, documentary-style script
- Each batch continues from previous context

### Stage 3: Text Chunking
- Aggregates all narration
- Splits into ≤3800 character chunks
- Prevents TTS API limit errors (4096 char max)
- `const MAX_LEN = 3800`

### Stage 4: Text-to-Speech
- Converts chunks to MP3 audio
- Uses OpenAI audio generation endpoint
- Outputs binary audio data

### Stage 5: Google Drive Upload
- Automatic upload via OAuth2
- Timestamped filenames: `narrating-video-using-vision-ai-YYYYMMDDHHMMSS.mp3`

## Credentials Required

- **OpenAI API**
  - Vision-capable chat model access
  - Text-to-Speech (audio) access

- **Google Drive OAuth2**
  - Google Drive API enabled
  - OAuth consent screen configured
  - Redirect URI: `http://localhost:5678/rest/oauth2-credential/callback`

## Setup

1. Import workflow JSON into n8n
2. Configure OpenAI API credentials
3. Configure Google Drive OAuth2 credentials
4. Execute via manual trigger

## Constraints

- TTS API enforces 4096-character limit (chunked to 3800 for safety)
- Frame extraction: 1–2 minutes for ~3MB videos
- Requires sufficient memory for video processing
- Best performance on local n8n instance

## Use Cases

- Automated video narration
- AI-generated documentaries
- Video summarization pipelines
- Accessibility (audio descriptions)
- Content repurposing (video → podcast)

## Technologies

- n8n (workflow automation)
- OpenAI GPT-4.1-mini (vision + text)
- OpenAI Text-to-Speech (MP3)
- Google Drive API v3
- OpenCV (Python)

## Future Improvements

- Merge audio chunks programmatically
- Add background music support
- Dynamic support for longer videos
- Parallelize frame processing
- Subtitle (SRT) generation

## Author

**Anuj Patel**  
AI / ML Engineer  
Focus: Vision AI, LLM pipelines, automation, applied ML systems
