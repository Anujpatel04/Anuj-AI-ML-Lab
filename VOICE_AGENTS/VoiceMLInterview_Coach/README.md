# Voice-Based ML Interview Coach

Streamlit app that runs a voice-based ML interview using OpenAI: speech-to-text (Whisper), chat (evaluation), and text-to-speech (TTS). Part of Anuj AI ML Lab.

## Features

- **Single-file app** (`app.py`): interview loop, scoring, and audio in one place
- **OpenAI**: STT (Whisper), Chat Completions (evaluation), TTS (questions and feedback)
- **Config**: `OPENAI_API_KEY` loaded from project root `.env` (no hardcoded keys)
- **Flow**: Ask question (voice) → record answer → transcribe → score (0–10) → spoken feedback → next question
- **Scope**: 5 ML questions (beginner–intermediate), conceptual and practical

## Requirements

- Python 3.10+
- Root `.env` with `OPENAI_API_KEY=your_key_here`

## Setup

```bash
pip install -r requirements.txt
```

Ensure the repo root (e.g. `Anuj-AI-ML-Lab`) has a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## Run

From this folder:

```bash
streamlit run app.py
```

Or from repo root:

```bash
streamlit run VOICE_AGENTS/VoiceMLInterview_Coach/app.py
```

## Usage

1. Click **Start Interview**
2. Listen to the first question (TTS)
3. Click **Answer (Record Voice)** and record (or upload an audio file)
4. Review transcription, score (0–10), and written evaluation
5. Listen to spoken feedback, then continue to the next question
6. After 5 questions, see final score and overall feedback

## Tech

- **Frontend**: Streamlit; `st.session_state` for question index, answers, scores, completion
- **STT**: OpenAI Whisper
- **LLM**: OpenAI Chat (e.g. gpt-4o-mini) for scoring and feedback text
- **TTS**: OpenAI `tts-1` for question and feedback audio
