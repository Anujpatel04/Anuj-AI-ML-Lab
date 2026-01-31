"""
Voice-Based ML Interview Coach
Single-file Streamlit app: OpenAI STT, Chat, TTS. API key from root .env.
"""

import io
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load .env from project root (Anuj-AI-ML-Lab)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NUM_QUESTIONS = 5
SYSTEM_PROMPT = (
    "You are a professional machine learning interviewer. "
    "You ask clear interview questions, evaluate answers fairly, and provide constructive spoken feedback. "
    "Your tone is calm, concise, and realistic."
)

ML_QUESTIONS = [
    "What is the bias-variance tradeoff, and why does it matter in model selection?",
    "How would you detect and handle overfitting in practice?",
    "Explain the difference between supervised and unsupervised learning with one example each.",
    "Why might we use cross-validation instead of a single train-test split?",
    "What is regularization, and how does it help with overfitting?",
]


def ensure_session_state():
    if "question_index" not in st.session_state:
        st.session_state.question_index = 0
    if "questions" not in st.session_state:
        st.session_state.questions = ML_QUESTIONS.copy()
    if "transcribed_answers" not in st.session_state:
        st.session_state.transcribed_answers = []
    if "scores" not in st.session_state:
        st.session_state.scores = []
    if "evaluations" not in st.session_state:
        st.session_state.evaluations = []
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    if "interview_complete" not in st.session_state:
        st.session_state.interview_complete = False
    if "current_question_audio" not in st.session_state:
        st.session_state.current_question_audio = None
    if "show_recorder" not in st.session_state:
        st.session_state.show_recorder = False


def get_client():
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is missing. Add it to the root .env file.")
        st.stop()
    return OpenAI(api_key=OPENAI_API_KEY)


def ask_question(client: OpenAI, question: str) -> bytes:
    """Generate TTS for the current interview question."""
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=question,
    )
    return response.content


def transcribe_audio(client: OpenAI, audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Speech-to-text via OpenAI Whisper."""
    file = io.BytesIO(audio_bytes)
    file.name = filename
    transcript = client.audio.transcriptions.create(model="whisper-1", file=file)
    return transcript.text.strip() if transcript.text else ""


def evaluate_answer(client: OpenAI, question: str, answer: str) -> tuple[int, str]:
    """Score 0-10 and return evaluation text (strengths, weak points)."""
    user = (
        f"Interview question: {question}\n\nCandidate answer: {answer}\n\n"
        "Reply with exactly two lines:\n"
        "1. A single integer score from 0 to 10.\n"
        "2. One short paragraph: strengths, then missing or weak points. Be concise."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    text = (resp.choices[0].message.content or "").strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    score = 5
    for line in lines:
        for part in line.replace(".", " ").split():
            if part.isdigit():
                score = max(0, min(10, int(part)))
                break
    evaluation = "\n".join(lines) if lines else text
    return score, evaluation


def feedback_for_tts(client: OpenAI, question: str, answer: str, score: int, evaluation: str) -> str:
    """Short spoken feedback (10–20 seconds)."""
    user = (
        f"Question: {question}\nAnswer summary: {answer[:200]}\nScore: {score}/10.\nEvaluation: {evaluation[:300]}.\n\n"
        "Write 2–3 short sentences of spoken interview feedback: encouraging, concise, realistic. "
        "No bullet points. Plain prose only, for text-to-speech."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_tokens=150,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_speech(client: OpenAI, text: str) -> bytes:
    """Text-to-speech for feedback."""
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text[:500],
    )
    return response.content


def render_final_screen():
    st.subheader("Interview complete")
    total = sum(st.session_state.scores)
    max_total = 10 * NUM_QUESTIONS
    st.metric("Final score", f"{total} / {max_total}")
    st.write("**Per-question scores:**", st.session_state.scores)
    if st.session_state.evaluations:
        st.write("**Evaluations**")
        for i, ev in enumerate(st.session_state.evaluations, 1):
            st.write(f"Q{i}:", ev)
    overall = (
        "You've completed the ML interview. Review the evaluations above to improve. "
        "Well done for finishing all questions."
    )
    st.write("**Overall:**", overall)
    client = get_client()
    audio_bytes = generate_speech(client, overall)
    st.audio(audio_bytes, format="audio/mp3")


def main():
    st.set_page_config(page_title="Voice ML Interview Coach", layout="centered")
    st.title("Voice-Based ML Interview Coach")
    ensure_session_state()
    client = get_client()

    if st.session_state.interview_complete:
        render_final_screen()
        return

    if not st.session_state.interview_started:
        if st.button("Start Interview"):
            st.session_state.interview_started = True
            st.session_state.question_index = 0
            st.session_state.transcribed_answers = []
            st.session_state.scores = []
            st.session_state.evaluations = []
            st.rerun()
        return

    idx = st.session_state.question_index
    questions = st.session_state.questions

    if idx >= len(questions):
        st.session_state.interview_complete = True
        st.rerun()
        return

    current_q = questions[idx]
    st.subheader("Current question")
    st.write(f"**Q{idx + 1}:** {current_q}")

    # Play question TTS once when we land on this question
    if st.session_state.current_question_audio is None or len(st.session_state.get("_played_index", [])) <= idx:
        with st.spinner("Generating question audio..."):
            q_audio = ask_question(client, current_q)
            st.session_state.current_question_audio = q_audio
            if "_played_index" not in st.session_state:
                st.session_state._played_index = []
            st.session_state._played_index.append(idx)
    if st.session_state.current_question_audio:
        st.audio(st.session_state.current_question_audio, format="audio/mp3")

    st.subheader("Your answer (voice)")
    audio_data = None
    if not st.session_state.show_recorder:
        if st.button("Answer (Record Voice)"):
            st.session_state.show_recorder = True
            st.rerun()
    else:
        audio_data = st.audio_input("Record your answer") if hasattr(st, "audio_input") else st.file_uploader("Upload recorded answer", type=["wav", "mp3", "webm", "m4a"])

    if st.session_state.show_recorder and audio_data is not None:
        audio_bytes = audio_data.read()
        if audio_bytes:
            with st.spinner("Transcribing..."):
                transcribed = transcribe_audio(client, audio_bytes, getattr(audio_data, "name", "audio.webm"))
            if transcribed:
                st.write("**Transcribed:**", transcribed)
                with st.spinner("Evaluating..."):
                    score, evaluation = evaluate_answer(client, current_q, transcribed)
                st.session_state.scores.append(score)
                st.session_state.evaluations.append(evaluation)
                st.session_state.transcribed_answers.append(transcribed)
                st.write("**Score:**", score, "/ 10")
                st.write("**Evaluation:**", evaluation)
                feedback_text = feedback_for_tts(client, current_q, transcribed, score, evaluation)
                with st.spinner("Generating feedback audio..."):
                    feedback_audio = generate_speech(client, feedback_text)
                st.audio(feedback_audio, format="audio/mp3")
                st.session_state.question_index = idx + 1
                st.session_state.current_question_audio = None
                st.session_state.show_recorder = False
                if st.session_state.question_index >= len(questions):
                    st.session_state.interview_complete = True
                st.success("Moving to next question.")
                st.rerun()

    st.write("**Running total:**", sum(st.session_state.scores), "/", 10 * (idx + 1) if st.session_state.scores else "—")


if __name__ == "__main__":
    main()
