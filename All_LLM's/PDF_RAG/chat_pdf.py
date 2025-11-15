import os
import tempfile
import streamlit as st
from embedchain import App
from dotenv import load_dotenv

# Load environment variables from .env file (from root directory)
# Get the absolute path to the .env file in the root directory
try:
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get to the root directory
    root_dir = os.path.dirname(script_dir)
    env_path = os.path.join(root_dir, '.env')
    # Load the .env file
    load_dotenv(env_path, override=True)
except Exception as e:
    # Fallback: try loading from current working directory or parent directories
    load_dotenv(override=True)

# Get DeepSeek API key from environment variable
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
# Strip any whitespace that might have been added
if DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = DEEPSEEK_API_KEY.strip()

def embedchain_bot(db_path, api_key):
    return App.from_config(
        config={
            "llm": {
                "provider": "openai", 
                "config": {
                    "model": "deepseek-chat", 
                    "temperature": 0.5, 
                    "api_key": api_key,
                    "base_url": "https://api.deepseek.com"
                }
            },
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            "embedder": {
                "provider": "huggingface", 
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            },
        }
    )

st.title("Chat with PDF using DeepSeek")
st.caption("This app allows you to chat with a PDF using DeepSeek API")

# Debug: Show API key status (without exposing the key)
if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
    st.sidebar.success("✅ API Key loaded successfully")
else:
    st.sidebar.error("❌ API Key not found")

# Check if API key is set
if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
    db_path = tempfile.mkdtemp()
    app = embedchain_bot(db_path, DEEPSEEK_API_KEY)

    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_file.getvalue())
            app.add(f.name, data_type="pdf_file")
        os.remove(f.name)
        st.success(f"Added {pdf_file.name} to knowledge base!")

    prompt = st.text_input("Ask a question about the PDF")

    if prompt:
        with st.spinner("Thinking..."):
            answer = app.chat(prompt)
            st.write("**Answer:**")
            st.write(answer)
else:
    st.error("❌ DeepSeek API key is not configured!")
    st.info("""
    **To set up your API key:**
    
    1. **Edit the .env file**: Open `.env` file in the root directory and set:
       ```
       DEEPSEEK_API_KEY=your-api-key-here
       ```
    
    2. **Or set environment variable**:
       ```bash
       export DEEPSEEK_API_KEY="your-api-key-here"
       ```
    """)

        