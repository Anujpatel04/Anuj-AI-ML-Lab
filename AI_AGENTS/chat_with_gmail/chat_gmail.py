import os
import tempfile
import streamlit as st
from embedchain import App
from dotenv import load_dotenv

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_llms_dir = os.path.dirname(script_dir)
    env_path = os.path.join(all_llms_dir, '.env')
    load_dotenv(env_path, override=True)
except Exception as e:
    load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
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

st.title("Chat with your Gmail Inbox 📧")
st.caption("This app allows you to chat with your Gmail inbox using DeepSeek API")

if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
    st.sidebar.success("✅ DeepSeek API Key loaded successfully")
else:
    st.sidebar.error("❌ API Key not found")

gmail_filter = "to: me label:inbox"

if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
    if 'app' not in st.session_state:
        db_path = tempfile.mkdtemp()
        st.session_state.app = embedchain_bot(db_path, DEEPSEEK_API_KEY)
        st.session_state.emails_loaded = False
    
    if not st.session_state.emails_loaded:
        with st.spinner("📧 Loading emails from your Gmail inbox..."):
            try:
                st.session_state.app.add(gmail_filter, data_type="gmail")
                st.session_state.emails_loaded = True
                st.success(f"✅ Added emails from Inbox to the knowledge base!")
            except Exception as e:
                st.error(f"❌ Error loading emails: {e}")
                st.info("💡 Make sure you have set up Gmail API credentials (credentials.json) in your working directory.")

    if st.session_state.emails_loaded:
        prompt = st.text_input("Ask any question about your emails")

        if prompt:
            try:
                with st.spinner("🤔 Thinking..."):
                    answer = st.session_state.app.chat(prompt)
                    st.write("**Answer:**")
                    st.write(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")
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
    
    **Note:** You also need to set up Gmail API credentials:
    - Go to [Google Cloud Console](https://console.cloud.google.com/)
    - Create a project and enable Gmail API
    - Create OAuth credentials and download as `credentials.json`
    - Place `credentials.json` in this directory
    """)
