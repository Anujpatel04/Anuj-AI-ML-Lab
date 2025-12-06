import os
import tempfile
import streamlit as st
from embedchain import App
from dotenv import load_dotenv

# Load environment variables from .env file (from All_LLM's directory)
# Get the absolute path to the .env file in All_LLM's directory
try:
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to All_LLM's directory
    all_llms_dir = os.path.dirname(script_dir)
    env_path = os.path.join(all_llms_dir, '.env')
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

# Define the embedchain_bot function
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

# Create Streamlit app
st.title("Chat with your Gmail Inbox 📧")
st.caption("This app allows you to chat with your Gmail inbox using DeepSeek API")

# Debug: Show API key status (without exposing the key)
if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
    st.sidebar.success("✅ DeepSeek API Key loaded successfully")
else:
    st.sidebar.error("❌ API Key not found")

# Set the Gmail filter statically
gmail_filter = "to: me label:inbox"

# Add the Gmail data to the knowledge base if the DeepSeek API key is provided
if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
    # Initialize session state
    if 'app' not in st.session_state:
        # Create a temporary directory to store the database
        db_path = tempfile.mkdtemp()
        # Create an instance of Embedchain App
        st.session_state.app = embedchain_bot(db_path, DEEPSEEK_API_KEY)
        st.session_state.emails_loaded = False
    
    # Load emails if not already loaded
    if not st.session_state.emails_loaded:
        with st.spinner("📧 Loading emails from your Gmail inbox..."):
            try:
                st.session_state.app.add(gmail_filter, data_type="gmail")
                st.session_state.emails_loaded = True
                st.success(f"✅ Added emails from Inbox to the knowledge base!")
            except Exception as e:
                st.error(f"❌ Error loading emails: {e}")
                st.info("💡 Make sure you have set up Gmail API credentials (credentials.json) in your working directory.")

    # Ask a question about the emails
    if st.session_state.emails_loaded:
        prompt = st.text_input("Ask any question about your emails")

        # Chat with the emails
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