"""Configuration settings for the Codebase Q&A Agent."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)


class Settings:
    """Application settings."""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    
    # Model settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    QA_MODEL: str = os.getenv("QA_MODEL", "deepseek-chat")
    USE_OPENAI_EMBEDDINGS: bool = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
    
    # Chunking settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval settings
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    
    # Paths
    INDEX_PATH: str = os.getenv("INDEX_PATH", ".codebase_index.faiss")
    CONTEXT_PATH: str = os.getenv("CONTEXT_PATH", ".mcp_context.json")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_api_key(cls) -> str:
        """Get API key for LLM (prefer DeepSeek, fallback to OpenAI)."""
        return cls.DEEPSEEK_API_KEY or cls.OPENAI_API_KEY or ""
    
    @classmethod
    def get_base_url(cls) -> Optional[str]:
        """Get base URL (only for DeepSeek)."""
        if cls.DEEPSEEK_API_KEY:
            return cls.DEEPSEEK_BASE_URL
        return None





