"""Embedding generation module for code chunks."""

from typing import List, Dict, Any, Optional
import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class CodeEmbedder:
    """Generates embeddings for code chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = False, api_key: Optional[str] = None):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            use_openai: Whether to use OpenAI embeddings
            api_key: OpenAI API key (required if use_openai=True)
        """
        self.use_openai = use_openai
        self.model = None
        self.model_name = model_name
        self.embedding_dim = None
        
        if use_openai:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required for OpenAI embeddings")
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.client = openai.OpenAI(api_key=api_key)
            self.embedding_dim = 1536  # OpenAI text-embedding-3-small dimension
            logger.info("Using OpenAI embeddings")
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers package required. Install with: pip install sentence-transformers")
            self.model = SentenceTransformer(model_name)
            # Get embedding dimension from model
            test_embedding = self.model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
            logger.info(f"Using sentence transformer model: {model_name} (dim={self.embedding_dim})")
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if self.embedding_dim is not None:
            return self.embedding_dim
        # Default for all-MiniLM-L6-v2
        return 384
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of code chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' field
        
        Returns:
            List of chunks with added 'embedding' field
        """
        if not chunks:
            return []
        
        texts = [chunk['content'] for chunk in chunks]
        
        if self.use_openai:
            embeddings = self._embed_with_openai(texts)
        else:
            embeddings = self._embed_with_sentence_transformer(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        return chunks
    
    def _embed_with_sentence_transformer(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence transformers."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text[:8000]  # Limit text length
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                logger.error(f"Error generating OpenAI embedding: {e}")
                # Fallback to zero vector
                embeddings.append([0.0] * 1536)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query text
        
        Returns:
            Embedding vector
        """
        if self.use_openai:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            return response.data[0].embedding
        else:
            embedding = self.model.encode([query], show_progress_bar=False)
            return embedding[0].tolist()

