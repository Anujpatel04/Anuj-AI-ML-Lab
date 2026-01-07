"""Vector store using FAISS for efficient similarity search."""

from typing import List, Dict, Any, Optional
import numpy as np
import logging
import pickle
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for code chunk embeddings."""
    
    def __init__(self, dimension: int = 384, index_path: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of embedding vectors
            index_path: Path to save/load FAISS index
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self._init_index()
    
    def _init_index(self) -> None:
        """Initialize or load FAISS index."""
        if self.index_path and Path(self.index_path).exists():
            try:
                self.load(self.index_path)
                logger.info(f"Loaded existing index from {self.index_path}")
            except Exception as e:
                logger.warning(f"Could not load index: {e}. Creating new index.")
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' field
        """
        if not chunks:
            return
        
        embeddings = []
        valid_chunks = []
        
        for chunk in chunks:
            if 'embedding' in chunk:
                embedding = chunk['embedding']
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                
                if embedding.shape[0] == self.dimension:
                    embeddings.append(embedding)
                    valid_chunks.append(chunk)
                else:
                    logger.warning(f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[0]}")
        
        if not embeddings:
            logger.warning("No valid embeddings to add")
            return
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store chunk metadata
        start_idx = len(self.chunks)
        self.chunks.extend(valid_chunks)
        
        logger.info(f"Added {len(valid_chunks)} chunks to vector store (total: {len(self.chunks)})")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of chunk dictionaries with similarity scores
        """
        if self.index.ntotal == 0:
            return []
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Ensure dimension matches
        if query_vector.shape[1] != self.dimension:
            logger.error(f"Query dimension mismatch: expected {self.dimension}, got {query_vector.shape[1]}")
            return []
        
        # Search
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(1.0 / (1.0 + distance))  # Convert distance to similarity
                chunk['distance'] = float(distance)
                results.append(chunk)
        
        return results
    
    def save(self, index_path: Optional[str] = None) -> None:
        """
        Save the index and chunks to disk.
        
        Args:
            index_path: Path to save (uses self.index_path if None)
        """
        save_path = index_path or self.index_path
        if not save_path:
            logger.warning("No index path specified, skipping save")
            return
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path))
        
        # Save chunks metadata
        chunks_path = save_path.with_suffix('.chunks.pkl')
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Saved index to {save_path}")
    
    def load(self, index_path: str) -> None:
        """
        Load index and chunks from disk.
        
        Args:
            index_path: Path to load from
        """
        index_path = Path(index_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d
        
        # Load chunks metadata
        chunks_path = index_path.with_suffix('.chunks.pkl')
        if chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
        else:
            logger.warning(f"Chunks file not found: {chunks_path}")
            self.chunks = []
        
        logger.info(f"Loaded index with {len(self.chunks)} chunks")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal,
            'dimension': self.dimension
        }





