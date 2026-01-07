"""Embeddings module for generating vector representations of code chunks."""

from .embedder import CodeEmbedder
from .chunker import CodeChunker

__all__ = ['CodeEmbedder', 'CodeChunker']





