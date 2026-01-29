"""
Embedding Provider and Search modules for Wagtail RAG.

This package contains:
- providers: Factory function for creating embedding instances from various providers
- search: Embedding-based similarity search logic (vector search, hybrid search, ranking)
"""

from .providers import get_embeddings, EmbeddingProviderFactory, BaseEmbeddingProvider
from .search import EmbeddingSearcher

__all__ = ['get_embeddings', 'EmbeddingSearcher', 'EmbeddingProviderFactory', 'BaseEmbeddingProvider']

