"""
Content Extraction and Indexing modules for Wagtail RAG.

This package contains:
- api_fields_extractor: Advanced API fields-based content extractor
- index_builder: Core indexing logic for building the RAG index
"""

# Export primary extractor
try:
    from .api_fields_extractor import (
        page_to_documents_api_extractor,
        WagtailAPIExtractor,
    )
except ImportError:
    page_to_documents_api_extractor = None
    WagtailAPIExtractor = None

# Export index building function
from .index_builder import build_rag_index

# Export components from index builder
from .index_builder import get_page_models, get_live_pages, ChromaStore

__all__ = [
    # Primary extractor
    "page_to_documents_api_extractor",
    "WagtailAPIExtractor",
    # Index building
    "build_rag_index",
    # Index builder components
    "get_page_models",
    "get_live_pages",
    "ChromaStore",  # Works with both ChromaDB and FAISS
]
