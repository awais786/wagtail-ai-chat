"""
Content Extraction and Indexing modules for Wagtail RAG.

This package contains:
- page_to_documents: Converts Wagtail pages to LangChain Document objects with chunking
- index_builder: Core indexing logic for building the RAG index
"""

# Export document conversion (converts pages to LangChain documents)
from .page_to_documents import (
    wagtail_page_to_documents,
    get_page_url,
    extract_text_from_streamfield,
)

# Export index building function
from .index_builder import build_rag_index

# Export components from index builder
from .index_builder import get_page_models, get_live_pages, ChromaStore

__all__ = [
    # Document conversion
    'wagtail_page_to_documents',
    'get_page_url',
    'extract_text_from_streamfield',
    # Index building
    'build_rag_index',
    # Index builder components
    'get_page_models',
    'get_live_pages',
    'ChromaStore',  # Works with both ChromaDB and FAISS
]

