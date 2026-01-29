"""
Content Extraction and Indexing modules for Wagtail RAG.

This package contains:
- content_extraction: Utility functions for extracting text from Wagtail pages
- indexing: Core indexing logic for building the RAG index
"""

# Export content extraction utility functions
from .content_extraction import (
    extract_page_content,
    get_page_url,
    extract_streamfield_text,
    clean_html,
)

# Export document extraction (moved to extractors.py)
from .extractors import wagtail_page_to_documents

# Export indexing function
from .indexer import build_rag_index

# Export components from indexer
from .indexer import get_page_models, get_live_pages, ChromaStore

__all__ = [
    'extract_page_content',
    'get_page_url',
    'extract_streamfield_text',
    'clean_html',
    'wagtail_page_to_documents',
    'build_rag_index',
    # New modular components
    'get_page_models',
    'get_live_pages',
    'ChromaStore',  # Works with both ChromaDB and FAISS
]

