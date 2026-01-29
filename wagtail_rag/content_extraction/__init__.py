"""
Content Extraction and Indexing modules for Wagtail RAG.

This package contains:
- content_extraction: Utility functions for extracting text from Wagtail pages
- indexing: Core indexing logic for building the RAG index
"""

# Export content extraction functions
from .content_extraction import (
    extract_page_content,
    get_page_url,
    extract_streamfield_text,
    clean_html,
    wagtail_page_to_documents,
)

# Export indexing function
from .indexing import build_rag_index

__all__ = [
    'extract_page_content',
    'get_page_url',
    'extract_streamfield_text',
    'clean_html',
    'wagtail_page_to_documents',
    'build_rag_index',
]

