"""
Content Extraction and Indexing modules for Wagtail RAG.

This package contains:
- text_extractors: Utility functions for extracting text from Wagtail pages
- page_to_documents: Converts Wagtail pages to LangChain Document objects with chunking
- index_builder: Core indexing logic for building the RAG index
"""

# Export text extraction utility functions
from .text_extractors import (
    extract_all_page_content_as_text,
    get_page_url,
    extract_text_from_streamfield,
    strip_html_tags_and_normalize_text,
)

# Backward compatibility aliases
extract_page_content = extract_all_page_content_as_text
extract_streamfield_text = extract_text_from_streamfield
clean_html = strip_html_tags_and_normalize_text

# Export document conversion (converts pages to LangChain documents)
from .page_to_documents import wagtail_page_to_documents

# Export index building function
from .index_builder import build_rag_index

# Export components from index builder
from .index_builder import get_page_models, get_live_pages, ChromaStore

__all__ = [
    # New clear function names
    'extract_all_page_content_as_text',
    'extract_text_from_streamfield',
    'strip_html_tags_and_normalize_text',
    # Backward compatibility aliases
    'extract_page_content',  # Alias for extract_all_page_content_as_text
    'extract_streamfield_text',  # Alias for extract_text_from_streamfield
    'clean_html',  # Alias for strip_html_tags_and_normalize_text
    # Other exports
    'get_page_url',
    'wagtail_page_to_documents',
    'build_rag_index',
    # New modular components
    'get_page_models',
    'get_live_pages',
    'ChromaStore',  # Works with both ChromaDB and FAISS
]

