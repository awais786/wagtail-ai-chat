"""
Convert Wagtail pages to LangChain Document objects.

This module handles converting Wagtail Page objects into LangChain Document objects
with intelligent chunking and metadata. It creates separate documents for title/intro
and chunks body content with title context for better semantic separation and retrieval.
"""

import logging
from typing import List, Optional, Callable, TYPE_CHECKING

from django.conf import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from wagtail.models import Page  # type: ignore
else:
    try:
        from wagtail.models import Page  # type: ignore
    except ImportError:
        Page = None  # type: ignore

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


def get_page_url(page: Page) -> str:
    """
    Get page URL safely, handling cases where it might not be available.

    Args:
        page: Wagtail Page instance

    Returns:
        Page URL or empty string if not available
    """
    try:
        return page.get_full_url() or page.url_path or ""
    except (AttributeError, RuntimeError) as e:
        logger.debug("get_page_url failed for page %s: %s", getattr(page, "pk", None), e)
        return ""


def _extract_strings_from_dict(data):
    """Recursively extract all string values from a nested dictionary."""
    strings = []
    for value in data.values():
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            strings.extend(_extract_strings_from_dict(value))
    return strings


def extract_text_from_streamfield(streamfield) -> str:
    """Extract text content from a Wagtail StreamField."""
    if not streamfield:
        return ""

    text_parts = []
    for block in streamfield:
        if hasattr(block, "value"):
            value = block.value
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, dict):
                text_parts.extend(_extract_strings_from_dict(value))
        elif isinstance(block, str):
            text_parts.append(block)

    return " ".join(text_parts)


def _chunk_streamfield(
    page: Page,
    base_metadata: dict,
    chunk_size: int,
    chunk_overlap: int,
    stdout: Optional[Callable[[str], None]] = None,
    streamfield_field_names: Optional[List[str]] = None,
) -> List[Document]:
    """Extract and chunk StreamField content from a page."""
    documents = []
    field_names = streamfield_field_names or ["body", "content", "backstory", "instructions"]
    
    # Collect text from all body fields
    body_texts = []
    for field_name in field_names:
        field_value = getattr(page, field_name, None)
        if field_value:
            text = extract_text_from_streamfield(field_value)
            if text and text.strip():
                body_texts.append({
                    "text": text,
                    "field": field_name,
                    "metadata": {**base_metadata, "section": "body", "field": field_name}
                })

    if not body_texts:
        return documents

    # Import text splitter
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    # Chunk each field's text
    title = (page.title or "").strip() or "(no title)"
    for body_item in body_texts:
        chunks = splitter.split_text(body_item["text"])
        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{base_metadata['page_id']}_body_{chunk_idx}"
            documents.append(Document(
                page_content=f"Title: {title}\n\n{chunk}",
                metadata={**body_item["metadata"], "chunk_index": chunk_idx, "doc_id": doc_id}
            ))

    return documents


def wagtail_page_to_documents(
    page: Page,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    stdout: Optional[Callable[[str], None]] = None,
    streamfield_field_names: Optional[List[str]] = None,
) -> List[Document]:
    """
    Convert a Wagtail page to Document objects with chunking.
    
    Creates separate documents for title, intro, and chunked body content.
    
    Args:
        page: Wagtail Page instance
        chunk_size: Size of text chunks (default: 500)
        chunk_overlap: Overlap between chunks (default: 75)
        stdout: Optional output function for debugging
        streamfield_field_names: Optional field names for body content

    Returns:
        List of Document objects
    """
    # Apply settings defaults
    chunk_size = getattr(settings, "WAGTAIL_RAG_CHUNK_SIZE", chunk_size)
    chunk_overlap = getattr(settings, "WAGTAIL_RAG_CHUNK_OVERLAP", chunk_overlap)

    # Get specific page type to access all fields
    page = page.specific
    page_title = page.title or ""
    
    base_meta = {
        "page_id": page.id,
        "page_type": page.__class__.__name__,
        "slug": getattr(page, "slug", ""),
        "title": page_title,
        "url": get_page_url(page),
    }

    documents = []

    # 1. Title document
    documents.append(Document(
        page_content=f"Title: {page_title or '(no title)'}",
        metadata={**base_meta, "section": "title", "chunk_index": 0, 
                 "doc_id": f"{page.id}_title_0"}
    ))

    # 2. Intro document (check common field names)
    intro_text = None
    for field_name in ['intro', 'introduction', 'lead', 'summary', 'description']:
        value = getattr(page, field_name, None)
        if value:
            intro_text = str(value) if not isinstance(value, str) else value
            break
    
    if intro_text:
        documents.append(Document(
            page_content=f"Introduction: {intro_text}",
            metadata={**base_meta, "section": "intro", "chunk_index": 0,
                     "doc_id": f"{page.id}_intro_0"}
        ))

    # 3. Body documents (chunked)
    body_docs = _chunk_streamfield(
        page, base_meta, chunk_size, chunk_overlap, 
        stdout, streamfield_field_names
    )
    documents.extend(body_docs)

    return documents

