"""
Document extraction from Wagtail pages.

This module handles converting Wagtail Page objects into LangChain Document objects
with intelligent chunking and metadata.
"""

from typing import List, Optional, Callable
from django.conf import settings
from wagtail.models import Page

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from wagtail_rag.content_extraction.content_extraction import (
    extract_streamfield_text,
    get_page_url,
)


def _chunk_streamfield(page: Page, base_metadata: dict, chunk_size: int, chunk_overlap: int, stdout: Optional[Callable[[str], None]] = None) -> List[Document]:
    """
    Extract and chunk StreamField content from a page.
    
    Returns a list of Document objects for the body content.
    """
    documents = []
    streamfield_fields = ['body', 'content', 'backstory', 'instructions']
    body_texts = []
    
    if stdout:
        stdout(f"  Checking for body content in fields: {', '.join(streamfield_fields)}")
    
    for field_name in streamfield_fields:
        if hasattr(page, field_name):
            field_value = getattr(page, field_name, None)
            if field_value:
                if stdout:
                    stdout(f"  Found field '{field_name}', extracting text...")
                streamfield_text = extract_streamfield_text(field_value)
                if streamfield_text and streamfield_text.strip():
                    body_texts.append({
                        "text": streamfield_text,
                        "field": field_name,
                        "metadata": {
                            **base_metadata,
                            "section": "body",
                            "field": field_name,
                        }
                    })
                    if stdout:
                        stdout(f"  Extracted {len(streamfield_text)} characters from '{field_name}'")
                elif stdout:
                    stdout(f"  Field '{field_name}' exists but contains no text")
            elif stdout:
                stdout(f"  Field '{field_name}' exists but is None/empty")
        elif stdout:
            stdout(f"  Field '{field_name}' not found on page")
    
    if not body_texts and stdout:
        stdout(f"  No body content found. Only title and intro will be indexed.")

    if body_texts:
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

        for body_item in body_texts:
            chunks = splitter.split_text(body_item["text"])
            if stdout:
                stdout(f"  Field: {body_item['field']} - Split into {len(chunks)} chunks")
            
            for chunk_idx, chunk in enumerate(chunks):
                # Include title for context (improves semantic understanding)
                page_id = body_item['metadata'].get('page_id', 'unknown')
                section = body_item['metadata'].get('section', 'body')
                chunk_doc_id = f"{page_id}_{section}_{chunk_idx}"
                chunk_doc = Document(
                    page_content=f"Title: {page.title}\n\n{chunk}",
                    metadata={**body_item["metadata"], "chunk_index": chunk_idx, "doc_id": chunk_doc_id},
                )
                documents.append(chunk_doc)
                
                if stdout:
                    field = body_item['metadata'].get('field', 'unknown')
                    stdout(f"  [Document {len(documents)}] Section: {section}, Field: {field}, Chunk {chunk_idx + 1}/{len(chunks)} (ID: {chunk_doc_id})")
                    stdout(f"    Content: {chunk_doc.page_content[:300]}{'...' if len(chunk_doc.page_content) > 300 else ''}")
                    stdout(f"    Metadata: {chunk_doc.metadata}")
                    stdout("")

    return documents


def wagtail_page_to_documents(
    page: Page,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    stdout: Optional[Callable[[str], None]] = None,
) -> List[Document]:
    """
    Convert a Wagtail page to multiple Document objects with intelligent chunking.
    
    This approach creates separate documents for title/intro and chunks body content
    with title context, providing better semantic separation and retrieval.
    
    Args:
        page: Wagtail Page instance (will be converted to specific page)
        chunk_size: Size of text chunks (default: 500)
        chunk_overlap: Overlap between chunks (default: 75)
        stdout: Optional output function to print documents (for debugging)
    
    Returns:
        List of Document objects ready for indexing
    """
    # Get chunk size/overlap from settings if using defaults
    if chunk_size == 500:
        chunk_size = getattr(settings, 'WAGTAIL_RAG_CHUNK_SIZE', 500)
    if chunk_overlap == 75:
        chunk_overlap = getattr(settings, 'WAGTAIL_RAG_CHUNK_OVERLAP', 75)
    
    # Get specific page type
    page = page.specific
    documents = []

    base_meta = {
        "page_id": page.id,
        "page_type": page.__class__.__name__,
        "slug": getattr(page, "slug", ""),
        "title": page.title,
        "url": get_page_url(page),
    }
    
    if stdout:
        # Debug: Show all fields on the page
        all_fields = [f.name for f in page._meta.get_fields() if hasattr(f, 'name')]
        streamfield_fields = ['body', 'content', 'backstory', 'instructions']
        found_fields = [f for f in streamfield_fields if f in all_fields]
        if not found_fields:
            stdout(f"  Debug: Page has {len(all_fields)} fields, but none of {streamfield_fields} were found.")
            stdout(f"  Available fields include: {', '.join(all_fields[:20])}{'...' if len(all_fields) > 20 else ''}")

    # Title
    title_doc_id = f"{base_meta['page_id']}_title_0"
    title_doc = Document(
        page_content=f"Title: {page.title}",
        metadata={**base_meta, "section": "title", "chunk_index": 0, "doc_id": title_doc_id},
    )
    documents.append(title_doc)
    
    if stdout:
        stdout(f"  [Document 1] Section: title (ID: {title_doc_id})")
        stdout(f"    Content: {title_doc.page_content}")
        stdout(f"    Metadata: {title_doc.metadata}")
        stdout("")

    # Intro (if exists) - check multiple common field names
    intro_fields = ['intro', 'introduction', 'lead', 'summary', 'description']
    intro_text = None
    for field_name in intro_fields:
        if hasattr(page, field_name):
            value = getattr(page, field_name, None)
            if value:
                # Extract string from various field types
                if isinstance(value, str):
                    intro_text = value
                elif hasattr(value, 'title'):
                    attr = value.title
                    intro_text = attr() if callable(attr) else attr
                elif hasattr(value, 'name'):
                    attr = value.name
                    intro_text = attr() if callable(attr) else attr
                else:
                    intro_text = str(value)
                if intro_text:
                    break
    
    if intro_text:
        intro_doc_id = f"{base_meta['page_id']}_intro_0"
        intro_doc = Document(
            page_content=f"Introduction: {intro_text}",
            metadata={**base_meta, "section": "intro", "chunk_index": 0, "doc_id": intro_doc_id},
        )
        documents.append(intro_doc)
        
        if stdout:
            stdout(f"  [Document {len(documents)}] Section: intro (ID: {intro_doc_id})")
            stdout(f"    Content: {intro_doc.page_content[:200]}{'...' if len(intro_doc.page_content) > 200 else ''}")
            stdout(f"    Metadata: {intro_doc.metadata}")
            stdout("")

    # Body handled by chunking function
    body_docs = _chunk_streamfield(page, base_meta, chunk_size, chunk_overlap, stdout)
    documents.extend(body_docs)

    if stdout:
        stdout(f"  Total documents created: {len(documents)}")
        if len(documents) == 1:
            stdout(f"  Note: Only title document created (no intro or body content found)")
        elif len(documents) == 2:
            stdout(f"  Note: Title and intro created (no body content found)")
        stdout("")

    return documents

