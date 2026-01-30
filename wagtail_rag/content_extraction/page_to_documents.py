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


def _extract_all_strings_from_dict_recursively(data):
    """
    Recursively extract all string values from a nested dictionary.
    
    Traverses the dictionary structure and collects all string values,
    handling nested dictionaries by recursively processing them.
    
    Args:
        data: Dictionary (may contain nested dictionaries)
        
    Returns:
        List of all string values found in the dictionary structure
    """
    strings = []
    for value in data.values():
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            strings.extend(_extract_all_strings_from_dict_recursively(value))
    return strings


def extract_text_from_streamfield(streamfield) -> str:
    """
    Extract all text content from a Wagtail StreamField block.
    
    Processes each block in the StreamField:
    - String blocks: added directly
    - Blocks with 'value' attribute: extracts string values or recursively processes dict values
    - Dict blocks: recursively extracts all string values from nested dictionaries
    
    Args:
        streamfield: Wagtail StreamField instance or iterable of blocks
        
    Returns:
        Single string containing all extracted text, space-separated
    """
    if not streamfield:
        return ""

    text_parts = []
    for block in streamfield:
        if hasattr(block, "value"):
            value = block.value
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, dict):
                # Extract text from dict values recursively
                text_parts.extend(_extract_all_strings_from_dict_recursively(value))
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
    """
    Extract and chunk StreamField/content field text from a page.

    When streamfield_field_names is provided (e.g. from WAGTAIL_RAG_MODELS "app.Model:*"),
    only those fields are used; otherwise defaults to body, content, backstory, instructions.
    """
    documents = []
    streamfield_fields = streamfield_field_names if streamfield_field_names is not None else ["body", "content", "backstory", "instructions"]
    body_texts = []

    if stdout:
        stdout(f"  Checking for body content in fields: {', '.join(streamfield_fields)}")
    
    for field_name in streamfield_fields:
        if hasattr(page, field_name):
            field_value = getattr(page, field_name, None)
            if field_value:
                if stdout:
                    stdout(f"  Found field '{field_name}', extracting text...")
                streamfield_text = extract_text_from_streamfield(field_value)
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
                page_id = body_item["metadata"].get("page_id", "unknown")
                section = body_item["metadata"].get("section", "body")
                chunk_doc_id = f"{page_id}_{section}_{chunk_idx}"
                title = (page.title or "").strip() or "(no title)"
                chunk_doc = Document(
                    page_content=f"Title: {title}\n\n{chunk}",
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
    streamfield_field_names: Optional[List[str]] = None,
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
        streamfield_field_names: Optional list of field names for body content (e.g. from "app.Model:*")

    Returns:
        List of Document objects ready for indexing
    """
    # Get chunk size/overlap from settings if using defaults
    if chunk_size == 500:
        chunk_size = getattr(settings, "WAGTAIL_RAG_CHUNK_SIZE", 500)
    if chunk_overlap == 75:
        chunk_overlap = getattr(settings, "WAGTAIL_RAG_CHUNK_OVERLAP", 75)

    # Resolve to the concrete page type (e.g. BreadPage) so we can access its fields
    page = page.specific
    documents = []

    page_title = page.title if page.title is not None else ""
    base_meta = {
        "page_id": page.id,
        "page_type": page.__class__.__name__,
        "slug": getattr(page, "slug", ""),
        "title": page_title,
        "url": get_page_url(page),
    }
    
    body_field_list = streamfield_field_names if streamfield_field_names is not None else ["body", "content", "backstory", "instructions"]
    if stdout:
        all_fields = [f.name for f in page._meta.get_fields() if hasattr(f, "name")]
        found_fields = [f for f in body_field_list if f in all_fields]
        if not found_fields:
            stdout(f"  Debug: Page has {len(all_fields)} fields, but none of {body_field_list} were found.")
            stdout(f"  Available fields include: {', '.join(all_fields[:20])}{'...' if len(all_fields) > 20 else ''}")

    # Title
    title_doc_id = f"{base_meta['page_id']}_title_0"
    title_doc = Document(
        page_content=f"Title: {page_title or '(no title)'}",
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

    # Body handled by chunking function (uses streamfield_field_names when provided, e.g. from "app.Model:*")
    body_docs = _chunk_streamfield(page, base_meta, chunk_size, chunk_overlap, stdout, streamfield_field_names=streamfield_field_names)
    documents.extend(body_docs)

    if stdout:
        stdout(f"  Total documents created: {len(documents)}")
        if len(documents) == 1:
            stdout(f"  Note: Only title document created (no intro or body content found)")
        elif len(documents) == 2:
            stdout(f"  Note: Title and intro created (no body content found)")
        stdout("")

    return documents

