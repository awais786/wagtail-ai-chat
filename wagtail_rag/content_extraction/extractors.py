"""Convert Wagtail pages to LangChain Document objects with chunking."""

from typing import List, Optional, Callable

from django.conf import settings
from wagtail.models import Page

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

from wagtail_rag.content_extraction.content_extraction import (
    extract_streamfield_text,
    get_page_url,
)

STREAMFIELD_FIELDS = ["body", "content", "backstory", "instructions"]
INTRO_FIELDS = ["intro", "introduction", "lead", "summary", "description"]


def _field_to_str(value):
    if not value:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "title"):
        attr = value.title
        return attr() if callable(attr) else attr
    if hasattr(value, "name"):
        attr = value.name
        return attr() if callable(attr) else attr
    return str(value)


def _chunk_streamfield(
    page: Page,
    base_metadata: dict,
    chunk_size: int,
    chunk_overlap: int,
    stdout: Optional[Callable[[str], None]] = None,
) -> List[Document]:
    documents = []
    body_texts = []

    for field_name in STREAMFIELD_FIELDS:
        if not hasattr(page, field_name):
            if stdout:
                stdout(f"  Field '{field_name}' not found on page")
            continue
        value = getattr(page, field_name, None)
        if not value:
            if stdout:
                stdout(f"  Field '{field_name}' exists but is None/empty")
            continue
        if stdout:
            stdout(f"  Found field '{field_name}', extracting text...")
        text = extract_streamfield_text(value)
        if not text or not text.strip():
            if stdout:
                stdout(f"  Field '{field_name}' exists but contains no text")
            continue
        body_texts.append(
            {
                "text": text,
                "field": field_name,
                "metadata": {**base_metadata, "section": "body", "field": field_name},
            }
        )
        if stdout:
            stdout(f"  Extracted {len(text)} characters from '{field_name}'")

    if not body_texts and stdout:
        stdout("  No body content found. Only title and intro will be indexed.")

    if body_texts:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
        for body_item in body_texts:
            chunks = splitter.split_text(body_item["text"])
            if stdout:
                stdout(f"  Field: {body_item['field']} - Split into {len(chunks)} chunks")
            page_id = body_item["metadata"].get("page_id", "unknown")
            section = body_item["metadata"].get("section", "body")
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{page_id}_{section}_{chunk_idx}"
                doc = Document(
                    page_content=f"Title: {page.title}\n\n{chunk}",
                    metadata={
                        **body_item["metadata"],
                        "chunk_index": chunk_idx,
                        "doc_id": doc_id,
                    },
                )
                documents.append(doc)
                if stdout:
                    stdout(f"  [Document {len(documents)}] {section} / {body_item['field']} chunk {chunk_idx + 1}/{len(chunks)} (ID: {doc_id})")

    return documents


def wagtail_page_to_documents(
    page: Page,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    stdout: Optional[Callable[[str], None]] = None,
) -> List[Document]:
    chunk_size = getattr(settings, "WAGTAIL_RAG_CHUNK_SIZE", chunk_size)
    chunk_overlap = getattr(settings, "WAGTAIL_RAG_CHUNK_OVERLAP", chunk_overlap)
    page = page.specific
    documents = []

    base_meta = {
        "page_id": page.id,
        "page_type": page.__class__.__name__,
        "slug": getattr(page, "slug", ""),
        "title": page.title,
        "url": get_page_url(page),
    }

    title_doc_id = f"{base_meta['page_id']}_title_0"
    documents.append(
        Document(
            page_content=f"Title: {page.title}",
            metadata={**base_meta, "section": "title", "chunk_index": 0, "doc_id": title_doc_id},
        )
    )
    if stdout:
        stdout(f"  [Document 1] Section: title (ID: {title_doc_id})")

    intro_text = None
    for field_name in INTRO_FIELDS:
        if hasattr(page, field_name):
            value = getattr(page, field_name, None)
            intro_text = _field_to_str(value)
            if intro_text:
                break

    if intro_text:
        intro_doc_id = f"{base_meta['page_id']}_intro_0"
        documents.append(
            Document(
                page_content=f"Introduction: {intro_text}",
                metadata={**base_meta, "section": "intro", "chunk_index": 0, "doc_id": intro_doc_id},
            )
        )
        if stdout:
            stdout(f"  [Document {len(documents)}] Section: intro (ID: {intro_doc_id})")

    body_docs = _chunk_streamfield(page, base_meta, chunk_size, chunk_overlap, stdout)
    documents.extend(body_docs)

    if stdout:
        stdout(f"  Total documents created: {len(documents)}")

    return documents
