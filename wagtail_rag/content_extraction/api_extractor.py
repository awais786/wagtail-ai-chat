"""
Simple extractor using Wagtail's API serialization.

Uses Wagtail's built-in methods to extract content, avoiding manual field type handling.
"""
import logging
from typing import List, Optional, Dict

from django.conf import settings
from django.utils.html import strip_tags

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SIZE_THRESHOLD = 2000
MIN_FIELD_LENGTH = 10


class WagtailAPIExtractor:
    """
    Simple extractor using Wagtail's API fields.
    
    Instead of manually parsing StreamFields, RichTextFields, etc.,
    this uses Wagtail's own serialization methods which handle all field types.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        size_threshold: Optional[int] = None
    ):
        self.chunk_size = chunk_size or getattr(settings, 'WAGTAIL_RAG_CHUNK_SIZE', DEFAULT_CHUNK_SIZE)
        self.chunk_overlap = chunk_overlap or getattr(settings, 'WAGTAIL_RAG_CHUNK_OVERLAP', DEFAULT_CHUNK_OVERLAP)
        self.size_threshold = size_threshold or getattr(settings, 'WAGTAIL_RAG_SIZE_THRESHOLD', DEFAULT_SIZE_THRESHOLD)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    @staticmethod
    def _get_page_url(page) -> str:
        """Get page URL safely."""
        try:
            return page.full_url
        except (AttributeError, RuntimeError):
            try:
                return page.url
            except (AttributeError, RuntimeError):
                return f"/page/{page.id}/"
    
    @staticmethod
    def _clean_text(text) -> str:
        """Clean text by stripping HTML and normalizing whitespace."""
        if not text:
            return ""
        # Strip HTML tags
        text = strip_tags(str(text))
        # Normalize whitespace
        text = " ".join(text.split())
        return text.strip()
    
    def _extract_from_dict(self, data: dict) -> str:
        """Extract text from dictionary structures."""
        text_parts = []
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(self._clean_text(value))
            elif hasattr(value, 'source'):  # RichText in dict
                text_parts.append(self._clean_text(value.source))
        return " ".join(text_parts) if text_parts else ""

    def _extract_from_list(self, data: list) -> str:
        """Extract text from list structures."""
        text_parts = []
        for item in data:
            if isinstance(item, str) and item.strip():
                text_parts.append(self._clean_text(item))
            elif hasattr(item, 'source'):
                text_parts.append(self._clean_text(item.source))
        return " ".join(text_parts) if text_parts else ""

    def _extract_from_block(self, block) -> Optional[str]:
        """Extract text from a StreamField block using multiple fallback strategies."""
        block_value = block.value

        # Method 1: Try render_as_block() first
        try:
            rendered = block.render_as_block()
            extracted_text = self._clean_text(rendered)
            if extracted_text:
                return extracted_text
        except Exception:
            pass

        # Method 2: Try direct value extraction
        # For RichText blocks
        if hasattr(block_value, 'source'):
            extracted_text = self._clean_text(block_value.source)
            if extracted_text:
                return extracted_text

        # For dict/structured blocks
        if isinstance(block_value, dict):
            extracted_text = self._extract_from_dict(block_value)
            if extracted_text:
                return extracted_text

        # For simple string values
        if isinstance(block_value, str):
            extracted_text = self._clean_text(block_value)
            if extracted_text:
                return extracted_text

        # For list blocks
        if isinstance(block_value, list):
            extracted_text = self._extract_from_list(block_value)
            if extracted_text:
                return extracted_text

        # Method 3: Last resort - stringify
        extracted_text = self._clean_text(str(block_value))
        if extracted_text:
            return extracted_text

        return None

    def _extract_field_value(self, page, field_name: str) -> Optional[str]:
        """
        Extract text from a field using Wagtail's rendering.
        
        This works for all field types: StreamField, RichTextField, TextField, etc.
        """
        try:
            value = getattr(page, field_name, None)
            if value is None:
                return None
            
            # For StreamField: extract raw content from each block
            if hasattr(value, '__iter__') and hasattr(value, 'stream_block'):
                parts = []
                for block in value:
                    try:
                        extracted_text = self._extract_from_block(block)
                        if extracted_text:
                            parts.append(extracted_text)
                    except Exception as e:
                        logger.warning(f"Block extraction failed for '{field_name}': {e}")

                return "\n\n".join(parts) if parts else None
            
            # For RichTextField: has .source property
            if hasattr(value, 'source'):
                return self._clean_text(value.source)
            
            # For simple text fields
            if isinstance(value, str):
                return self._clean_text(value)
            
            # For other types, convert to string
            text = str(value)
            return self._clean_text(text) if text else None
            
        except Exception as e:
            logger.error(f"Error extracting field '{field_name}': {e}")
            return None
    
    def extract_page(self, page) -> List[Document]:
        """
        Extract content from a Wagtail page using API-style field access.
        
        This is simpler than manual parsing because we let Wagtail handle
        the complexity of different field types.
        """
        # Get all API fields defined on the page
        api_fields = []
        if hasattr(page, 'api_fields'):
            api_fields = [f.name for f in page.api_fields if hasattr(f, 'name')]
        
        # Fallback: use common field names or configured defaults
        if not api_fields:
            api_fields = getattr(
                settings, 
                'WAGTAIL_RAG_DEFAULT_FIELDS', 
                ['introduction', 'body', 'content', 'backstory']
            )

        # Base metadata
        metadata = {
            "page_id": page.id,
            "page_type": page.__class__.__name__,
            "title": str(page.title),
            "slug": getattr(page, 'slug', ''),
            "url": self._get_page_url(page),
        }
        
        # Extract all fields
        sections = {}
        sections['title'] = str(page.title)
        extracted_fields = ['title']

        for field_name in api_fields:
            text = self._extract_field_value(page, field_name)
            if text and len(text.strip()) > MIN_FIELD_LENGTH:
                sections[field_name] = text
                extracted_fields.append(field_name)

        # Combine all sections
        full_content = "\n\n".join(sections.values())
        content_length = len(full_content)
        
        logger.debug(
            f"Page '{page.title}' (ID: {page.id}): extracted {len(extracted_fields)} fields "
            f"({', '.join(extracted_fields)}), {content_length} chars total"
        )

        # Small page: single document
        if content_length <= self.size_threshold:
            return [Document(
                page_content=full_content,
                metadata={
                    **metadata,
                    "section": "full",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "content_length": content_length,
                    "doc_id": f"{page.id}_full_0",
                }
            )]
        
        # Large page: chunk it
        chunks = self.text_splitter.split_text(full_content)
        logger.debug(f"Created {len(chunks)} chunks for page {page.id}")

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "section": "full",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_length": len(chunk),
                    "doc_id": f"{page.id}_chunk_{i}",
                }
            ))
        
        return documents


# Adapter function for compatibility with index_builder
def page_to_documents_api_extractor(page, stdout=None) -> List[Document]:
    """
    Extract documents from a Wagtail page using API-style extraction.
    
    This is a simpler, more reliable alternative to manual field parsing.

    Args:
        page: Wagtail page to extract content from
        stdout: Optional output function for logging (unused, for compatibility)

    Returns:
        List of Document objects
    """
    extractor = WagtailAPIExtractor(
        chunk_size=getattr(settings, 'WAGTAIL_RAG_CHUNK_SIZE', DEFAULT_CHUNK_SIZE),
        chunk_overlap=getattr(settings, 'WAGTAIL_RAG_CHUNK_OVERLAP', DEFAULT_CHUNK_OVERLAP),
        size_threshold=getattr(settings, 'WAGTAIL_RAG_SIZE_THRESHOLD', DEFAULT_SIZE_THRESHOLD),
    )
    return extractor.extract_page(page)

