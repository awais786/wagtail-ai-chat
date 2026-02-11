"""
Simple extractor using Wagtail's API serialization.

Uses Wagtail's built-in methods to extract content, avoiding manual field type handling.
"""
import logging
from typing import List, Dict, Optional
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


class WagtailAPIExtractor:
    """
    Simple extractor using Wagtail's API fields.
    
    Instead of manually parsing StreamFields, RichTextFields, etc.,
    this uses Wagtail's own serialization methods which handle all field types.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        size_threshold: int = 2000
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.size_threshold = size_threshold
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
    
    def _extract_field_value(self, page, field_name: str, verbose: bool = True) -> Optional[str]:
        """
        Extract text from a field using Wagtail's rendering.
        
        This works for all field types: StreamField, RichTextField, TextField, etc.
        """
        try:
            value = getattr(page, field_name, None)
            if value is None:
                if verbose:
                    print(f"    ⓘ Field '{field_name}' is None")
                return None
            
            # For StreamField: extract raw content from each block
            if hasattr(value, '__iter__') and hasattr(value, 'stream_block'):
                # It's a StreamField
                if verbose:
                    print(f"    → Processing StreamField '{field_name}' with {len(value)} blocks")
                parts = []
                for idx, block in enumerate(value):
                    block_type = block.block_type
                    block_value = block.value
                    
                    if verbose:
                        print(f"      🔹 Block {idx+1}: type={block_type}, value_type={type(block_value).__name__}")
                    
                    try:
                        # Try multiple extraction methods
                        extracted_text = None
                        
                        # Method 1: Try render_as_block() first
                        try:
                            rendered = block.render_as_block()
                            extracted_text = self._clean_text(rendered)
                            if verbose and extracted_text:
                                print(f"        ✓ Method 1 (render_as_block): {len(extracted_text)} chars")
                        except Exception as e:
                            if verbose:
                                print(f"        ⚠ Method 1 failed: {e}")
                        
                        # Method 2: If that didn't work or was empty, try direct value
                        if not extracted_text:
                            # For RichText blocks
                            if hasattr(block_value, 'source'):
                                extracted_text = self._clean_text(block_value.source)
                                if verbose and extracted_text:
                                    print(f"        ✓ Method 2 (RichText.source): {len(extracted_text)} chars")
                            # For dict/structured blocks
                            elif isinstance(block_value, dict):
                                # Extract all text values from dict
                                text_parts = []
                                for k, v in block_value.items():
                                    if isinstance(v, str) and v.strip():
                                        text_parts.append(self._clean_text(v))
                                    elif hasattr(v, 'source'):  # RichText in dict
                                        text_parts.append(self._clean_text(v.source))
                                if text_parts:
                                    extracted_text = " ".join(text_parts)
                                    if verbose and extracted_text:
                                        print(f"        ✓ Method 2 (dict extraction): {len(extracted_text)} chars")
                            # For simple values
                            elif isinstance(block_value, str):
                                extracted_text = self._clean_text(block_value)
                                if verbose and extracted_text:
                                    print(f"        ✓ Method 2 (string): {len(extracted_text)} chars")
                            # For list blocks
                            elif isinstance(block_value, list):
                                text_parts = []
                                for item in block_value:
                                    if isinstance(item, str):
                                        text_parts.append(self._clean_text(item))
                                    elif hasattr(item, 'source'):
                                        text_parts.append(self._clean_text(item.source))
                                if text_parts:
                                    extracted_text = " ".join(text_parts)
                                    if verbose and extracted_text:
                                        print(f"        ✓ Method 2 (list extraction): {len(extracted_text)} chars")
                        
                        # Method 3: Last resort - stringify
                        if not extracted_text:
                            extracted_text = self._clean_text(str(block_value))
                            if verbose and extracted_text:
                                print(f"        ✓ Method 3 (str fallback): {len(extracted_text)} chars")
                        
                        if extracted_text:
                            parts.append(extracted_text)
                        elif verbose:
                            print(f"        ⊘ No content extracted from this block")
                            
                    except Exception as e:
                        if verbose:
                            print(f"        ✗ All methods failed: {e}")
                
                if verbose:
                    print(f"      → Total blocks with content: {len(parts)}/{len(value)}")
                return "\n\n".join(parts) if parts else None
            
            # For RichTextField: has .source property
            if hasattr(value, 'source'):
                if verbose:
                    print(f"    → Processing RichTextField '{field_name}'")
                return self._clean_text(value.source)
            
            # For simple text fields
            if isinstance(value, str):
                if verbose:
                    print(f"    → Processing string field '{field_name}'")
                return self._clean_text(value)
            
            # For other types, convert to string
            if verbose:
                print(f"    → Processing field '{field_name}' as {type(value).__name__}")
            text = str(value)
            return self._clean_text(text) if text else None
            
        except Exception as e:
            print(f"    ✗ Error extracting field '{field_name}': {e}")
            import traceback
            print(f"      {traceback.format_exc()}")
            return None
    
    def extract_page(self, page, stdout=None) -> List[Document]:
        """
        Extract content from a Wagtail page using API-style field access.
        
        This is simpler than manual parsing because we let Wagtail handle
        the complexity of different field types.
        """
        # Get all API fields defined on the page
        api_fields = []
        if hasattr(page, 'api_fields'):
            api_fields = [f.name for f in page.api_fields if hasattr(f, 'name')]
        
        # Fallback: use common field names
        if not api_fields:
            api_fields = ['introduction', 'body', 'content', 'backstory']
        
        print(f"  📋 API fields to extract: {', '.join(api_fields)}")
        
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
        print(f"  ✓ Extracted 'title': {sections['title'][:80]}...")
        
        for field_name in api_fields:
            print(f"\n  🔍 Extracting field: '{field_name}'")
            text = self._extract_field_value(page, field_name, verbose=True)
            if text and len(text.strip()) > 10:  # Skip very short fields
                sections[field_name] = text
                preview = text[:150].replace('\n', ' ')
                print(f"  ✓ Extracted '{field_name}': {len(text)} chars")
                print(f"    Preview: {preview}...")
            elif text:
                print(f"  ⊘ Skipped '{field_name}': too short ({len(text)} chars)")
            else:
                print(f"  ⊘ Skipped '{field_name}': no content")
        
        # Combine all sections
        full_content = "\n\n".join(sections.values())
        content_length = len(full_content)
        
        print(f"  📊 Total content length: {content_length} chars")
        
        # Small page: single document
        if content_length <= self.size_threshold:
            print(f"  ✓ Creating single document (below {self.size_threshold} char threshold)")
            return [Document(
                page_content=full_content,
                metadata={
                    **metadata,
                    "section": "full",
                    "doc_id": f"{page.id}_full_0",
                }
            )]
        
        # Large page: chunk it
        print(f"  ✂️  Content exceeds threshold, chunking with size={self.chunk_size}, overlap={self.chunk_overlap}")
        chunks = self.text_splitter.split_text(full_content)
        print(f"  ✓ Created {len(chunks)} chunks")
        
        documents = []
        for i, chunk in enumerate(chunks):
            preview = chunk[:100].replace('\n', ' ')
            print(f"    Chunk {i+1}/{len(chunks)}: {len(chunk)} chars - {preview}...")
            documents.append(Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "section": "full",
                    "chunk_index": i,
                    "doc_id": f"{page.id}_chunk_{i}",
                }
            ))
        
        return documents


# Adapter function for compatibility with index_builder
def page_to_documents_api_extractor(page, stdout=None) -> List[Document]:
    """
    Extract documents from a Wagtail page using API-style extraction.
    
    This is a simpler, more reliable alternative to manual field parsing.
    """
    extractor = WagtailAPIExtractor(
        chunk_size=getattr(settings, 'WAGTAIL_RAG_CHUNK_SIZE', 1000),
        chunk_overlap=getattr(settings, 'WAGTAIL_RAG_CHUNK_OVERLAP', 200),
        size_threshold=getattr(settings, 'WAGTAIL_RAG_SIZE_THRESHOLD', 2000),
    )
    return extractor.extract_page(page, stdout=stdout)
