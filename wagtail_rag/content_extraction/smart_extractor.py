"""
Generic SmartWagtailExtractor with adaptive chunking for any Wagtail page model.

- Automatically detects and extracts fields from any page model
- Small pages (content <= size_threshold): one document per page
- Large pages: chunk by section (title, intro, body, metadata) or as full text
- Works with WAGTAIL_RAG_MODELS settings to index specified models
"""
import json
import logging
from typing import List, Dict, Optional

from django.conf import settings
from django.db.models import ForeignKey, ManyToManyField

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

from wagtail.fields import StreamField as WagtailStreamField

from .page_to_documents import get_page_url

logger = logging.getLogger(__name__)


class SmartWagtailExtractor:
    """
    Intelligent extraction with adaptive chunking for any Wagtail page model.
    Automatically detects available fields and extracts content accordingly.
    """
    
    # Field name patterns for different content types
    INTRO_FIELD_PATTERNS = [
        "introduction", "intro", "description", "summary", 
        "excerpt", "lead", "standfirst"
    ]
    
    BODY_FIELD_PATTERNS = [
        "body", "content", "main_content", "text", 
        "streamfield", "page_body"
    ]
    
    # System fields to always skip
    SKIP_FIELDS = {
        'id', 'path', 'depth', 'numchild', 'title', 'draft_title', 'slug',
        'content_type', 'live', 'has_unpublished_changes', 'owner',
        'locked', 'locked_at', 'locked_by', 'latest_revision', 
        'latest_revision_created_at', 'live_revision', 'first_published_at', 
        'last_published_at', 'go_live_at', 'expire_at', 'expired', 
        'search_description', 'seo_title', 'show_in_menus', 'url_path', 
        'translation_key', 'locale', 'alias_of', 'page_ptr', 
        'wagtail_admin_comments', 'revisions', 'workflow_states',
        'group_permissions', 'view_restrictions',
    }
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        size_threshold: int = 2000,
        chunk_by_section: bool = True,
        max_metadata_items: int = 10,
        max_text_field_length: int = 500,
        intro_patterns: Optional[List[str]] = None,
        body_patterns: Optional[List[str]] = None,
        skip_fields: Optional[set] = None
    ):
        """
        Generic Wagtail page extractor that works with any page model.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks for context preservation
            size_threshold: If page content exceeds this, use chunking
            chunk_by_section: Whether to respect section boundaries when chunking
            max_metadata_items: Max items to extract from ManyToMany fields
            max_text_field_length: Max length for text fields in metadata (increased to 500)
            intro_patterns: Custom field name patterns for introduction fields
            body_patterns: Custom field name patterns for body/main content fields
            skip_fields: Additional system fields to skip during extraction
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.size_threshold = size_threshold
        self.chunk_by_section = chunk_by_section
        self.max_metadata_items = max_metadata_items
        self.max_text_field_length = max_text_field_length
        
        # Allow custom patterns (for site-specific field naming)
        if intro_patterns:
            self.INTRO_FIELD_PATTERNS = intro_patterns
        if body_patterns:
            self.BODY_FIELD_PATTERNS = body_patterns
        if skip_fields:
            self.SKIP_FIELDS = self.SKIP_FIELDS.union(skip_fields)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    @staticmethod
    def extract_streamfield_text(body) -> str:
        """
        Enhanced StreamField extraction with better formatting.
        Works with any StreamField structure.
        """
        if not body:
            return ""
        
        text_parts = []
        
        try:
            for block in body:
                block_type = block.block_type
                block_value = block.value
                
                # Handle different block types
                if block_type in ('heading', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                    text_parts.append(f"\n## {block_value}\n")
                
                elif block_type in ('paragraph', 'text', 'rich_text'):
                    # Handle RichText objects
                    if hasattr(block_value, 'source'):
                        text_parts.append(block_value.source)
                    elif hasattr(block_value, '__html__'):
                        text_parts.append(str(block_value.__html__()))
                    else:
                        text_parts.append(str(block_value))
                
                elif block_type in ('quote', 'blockquote'):
                    text_parts.append(f"\n> {block_value}\n")
                
                elif block_type in ('list', 'ul', 'ol'):
                    if isinstance(block_value, (list, tuple)):
                        text_parts.append("\n".join(f"- {item}" for item in block_value))
                    else:
                        text_parts.append(str(block_value))
                
                elif block_type in ('image', 'img'):
                    # Optionally include image captions/alt text
                    if hasattr(block_value, 'caption'):
                        text_parts.append(f"[Image: {block_value.caption}]")
                    else:
                        text_parts.append("[Image]")
                
                elif block_type == 'embed':
                    if hasattr(block_value, 'url'):
                        text_parts.append(f"[Embed: {block_value.url}]")
                    else:
                        text_parts.append("[Embed]")
                
                elif block_type == 'code':
                    text_parts.append(f"```\n{block_value}\n```")
                
                elif block_type == 'table':
                    # Extract table content as text
                    if hasattr(block_value, 'data'):
                        table_text = SmartWagtailExtractor._extract_table_text(block_value.data)
                        if table_text:
                            text_parts.append(table_text)
                    else:
                        text_parts.append(str(block_value))
                
                else:
                    # Generic handling for unknown block types
                    # Try to extract meaningful text
                    if hasattr(block_value, 'source'):
                        text_parts.append(block_value.source)
                    elif isinstance(block_value, dict):
                        # Try to extract text from common dict keys
                        text = SmartWagtailExtractor._extract_dict_text(block_value)
                        if text:
                            text_parts.append(text)
                    else:
                        text_parts.append(str(block_value))
        
        except Exception as e:
            logger.warning(f"Error extracting StreamField: {e}")
            # Fallback to string representation
            try:
                text_parts.append(str(body))
            except:
                pass
        
        return "\n\n".join(filter(None, text_parts))
    
    @staticmethod
    def _extract_table_text(table_data) -> str:
        """Extract text from table data structure"""
        try:
            if isinstance(table_data, (list, tuple)):
                rows = []
                for row in table_data:
                    if isinstance(row, (list, tuple)):
                        rows.append(" | ".join(str(cell) for cell in row))
                return "\n".join(rows)
        except:
            pass
        return ""
    
    @staticmethod
    def _extract_dict_text(data: dict) -> str:
        """Extract text from dictionary (common in custom blocks)"""
        text_keys = ['text', 'content', 'value', 'title', 'heading', 'caption', 'description']
        parts = []
        
        for key in text_keys:
            if key in data and data[key]:
                parts.append(str(data[key]))
        
        return " ".join(parts)
    
    def _find_field_by_patterns(self, page, patterns: List[str]) -> tuple:
        """
        Find first matching field from patterns.
        Returns (field_name, field_value) or (None, None)
        """
        for pattern in patterns:
            value = getattr(page, pattern, None)
            if value:
                return pattern, value
        return None, None

    @staticmethod
    def _to_plain_text(value) -> str:
        """Get plain text from RichText, string, or other value."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if hasattr(value, "source"):
            return (value.source or "").strip()
        return str(value).strip()

    def _extract_body_text(self, body_value) -> str:
        """
        Extract text from body field: StreamField (iterable blocks) or RichTextField (.source).
        """
        if not body_value:
            return ""
        # Try StreamField first (iterable of blocks)
        try:
            if hasattr(body_value, "__iter__") and not isinstance(body_value, (str, dict)):
                text = self.extract_streamfield_text(body_value)
                if text and text.strip():
                    return text
        except Exception:
            pass
        # Fallback: RichTextField (.source) or single value
        return self._to_plain_text(body_value)
    
    def extract_sections(self, page) -> Dict[str, str]:
        """
        Extract page content organized by sections.
        Works with any Wagtail page model by detecting available fields.
        
        Generic approach:
        - Detects intro, body, and metadata fields dynamically
        - Creates natural narrative structure without model-specific assumptions
        - Works consistently across all Wagtail sites
        """
        from collections import OrderedDict
        sections = OrderedDict()
        
        title = getattr(page, "title", None) or "Untitled"
        page_type = page.__class__.__name__
        
        # 1. Find introduction field (generic patterns)
        intro_field, intro_value = self._find_field_by_patterns(
            page, self.INTRO_FIELD_PATTERNS
        )
        intro_text = None
        if intro_value:
            intro_text = self._to_plain_text(intro_value)
        
        # 2. Find main body field (generic patterns)
        body_field, body_value = self._find_field_by_patterns(
            page, self.BODY_FIELD_PATTERNS
        )
        
        # 3. Extract all other metadata fields
        metadata_parts = self._extract_metadata_fields(page, intro_field, body_field)
        
        # 4. Build OVERVIEW section - natural narrative style
        narrative_parts = []
        
        # Opening: Simple subject statement (works for any page type)
        narrative_parts.append(f"{title}.")
        
        # Add intro text immediately after title (if present)
        if intro_text:
            narrative_parts.append(intro_text)
        
        # Add metadata as natural sentences
        # Convert "Label: Value" → "The label is value."
        for metadata_line in metadata_parts:
            if ": " in metadata_line:
                label, value = metadata_line.split(": ", 1)
                # Clean label for readability
                clean_label = label.lower().replace("_", " ")
                narrative_parts.append(f"The {clean_label} is {value}.")
            else:
                # Already a sentence or doesn't fit pattern
                narrative_parts.append(metadata_line)
        
        # Combine into cohesive overview
        sections["overview"] = " ".join(narrative_parts)
        
        # 5. Main body content (detailed content in separate section)
        if body_value:
            body_text = self._extract_body_text(body_value)
            if body_text:
                sections["body"] = body_text
        
        return sections
    
    def _extract_metadata_fields(
        self, 
        page, 
        intro_field: Optional[str], 
        body_field: Optional[str]
    ) -> List[str]:
        """Extract all other fields as metadata"""
        metadata_parts = []
        
        # Build skip list
        skip_fields = self.SKIP_FIELDS.copy()
        skip_fields.update(self.INTRO_FIELD_PATTERNS)
        skip_fields.update(self.BODY_FIELD_PATTERNS)
        
        try:
            # Iterate through model fields
            for field in page._meta.get_fields():
                field_name = field.name
                
                # Skip unwanted fields
                if (field_name in skip_fields or 
                    field_name.startswith('_') or
                    field_name.endswith('_ptr')):
                    continue
                
                field_value = getattr(page, field_name, None)
                
                if field_value is None:
                    continue
                
                # Handle StreamFields that aren't the main body/intro
                # (e.g., RecipePage has both 'body' and 'backstory' StreamFields)
                if isinstance(field, WagtailStreamField):
                    # Only extract if NOT already processed as intro or body
                    if field_name != intro_field and field_name != body_field:
                        streamfield_text = self.extract_streamfield_text(field_value)
                        if streamfield_text:
                            field_label = field.verbose_name or field_name.replace("_", " ").title()
                            # Truncate if too long (StreamFields can be large)
                            if len(streamfield_text) > self.max_text_field_length:
                                streamfield_text = streamfield_text[: self.max_text_field_length] + "..."
                            metadata_parts.append(f"{field_label}: {streamfield_text}")
                    continue
                
                # Handle ForeignKey relationships
                if isinstance(field, ForeignKey):
                    # Skip image/file fields (will be handled in base metadata)
                    if any(x in field_name.lower() for x in ['image', 'file', 'document']):
                        continue
                    
                    # Get display name from related object
                    name = (getattr(field_value, 'name', None) or 
                           getattr(field_value, 'title', None) or 
                           str(field_value))
                    
                    field_label = field.verbose_name or field_name.replace('_', ' ').title()
                    metadata_parts.append(f"{field_label}: {name}")
                
                # Handle ManyToMany relationships
                elif isinstance(field, ManyToManyField):
                    try:
                        if hasattr(field_value, 'exists') and field_value.exists():
                            items = []
                            for item in field_value.all()[:self.max_metadata_items]:
                                item_name = (getattr(item, 'name', None) or 
                                           getattr(item, 'title', None) or 
                                           str(item))
                                items.append(item_name)
                            
                            if items:
                                field_label = field.verbose_name or field_name.replace('_', ' ').title()
                                metadata_parts.append(f"{field_label}: {', '.join(items)}")
                    except Exception as e:
                        logger.debug(f"Error extracting M2M field {field_name}: {e}")
                
                # Handle RichText / other text-like (e.g. RichTextField not in intro/body patterns)
                if hasattr(field_value, "source") and field_value.source:
                    text = (field_value.source or "").strip()
                    if text:
                        if len(text) > self.max_text_field_length:
                            text = text[: self.max_text_field_length] + "..."
                        field_label = field.verbose_name or field_name.replace("_", " ").title()
                        metadata_parts.append(f"{field_label}: {text}")
                    continue

                # Handle simple fields (text, numbers, booleans, dates)
                elif isinstance(field_value, (str, int, float, bool)):
                    if isinstance(field_value, str):
                        if not field_value.strip():
                            continue
                        # Include long text (e.g. address) but truncate so we don't drop it
                        if len(field_value) > self.max_text_field_length:
                            field_value = field_value[: self.max_text_field_length] + "..."
                    field_label = field.verbose_name or field_name.replace("_", " ").title()
                    metadata_parts.append(f"{field_label}: {field_value}")
        
        except Exception as e:
            logger.debug(f"Error extracting metadata from page {page.id}: {e}")
        
        return metadata_parts
    
    def create_base_metadata(self, page) -> Dict:
        """Create base metadata common to all documents"""
        metadata = {
            "page_id": page.id,
            "page_type": page.__class__.__name__,
            "title": getattr(page, "title", "") or "",
            "slug": getattr(page, "slug", ""),
            "url": self._get_page_url(page),
        }
        
        # Add timestamps
        first_pub = getattr(page, "first_published_at", None)
        if first_pub:
            metadata["first_published_at"] = first_pub.isoformat()
        
        last_pub = getattr(page, "last_published_at", None)
        if last_pub:
            metadata["last_published_at"] = last_pub.isoformat()
        
        # Add image if exists (common pattern)
        for img_field in ['image', 'header_image', 'featured_image', 'main_image']:
            image = getattr(page, img_field, None)
            if image and hasattr(image, 'file') and image.file:
                metadata["has_image"] = True
                metadata["image_url"] = image.file.url
                metadata["image_field"] = img_field
                break
        
        return metadata
    
    @staticmethod
    def _get_page_url(page) -> str:
        """Safely get page URL (delegates to shared get_page_url for consistency)."""
        return get_page_url(page)
    
    def extract_page_adaptive(self, page) -> List[Document]:
        """
        Adaptively extract page - uses chunking only when needed
        
        Returns:
            List of Document objects
        """
        base_metadata = self.create_base_metadata(page)
        sections = self.extract_sections(page)
        
        # Combine all sections
        full_content = "\n\n".join(sections.values())
        content_length = len(full_content)
        
        # Decision: chunk or not?
        if content_length <= self.size_threshold:
            # Small page: single document
            return [Document(
                page_content=full_content,
                metadata={
                    **base_metadata,
                    "section": "full",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "doc_id": f"{page.id}_full_0",
                    "content_length": content_length
                }
            )]
        
        else:
            # Large page: chunk it
            if self.chunk_by_section:
                return self._chunk_by_sections(page, sections, base_metadata)
            else:
                return self._chunk_full_content(page, full_content, base_metadata)
    
    def _chunk_by_sections(
        self, 
        page, 
        sections: Dict[str, str], 
        base_metadata: Dict
    ) -> List[Document]:
        """Chunk large pages by respecting section boundaries"""
        documents = []
        chunk_counter = 0
        
        for section_name, section_content in sections.items():
            if len(section_content) <= self.chunk_size:
                # Section fits in one chunk
                documents.append(Document(
                    page_content=section_content,
                    metadata={
                        **base_metadata,
                        "section": section_name,
                        "chunk_index": chunk_counter,
                        "doc_id": f"{page.id}_{section_name}_{chunk_counter}",
                        "content_length": len(section_content)
                    }
                ))
                chunk_counter += 1
            else:
                # Section needs splitting
                chunks = self.text_splitter.split_text(section_content)
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            **base_metadata,
                            "section": section_name,
                            "chunk_index": chunk_counter,
                            "section_chunk": i,
                            "doc_id": f"{page.id}_{section_name}_{chunk_counter}",
                            "content_length": len(chunk)
                        }
                    ))
                    chunk_counter += 1
        
        # Add total chunks to all documents
        for doc in documents:
            doc.metadata["total_chunks"] = len(documents)
        
        return documents
    
    def _chunk_full_content(
        self, 
        page, 
        full_content: str, 
        base_metadata: Dict
    ) -> List[Document]:
        """Chunk full content without respecting sections"""
        chunks = self.text_splitter.split_text(full_content)
        documents = []
        
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    **base_metadata,
                    "section": "full",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "doc_id": f"{page.id}_chunk_{i}",
                    "content_length": len(chunk)
                }
            ))
        
        return documents
    
    def batch_extract_pages(self, page_queryset, optimize_queries: bool = True) -> List[Document]:
        """
        Extract multiple pages with adaptive chunking.
        
        Args:
            page_queryset: QuerySet of Wagtail pages
            optimize_queries: Whether to attempt query optimization (select_related, prefetch_related)
        
        Returns:
            List of Document objects
        """
        all_documents = []
        
        # Attempt to optimize queries if requested
        if optimize_queries:
            try:
                # Generic optimization - prefetch common relationships
                queryset = page_queryset.select_related('owner').prefetch_related('tags')
            except Exception:
                queryset = page_queryset
        else:
            queryset = page_queryset
        
        for page in queryset:
            try:
                # Use .specific to get the actual page type with all fields
                specific_page = page.specific if hasattr(page, 'specific') else page
                
                docs = self.extract_page_adaptive(specific_page)
                all_documents.extend(docs)
                
            except Exception as e:
                logger.warning(
                    "Error extracting page %s (%s): %s", 
                    getattr(page, "id", "?"),
                    page.__class__.__name__,
                    e
                )
                continue
        
        return all_documents


def get_extractor_from_settings() -> SmartWagtailExtractor:
    """
    Create extractor instance from Django settings.
    
    All settings are optional and have sensible defaults for any Wagtail site.
    
    Settings:
        WAGTAIL_RAG_CHUNK_SIZE: Maximum characters per chunk (default: 1000)
        WAGTAIL_RAG_CHUNK_OVERLAP: Overlap between chunks (default: 200)
        WAGTAIL_RAG_NEW_EXTRACTOR_SIZE_THRESHOLD: Threshold for chunking (default: 2000)
        WAGTAIL_RAG_NEW_EXTRACTOR_CHUNK_BY_SECTION: Whether to chunk by section (default: True)
        WAGTAIL_RAG_MAX_METADATA_ITEMS: Max items from M2M fields (default: 10)
        WAGTAIL_RAG_MAX_TEXT_FIELD_LENGTH: Max length for text fields in metadata (default: 500)
        WAGTAIL_RAG_INTRO_PATTERNS: Custom intro field patterns (default: standard patterns)
        WAGTAIL_RAG_BODY_PATTERNS: Custom body field patterns (default: standard patterns)
        WAGTAIL_RAG_SKIP_FIELDS: Additional fields to skip (default: standard system fields)
    """
    return SmartWagtailExtractor(
        chunk_size=getattr(settings, "WAGTAIL_RAG_CHUNK_SIZE", 1000),
        chunk_overlap=getattr(settings, "WAGTAIL_RAG_CHUNK_OVERLAP", 200),
        size_threshold=getattr(settings, "WAGTAIL_RAG_NEW_EXTRACTOR_SIZE_THRESHOLD", 2000),
        chunk_by_section=getattr(settings, "WAGTAIL_RAG_NEW_EXTRACTOR_CHUNK_BY_SECTION", True),
        max_metadata_items=getattr(settings, "WAGTAIL_RAG_MAX_METADATA_ITEMS", 10),
        max_text_field_length=getattr(settings, "WAGTAIL_RAG_MAX_TEXT_FIELD_LENGTH", 500),
        intro_patterns=getattr(settings, "WAGTAIL_RAG_INTRO_PATTERNS", None),
        body_patterns=getattr(settings, "WAGTAIL_RAG_BODY_PATTERNS", None),
        skip_fields=getattr(settings, "WAGTAIL_RAG_SKIP_FIELDS", None),
    )


def page_to_documents(page) -> List[Document]:
    """
    Main entry point for converting a Wagtail page to documents.
    Uses settings-based configuration.

    Args:
        page: Wagtail Page instance (will be converted to .specific)

    Returns:
        List of Document objects
    """
    extractor = get_extractor_from_settings()

    # Ensure we're working with the specific page type
    specific_page = page.specific if hasattr(page, "specific") else page

    return extractor.extract_page_adaptive(specific_page)


def page_to_documents_new_extractor(page, *, stdout=None):
    """
    Adapter for index builder and compare_extractors: use SmartWagtailExtractor for any page.

    When WAGTAIL_RAG_USE_NEW_EXTRACTOR is True, the index builder calls this. Returns a list
    of documents from the new extractor, or None on error so the chunked extractor can be used.

    Args:
        page: Wagtail Page (will use page.specific)
        stdout: Optional output function for logging

    Returns:
        List[Document] from SmartWagtailExtractor, or None to fall back to chunked extractor.
    """
    try:
        specific_page = page.specific if hasattr(page, "specific") else page
        extractor = get_extractor_from_settings()
        docs = extractor.extract_page_adaptive(specific_page)
        if stdout and docs:
            stdout(
                f"  [New extractor] {len(docs)} document(s) for {specific_page.__class__.__name__} ID {specific_page.id}"
            )
        return docs
    except Exception as e:
        logger.debug(
            "New extractor failed for page %s, fallback to chunked: %s",
            getattr(page, "id", "?"),
            e,
        )
        return None


def batch_index_models(models_list: Optional[List[str]] = None) -> List[Document]:
    """
    Batch index all pages from specified models.
    
    Args:
        models_list: List of model strings like ["breads.BreadPage", "locations.LocationPage"]
                    If None, uses WAGTAIL_RAG_MODELS from settings
    
    Returns:
        List of all extracted documents
    """
    from django.apps import apps
    from wagtail.models import Page
    
    if models_list is None:
        models_list = getattr(settings, 'WAGTAIL_RAG_MODELS', [])
    
    extractor = get_extractor_from_settings()
    all_documents = []
    
    for model_str in models_list:
        try:
            # Parse model string (e.g., "breads.BreadPage" or "locations.LocationPage:*")
            model_path = model_str.split(':')[0]  # Remove any suffix like ":*"
            app_label, model_name = model_path.split('.')
            
            # Get the model class
            model_class = apps.get_model(app_label, model_name)
            
            # Get live, public pages of this type
            pages = model_class.objects.live().public()
            
            logger.info(f"Extracting {pages.count()} pages from {model_str}")
            
            # Extract documents
            documents = extractor.batch_extract_pages(pages)
            all_documents.extend(documents)
            
            logger.info(f"Created {len(documents)} documents from {model_str}")
            
        except Exception as e:
            logger.error(f"Error processing model {model_str}: {e}")
            continue
    
    logger.info(f"Total: {len(all_documents)} documents from {len(models_list)} models")
    return all_documents

