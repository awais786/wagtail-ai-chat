"""
Simple extractor using Wagtail's API serialization.

Uses Wagtail's built-in methods to extract content, avoiding manual field type handling.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

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
DEFAULT_FIELDS = ["introduction", "body", "content", "backstory"]
SYSTEM_FIELDS = {
    "id",
    "path",
    "depth",
    "title",
    "slug",
    "draft_title",
    "content_type",
    "live",
    "has_unpublished_changes",
    "owner",
    "locked",
    "locked_at",
    "locked_by",
    "latest_revision",
    "latest_revision_created_at",
    "live_revision",
    "first_published_at",
    "last_published_at",
    "go_live_at",
    "expire_at",
    "expired",
    "search_description",
    "seo_title",
    "show_in_menus",
    "url_path",
    "translation_key",
    "locale",
    "alias_of",
}
CORE_SKIP_FIELDS = {"id", "path", "depth", "title", "slug"}


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
        size_threshold: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or getattr(
            settings, "WAGTAIL_RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE
        )
        self.chunk_overlap = chunk_overlap or getattr(
            settings, "WAGTAIL_RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP
        )
        self.size_threshold = size_threshold or getattr(
            settings, "WAGTAIL_RAG_SIZE_THRESHOLD", DEFAULT_SIZE_THRESHOLD
        )
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
    def _get_available_content_fields(page) -> List[str]:
        """
        Get all available content fields on a page (for debugging).
        Shows what fields exist, even if they're empty.
        """
        from wagtail.fields import StreamField, RichTextField
        from django.db.models import TextField, ForeignKey, ManyToManyField

        available_fields = []

        try:
            for field in page._meta.get_fields():
                field_name = field.name

                # Skip system fields
                if field_name.startswith("_") or field_name in SYSTEM_FIELDS:
                    continue

                # Skip relationship fields
                if isinstance(field, (ForeignKey, ManyToManyField)):
                    continue

                # List all content fields (even empty ones)
                if isinstance(field, (StreamField, RichTextField, TextField)):
                    available_fields.append(field_name)
        except Exception as e:
            logger.debug(f"Error listing fields for {page.__class__.__name__}: {e}")

        return available_fields

    @staticmethod
    def _discover_content_fields(page) -> List[str]:
        """
        Auto-discover content fields on a page by inspecting field types.
        Looks for StreamFields, RichTextFields, and TextFields.
        Excludes ForeignKeys and ManyToMany relationships.
        """
        from wagtail.fields import StreamField, RichTextField
        from django.db.models import TextField, ForeignKey, ManyToManyField

        content_fields = []

        try:
            for field in page._meta.get_fields():
                field_name = field.name

                # Skip system fields
                if field_name.startswith("_") or field_name in CORE_SKIP_FIELDS:
                    continue

                # Skip relationship fields
                if isinstance(field, (ForeignKey, ManyToManyField)):
                    continue

                # Check if it's a content field
                if isinstance(field, (StreamField, RichTextField, TextField)):
                    # Verify the field has content
                    value = getattr(page, field_name, None)
                    if value:
                        content_fields.append(field_name)
        except Exception as e:
            logger.debug(f"Error discovering fields for {page.__class__.__name__}: {e}")

        return content_fields

    @staticmethod
    def _resolve_candidate_fields(page) -> Tuple[List[str], str]:
        """
        Determine candidate fields in priority order:
        1) model api_fields
        2) configured defaults
        3) auto-discovered content fields (if defaults missing on model)
        """
        api_fields: List[str] = []
        field_source = None

        if hasattr(page, "api_fields"):
            api_fields = [f.name for f in page.api_fields if hasattr(f, "name")]
            if api_fields:
                field_source = f"model api_fields: {', '.join(api_fields)}"
                logger.debug(
                    f"Using api_fields from {page.__class__.__name__}: {', '.join(api_fields)}"
                )

        if not api_fields:
            api_fields = getattr(settings, "WAGTAIL_RAG_DEFAULT_FIELDS", DEFAULT_FIELDS)
            field_source = "default fields"
            logger.debug(
                f"No api_fields, trying default fields: {', '.join(api_fields)}"
            )

        if not any(hasattr(page, field) for field in api_fields):
            logger.debug(
                f"No standard fields found on {page.__class__.__name__}, auto-discovering..."
            )
            discovered_fields = WagtailAPIExtractor._discover_content_fields(page)
            if discovered_fields:
                api_fields = discovered_fields
                field_source = f"auto-discovered: {', '.join(discovered_fields)}"
                logger.debug(f"Discovered fields: {', '.join(discovered_fields)}")

        return api_fields, field_source or "unknown"

    @staticmethod
    def _build_metadata(page) -> Dict[str, str]:
        """Build common metadata shared by full and chunked documents."""
        metadata = {
            "page_id": page.id,
            "page_type": page.__class__.__name__,
            "title": str(page.title),
            "slug": getattr(page, "slug", ""),
            "url": WagtailAPIExtractor._get_page_url(page),
        }
        if getattr(page, "last_published_at", None):
            metadata["last_published_at"] = page.last_published_at.isoformat()
        return metadata

    @staticmethod
    def _clean_text(text) -> str:
        """Clean inline text (single block value) by stripping HTML and normalizing whitespace.

        Intentionally collapses whitespace — appropriate for individual block values
        where paragraph structure within a single block is not meaningful.
        Use _clean_field_text() when processing a whole field that may contain paragraphs.
        """
        if not text:
            return ""
        text = strip_tags(str(text))
        text = " ".join(text.split())
        return text.strip()

    @staticmethod
    def _clean_field_text(text) -> str:
        """Clean a whole field's text while preserving paragraph structure.

        Unlike _clean_text, this keeps paragraph breaks so that
        RecursiveCharacterTextSplitter can split on them. Used when processing
        RichTextField.source or any pre-assembled multi-paragraph content.
        """
        if not text:
            return ""
        # Convert common HTML block elements to paragraph breaks before stripping tags,
        # so that <p>, <br>, <li> etc. produce readable line breaks rather than
        # running all sentences together.
        text = re.sub(r"<br\s*/?>", "\n", str(text), flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</h[1-6]>", "\n\n", text, flags=re.IGNORECASE)
        text = strip_tags(text)
        # Normalize whitespace within each paragraph, but preserve paragraph breaks.
        paragraphs = re.split(r"\n{2,}", text)
        cleaned = [" ".join(p.split()) for p in paragraphs if p.strip()]
        return "\n\n".join(cleaned)

    def _extract_from_dict(self, data: dict) -> str:
        """Extract text from dictionary structures."""
        text_parts = []
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(self._clean_text(value))
            elif hasattr(value, "source"):  # RichText in dict
                text_parts.append(self._clean_text(value.source))
        return " ".join(text_parts) if text_parts else ""

    def _extract_from_list(self, data: list) -> str:
        """Extract text from list structures."""
        text_parts = []
        for item in data:
            if isinstance(item, str) and item.strip():
                text_parts.append(self._clean_text(item))
            elif hasattr(item, "source"):
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
        if hasattr(block_value, "source"):
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
            if hasattr(value, "__iter__") and hasattr(value, "stream_block"):
                parts = []
                for block in value:
                    try:
                        extracted_text = self._extract_from_block(block)
                        if extracted_text:
                            parts.append(extracted_text)
                    except Exception as e:
                        logger.warning(
                            f"Block extraction failed for '{field_name}': {e}"
                        )

                return "\n\n".join(parts) if parts else None

            # For RichTextField: has .source property (HTML).
            # Use _clean_field_text to preserve paragraph breaks.
            if hasattr(value, "source"):
                return self._clean_field_text(value.source)

            # For simple text fields — preserve paragraph breaks.
            if isinstance(value, str):
                return self._clean_field_text(value)

            # For other types, convert to string
            text = str(value)
            return self._clean_text(text) if text else None

        except Exception as e:
            logger.error(f"Error extracting field '{field_name}': {e}")
            return None

    def _make_chunk_header(self, title: str, field_name: str) -> str:
        """Return the context header prepended to every chunk.

        Embedding the page title and section name directly in the chunk text
        (not just in metadata) means the LLM always has source context even
        when metadata is not passed through to the prompt.
        """
        return f"Page: {title}\nSection: {field_name}\n\n"

    def extract_page(self, page) -> List[Document]:
        """Extract content from a Wagtail page using API-style field access.

        Each content field is chunked independently so that chunks never
        span multiple fields, section metadata is accurate, and each chunk
        carries a title+section header for LLM context.
        """
        # Debug: Show all available content fields on this page
        available_fields = self._get_available_content_fields(page)
        if available_fields:
            logger.info(
                f"Available content fields on {page.__class__.__name__}: {', '.join(available_fields)}"
            )

        api_fields, field_source = self._resolve_candidate_fields(page)

        # Base metadata shared across all documents from this page
        base_metadata = self._build_metadata(page)
        base_metadata["field_source"] = field_source

        title = str(page.title)
        documents: List[Document] = []
        extracted_fields: List[str] = []

        for field_name in api_fields:
            text = self._extract_field_value(page, field_name)
            if not text or len(text.strip()) <= MIN_FIELD_LENGTH:
                continue

            extracted_fields.append(field_name)
            header = self._make_chunk_header(title, field_name)

            if len(text) + len(header) <= self.size_threshold:
                # Field fits in a single document — no splitting needed.
                documents.append(
                    Document(
                        page_content=f"{header}{text}",
                        metadata={
                            **base_metadata,
                            "section": field_name,
                            "chunk_index": 0,
                            "total_chunks": 1,
                            "content_length": len(text),
                        },
                    )
                )
            else:
                # Field is large — split into chunks, each with the same header.
                chunks = self.text_splitter.split_text(text)
                logger.debug(
                    f"Page '{title}' field '{field_name}': split into {len(chunks)} chunks"
                )
                for i, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            page_content=f"{header}{chunk}",
                            metadata={
                                **base_metadata,
                                "section": field_name,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "content_length": len(chunk),
                            },
                        )
                    )

        # Log extraction summary
        logger.debug(
            f"Page '{title}' (ID: {page.id}): extracted {len(extracted_fields)} field(s) "
            f"({', '.join(extracted_fields) or 'none'}), produced {len(documents)} document(s)"
        )

        if not documents:
            logger.warning(
                f"No content fields found for {page.__class__.__name__} (ID: {page.id}). "
                "Returning empty list to trigger fallback extractor."
            )
            return []

        # Stamp extracted_fields on every document now that we know the full list.
        extracted_fields_str = ", ".join(["title"] + extracted_fields)
        for doc in documents:
            doc.metadata["extracted_fields"] = extracted_fields_str

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
        chunk_size=getattr(settings, "WAGTAIL_RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=getattr(
            settings, "WAGTAIL_RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP
        ),
        size_threshold=getattr(
            settings, "WAGTAIL_RAG_SIZE_THRESHOLD", DEFAULT_SIZE_THRESHOLD
        ),
    )
    return extractor.extract_page(page)
