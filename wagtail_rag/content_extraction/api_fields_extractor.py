"""
Converts Wagtail pages into LangChain Documents for RAG indexing.

For each page, the extractor:
  1. Resolves candidate fields (api_fields → settings defaults → auto-scan)
  2. Extracts and cleans text from each field (StreamField, RichTextField, TextField)
  3. Chunks large fields with RecursiveCharacterTextSplitter
  4. Returns Documents with a 'Page / Section' header and rich metadata
"""

import logging
import re
from typing import List, Optional

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

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SIZE_THRESHOLD = 2000
MIN_FIELD_LENGTH = 10
DEFAULT_FIELDS = ["introduction", "body", "content", "backstory"]

# Fields on every Wagtail page that carry no user content.
_SKIP_FIELDS = {
    "id",
    "pk",
    "path",
    "depth",
    "numchild",
    "url_path",
    "title",
    "slug",
    "draft_title",
    "content_type",
    "content_type_id",
    "live",
    "has_unpublished_changes",
    "owner",
    "locked",
    "locked_at",
    "locked_by",
    "latest_revision",
    "latest_revision_id",
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
    "translation_key",
    "locale",
    "locale_id",
    "alias_of",
}


class WagtailAPIExtractor:
    """Extracts text content from a Wagtail page and returns LangChain Documents."""

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

    # -------------------------------------------------------------------------
    # Field discovery
    # -------------------------------------------------------------------------

    @staticmethod
    def _scan_content_fields(page, require_content: bool = False) -> List[str]:
        """Return content field names on a page, optionally skipping empty ones.

        Args:
            require_content: When True, only include fields that have a value.
        """
        from django.db.models import ForeignKey, ManyToManyField, TextField
        from wagtail.fields import RichTextField, StreamField

        fields = []
        try:
            for field in page._meta.get_fields():
                name = getattr(field, "name", None)
                if not name or name.startswith("_") or name in _SKIP_FIELDS:
                    continue
                if isinstance(field, (ForeignKey, ManyToManyField)):
                    continue
                if isinstance(field, (StreamField, RichTextField, TextField)):
                    if require_content and not getattr(page, name, None):
                        continue
                    fields.append(name)
        except Exception as exc:
            logger.debug(
                "Error scanning fields for %s: %s", page.__class__.__name__, exc
            )
        return fields

    @staticmethod
    def _resolve_candidate_fields(page) -> tuple[List[str], str]:
        """Resolve fields to extract, in priority order:

        1. page.api_fields (Wagtail declared API fields)
        2. WAGTAIL_RAG_DEFAULT_FIELDS setting (or built-in defaults)
        3. Auto-scan — when no default fields exist on the model
        """
        if hasattr(page, "api_fields"):
            names = [f.name for f in page.api_fields if hasattr(f, "name")]
            if names:
                logger.debug(
                    "Using api_fields from %s: %s",
                    page.__class__.__name__,
                    ", ".join(names),
                )
                return names, f"model api_fields: {', '.join(names)}"

        defaults = getattr(settings, "WAGTAIL_RAG_DEFAULT_FIELDS", DEFAULT_FIELDS)
        if any(hasattr(page, f) for f in defaults):
            logger.debug(
                "Using default fields for %s: %s",
                page.__class__.__name__,
                ", ".join(defaults),
            )
            return list(defaults), "default fields"

        discovered = WagtailAPIExtractor._scan_content_fields(
            page, require_content=True
        )
        if discovered:
            logger.debug(
                "Auto-discovered fields for %s: %s",
                page.__class__.__name__,
                ", ".join(discovered),
            )
            return discovered, f"auto-discovered: {', '.join(discovered)}"

        return list(defaults), "default fields (fallback)"

    # -------------------------------------------------------------------------
    # Text cleaning
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_text(text) -> str:
        """Strip HTML and collapse all whitespace to single spaces.

        Use for individual block values where paragraph structure is not meaningful.
        Use _clean_field_text() for multi-paragraph field content.
        """
        if not text:
            return ""
        return " ".join(strip_tags(str(text)).split()).strip()

    @staticmethod
    def _clean_field_text(text) -> str:
        """Strip HTML while preserving paragraph breaks for the text splitter.

        Converts block-level HTML elements (<p>, <br>, <li>, headings) to
        newlines before stripping tags so sentences don't run together.
        """
        if not text:
            return ""
        text = str(text)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</h[1-6]>", "\n\n", text, flags=re.IGNORECASE)
        text = strip_tags(text)
        paragraphs = re.split(r"\n{2,}", text)
        return "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())

    # -------------------------------------------------------------------------
    # StreamField extraction
    # -------------------------------------------------------------------------

    def _extract_from_dict(self, data: dict) -> str:
        parts = [
            self._clean_text(v.source if hasattr(v, "source") else v)
            for v in data.values()
            if isinstance(v, str) and v.strip() or hasattr(v, "source")
        ]
        return " ".join(parts)

    def _extract_from_list(self, data: list) -> str:
        parts = [
            self._clean_text(item.source if hasattr(item, "source") else item)
            for item in data
            if isinstance(item, str) and item.strip() or hasattr(item, "source")
        ]
        return " ".join(parts)

    def _extract_from_block(self, block) -> Optional[str]:
        """Extract text from a StreamField block, trying multiple strategies."""
        # 1. Rendered HTML (most complete)
        try:
            text = self._clean_text(block.render_as_block())
            if text:
                return text
        except Exception:
            pass

        value = block.value

        # 2. RichText block
        if hasattr(value, "source"):
            text = self._clean_text(value.source)
            if text:
                return text

        # 3. Structured / dict block
        if isinstance(value, dict):
            text = self._extract_from_dict(value)
            if text:
                return text

        # 4. Plain string
        if isinstance(value, str):
            text = self._clean_text(value)
            if text:
                return text

        # 5. List block
        if isinstance(value, list):
            text = self._extract_from_list(value)
            if text:
                return text

        # 6. Last resort — skip if it looks like a model repr (e.g. "App.Model.None")
        text = str(value)
        if text.endswith("None") and "." in text:
            return None
        return self._clean_text(text) or None

    # -------------------------------------------------------------------------
    # Field-level extraction
    # -------------------------------------------------------------------------

    def _extract_field_value(self, page, field_name: str) -> Optional[str]:
        """Extract cleaned text from a single page field."""
        try:
            value = getattr(page, field_name, None)
            if value is None:
                return None

            # StreamField
            if hasattr(value, "__iter__") and hasattr(value, "stream_block"):
                parts = []
                for block in value:
                    try:
                        text = self._extract_from_block(block)
                        if text:
                            parts.append(text)
                    except Exception as exc:
                        logger.warning(
                            "Block extraction failed for '%s': %s", field_name, exc
                        )
                return "\n\n".join(parts) or None

            # RichTextField
            if hasattr(value, "source"):
                return self._clean_field_text(value.source)

            # Plain text
            if isinstance(value, str):
                return self._clean_field_text(value)

            # Other types — skip model reprs like "app.Model.None"
            text = str(value)
            if not text or (text.endswith("None") and "." in text):
                return None
            return self._clean_text(text)

        except Exception as exc:
            logger.error("Error extracting field '%s': %s", field_name, exc)
            return None

    # -------------------------------------------------------------------------
    # Page extraction
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_page_url(page) -> str:
        for attr in ("full_url", "url"):
            try:
                return getattr(page, attr)
            except (AttributeError, RuntimeError):
                continue
        return f"/page/{page.id}/"

    @staticmethod
    def _build_metadata(page) -> dict:
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

    def extract_page(self, page) -> List[Document]:
        """Extract and chunk all content fields from a Wagtail page.

        Each field is chunked independently — chunks never span fields,
        ensuring section metadata is accurate and the LLM knows the source.
        """
        if logger.isEnabledFor(logging.DEBUG):
            available = self._scan_content_fields(page)
            if available:
                logger.debug(
                    "Content fields on %s: %s",
                    page.__class__.__name__,
                    ", ".join(available),
                )

        candidate_fields, field_source = self._resolve_candidate_fields(page)
        base_metadata = self._build_metadata(page)
        base_metadata["field_source"] = field_source

        title = str(page.title)
        documents: List[Document] = []
        extracted_fields: List[str] = []

        for field_name in candidate_fields:
            text = self._extract_field_value(page, field_name)
            if not text or len(text.strip()) <= MIN_FIELD_LENGTH:
                continue

            extracted_fields.append(field_name)
            header = f"Page: {title}\nSection: {field_name}\n\n"

            if len(text) + len(header) <= self.size_threshold:
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
                chunks = self.text_splitter.split_text(text)
                logger.debug(
                    "Page '%s' field '%s': %d chunk(s)", title, field_name, len(chunks)
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

        if not documents:
            logger.warning(
                "No content extracted from %s (ID: %s)",
                page.__class__.__name__,
                page.id,
            )
            return []

        extracted_fields_str = ", ".join(["title"] + extracted_fields)
        for doc in documents:
            doc.metadata["extracted_fields"] = extracted_fields_str

        logger.debug(
            "Page '%s' (ID: %s): %d field(s), %d document(s)",
            title,
            page.id,
            len(extracted_fields),
            len(documents),
        )
        return documents


def page_to_documents_api_extractor(page) -> List[Document]:
    """Extract LangChain Documents from a Wagtail page."""
    return WagtailAPIExtractor().extract_page(page)
