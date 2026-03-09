"""
Converts Wagtail pages into LangChain Documents for RAG indexing.

Field resolution per page:
  1. Explicit list in settings  → ["introduction", "body", "address"]
  2. "*" in settings            → Wagtail search_fields (text-only, curated)

Each field is chunked independently:
  - StreamField  → per-block chunking (preserves Wagtail structure)
  - Other fields → RecursiveCharacterTextSplitter
"""

import logging
import re
from typing import List, Optional

from django.conf import settings
from django.utils.html import strip_tags

from wagtail_rag.conf import conf

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 100
MIN_FIELD_LENGTH = 10

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
    ):
        self.chunk_size = chunk_size or getattr(
            settings, "WAGTAIL_RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE
        )
        self.chunk_overlap = chunk_overlap or getattr(
            settings, "WAGTAIL_RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP
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
    def _scan_search_fields(page) -> List[str]:
        """Return field names from Wagtail's search_fields (SearchField only).

        Only SearchField instances are included; FilterField, AutocompleteField,
        and RelatedFields are ignored.
        """
        try:
            from wagtail.search.index import SearchField as WagtailSearchField

            return [
                sf.field_name
                for sf in getattr(page, "search_fields", [])
                if isinstance(sf, WagtailSearchField)
                and sf.field_name not in _SKIP_FIELDS
                and getattr(page, sf.field_name, None)
            ]
        except Exception as exc:
            logger.debug(
                "Error scanning search_fields for %s: %s", page.__class__.__name__, exc
            )
            return []

    @staticmethod
    def _resolve_candidate_fields(page) -> tuple[List[str], str]:
        """Resolve fields to extract:

        - explicit list in settings → use those fields
        - "*" in settings           → use Wagtail search_fields
        """
        from_settings = conf.indexing.fields_for(page)
        if from_settings:
            return from_settings, f"settings: {', '.join(from_settings)}"

        search_fields = WagtailAPIExtractor._scan_search_fields(page)
        return search_fields, f"search_fields: {', '.join(search_fields)}"

    # -------------------------------------------------------------------------
    # Text cleaning
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_text(text) -> str:
        """Strip HTML and collapse whitespace."""
        if not text:
            return ""
        return " ".join(strip_tags(str(text)).split()).strip()

    @staticmethod
    def _clean_field_text(text) -> str:
        """Strip HTML while preserving paragraph breaks for the text splitter."""
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
    # StreamField block extraction
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
        """Extract text from a single StreamField block."""
        try:
            text = self._clean_text(block.render_as_block())
            if text:
                return text
        except Exception:
            pass

        value = block.value

        if hasattr(value, "source"):
            return self._clean_text(value.source) or None
        if isinstance(value, dict):
            return self._extract_from_dict(value) or None
        if isinstance(value, str):
            return self._clean_text(value) or None
        if isinstance(value, list):
            return self._extract_from_list(value) or None

        text = str(value)
        if text.endswith("None") and "." in text:
            return None
        return self._clean_text(text) or None

    def _chunk_streamfield(
        self, page, field_name: str, base_metadata: dict, title: str
    ) -> List[Document]:
        """Chunk a StreamField per-block so chunks never span unrelated blocks."""
        value = getattr(page, field_name, None)
        if not value:
            return []

        header = f"Page: {title}\nSection: {field_name}\n\n"
        documents: List[Document] = []
        chunk_index = 0

        for block in value:
            try:
                block_text = self._extract_from_block(block)
            except Exception as exc:
                logger.warning("Block extraction failed for '%s': %s", field_name, exc)
                continue

            if not block_text or len(block_text.strip()) <= MIN_FIELD_LENGTH:
                continue

            block_type = getattr(block, "block_type", "block")
            block_meta = {
                **base_metadata,
                "section": field_name,
                "block_type": block_type,
            }

            chunks = self.text_splitter.split_text(block_text)
            for i, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=f"{header}{chunk}",
                        metadata={
                            **block_meta,
                            "chunk_index": chunk_index,
                            "block_chunk_index": i,
                            "total_block_chunks": len(chunks),
                            "content_length": len(chunk),
                        },
                    )
                )
                chunk_index += 1

        return documents

    # -------------------------------------------------------------------------
    # Plain field extraction
    # -------------------------------------------------------------------------

    def _extract_field_value(self, page, field_name: str) -> Optional[str]:
        """Extract cleaned text from a non-StreamField page field."""
        try:
            value = getattr(page, field_name, None)
            if value is None:
                return None
            if hasattr(value, "source"):
                return self._clean_field_text(value.source)
            if isinstance(value, str):
                return self._clean_field_text(value)
            text = str(value)
            if not text or (text.endswith("None") and "." in text):
                return None
            return self._clean_text(text)
        except Exception as exc:
            logger.error("Error extracting field '%s': %s", field_name, exc)
            return None

    def _chunk_plain_field(
        self, text: str, field_name: str, base_metadata: dict, title: str
    ) -> List[Document]:
        """Chunk a plain text / RichTextField field."""
        header = f"Page: {title}\nSection: {field_name}\n\n"
        chunks = self.text_splitter.split_text(text)
        return [
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
            for i, chunk in enumerate(chunks)
        ]

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

    @staticmethod
    def _is_streamfield(page, field_name: str) -> bool:
        try:
            from wagtail.fields import StreamField

            field = page._meta.get_field(field_name)
            return isinstance(field, StreamField)
        except Exception:
            return False

    def extract_page(self, page) -> List[Document]:
        """Extract and chunk all content fields from a Wagtail page.

        Always includes a title document. StreamField fields are chunked
        per-block; other fields use RecursiveCharacterTextSplitter.
        """
        candidate_fields, field_source = self._resolve_candidate_fields(page)
        base_metadata = self._build_metadata(page)
        base_metadata["field_source"] = field_source

        title = str(page.title)
        documents: List[Document] = []
        extracted_fields: List[str] = []

        documents.append(
            Document(
                page_content=f"Page: {title}\nSection: title\n\n{title}",
                metadata={
                    **base_metadata,
                    "section": "title",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "content_length": len(title),
                },
            )
        )

        for field_name in candidate_fields:
            if self._is_streamfield(page, field_name):
                field_docs = self._chunk_streamfield(
                    page, field_name, base_metadata, title
                )
            else:
                text = self._extract_field_value(page, field_name)
                if not text or len(text.strip()) <= MIN_FIELD_LENGTH:
                    continue
                field_docs = self._chunk_plain_field(
                    text, field_name, base_metadata, title
                )

            if field_docs:
                extracted_fields.append(field_name)
                documents.extend(field_docs)

        if len(documents) == 1:
            logger.warning(
                "No content extracted from %s (ID: %s)",
                page.__class__.__name__,
                page.id,
            )

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
