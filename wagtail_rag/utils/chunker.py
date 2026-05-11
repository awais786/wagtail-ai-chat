# wagtail_rag/utils/chunker.py
import re
import unicodedata
from typing import Iterable, List, Tuple, Optional, Dict, Any


# Tokenizer loaders (try transformers, then tiktoken fallback)
def _load_transformers_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return None


def _load_tiktoken_encoder(model_hint: Optional[str] = None):
    try:
        import tiktoken

        # prefer cl100k_base for OpenAI-like models; fallback to encoding_for_model if hint provided
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            if model_hint:
                return tiktoken.encoding_for_model(model_hint)
            return None
    except Exception:
        return None


def get_tokenizer_for_embedding(model_name: Optional[str]) -> Any:
    """
    Return a tokenizer-like object with .encode(text, add_special_tokens=False)
    or a small fallback that approximates tokens by words.
    """
    if not model_name:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = _load_transformers_tokenizer(model_name)
    if tok:
        return tok
    enc = _load_tiktoken_encoder(model_name)
    if enc:

        class TiktokenAdapter:
            def encode(self, text, add_special_tokens=False):
                return enc.encode(text or "")

        return TiktokenAdapter()

    # fallback: simple whitespace-based pseudo-tokenizer
    class SimpleTokenizer:
        def encode(self, text, add_special_tokens=False):
            if not text:
                return []
            words = re.findall(r"\S+", text)
            # scale factor to approximate subword tokens
            return ["w"] * max(1, int(len(words) * 1.3))

    return SimpleTokenizer()


# Basic sentence splitter (replaceable with spacy/nltk)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def simple_sentence_split(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def tokens_of(text: str, tokenizer) -> int:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)
    except Exception:
        # fallback: approximate
        words = re.findall(r"\S+", text or "")
        return max(1, int(len(words) * 1.3)) if words else 0


def paragraph_token_chunker(
    text: str,
    tokenizer,
    chunk_size: int = 800,
    overlap: int = 128,
) -> Iterable[str]:
    """
    Paragraph-first, token-aware chunker.
    Yields chunk strings whose token length <= chunk_size (except possibly last).
    """
    if not text or not text.strip():
        return
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    for p in paragraphs:
        if tokens_of(p, tokenizer) <= chunk_size:
            yield p
            continue
        # split long paragraph by sentences into token windows
        sentences = simple_sentence_split(p)
        window: List[str] = []
        window_tokens = 0
        i = 0
        while i < len(sentences):
            s = sentences[i]
            s_tokens = tokens_of(s, tokenizer)
            if window_tokens + s_tokens <= chunk_size:
                window.append(s)
                window_tokens += s_tokens
                i += 1
            else:
                if window:
                    yield " ".join(window)
                # build overlap window (keep last sentences until overlap satisfied)
                overlap_tokens = 0
                new_window: List[str] = []
                for sent in reversed(window):
                    tlen = tokens_of(sent, tokenizer)
                    if overlap_tokens + tlen > overlap:
                        break
                    new_window.insert(0, sent)
                    overlap_tokens += tlen
                window = new_window
                window_tokens = sum(tokens_of(x, tokenizer) for x in window)
        if window:
            yield " ".join(window)


# -------------------------
# Page-level generic exporter
# -------------------------
def _ascii_fold(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _clean_html_preserve_paragraphs(value: Any) -> str:
    """
    Minimal HTML cleaning that preserves paragraph breaks for chunking.
    Uses Django's strip_tags if available, otherwise naive removal.
    """
    try:
        from django.utils.html import strip_tags

        text = str(value)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</h[1-6]>", "\n\n", text, flags=re.IGNORECASE)
        paragraphs = re.split(r"\n{2,}", strip_tags(text))
        return "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())
    except Exception:
        # fallback: plain string
        return " ".join(str(value).split())


def _is_streamfield(page, field_name: str) -> bool:
    try:
        from wagtail.fields import StreamField

        field = page._meta.get_field(field_name)
        return isinstance(field, StreamField)
    except Exception:
        return False


def _extract_streamfield_text(page, field_name: str) -> str:
    value = getattr(page, field_name, None)
    if not value:
        return ""
    parts = []
    for block in value:
        try:
            # prefer block.render_as_block() then block.value
            if hasattr(block, "render_as_block"):
                txt = block.render_as_block()
                if txt:
                    parts.append(_clean_html_preserve_paragraphs(txt))
                    continue
        except Exception:
            pass
        val = getattr(block, "value", block)
        if isinstance(val, str):
            parts.append(_clean_html_preserve_paragraphs(val))
        elif isinstance(val, dict):
            # collect string fields
            parts.extend(
                _clean_html_preserve_paragraphs(v)
                for v in val.values()
                if isinstance(v, str)
            )
        elif isinstance(val, list):
            parts.extend(
                _clean_html_preserve_paragraphs(i) for i in val if isinstance(i, str)
            )
        else:
            parts.append(_clean_html_preserve_paragraphs(str(val)))
    return " ".join(p for p in parts if p)


def _discover_text_fields(page) -> List[str]:
    """
    Auto-discover candidate text fields on a Wagtail page.
    Returns list of field names and any *_search_text properties are prioritized.
    """
    skip_fields = {
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
    fields = []
    # explicit *_search_text properties
    for attr in dir(page.__class__):
        if attr.endswith("_search_text") and not attr.startswith("_"):
            try:
                val = getattr(page, attr)
                if callable(val):
                    val = val()
            except Exception:
                val = getattr(page, attr, None)
            if isinstance(val, str) and val.strip():
                fields.append(attr)
    # model concrete fields
    for field in getattr(page._meta, "concrete_fields", []) + getattr(
        page._meta, "private_fields", []
    ):
        name = getattr(field, "name", None)
        if not name or name in skip_fields:
            continue
        if getattr(field, "is_relation", False):
            continue
        try:
            internal_type = field.get_internal_type()
        except Exception:
            internal_type = field.__class__.__name__
        if internal_type in {"CharField", "TextField", "RichTextField", "StreamField"}:
            fields.append(name)
    # dedupe preserving order
    seen = set()
    out = []
    for f in fields:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def page_to_chunks(
    page,
    tokenizer,
    chunk_size: int = 800,
    overlap: int = 128,
    emit_full_blob: bool = True,
) -> Iterable[Dict[str, Any]]:
    """
    Generic exporter: yields dicts with keys:
      - text: chunk text (string)
      - metadata: dict with page_id, page_type, title, section, chunk_index, total_chunks, content_length, url
    """
    title = getattr(page, "title", "") or ""
    page_id = getattr(page, "id", None)
    page_type = page.__class__.__name__
    url = None
    for attr in ("full_url", "url"):
        try:
            url = getattr(page, attr)
            break
        except Exception:
            url = None
    if not url:
        url = f"/page/{page_id}/"

    candidate_fields = _discover_text_fields(page)
    # always include title as first small chunk
    canonical_sections: List[Tuple[str, str]] = [("title", title)]
    # yield title doc
    yield {
        "text": f"Page: {title}\nSection: title\n\n{title}",
        "metadata": {
            "page_id": page_id,
            "page_type": page_type,
            "title": title,
            "url": url,
            "section": "title",
            "chunk_kind": "field",
            "chunk_index": 0,
            "total_chunks": 1,
            "content_length": len(title),
        },
    }

    for field_name in candidate_fields:
        try:
            if _is_streamfield(page, field_name):
                field_text = _extract_streamfield_text(page, field_name)
            else:
                raw = getattr(page, field_name, None)
                if raw is None:
                    continue
                # prefer .source for rich text-like objects
                if hasattr(raw, "source"):
                    field_text = _clean_html_preserve_paragraphs(raw.source)
                else:
                    field_text = _clean_html_preserve_paragraphs(raw)
            if not field_text or not field_text.strip():
                continue
            # collect for canonical blob
            canonical_sections.append((field_name, field_text))
            # chunk field_text
            chunks = list(
                paragraph_token_chunker(
                    field_text, tokenizer, chunk_size=chunk_size, overlap=overlap
                )
            )
            for i, chunk in enumerate(chunks):
                yield {
                    "text": f"Page: {title}\nSection: {field_name}\n\n{chunk}",
                    "metadata": {
                        "page_id": page_id,
                        "page_type": page_type,
                        "title": title,
                        "url": url,
                        "section": field_name,
                        "chunk_kind": "field",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "content_length": len(chunk),
                    },
                }
        except Exception:
            # skip problematic fields but continue
            continue

    # canonical full blob
    if emit_full_blob and canonical_sections:
        body = "\n\n".join(f"[{n}]\n{t}" for n, t in canonical_sections if t)
        if body.strip():
            yield {
                "text": f"Page: {title}\nSection: full_page\n\n{body}",
                "metadata": {
                    "page_id": page_id,
                    "page_type": page_type,
                    "title": title,
                    "url": url,
                    "section": "full_page",
                    "chunk_kind": "canonical_full_blob",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "content_length": len(body),
                },
            }
