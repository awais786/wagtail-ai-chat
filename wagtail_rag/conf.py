"""
Centralised settings access for Wagtail RAG.

All configuration lives under a single WAGTAIL_RAG dict.
Flat WAGTAIL_RAG_* keys are still read as fallbacks so existing
deployments continue to work without changes.

    WAGTAIL_RAG = {
        "embedding": {
            "provider": "openai",           # "openai" | "sentence-transformers" | "huggingface"
            "model":    "text-embedding-3-small",
        },
        "llm": {
            "provider":                "openai",   # "openai" | "ollama" | "anthropic"
            "model":                   "gpt-4o",
            "max_context_chars":       8000,       # 0 = unlimited
            "enable_history":          True,
            "history_recent_messages": 6,
        },
        "vector_store": {
            "backend":    "faiss",          # "faiss" | "chroma" | "pgvector"
            "path":       "/path/to/index",
            "collection": "wagtail_rag",
            # "connection_string": "postgresql+psycopg2://..."  # pgvector only
        },
        "indexing": {
            "chunk_size":      1500,
            "chunk_overlap":   100,
            "batch_size":      100,
            "skip_if_indexed": True,
            "prune_deleted":   True,
            "models": {
                # ["f1", "f2"] → use exactly these fields
                # "*"          → use Wagtail search_fields automatically
                "locations.LocationPage": ["introduction", "body", "address"],
                "breads.BreadPage":       "*",
            },
        },
        "search": {
            "k":                  8,    # chunks retrieved per query
            "max_sources":        3,    # unique pages shown as sources
            "use_hybrid":         True, # combine vector + Wagtail full-text search
            "use_query_expansion": True, # MultiQueryRetriever
        },
        "api": {
            "max_question_length":   150,  # characters; 0 = unlimited
            "max_request_body_size": 1048576,  # bytes (1 MB)
            "rate_limit_per_minute": 0,    # 0 = disabled
        },
    }
"""

import os

from django.conf import settings as django_settings


def _root() -> dict:
    return getattr(django_settings, "WAGTAIL_RAG", {}) or {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_bool(group: dict, key: str, flat_key: str, default: bool) -> bool:
    """Read a bool from the group dict, falling back to a flat setting."""
    v = group.get(key)
    if v is not None:
        return bool(v)
    return bool(getattr(django_settings, flat_key, default))


def _get_int(group: dict, key: str, flat_key: str, default: int) -> int:
    """Read an int from the group dict, falling back to a flat setting.

    Unlike ``or``, this preserves 0 as an intentional value (e.g. 'unlimited').
    """
    v = group.get(key)
    if v is not None:
        return int(v)
    flat = getattr(django_settings, flat_key, None)
    if flat is not None:
        return int(flat)
    return default


# ---------------------------------------------------------------------------
# Config groups
# ---------------------------------------------------------------------------


class _EmbeddingConf:
    def _group(self) -> dict:
        return _root().get("embedding") or {}

    @property
    def provider(self) -> str:
        return self._group().get("provider") or getattr(
            django_settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface"
        )

    @property
    def model(self):
        return self._group().get("model") or getattr(
            django_settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None
        )


class _LLMConf:
    def _group(self) -> dict:
        return _root().get("llm") or {}

    @property
    def provider(self) -> str:
        return self._group().get("provider") or getattr(
            django_settings, "WAGTAIL_RAG_LLM_PROVIDER", "ollama"
        )

    @property
    def model(self):
        return self._group().get("model") or getattr(
            django_settings, "WAGTAIL_RAG_MODEL_NAME", None
        )

    @property
    def max_context_chars(self) -> int:
        return _get_int(
            self._group(), "max_context_chars", "WAGTAIL_RAG_MAX_CONTEXT_CHARS", 0
        )

    @property
    def enable_history(self) -> bool:
        return _get_bool(
            self._group(), "enable_history", "WAGTAIL_RAG_ENABLE_CHAT_HISTORY", True
        )

    @property
    def history_recent_messages(self) -> int:
        return _get_int(
            self._group(),
            "history_recent_messages",
            "WAGTAIL_RAG_CHAT_HISTORY_RECENT_MESSAGES",
            6,
        )


class _VectorStoreConf:
    def _group(self) -> dict:
        return _root().get("vector_store") or {}

    @property
    def backend(self) -> str:
        return self._group().get("backend") or getattr(
            django_settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss"
        )

    @property
    def path(self) -> str:
        return (
            self._group().get("path")
            or getattr(django_settings, "WAGTAIL_RAG_CHROMA_PATH", None)
            or os.path.join(django_settings.BASE_DIR, "faiss_index")
        )

    @property
    def collection(self) -> str:
        return self._group().get("collection") or getattr(
            django_settings, "WAGTAIL_RAG_COLLECTION_NAME", "wagtail_rag"
        )

    @property
    def connection_string(self):
        return self._group().get("connection_string") or getattr(
            django_settings, "WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING", None
        )


class _IndexingConf:
    def _group(self) -> dict:
        return _root().get("indexing") or {}

    @property
    def models(self) -> dict:
        """Return {model_name: fields} dict.

        Falls back to flat WAGTAIL_RAG_MODELS list (all models get "*").
        """
        from_group = self._group().get("models")
        if isinstance(from_group, dict):
            return from_group
        flat = list(getattr(django_settings, "WAGTAIL_RAG_MODELS", None) or [])
        return {name: "*" for name in flat}

    def fields_for(self, model) -> list:
        """Return explicit field list for this model, or [] to use search_fields."""
        model_name = f"{model._meta.app_label}.{model.__class__.__name__}"
        v = self.models.get(model_name) or self.models.get(model.__class__.__name__)
        return v if isinstance(v, list) else []

    @property
    def chunk_size(self) -> int:
        return _get_int(self._group(), "chunk_size", "WAGTAIL_RAG_CHUNK_SIZE", 1500)

    @property
    def chunk_overlap(self) -> int:
        return _get_int(
            self._group(), "chunk_overlap", "WAGTAIL_RAG_CHUNK_OVERLAP", 100
        )

    @property
    def batch_size(self) -> int:
        return _get_int(
            self._group(), "batch_size", "WAGTAIL_RAG_EMBEDDING_BATCH_SIZE", 100
        )

    @property
    def skip_if_indexed(self) -> bool:
        return _get_bool(
            self._group(), "skip_if_indexed", "WAGTAIL_RAG_SKIP_IF_INDEXED", True
        )

    @property
    def prune_deleted(self) -> bool:
        return _get_bool(
            self._group(), "prune_deleted", "WAGTAIL_RAG_PRUNE_DELETED", True
        )


class _SearchConf:
    def _group(self) -> dict:
        return _root().get("search") or {}

    @property
    def k(self) -> int:
        return _get_int(self._group(), "k", "WAGTAIL_RAG_RETRIEVE_K", 8)

    @property
    def max_sources(self) -> int:
        return _get_int(self._group(), "max_sources", "WAGTAIL_RAG_MAX_SOURCES", 3)

    @property
    def use_hybrid(self) -> bool:
        return _get_bool(
            self._group(), "use_hybrid", "WAGTAIL_RAG_USE_HYBRID_SEARCH", True
        )

    @property
    def use_query_expansion(self) -> bool:
        return _get_bool(
            self._group(),
            "use_query_expansion",
            "WAGTAIL_RAG_USE_LLM_QUERY_EXPANSION",
            True,
        )

    @property
    def search_k(self) -> int:
        """K for the lower-level search_with_embeddings() (score-based search)."""
        return _get_int(self._group(), "search_k", "WAGTAIL_RAG_SEARCH_K", 10)

    @property
    def title_boost_max_score(self):
        """Maximum similarity score for title boosting (None = no cap)."""
        v = self._group().get("title_boost_max_score")
        if v is not None:
            return v
        return getattr(django_settings, "WAGTAIL_RAG_TITLE_BOOST_MAX_SCORE", None)


class _APIConf:
    def _group(self) -> dict:
        return _root().get("api") or {}

    @property
    def max_question_length(self) -> int:
        return _get_int(
            self._group(), "max_question_length", "WAGTAIL_RAG_MAX_QUESTION_LENGTH", 150
        )

    @property
    def max_request_body_size(self) -> int:
        return _get_int(
            self._group(),
            "max_request_body_size",
            "WAGTAIL_RAG_MAX_REQUEST_BODY_SIZE",
            1024 * 1024,
        )

    @property
    def rate_limit_per_minute(self) -> int:
        return _get_int(
            self._group(),
            "rate_limit_per_minute",
            "WAGTAIL_RAG_RATE_LIMIT_PER_MINUTE",
            0,
        )


class _Conf:
    embedding = _EmbeddingConf()
    llm = _LLMConf()
    vector_store = _VectorStoreConf()
    indexing = _IndexingConf()
    search = _SearchConf()
    api = _APIConf()


conf = _Conf()
