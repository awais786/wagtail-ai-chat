"""
Centralised settings access for Wagtail RAG.

Preferred configuration uses a single grouped dict:

    WAGTAIL_RAG = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o",
        },
        "vector_store": {
            "backend":    "faiss",          # "faiss" | "chroma" | "pgvector"
            "path":       "/path/to/index",
            "collection": "wagtail_rag",
            # "connection_string": "postgresql+psycopg2://..."  # pgvector only
        },
    }

Flat legacy settings (WAGTAIL_RAG_VECTOR_STORE_BACKEND, WAGTAIL_RAG_MODEL_NAME, …)
are still supported as fallbacks.

Import and use:
    from wagtail_rag.conf import conf
    conf.embedding.provider      # → "openai"
    conf.llm.model               # → "gpt-4o"
    conf.vector_store.backend    # → "faiss"
"""

from django.conf import settings as django_settings


def _root() -> dict:
    return getattr(django_settings, "WAGTAIL_RAG", {}) or {}


class _EmbeddingConf:
    """Accessor for WAGTAIL_RAG['embedding'] settings."""

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
    """Accessor for WAGTAIL_RAG['llm'] settings."""

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


class _VectorStoreConf:
    """Accessor for WAGTAIL_RAG['vector_store'] settings."""

    def _group(self) -> dict:
        return _root().get("vector_store") or {}

    @property
    def backend(self) -> str:
        return self._group().get("backend") or getattr(
            django_settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss"
        )

    @property
    def path(self) -> str:
        import os

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


class _Conf:
    embedding = _EmbeddingConf()
    llm = _LLMConf()
    vector_store = _VectorStoreConf()


conf = _Conf()
