"""
Centralised settings access for Wagtail RAG.

    WAGTAIL_RAG = {
        "embedding": {
            "provider": "openai",           # "openai" | "sentence-transformers" | "huggingface"
            "model":    "text-embedding-3-small",
        },
        "llm": {
            "provider": "openai",           # "openai" | "ollama" | "anthropic"
            "model":    "gpt-4o",
        },
        "vector_store": {
            "backend":    "faiss",          # "faiss" | "chroma" | "pgvector"
            "path":       "/path/to/index",
            "collection": "wagtail_rag",
            # "connection_string": "postgresql+psycopg2://..."  # pgvector only
        },
        "indexing": {
            "models": {
                # ["f1", "f2"] → use exactly these fields
                # "*"          → use Wagtail search_fields automatically
                "locations.LocationPage": ["introduction", "body", "address"],
                "breads.BreadPage":       "*",
                "blog.BlogPage":          "*",
            },
        },
    }
"""

from django.conf import settings as django_settings


def _root() -> dict:
    return getattr(django_settings, "WAGTAIL_RAG", {}) or {}


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


class _Conf:
    embedding    = _EmbeddingConf()
    llm          = _LLMConf()
    vector_store = _VectorStoreConf()
    indexing     = _IndexingConf()


conf = _Conf()
