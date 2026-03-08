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
    }

Flat legacy settings (WAGTAIL_RAG_EMBEDDING_PROVIDER, WAGTAIL_RAG_MODEL_NAME, …)
are still supported as fallbacks.

Import and use:
    from wagtail_rag.conf import conf
    conf.embedding.provider  # → "openai"
    conf.llm.model           # → "gpt-4o"
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
        return (
            self._group().get("provider")
            or getattr(django_settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface")
        )

    @property
    def model(self):
        return (
            self._group().get("model")
            or getattr(django_settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None)
        )


class _LLMConf:
    """Accessor for WAGTAIL_RAG['llm'] settings."""

    def _group(self) -> dict:
        return _root().get("llm") or {}

    @property
    def provider(self) -> str:
        return (
            self._group().get("provider")
            or getattr(django_settings, "WAGTAIL_RAG_LLM_PROVIDER", "ollama")
        )

    @property
    def model(self):
        return (
            self._group().get("model")
            or getattr(django_settings, "WAGTAIL_RAG_MODEL_NAME", None)
        )


class _Conf:
    embedding = _EmbeddingConf()
    llm = _LLMConf()


conf = _Conf()
