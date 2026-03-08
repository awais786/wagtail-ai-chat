"""
Embedding Provider Factory for Wagtail RAG.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from django.conf import settings

logger = logging.getLogger(__name__)

PROVIDER_DEFAULTS: dict[str, str] = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "hf": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "sentence_transformers": "all-MiniLM-L6-v2",
    "sentence-transformers": "all-MiniLM-L6-v2",
}


def _import_class(paths: list[tuple[str, str]]) -> type | None:
    """Try each (module_path, class_name) pair and return the first found class."""
    for module_path, class_name in paths:
        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    return None


# ============================================================================
# Provider classes
# ============================================================================


class BaseEmbeddingProvider(ABC):
    """Base class for embedding provider implementations."""

    def __init__(self, django_settings):
        self.settings = django_settings

    @abstractmethod
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        """Create and return a provider-specific embedding instance."""


class HuggingFaceProvider(BaseEmbeddingProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        cls = _import_class([
            ("langchain_huggingface", "HuggingFaceEmbeddings"),
            ("langchain_community.embeddings", "HuggingFaceEmbeddings"),
            ("langchain.embeddings", "HuggingFaceEmbeddings"),
        ])
        if cls is None:
            raise ImportError(
                "HuggingFace embeddings are not installed. "
                "Install: pip install sentence-transformers langchain-huggingface"
            )
        if not model_name:
            raise ValueError("model_name is required for HuggingFace embeddings")
        return cls(model_name=model_name, **kwargs)


class OpenAIProvider(BaseEmbeddingProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        cls = _import_class([
            ("langchain_openai", "OpenAIEmbeddings"),
            ("langchain.embeddings", "OpenAIEmbeddings"),
        ])
        if cls is None:
            raise ImportError(
                "OpenAI embeddings are not installed. "
                "Install: pip install langchain-openai"
            )
        if not model_name:
            raise ValueError("model_name is required for OpenAI embeddings")
        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "OPENAI_API_KEY", None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in settings or passed as api_key")
        return cls(model=model_name, api_key=api_key, **kwargs)


class OllamaProvider(BaseEmbeddingProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        cls = _import_class([
            ("langchain_ollama", "OllamaEmbeddings"),
            ("langchain.embeddings", "OllamaEmbeddings"),
        ])
        if cls is None:
            raise ImportError(
                "Ollama embeddings are not installed. "
                "Install: pip install langchain-ollama"
            )
        if not model_name:
            raise ValueError("model_name is required for Ollama embeddings")
        return cls(model=model_name, **kwargs)


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Provider for sentence-transformers models via langchain_community.

    Distinct from HuggingFaceProvider: uses SentenceTransformerEmbeddings
    (langchain_community) without the full HuggingFace hub stack.
    Falls back to HuggingFaceEmbeddings if SentenceTransformerEmbeddings is unavailable.
    """

    def create(self, model_name: Optional[str], **kwargs) -> Any:
        if not model_name:
            raise ValueError("model_name is required for sentence-transformers embeddings")

        cls = _import_class([
            ("langchain_community.embeddings", "SentenceTransformerEmbeddings"),
        ])
        if cls is not None:
            return cls(model_name=model_name, **kwargs)

        # Fallback: HuggingFaceEmbeddings also wraps sentence-transformers.
        cls = _import_class([
            ("langchain_huggingface", "HuggingFaceEmbeddings"),
            ("langchain_community.embeddings", "HuggingFaceEmbeddings"),
        ])
        if cls is not None:
            logger.debug(
                "SentenceTransformerEmbeddings unavailable; falling back to HuggingFaceEmbeddings"
            )
            return cls(model_name=model_name, **kwargs)

        raise ImportError(
            "sentence-transformers embeddings are not installed. "
            "Install: pip install sentence-transformers langchain-community"
        )


# ============================================================================
# Factory
# ============================================================================


class EmbeddingProviderFactory:
    """Create embedding instances for supported providers."""

    PROVIDER_MAP: dict[str, type[BaseEmbeddingProvider]] = {
        "huggingface": HuggingFaceProvider,
        "hf": HuggingFaceProvider,
        "sentence_transformers": SentenceTransformerProvider,
        "sentence-transformers": SentenceTransformerProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
    }

    def __init__(self, django_settings=None):
        self.settings = django_settings or settings

    def _is_model_compatible(self, provider: str, model_name: str) -> bool:
        """Best-effort compatibility check to avoid obvious provider/model mismatches."""
        if not model_name:
            return False
        provider = provider.lower()
        if provider == "openai":
            return model_name.startswith("text-embedding-")
        if provider in {"huggingface", "hf"}:
            return "/" in model_name and not model_name.startswith("text-embedding-")
        if provider in {"sentence-transformers", "sentence_transformers"}:
            return (
                not model_name.startswith("text-embedding-")
                and not model_name.startswith("gpt-")
            )
        if provider == "ollama":
            return "/" not in model_name and not model_name.startswith("text-embedding-")
        return True

    def _resolve_setting_model_name(self, provider: str) -> Optional[str]:
        """Resolve model name from a provider-specific setting, then the global setting."""
        provider_setting_keys: dict[str, str] = {
            "openai": "WAGTAIL_RAG_OPENAI_EMBEDDING_MODEL",
            "ollama": "WAGTAIL_RAG_OLLAMA_EMBEDDING_MODEL",
            "huggingface": "WAGTAIL_RAG_HUGGINGFACE_EMBEDDING_MODEL",
            "hf": "WAGTAIL_RAG_HUGGINGFACE_EMBEDDING_MODEL",
        }
        key = provider_setting_keys.get(provider)
        if key:
            value = getattr(self.settings, key, None)
            if value:
                return value
        return getattr(self.settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None)

    def _resolve_model_name(self, provider: str, model_name: Optional[str]) -> str:
        """Resolve model name from explicit value, settings, or provider defaults."""
        if model_name:
            return model_name

        settings_model = self._resolve_setting_model_name(provider)
        if settings_model:
            if not self._is_model_compatible(provider, settings_model):
                logger.warning(
                    "Ignoring incompatible embedding model '%s' for provider '%s'; "
                    "falling back to provider default.",
                    settings_model,
                    provider,
                )
            else:
                return settings_model

        default = PROVIDER_DEFAULTS.get(provider)
        if not default:
            raise ValueError(
                f"model_name must be specified for embedding provider '{provider}'"
            )
        return default

    @classmethod
    def register(cls, name: str, provider_class: type[BaseEmbeddingProvider]) -> None:
        """Register a new embedding provider dynamically.

        Allows other Django apps to extend the provider registry.
        Typically called in AppConfig.ready().
        """
        cls.PROVIDER_MAP[name.lower()] = provider_class

    def get(
        self, provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs
    ) -> Any:
        """Get an embedding instance for the specified provider.

        Args:
            provider: Provider name (defaults to WAGTAIL_RAG_EMBEDDING_PROVIDER or 'huggingface')
            model_name: Optional model name (most providers require this)
            **kwargs: Provider-specific arguments
        """
        provider_key = (
            provider or getattr(self.settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface")
        ).lower()
        resolved_model = self._resolve_model_name(provider_key, model_name)
        logger.info(
            "Initializing embeddings: provider='%s', model='%s'", provider_key, resolved_model
        )

        provider_cls = self.PROVIDER_MAP.get(provider_key)
        if not provider_cls:
            canonical = sorted({
                k for k in self.PROVIDER_MAP
                if k not in ("hf", "sentence-transformers", "sentence_transformers")
            })
            raise ValueError(
                f"Unknown embedding provider: {provider_key!r}. "
                f"Supported: {', '.join(canonical)}"
            )

        return provider_cls(self.settings).create(resolved_model, **kwargs)


def get_embeddings(
    provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs
) -> Any:
    """Create and return an embedding instance for the given provider.

    Creates a fresh EmbeddingProviderFactory on each call so that Django
    settings are always read at call time rather than at import time.
    This is important for tests that override settings and for apps that
    configure settings after the module is first imported.
    """
    return EmbeddingProviderFactory().get(provider=provider, model_name=model_name, **kwargs)


__all__ = [
    "EmbeddingProviderFactory",
    "get_embeddings",
    "BaseEmbeddingProvider",
    "SentenceTransformerProvider",
]
