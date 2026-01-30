"""
Embedding Provider Factory for Wagtail RAG.

Refactored to use provider classes derived from BaseEmbeddingProvider for clarity
and easier extension/testing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from django.conf import settings

logger = logging.getLogger(__name__)

# Provider-specific default models
PROVIDER_DEFAULTS: Dict[str, str] = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "hf": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-ada-002",
    "sentence_transformers": "sentence-transformers/all-MiniLM-L6-v2",
}


# --- Provider classes -------------------------------------------------
class BaseEmbeddingProvider:
    """Base class for embedding provider implementations.

    Implementations should override create(model_name, **kwargs) and perform any
    provider-specific validation (e.g. API key presence) and imports lazily.
    """

    def __init__(self, django_settings):
        # explicit settings parameter (caller/factory passes settings)
        self.settings = django_settings

    def create(self, model_name: Optional[str], **kwargs) -> Any:
        """Create and return a provider-specific embedding instance.

        Subclasses must implement this method.
        
        Args:
            model_name: Optional model name. Most providers require this, but some
                       custom providers might allow None.
            **kwargs: Provider-specific arguments
        """
        raise NotImplementedError


class HuggingFaceProvider(BaseEmbeddingProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        try:
            # Prefer the standard langchain embedding interface when available
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
        except Exception:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
            except Exception:
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
                except Exception as e:
                    raise ImportError(
                        "HuggingFace embeddings are not installed. Install: pip install sentence-transformers langchain-huggingface"
                    ) from e
        
        if not model_name:
            raise ValueError("model_name is required for HuggingFace embeddings")
        
        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)


class OpenAIProvider(BaseEmbeddingProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        try:
            # try common package paths
            try:
                from langchain_openai import OpenAIEmbeddings  # type: ignore
            except Exception:
                from langchain.embeddings import OpenAIEmbeddings  # type: ignore
        except Exception as e:
            raise ImportError("OpenAI embeddings are not installed. Install: pip install langchain-openai") from e

        if not model_name:
            raise ValueError("model_name is required for OpenAI embeddings")
        
        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "OPENAI_API_KEY", None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in settings or passed as api_key")
        return OpenAIEmbeddings(model=model_name, api_key=api_key, **kwargs)


# --- Factory using provider classes -----------------------------------
class EmbeddingProviderFactory:
    """Create embedding instances for supported providers.

    This factory delegates to the provider classes above.
    """

    PROVIDER_MAP: Dict[str, Type[BaseEmbeddingProvider]] = {
        "huggingface": HuggingFaceProvider,
        "hf": HuggingFaceProvider,
        "sentence_transformers": HuggingFaceProvider,
        "sentence-transformers": HuggingFaceProvider,
        "openai": OpenAIProvider,
    }

    def __init__(self, django_settings=None):
        self.settings = django_settings or settings

    def _resolve_model_name(self, provider: str, model_name: Optional[str]) -> str:
        """Resolve model name from explicit value, settings, or provider defaults."""
        if model_name:
            return model_name
        settings_model = getattr(self.settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None)
        if settings_model:
            return settings_model
        default = PROVIDER_DEFAULTS.get(provider)
        if not default:
            raise ValueError(f"model_name must be specified for embedding provider '{provider}'")
        return default

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseEmbeddingProvider]) -> None:
        """Register a new embedding provider dynamically.
        
        This allows other Django apps to extend the provider registry.
        Typically called in AppConfig.ready().
        
        Args:
            name: Provider name (will be lowercased)
            provider_class: Class that inherits from BaseEmbeddingProvider
        """
        cls.PROVIDER_MAP[name.lower()] = provider_class

    def get(self, provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> Any:
        """Get an embedding instance for the specified provider.
        
        Args:
            provider: Provider name (defaults to WAGTAIL_RAG_EMBEDDING_PROVIDER or 'huggingface')
            model_name: Optional model name (most providers require this)
            **kwargs: Provider-specific arguments
        """
        provider_key = (provider or getattr(self.settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface")).lower()
        resolved_model_name = self._resolve_model_name(provider_key, model_name)
        logger.info("Initializing embeddings with provider='%s', model='%s'", provider_key, resolved_model_name)

        provider_cls = self.PROVIDER_MAP.get(provider_key)
        if not provider_cls:
            canonical = sorted({k for k in self.PROVIDER_MAP.keys() if k not in ("hf", "sentence-transformers", "sentence_transformers")})
            raise ValueError(f"Unknown embedding provider: {provider_key}. Supported providers: {', '.join(canonical)}")

        # Instantiate the provider strategy
        strategy = provider_cls(self.settings)
        # Pass the resolved model_name
        return strategy.create(resolved_model_name, **dict(kwargs))


# Convenience module-level helper
_factory = EmbeddingProviderFactory()


def get_embeddings(provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> Any:
    """Thin wrapper around EmbeddingProviderFactory.get for backward compatibility."""
    return _factory.get(provider=provider, model_name=model_name, **kwargs)


__all__ = ["EmbeddingProviderFactory", "get_embeddings", "BaseEmbeddingProvider"]
