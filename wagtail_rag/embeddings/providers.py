"""
Embedding Provider Factory for Wagtail RAG.

Refactored to use provider classes derived from BaseEmbeddingProvider for clarity
and easier extension/testing.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from django.conf import settings

logger = logging.getLogger(__name__)

# Provider-specific default models
PROVIDER_DEFAULTS: Dict[str, str] = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "hf": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-ada-002",
    "cohere": "embed-english-v3.0",
    "google": "models/embedding-001",
    "gemini": "models/embedding-001",
    "sentence_transformers": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2",
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

    def create(self, model_name: str, **kwargs) -> Any:
        """Create and return a provider-specific embedding instance.

        Subclasses must implement this method.
        """
        raise NotImplementedError


class HuggingFaceProvider(BaseEmbeddingProvider):
    def create(self, model_name: str, **kwargs) -> Any:
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
        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)


class OpenAIProvider(BaseEmbeddingProvider):
    def create(self, model_name: str, **kwargs) -> Any:
        try:
            # try common package paths
            try:
                from langchain_openai import OpenAIEmbeddings  # type: ignore
            except Exception:
                from langchain.embeddings import OpenAIEmbeddings  # type: ignore
        except Exception as e:
            raise ImportError("OpenAI embeddings are not installed. Install: pip install langchain-openai") from e

        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "OPENAI_API_KEY", None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in settings or passed as api_key")
        return OpenAIEmbeddings(model=model_name, api_key=api_key, **kwargs)


class CohereProvider(BaseEmbeddingProvider):
    def create(self, model_name: str, **kwargs) -> Any:
        try:
            from langchain_community.embeddings import CohereEmbeddings  # type: ignore
        except Exception as e:
            raise ImportError("Cohere embeddings are not installed. Install: pip install langchain-community cohere") from e

        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "COHERE_API_KEY", None)
        if not api_key:
            raise ValueError("COHERE_API_KEY must be set in settings or passed as api_key")
        return CohereEmbeddings(model=model_name, cohere_api_key=api_key, **kwargs)


class GoogleProvider(BaseEmbeddingProvider):
    def create(self, model_name: str, **kwargs) -> Any:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
        except Exception as e:
            raise ImportError("Google embeddings are not installed. Install: pip install langchain-google-genai") from e

        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "GOOGLE_API_KEY", None)
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be set in settings or passed as api_key")
        # google package expects google_api_key kwarg
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key, **kwargs)


class CustomProvider(BaseEmbeddingProvider):
    def create(self, model_name: str, **kwargs) -> Any:
        custom_factory = getattr(self.settings, "WAGTAIL_RAG_CUSTOM_EMBEDDING_FACTORY", None)
        if not custom_factory:
            raise ValueError("WAGTAIL_RAG_CUSTOM_EMBEDDING_FACTORY must be set for custom provider")
        if not callable(custom_factory):
            raise ValueError("WAGTAIL_RAG_CUSTOM_EMBEDDING_FACTORY must be a callable")
        return custom_factory(model_name=model_name, **kwargs)


# --- Factory using provider classes -----------------------------------
class EmbeddingProviderFactory:
    """Create embedding instances for supported providers.

    This factory delegates to the provider classes above.
    """

    PROVIDER_MAP: Dict[str, Callable[..., BaseEmbeddingProvider]] = {
        "huggingface": HuggingFaceProvider,
        "hf": HuggingFaceProvider,
        "sentence_transformers": HuggingFaceProvider,
        "sentence-transformers": HuggingFaceProvider,
        "openai": OpenAIProvider,
        "cohere": CohereProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,
        "custom": CustomProvider,
    }

    def __init__(self, django_settings=None):
        self.settings = django_settings or settings

    def _resolve_model_name(self, provider: str, model_name: Optional[str]) -> str:
        if model_name:
            return model_name
        settings_model = getattr(self.settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None)
        if settings_model:
            return settings_model
        default = PROVIDER_DEFAULTS.get(provider)
        if not default:
            raise ValueError(f"model_name must be specified for embedding provider '{provider}'")
        return default

    def get(self, provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> Any:
        provider_key = (provider or getattr(self.settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface")).lower()
        model_name = self._resolve_model_name(provider_key, model_name)
        logger.info("Initializing embeddings with provider='%s', model='%s'", provider_key, model_name)

        provider_cls = self.PROVIDER_MAP.get(provider_key)
        if not provider_cls:
            canonical = sorted({k for k in self.PROVIDER_MAP.keys() if k not in ("hf", "gemini", "sentence-transformers", "sentence_transformers")})
            raise ValueError(f"Unknown embedding provider: {provider_key}. Supported providers: {', '.join(canonical)}")

        # instantiate provider and delegate create (factory passes settings explicitly)
        provider_instance = provider_cls(self.settings)
        # pass a shallow copy of kwargs to avoid mutating caller dict
        return provider_instance.create(model_name, **dict(kwargs))


# Convenience module-level helper
_factory = EmbeddingProviderFactory()


def get_embeddings(provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> Any:
    """Thin wrapper around EmbeddingProviderFactory.get for backward compatibility."""
    return _factory.get(provider=provider, model_name=model_name, **kwargs)


__all__ = ["EmbeddingProviderFactory", "get_embeddings", "BaseEmbeddingProvider"]
