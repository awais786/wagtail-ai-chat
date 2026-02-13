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
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "sentence_transformers": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2",  # Same as above (hyphen variant)
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
        # Try multiple import paths for HuggingFace embeddings
        HuggingFaceEmbeddings = None
        import_paths = [
            ("langchain.embeddings", "HuggingFaceEmbeddings"),
            ("langchain_huggingface", "HuggingFaceEmbeddings"),
            ("langchain_community.embeddings", "HuggingFaceEmbeddings"),
        ]
        
        for module_path, class_name in import_paths:
            try:
                module = __import__(module_path, fromlist=[class_name])
                HuggingFaceEmbeddings = getattr(module, class_name)
                break
            except (ImportError, AttributeError):
                continue
        
        if HuggingFaceEmbeddings is None:
            raise ImportError(
                "HuggingFace embeddings are not installed. Install: pip install sentence-transformers langchain-huggingface"
            )
        
        if not model_name:
            raise ValueError("model_name is required for HuggingFace embeddings")
        
        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)


class OpenAIProvider(BaseEmbeddingProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        # Try multiple import paths for OpenAI embeddings
        OpenAIEmbeddings = None
        import_paths = [
            ("langchain_openai", "OpenAIEmbeddings"),
            ("langchain.embeddings", "OpenAIEmbeddings"),
        ]
        
        for module_path, class_name in import_paths:
            try:
                module = __import__(module_path, fromlist=[class_name])
                OpenAIEmbeddings = getattr(module, class_name)
                break
            except (ImportError, AttributeError):
                continue
        
        if OpenAIEmbeddings is None:
            raise ImportError("OpenAI embeddings are not installed. Install: pip install langchain-openai")

        if not model_name:
            raise ValueError("model_name is required for OpenAI embeddings")
        
        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "OPENAI_API_KEY", None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in settings or passed as api_key")
        return OpenAIEmbeddings(model=model_name, api_key=api_key, **kwargs)


class OllamaProvider(BaseEmbeddingProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        # Try multiple import paths for Ollama embeddings
        OllamaEmbeddings = None
        import_paths = [
            ("langchain_ollama", "OllamaEmbeddings"),
            ("langchain.embeddings", "OllamaEmbeddings"),
        ]
        
        for module_path, class_name in import_paths:
            try:
                module = __import__(module_path, fromlist=[class_name])
                OllamaEmbeddings = getattr(module, class_name)
                break
            except (ImportError, AttributeError):
                continue
        
        if OllamaEmbeddings is None:
            raise ImportError("Ollama embeddings are not installed. Install: pip install langchain-ollama")

        if not model_name:
            raise ValueError("model_name is required for Ollama embeddings")
        
        return OllamaEmbeddings(model=model_name, **kwargs)

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
        "ollama": OllamaProvider,
    }

    def __init__(self, django_settings=None):
        self.settings = django_settings or settings

    def _is_model_compatible(self, provider: str, model_name: str) -> bool:
        """Best-effort compatibility check to avoid obvious provider/model mismatches."""
        if not model_name:
            return False
        provider = provider.lower()
        hf_aliases = {"huggingface", "hf", "sentence-transformers", "sentence_transformers"}

        if provider == "openai":
            return model_name.startswith("text-embedding-")
        if provider in hf_aliases:
            # HF models are usually repo-style names like org/model.
            return "/" in model_name and not model_name.startswith("text-embedding-")
        if provider == "ollama":
            # Ollama model names are typically short local tags, not HF repo paths or OpenAI names.
            return ("/" not in model_name) and (not model_name.startswith("text-embedding-"))
        return True

    def _resolve_setting_model_name(self, provider: str) -> Optional[str]:
        """
        Resolve model name from provider-specific setting first, then global setting.
        """
        provider_setting_key_map = {
            "openai": "WAGTAIL_RAG_OPENAI_EMBEDDING_MODEL",
            "ollama": "WAGTAIL_RAG_OLLAMA_EMBEDDING_MODEL",
            "huggingface": "WAGTAIL_RAG_HUGGINGFACE_EMBEDDING_MODEL",
            "hf": "WAGTAIL_RAG_HUGGINGFACE_EMBEDDING_MODEL",
            "sentence-transformers": "WAGTAIL_RAG_HUGGINGFACE_EMBEDDING_MODEL",
            "sentence_transformers": "WAGTAIL_RAG_HUGGINGFACE_EMBEDDING_MODEL",
        }
        provider_setting_key = provider_setting_key_map.get(provider)
        if provider_setting_key:
            provider_model = getattr(self.settings, provider_setting_key, None)
            if provider_model:
                return provider_model
        return getattr(self.settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None)

    def _resolve_model_name(self, provider: str, model_name: Optional[str]) -> str:
        """Resolve model name from explicit value, settings, or provider defaults."""
        if model_name:
            return model_name

        settings_model = self._resolve_setting_model_name(provider)
        if settings_model:
            if not self._is_model_compatible(provider, settings_model):
                logger.warning(
                    "Ignoring incompatible embedding model '%s' for provider '%s'; falling back to provider default.",
                    settings_model,
                    provider,
                )
            else:
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
