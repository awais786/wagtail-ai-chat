"""
LLM Provider Factory for Wagtail RAG.

Refactored to use provider classes derived from BaseLLMProvider for clarity
and easier extension/testing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from django.conf import settings

logger = logging.getLogger(__name__)

# Provider-specific default models
# Note: For Ollama, make sure the model is installed: ollama pull <model_name>
PROVIDER_DEFAULTS: Dict[str, Optional[str]] = {
    "ollama": None,  # No default - user must specify model or set WAGTAIL_RAG_MODEL_NAME
    "openai": "gpt-4",
    "anthropic": "claude-3-sonnet-20240229",
    "claude": "claude-3-sonnet-20240229",
    "huggingface": None,
    "hf": None,
}


# --- Provider classes -------------------------------------------------
class BaseLLMProvider:
    """Base class for LLM provider implementations.

    Implementations should override load(model_name, **kwargs) and perform any
    provider-specific validation (e.g. API key presence) and imports lazily.
    """

    def __init__(self, django_settings):
        # explicit settings parameter (caller/factory passes settings)
        self.settings = django_settings

    def create(self, model_name: Optional[str], **kwargs) -> Any:
        """Create and return a provider-specific LLM/chat model instance.

        Subclasses must implement this method.
        
        Args:
            model_name: Optional model name. Some providers (e.g., HuggingFace with endpoint_url)
                       may allow None if an endpoint_url is provided.
            **kwargs: Provider-specific arguments
        """
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        """Create Ollama LLM instance, preferring ChatOllama over legacy Ollama.
        
        Args:
            model_name: Model name (e.g., 'mistral', 'llama2', 'phi'). 
                       Must be installed in Ollama: ollama pull <model_name>
        """
        if not model_name:
            raise ValueError(
                "model_name is required for Ollama provider. "
                "Set WAGTAIL_RAG_MODEL_NAME in settings or install a model: ollama pull <model_name>"
            )
        
        try:
            # Modern LangChain standard
            from langchain_community.chat_models import ChatOllama  # type: ignore
            return ChatOllama(model=model_name, **kwargs)
        except (ImportError, ModuleNotFoundError):
            try:
                # Legacy fallback
                from langchain_community.llms import Ollama  # type: ignore
                return Ollama(model=model_name, **kwargs)
            except ImportError as e:
                raise ImportError(
                    "Ollama not found. Run: pip install langchain-community ollama"
                ) from e
        except Exception as e:
            # Catch model not found errors and provide helpful message
            error_msg = str(e).lower()
            if "model" in error_msg and ("not found" in error_msg or "does not exist" in error_msg):
                raise ValueError(
                    f"Ollama model '{model_name}' not found. "
                    f"Install it with: ollama pull {model_name}\n"
                    f"Or set WAGTAIL_RAG_MODEL_NAME to a different model in your settings."
                ) from e
            raise


class OpenAIProvider(BaseLLMProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as e:
            raise ImportError("OpenAI provider not available. Install: pip install langchain-openai") from e

        if not model_name:
            raise ValueError("model_name is required for OpenAI provider")
        
        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "OPENAI_API_KEY", None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in settings or passed as api_key")

        return ChatOpenAI(model=model_name, api_key=api_key, **kwargs)


class AnthropicProvider(BaseLLMProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except Exception as e:
            raise ImportError("Anthropic provider not available. Install: pip install langchain-anthropic") from e

        if not model_name:
            raise ValueError("model_name is required for Anthropic provider")
        
        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "ANTHROPIC_API_KEY", None)
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in settings or passed as api_key")

        return ChatAnthropic(model=model_name, api_key=api_key, **kwargs)


class HuggingFaceProvider(BaseLLMProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        """Create HuggingFace LLM instance.
        
        Supports both local transformers pipeline and hosted endpoint.
        If endpoint_url is provided, model_name can be None.
        Task can be overridden via kwargs (default: "text-generation").
        """
        # Prefer local transformers pipeline if available, otherwise use hosted endpoint
        try:
            from langchain_huggingface import HuggingFacePipeline  # type: ignore
            from transformers import pipeline  # type: ignore

            # Allow task override via kwargs (default: "text-generation")
            task = kwargs.pop("task", "text-generation")
            model_id = kwargs.pop("model_id", model_name)
            
            if not model_id:
                raise ValueError("model_name or model_id is required for HuggingFace pipeline")
            
            # Filter out endpoint_url and other non-pipeline kwargs
            pipeline_kwargs = {k: v for k, v in kwargs.items() if k not in ("endpoint_url", "api_key")}
            pipe = pipeline(task, model=model_id, **pipeline_kwargs)
            return HuggingFacePipeline(pipeline=pipe)
        except Exception:
            try:
                from langchain_community.llms import HuggingFaceEndpoint  # type: ignore
                endpoint_url = kwargs.pop("endpoint_url", None) or getattr(self.settings, "HUGGINGFACE_ENDPOINT_URL", None)
                api_key = kwargs.pop("api_key", None) or getattr(self.settings, "HUGGINGFACE_API_KEY", None)

                # If endpoint_url not provided, construct from model_name
                if not endpoint_url:
                    if not model_name:
                        raise ValueError("Either endpoint_url or model_name must be provided for HuggingFace endpoint")
                    endpoint_url = f"https://api-inference.huggingface.co/models/{model_name}"

                return HuggingFaceEndpoint(endpoint_url=endpoint_url, huggingfacehub_api_token=api_key, **kwargs)
            except Exception as e:
                raise ImportError("HuggingFace provider not available. Install: pip install langchain-huggingface transformers") from e


# --- Factory using provider classes -----------------------------------
class LLMProviderFactory:
    """Create LLM/chat model instances for supported providers.

    This factory delegates to the provider classes above.
    """

    PROVIDER_MAP: Dict[str, Type[BaseLLMProvider]] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
        "hf": HuggingFaceProvider,
    }

    def __init__(self, django_settings=None):
        self.settings = django_settings or settings

    def _resolve_model_name(self, provider: str, model_name: Optional[str]) -> Optional[str]:
        """Resolve model name from explicit value, settings, or provider defaults."""
        if model_name:
            return model_name
        settings_model = getattr(self.settings, "WAGTAIL_RAG_MODEL_NAME", None)
        if settings_model:
            return settings_model
        return PROVIDER_DEFAULTS.get(provider)

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """Register a new LLM provider dynamically.
        
        This allows other Django apps to extend the provider registry.
        Typically called in AppConfig.ready().
        
        Args:
            name: Provider name (will be lowercased)
            provider_class: Class that inherits from BaseLLMProvider
        """
        cls.PROVIDER_MAP[name.lower()] = provider_class

    def get(self, provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> Any:
        """Get an LLM instance for the specified provider.
        
        Args:
            provider: Provider name (defaults to WAGTAIL_RAG_LLM_PROVIDER or 'ollama')
            model_name: Optional model name. Some providers (e.g., HuggingFace with endpoint_url)
                       may allow None if an endpoint_url is provided.
            **kwargs: Provider-specific arguments
        """
        provider_key = (provider or getattr(self.settings, "WAGTAIL_RAG_LLM_PROVIDER", "ollama")).lower()
        resolved_model_name = self._resolve_model_name(provider_key, model_name)
        logger.info("Initializing LLM with provider='%s', model='%s'", provider_key, resolved_model_name)

        provider_cls = self.PROVIDER_MAP.get(provider_key)
        if not provider_cls:
            canonical = sorted({k for k in self.PROVIDER_MAP.keys() if k not in ("hf", "claude")})
            raise ValueError(f"Unknown LLM provider: {provider_key}. Supported providers: {', '.join(canonical)}")

        # Instantiate the provider strategy
        strategy = provider_cls(self.settings)
        # Pass the resolved model_name (may be None for some providers like HuggingFace with endpoint_url)
        return strategy.create(resolved_model_name, **dict(kwargs))


# Convenience module-level helper
_factory = LLMProviderFactory()


def get_llm(provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> Any:
    """Thin wrapper around LLMProviderFactory.get for backward compatibility."""
    return _factory.get(provider=provider, model_name=model_name, **kwargs)


__all__ = ["LLMProviderFactory", "get_llm", "BaseLLMProvider"]
