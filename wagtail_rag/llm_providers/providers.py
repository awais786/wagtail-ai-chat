"""
LLM Provider Factory for Wagtail RAG.

Refactored to use provider classes derived from BaseLLMProvider for clarity
and easier extension/testing.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from django.conf import settings

logger = logging.getLogger(__name__)

# Provider-specific default models
PROVIDER_DEFAULTS: Dict[str, Optional[str]] = {
    "ollama": "mistral",
    "openai": "gpt-4",
    "anthropic": "claude-3-sonnet-20240229",
    "claude": "claude-3-sonnet-20240229",
    "google": "gemini-pro",
    "gemini": "gemini-pro",
    "cohere": "command",
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

    def create(self, model_name: str, **kwargs) -> Any:
        """Create and return a provider-specific LLM/chat model instance.

        Subclasses must implement this method (previously named `load`).
        """
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    def create(self, model_name: str, **kwargs) -> Any:
        # allow either ChatOllama or Ollama depending on langchain version
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
            return ChatOllama(model=model_name, **kwargs)
        except Exception:
            try:
                from langchain_community.llms import Ollama  # type: ignore
                return Ollama(model=model_name, **kwargs)
            except Exception as e:
                raise ImportError(
                    "Ollama provider not available. Install: pip install langchain-community ollama"
                ) from e


class OpenAIProvider(BaseLLMProvider):
    def create(self, model_name: str, **kwargs) -> Any:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as e:
            raise ImportError("OpenAI provider not available. Install: pip install langchain-openai") from e

        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "OPENAI_API_KEY", None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in settings or passed as api_key")

        return ChatOpenAI(model=model_name, api_key=api_key, **kwargs)


class AnthropicProvider(BaseLLMProvider):
    def create(self, model_name: str, **kwargs) -> Any:
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except Exception as e:
            raise ImportError("Anthropic provider not available. Install: pip install langchain-anthropic") from e

        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "ANTHROPIC_API_KEY", None)
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in settings or passed as api_key")

        return ChatAnthropic(model=model_name, api_key=api_key, **kwargs)


class CohereProvider(BaseLLMProvider):
    def create(self, model_name: str, **kwargs) -> Any:
        try:
            from langchain_community.llms import Cohere  # type: ignore
        except Exception as e:
            raise ImportError("Cohere provider not available. Install: pip install langchain-community cohere") from e

        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "COHERE_API_KEY", None)
        if not api_key:
            raise ValueError("COHERE_API_KEY must be set in settings or passed as api_key")

        return Cohere(model=model_name, cohere_api_key=api_key, **kwargs)


class GoogleProvider(BaseLLMProvider):
    def create(self, model_name: str, **kwargs) -> Any:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except Exception as e:
            raise ImportError("Google Generative AI provider not available. Install: pip install langchain-google-genai") from e

        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "GOOGLE_API_KEY", None)
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be set in settings or passed as api_key")

        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, **kwargs)


class HuggingFaceProvider(BaseLLMProvider):
    def create(self, model_name: str, **kwargs) -> Any:
        # Prefer local transformers pipeline if available, otherwise use hosted endpoint
        try:
            from langchain_huggingface import HuggingFacePipeline  # type: ignore
            from transformers import pipeline  # type: ignore

            model_id = kwargs.pop("model_id", model_name)
            pipe = pipeline("text-generation", model=model_id, **{k: v for k, v in kwargs.items() if k != "endpoint_url"})
            return HuggingFacePipeline(pipeline=pipe)
        except Exception:
            try:
                from langchain_community.llms import HuggingFaceEndpoint  # type: ignore
                endpoint_url = kwargs.pop("endpoint_url", None) or getattr(self.settings, "HUGGINGFACE_ENDPOINT_URL", None)
                api_key = kwargs.pop("api_key", None) or getattr(self.settings, "HUGGINGFACE_API_KEY", None)

                if not endpoint_url and model_name:
                    endpoint_url = f"https://api-inference.huggingface.co/models/{model_name}"

                return HuggingFaceEndpoint(endpoint_url=endpoint_url, huggingfacehub_api_token=api_key, **kwargs)
            except Exception as e:
                raise ImportError("HuggingFace provider not available. Install: pip install langchain-huggingface transformers") from e


class CustomProvider(BaseLLMProvider):
    def create(self, model_name: str, **kwargs) -> Any:
         custom_factory = getattr(self.settings, "WAGTAIL_RAG_CUSTOM_LLM_FACTORY", None)
         if not custom_factory:
             raise ValueError("WAGTAIL_RAG_CUSTOM_LLM_FACTORY must be set for custom provider")
         if not callable(custom_factory):
             raise ValueError("WAGTAIL_RAG_CUSTOM_LLM_FACTORY must be a callable")
         return custom_factory(model_name=model_name, **kwargs)


# --- Factory using provider classes -----------------------------------
class LLMProviderFactory:
    """Create LLM/chat model instances for supported providers.

    This factory delegates to the provider classes above.
    """

    PROVIDER_MAP: Dict[str, Callable[..., BaseLLMProvider]] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
        "hf": HuggingFaceProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,
        "cohere": CohereProvider,
        "custom": CustomProvider,
    }

    def __init__(self, django_settings=None):
        self.settings = django_settings or settings

    def _resolve_model_name(self, provider: str, model_name: Optional[str]) -> Optional[str]:
        if model_name:
            return model_name
        settings_model = getattr(self.settings, "WAGTAIL_RAG_MODEL_NAME", None)
        if settings_model:
            return settings_model
        return PROVIDER_DEFAULTS.get(provider)

    def get(self, provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> Any:
        provider_key = (provider or getattr(self.settings, "WAGTAIL_RAG_LLM_PROVIDER", "ollama")).lower()
        model_name = self._resolve_model_name(provider_key, model_name)
        logger.info("Initializing LLM with provider='%s', model='%s'", provider_key, model_name)

        provider_cls = self.PROVIDER_MAP.get(provider_key)
        if not provider_cls:
            canonical = sorted({k for k in self.PROVIDER_MAP.keys() if k not in ("hf", "claude", "gemini")})
            raise ValueError(f"Unknown LLM provider: {provider_key}. Supported providers: {', '.join(canonical)}")

        # instantiate provider and delegate load (factory passes settings explicitly)
        provider_instance = provider_cls(self.settings)
        # pass a shallow copy of kwargs to avoid mutating caller dict
        return provider_instance.create(model_name or "", **dict(kwargs))


# Convenience module-level helper
_factory = LLMProviderFactory()


def get_llm(provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> Any:
    """Thin wrapper around LLMProviderFactory.get for backward compatibility."""
    return _factory.get(provider=provider, model_name=model_name, **kwargs)


__all__ = ["LLMProviderFactory", "get_llm", "BaseLLMProvider"]
