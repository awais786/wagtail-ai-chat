"""
LLM Provider Factory for Wagtail RAG.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from django.conf import settings

logger = logging.getLogger(__name__)

PROVIDER_DEFAULTS: dict[str, Optional[str]] = {
    "ollama": None,  # No default — user must specify model or set WAGTAIL_RAG_MODEL_NAME
    "openai": "gpt-4",
    "anthropic": "claude-3-sonnet-20240229",
    "huggingface": None,
    "hf": None,
}


# ============================================================================
# Provider classes
# ============================================================================


class BaseLLMProvider(ABC):
    """Base class for LLM provider implementations."""

    def __init__(self, django_settings):
        self.settings = django_settings

    @abstractmethod
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        """Create and return a provider-specific LLM/chat model instance."""


class OllamaProvider(BaseLLMProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        if not model_name:
            raise ValueError(
                "model_name is required for Ollama provider. "
                "Set WAGTAIL_RAG_MODEL_NAME in settings or run: ollama pull <model_name>"
            )

        for module_path, class_name in [
            ("langchain_community.chat_models", "ChatOllama"),
            ("langchain_community.llms", "Ollama"),
        ]:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                return cls(model=model_name, **kwargs)
            except (ImportError, AttributeError):
                continue

        raise ImportError("Ollama not found. Run: pip install langchain-community ollama")


class OpenAIProvider(BaseLLMProvider):
    def create(self, model_name: Optional[str], **kwargs) -> Any:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OpenAI provider not available. Install: pip install langchain-openai"
            ) from e

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
        except ImportError as e:
            raise ImportError(
                "Anthropic provider not available. Install: pip install langchain-anthropic"
            ) from e

        if not model_name:
            raise ValueError("model_name is required for Anthropic provider")

        api_key = kwargs.pop("api_key", None) or getattr(self.settings, "ANTHROPIC_API_KEY", None)
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in settings or passed as api_key")

        return ChatAnthropic(model=model_name, api_key=api_key, **kwargs)


class HuggingFaceProvider(BaseLLMProvider):
    """Supports both a local transformers pipeline and a hosted HuggingFace endpoint.

    If endpoint_url is provided (via kwargs or HUGGINGFACE_ENDPOINT_URL setting),
    uses HuggingFaceEndpoint. Otherwise uses a local HuggingFacePipeline.
    """

    def create(self, model_name: Optional[str], **kwargs) -> Any:
        endpoint_url = kwargs.get("endpoint_url") or getattr(
            self.settings, "HUGGINGFACE_ENDPOINT_URL", None
        )
        api_key = kwargs.get("api_key") or getattr(self.settings, "HUGGINGFACE_API_KEY", None)

        if endpoint_url:
            try:
                from langchain_community.llms import HuggingFaceEndpoint  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "HuggingFace endpoint mode not available. "
                    "Install: pip install langchain-community"
                ) from e
            endpoint_kwargs = {
                k: v for k, v in kwargs.items() if k not in ("endpoint_url", "api_key")
            }
            return HuggingFaceEndpoint(
                endpoint_url=endpoint_url,
                huggingfacehub_api_token=api_key,
                **endpoint_kwargs,
            )

        try:
            from langchain_huggingface import HuggingFacePipeline  # type: ignore
            from transformers import pipeline  # type: ignore
        except ImportError:
            # Local pipeline unavailable; fall back to hosted inference API.
            try:
                from langchain_community.llms import HuggingFaceEndpoint  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "HuggingFace provider not available. "
                    "Install: pip install langchain-huggingface transformers"
                ) from e

            model_id = kwargs.get("model_id", model_name)
            if not model_id:
                raise ValueError(
                    "Either endpoint_url or model_name must be provided for HuggingFace"
                )
            inferred_url = f"https://api-inference.huggingface.co/models/{model_id}"
            endpoint_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("model_id", "task", "endpoint_url", "api_key")
            }
            return HuggingFaceEndpoint(
                endpoint_url=inferred_url,
                huggingfacehub_api_token=api_key,
                **endpoint_kwargs,
            )

        task = kwargs.pop("task", "text-generation")
        model_id = kwargs.pop("model_id", model_name)
        if not model_id:
            raise ValueError("model_name or model_id is required for HuggingFace pipeline")

        pipeline_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("endpoint_url", "api_key")
        }
        try:
            pipe = pipeline(task, model=model_id, **pipeline_kwargs)
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize local HuggingFace pipeline. "
                "Provide endpoint_url to use hosted inference, or fix local transformers setup."
            ) from e


# ============================================================================
# Factory
# ============================================================================


class LLMProviderFactory:
    """Create LLM/chat model instances for supported providers."""

    PROVIDER_MAP: dict[str, type[BaseLLMProvider]] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
        "hf": HuggingFaceProvider,
    }

    def __init__(self, django_settings=None):
        self.settings = django_settings or settings

    def _is_model_compatible(self, provider: str, model_name: str) -> bool:
        """Best-effort guard against obvious provider/model mismatches."""
        if not model_name:
            return False
        lower = model_name.lower()
        if provider == "openai":
            return not lower.startswith("claude")
        if provider == "anthropic":
            return lower.startswith("claude")
        if provider == "ollama":
            return not lower.startswith(("gpt-", "claude"))
        return True

    def _resolve_setting_model_name(self, provider: str) -> Optional[str]:
        """Resolve model name from grouped setting, provider-specific setting, then global setting."""
        # Grouped setting takes priority
        group = (getattr(self.settings, "WAGTAIL_RAG", {}) or {}).get("llm") or {}
        if group.get("model"):
            return group["model"]

        provider_setting_keys: dict[str, str] = {
            "openai": "WAGTAIL_RAG_OPENAI_MODEL_NAME",
            "anthropic": "WAGTAIL_RAG_ANTHROPIC_MODEL_NAME",
            "ollama": "WAGTAIL_RAG_OLLAMA_MODEL_NAME",
            "huggingface": "WAGTAIL_RAG_HUGGINGFACE_MODEL_NAME",
            "hf": "WAGTAIL_RAG_HUGGINGFACE_MODEL_NAME",
        }
        key = provider_setting_keys.get(provider)
        if key:
            value = getattr(self.settings, key, None)
            if value:
                return value
        return getattr(self.settings, "WAGTAIL_RAG_MODEL_NAME", None)

    def _resolve_model_name(self, provider: str, model_name: Optional[str]) -> Optional[str]:
        """Resolve model name from explicit value, settings, or provider defaults."""
        if model_name:
            return model_name

        settings_model = self._resolve_setting_model_name(provider)
        if settings_model:
            if self._is_model_compatible(provider, settings_model):
                return settings_model
            logger.warning(
                "Ignoring incompatible model '%s' for provider '%s'; "
                "falling back to provider default.",
                settings_model,
                provider,
            )

        return PROVIDER_DEFAULTS.get(provider)

    @classmethod
    def register(cls, name: str, provider_class: type[BaseLLMProvider]) -> None:
        """Register a new LLM provider dynamically.

        Allows other Django apps to extend the provider registry.
        Typically called in AppConfig.ready().
        """
        cls.PROVIDER_MAP[name.lower()] = provider_class

    def get(
        self, provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs
    ) -> Any:
        """Get an LLM instance for the specified provider.

        Args:
            provider: Provider name (defaults to WAGTAIL_RAG_LLM_PROVIDER or 'ollama')
            model_name: Optional model name (required for most providers)
            **kwargs: Provider-specific arguments
        """
        group = (getattr(self.settings, "WAGTAIL_RAG", {}) or {}).get("llm") or {}
        default_provider = (
            group.get("provider")
            or getattr(self.settings, "WAGTAIL_RAG_LLM_PROVIDER", "ollama")
        )
        provider_key = (provider or default_provider).lower()
        resolved_model = self._resolve_model_name(provider_key, model_name)
        logger.info(
            "Initializing LLM: provider='%s', model='%s'", provider_key, resolved_model
        )

        provider_cls = self.PROVIDER_MAP.get(provider_key)
        if not provider_cls:
            canonical = sorted({k for k in self.PROVIDER_MAP if k != "hf"})
            raise ValueError(
                f"Unknown LLM provider: {provider_key!r}. "
                f"Supported: {', '.join(canonical)}"
            )

        return provider_cls(self.settings).create(resolved_model, **kwargs)


def get_llm(
    provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs
) -> Any:
    """Create and return an LLM instance for the given provider.

    Creates a fresh LLMProviderFactory on each call so that Django settings
    are always read at call time rather than at import time.
    """
    return LLMProviderFactory().get(provider=provider, model_name=model_name, **kwargs)


__all__ = ["LLMProviderFactory", "get_llm", "BaseLLMProvider"]
