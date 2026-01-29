"""
LLM Provider Factory for Wagtail RAG.

This module provides a factory function to create LLM instances from various providers
(OpenAI, Anthropic, Ollama, HuggingFace, Google, Cohere, etc.)
"""
import logging

from django.conf import settings

logger = logging.getLogger(__name__)

# Provider-specific default models
PROVIDER_DEFAULTS = {
    'ollama': 'mistral',
    'openai': 'gpt-4',
    'anthropic': 'claude-3-sonnet-20240229',
    'claude': 'claude-3-sonnet-20240229',
    'google': 'gemini-pro',
    'gemini': 'gemini-pro',
    'cohere': 'command',
    'huggingface': None,
    'hf': None,
}


def _get_model_name(provider, model_name):
    """
    Resolve the model name to use based on provider and settings.

    Args:
        provider: The LLM provider name
        model_name: Explicitly specified model name (or None)

    Returns:
        str: The resolved model name
    """
    # If model_name is explicitly provided, use it
    if model_name:
        return model_name

    # Try to get from settings
    settings_model = getattr(settings, 'WAGTAIL_RAG_MODEL_NAME', None)
    if settings_model:
        return settings_model

    # Fall back to provider default
    default_model = PROVIDER_DEFAULTS.get(provider)
    if not default_model:
        raise ValueError(f"model_name must be specified for provider '{provider}'")

    return default_model


def get_llm(provider=None, model_name=None, **kwargs):
    """
    Factory function to create LLM instances from various providers.

    Args:
        provider: LLM provider type ('ollama', 'openai', 'anthropic', 'huggingface', etc.)
                 If None, uses WAGTAIL_RAG_LLM_PROVIDER setting or defaults to 'ollama'
        model_name: Name of the model to use. If None, uses WAGTAIL_RAG_MODEL_NAME setting
                   or provider default
        **kwargs: Additional provider-specific arguments (e.g., api_key, temperature)

    Returns:
        LangChain LLM/ChatModel instance

    Raises:
        ValueError: If provider is unknown or required configuration is missing
        ImportError: If required dependencies are not installed
    """
    provider = (provider or getattr(settings, 'WAGTAIL_RAG_LLM_PROVIDER', 'ollama')).lower()
    model_name = _get_model_name(provider, model_name)

    logger.info(f"Initializing LLM with provider='{provider}', model='{model_name}'")
    
    # Ollama (default, local LLM)
    if provider == 'ollama':
        try:
            from langchain_community.llms import Ollama
            return Ollama(model=model_name, **kwargs)
        except ImportError as e:
            raise ImportError(
                "Ollama not available. Install with: pip install langchain-community ollama"
            ) from e

    # OpenAI
    elif provider == 'openai':
        try:
            from langchain_openai import ChatOpenAI
            api_key = kwargs.pop('api_key', None) or getattr(settings, 'OPENAI_API_KEY', None)
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set in settings or passed as api_key")
            return ChatOpenAI(model=model_name, api_key=api_key, **kwargs)
        except ImportError as e:
            raise ImportError(
                "OpenAI not available. Install with: pip install langchain-openai"
            ) from e

    # Anthropic (Claude)
    elif provider in ('anthropic', 'claude'):
        try:
            from langchain_anthropic import ChatAnthropic
            api_key = kwargs.pop('api_key', None) or getattr(settings, 'ANTHROPIC_API_KEY', None)
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY must be set in settings or passed as api_key")
            return ChatAnthropic(model=model_name, api_key=api_key, **kwargs)
        except ImportError as e:
            raise ImportError(
                "Anthropic not available. Install with: pip install langchain-anthropic"
            ) from e

    # HuggingFace (local or hosted)
    elif provider in ('huggingface', 'hf'):
        try:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import pipeline

            model_id = kwargs.pop('model_id', model_name)
            pipe = pipeline("text-generation", model=model_id, **kwargs)
            return HuggingFacePipeline(pipeline=pipe)
        except ImportError:
            try:
                from langchain_community.llms import HuggingFaceEndpoint
                endpoint_url = kwargs.pop('endpoint_url', None) or getattr(settings, 'HUGGINGFACE_ENDPOINT_URL', None)
                api_key = kwargs.pop('api_key', None) or getattr(settings, 'HUGGINGFACE_API_KEY', None)

                if not endpoint_url:
                    endpoint_url = f"https://api-inference.huggingface.co/models/{model_name}"

                return HuggingFaceEndpoint(
                    endpoint_url=endpoint_url,
                    huggingfacehub_api_token=api_key,
                    **kwargs
                )
            except ImportError as e:
                raise ImportError(
                    "HuggingFace not available. Install with: pip install langchain-huggingface transformers"
                ) from e

    # Google (Gemini)
    elif provider in ('google', 'gemini'):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = kwargs.pop('api_key', None) or getattr(settings, 'GOOGLE_API_KEY', None)
            if not api_key:
                raise ValueError("GOOGLE_API_KEY must be set in settings or passed as api_key")
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, **kwargs)
        except ImportError as e:
            raise ImportError(
                "Google Generative AI not available. Install with: pip install langchain-google-genai"
            ) from e

    # Cohere
    elif provider == 'cohere':
        try:
            from langchain_community.llms import Cohere
            api_key = kwargs.pop('api_key', None) or getattr(settings, 'COHERE_API_KEY', None)
            if not api_key:
                raise ValueError("COHERE_API_KEY must be set in settings or passed as api_key")
            return Cohere(model=model_name, cohere_api_key=api_key, **kwargs)
        except ImportError as e:
            raise ImportError(
                "Cohere not available. Install with: pip install langchain-community cohere"
            ) from e

    # Custom LLM (pass a callable that returns an LLM instance)
    elif provider == 'custom':
        custom_llm_func = getattr(settings, 'WAGTAIL_RAG_CUSTOM_LLM_FACTORY', None)
        if not custom_llm_func:
            raise ValueError("WAGTAIL_RAG_CUSTOM_LLM_FACTORY must be set for custom provider")
        if not callable(custom_llm_func):
            raise ValueError("WAGTAIL_RAG_CUSTOM_LLM_FACTORY must be a callable")
        return custom_llm_func(model_name=model_name, **kwargs)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: ollama, openai, anthropic, huggingface, google, cohere, custom"
        )

