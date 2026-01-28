"""
Embedding Provider Factory for Wagtail RAG.

This module provides a factory function to create embedding instances from various providers
(OpenAI, HuggingFace, Cohere, Google, etc.)
"""
import logging

from django.conf import settings

logger = logging.getLogger(__name__)

# Provider-specific default models
PROVIDER_DEFAULTS = {
    'huggingface': 'sentence-transformers/all-MiniLM-L6-v2',
    'hf': 'sentence-transformers/all-MiniLM-L6-v2',
    'openai': 'text-embedding-ada-002',
    'cohere': 'embed-english-v3.0',
    'google': 'models/embedding-001',
    'gemini': 'models/embedding-001',
    'sentence_transformers': 'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers': 'sentence-transformers/all-MiniLM-L6-v2',
}


def _get_model_name(provider, model_name):
    """
    Resolve the embedding model name to use based on provider and settings.

    Args:
        provider: The embedding provider name
        model_name: Explicitly specified model name (or None)

    Returns:
        str: The resolved model name
    """
    # If model_name is explicitly provided, use it
    if model_name:
        return model_name

    # Try to get from settings
    settings_model = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_MODEL', None)
    if settings_model:
        return settings_model

    # Fall back to provider default
    default_model = PROVIDER_DEFAULTS.get(provider)
    if not default_model:
        raise ValueError(f"model_name must be specified for embedding provider '{provider}'")

    return default_model


def get_embeddings(provider=None, model_name=None, **kwargs):
    """
    Factory function to create embedding instances from various providers.
    
    Args:
        provider: Embedding provider type ('huggingface', 'openai', 'cohere', 'google', etc.)
                 If None, uses WAGTAIL_RAG_EMBEDDING_PROVIDER setting or defaults to 'huggingface'
        model_name: Name of the embedding model to use. If None, uses WAGTAIL_RAG_EMBEDDING_MODEL
                   setting or provider default
        **kwargs: Additional provider-specific arguments (e.g., api_key)

    Returns:
        LangChain Embeddings instance

    Raises:
        ValueError: If provider is unknown or required configuration is missing
        ImportError: If required dependencies are not installed
    """
    provider = (provider or getattr(settings, 'WAGTAIL_RAG_EMBEDDING_PROVIDER', 'huggingface')).lower()
    model_name = _get_model_name(provider, model_name)

    logger.info(f"Initializing embeddings with provider='{provider}', model='{model_name}'")
    
    # HuggingFace (default, local, free)
    if provider in ('huggingface', 'hf', 'sentence_transformers', 'sentence-transformers'):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError:
                try:
                    from langchain.embeddings import HuggingFaceEmbeddings
                except ImportError as e:
                    raise ImportError(
                        "HuggingFace embeddings not available. "
                        "Install with: pip install langchain-huggingface sentence-transformers\n"
                        "Or alternatively: pip install langchain-community sentence-transformers"
                    ) from e
        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)

    # OpenAI
    elif provider == 'openai':
        try:
            from langchain_openai import OpenAIEmbeddings
            api_key = kwargs.pop('api_key', None) or getattr(settings, 'OPENAI_API_KEY', None)
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set in settings or passed as api_key")
            return OpenAIEmbeddings(model=model_name, api_key=api_key, **kwargs)
        except ImportError as e:
            raise ImportError(
                "OpenAI embeddings not available. Install with: pip install langchain-openai"
            ) from e

    # Cohere
    elif provider == 'cohere':
        try:
            from langchain_community.embeddings import CohereEmbeddings
            api_key = kwargs.pop('api_key', None) or getattr(settings, 'COHERE_API_KEY', None)
            if not api_key:
                raise ValueError("COHERE_API_KEY must be set in settings or passed as api_key")
            return CohereEmbeddings(model=model_name, cohere_api_key=api_key, **kwargs)
        except ImportError as e:
            raise ImportError(
                "Cohere embeddings not available. Install with: pip install langchain-community cohere"
            ) from e

    # Google (Gemini)
    elif provider in ('google', 'gemini'):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = kwargs.pop('api_key', None) or getattr(settings, 'GOOGLE_API_KEY', None)
            if not api_key:
                raise ValueError("GOOGLE_API_KEY must be set in settings or passed as api_key")
            return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key, **kwargs)
        except ImportError as e:
            raise ImportError(
                "Google embeddings not available. Install with: pip install langchain-google-genai"
            ) from e

    # Custom embeddings (pass a callable that returns an Embeddings instance)
    elif provider == 'custom':
        custom_emb_func = getattr(settings, 'WAGTAIL_RAG_CUSTOM_EMBEDDING_FACTORY', None)
        if not custom_emb_func:
            raise ValueError("WAGTAIL_RAG_CUSTOM_EMBEDDING_FACTORY must be set for custom provider")
        if not callable(custom_emb_func):
            raise ValueError("WAGTAIL_RAG_CUSTOM_EMBEDDING_FACTORY must be a callable")
        return custom_emb_func(model_name=model_name, **kwargs)

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported providers: huggingface, openai, cohere, google, sentence_transformers, custom"
        )

