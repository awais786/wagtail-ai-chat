"""
Embedding Provider Factory for Wagtail RAG.

This module provides a factory function to create embedding instances from various providers
(OpenAI, HuggingFace, Cohere, Google, etc.)
"""
import logging

from django.conf import settings

# Setup logging
logger = logging.getLogger(__name__)


def get_embeddings(provider=None, model_name=None, **kwargs):
    """
    Factory function to create embedding instances from various providers.
    
    Args:
        provider: Embedding provider type ('huggingface', 'openai', 'cohere', 'google', etc.)
                 If None, uses WAGTAIL_RAG_EMBEDDING_PROVIDER setting or defaults to 'huggingface'
        model_name: Name of the embedding model to use
        **kwargs: Additional provider-specific arguments
    
    Returns:
        LangChain Embeddings instance
    """
    if provider is None:
        provider = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_PROVIDER', 'huggingface')
    
    provider = provider.lower()
    
    # Set default model based on provider if not specified
    if model_name is None:
        # Provider-specific defaults
        provider_defaults = {
            'huggingface': 'sentence-transformers/all-MiniLM-L6-v2',
            'hf': 'sentence-transformers/all-MiniLM-L6-v2',
            'openai': 'text-embedding-ada-002',
            'cohere': 'embed-english-v3.0',
            'google': 'models/embedding-001',
        }
        default_model = provider_defaults.get(provider)
        if default_model:
            model_name = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_MODEL', default_model)
        else:
            model_name = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_MODEL', None)
            if not model_name:
                raise ValueError(f"model_name must be specified for embedding provider '{provider}'")
    
    logger.info(f"Initializing embeddings with provider='{provider}', model='{model_name}'")
    
    # HuggingFace (default, local, free)
    if provider == 'huggingface' or provider == 'hf':
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
            except ImportError:
                try:
                    from langchain.embeddings import HuggingFaceEmbeddings
                    return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
                except ImportError:
                    raise ImportError(
                        "HuggingFace embeddings not available. "
                        "Install with: pip install langchain-huggingface or langchain-community"
                    )
    
    # OpenAI
    elif provider == 'openai':
        try:
            from langchain_openai import OpenAIEmbeddings
            api_key = kwargs.get('api_key') or getattr(settings, 'OPENAI_API_KEY', None)
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set in settings or passed as api_key")
            return OpenAIEmbeddings(model=model_name, api_key=api_key, **{k: v for k, v in kwargs.items() if k != 'api_key'})
        except ImportError:
            raise ImportError(
                "OpenAI embeddings not available. Install with: pip install langchain-openai"
            )
    
    # Cohere
    elif provider == 'cohere':
        try:
            from langchain_community.embeddings import CohereEmbeddings
            api_key = kwargs.get('api_key') or getattr(settings, 'COHERE_API_KEY', None)
            if not api_key:
                raise ValueError("COHERE_API_KEY must be set in settings or passed as api_key")
            return CohereEmbeddings(model=model_name, cohere_api_key=api_key, **{k: v for k, v in kwargs.items() if k != 'api_key'})
        except ImportError:
            raise ImportError(
                "Cohere embeddings not available. Install with: pip install langchain-community cohere"
            )
    
    # Google
    elif provider == 'google' or provider == 'gemini':
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = kwargs.get('api_key') or getattr(settings, 'GOOGLE_API_KEY', None)
            if not api_key:
                raise ValueError("GOOGLE_API_KEY must be set in settings or passed as api_key")
            return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key, **{k: v for k, v in kwargs.items() if k != 'api_key'})
        except ImportError:
            raise ImportError(
                "Google embeddings not available. Install with: pip install langchain-google-genai"
            )
    
    # Sentence Transformers (direct, local)
    elif provider == 'sentence_transformers' or provider == 'sentence-transformers':
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            # Sentence transformers models work through HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        except ImportError:
            raise ImportError(
                "Sentence transformers not available. Install with: pip install langchain-community sentence-transformers"
            )
    
    # Custom embeddings (pass a callable that returns an Embeddings instance)
    elif provider == 'custom':
        custom_emb_func = getattr(settings, 'WAGTAIL_RAG_CUSTOM_EMBEDDING_FACTORY', None)
        if not custom_emb_func:
            raise ValueError("WAGTAIL_RAG_CUSTOM_EMBEDDING_FACTORY must be set for custom provider")
        if callable(custom_emb_func):
            return custom_emb_func(model_name=model_name, **kwargs)
        else:
            raise ValueError("WAGTAIL_RAG_CUSTOM_EMBEDDING_FACTORY must be a callable")
    
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported providers: huggingface, openai, cohere, google, sentence_transformers, custom"
        )

