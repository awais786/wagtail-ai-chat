"""
LLM Provider Factory for Wagtail RAG.

This module provides a factory function to create LLM instances from various providers
(OpenAI, Anthropic, Ollama, HuggingFace, Google, Cohere, etc.)
"""
import logging

from django.conf import settings

# Setup logging
logger = logging.getLogger(__name__)


def get_llm(provider=None, model_name=None, **kwargs):
    """
    Factory function to create LLM instances from various providers.
    
    Args:
        provider: LLM provider type ('ollama', 'openai', 'anthropic', 'huggingface', etc.)
                 If None, uses WAGTAIL_RAG_LLM_PROVIDER setting or defaults to 'ollama'
        model_name: Name of the model to use
        **kwargs: Additional provider-specific arguments
    
    Returns:
        LangChain LLM/ChatModel instance
    """
    if provider is None:
        provider = getattr(settings, 'WAGTAIL_RAG_LLM_PROVIDER', 'ollama')
    
    provider = provider.lower()
    
    # Provider-specific defaults
    provider_defaults = {
        'ollama': 'mistral',
        'openai': 'gpt-4',
        'anthropic': 'claude-3-sonnet-20240229',
        'claude': 'claude-3-sonnet-20240229',
        'google': 'gemini-pro',
        'gemini': 'gemini-pro',
        'cohere': 'command',
        'huggingface': None,  # No default, must specify
        'hf': None,  # No default, must specify
    }
    default_model = provider_defaults.get(provider)
    
    # Model validation lists
    ollama_models = ['mistral', 'llama2', 'phi', 'gemma', 'codellama', 'neural-chat', 'llama', 'orca']
    openai_models = ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-3.5']
    
    # Set default model based on provider if not specified
    if model_name is None:
        if default_model:
            # Get model from settings, but if it's None, use provider default
            settings_model = getattr(settings, 'WAGTAIL_RAG_MODEL_NAME', None)
            
            # Check if settings model is appropriate for the provider
            if settings_model:
                # If provider is OpenAI but model is an Ollama model, use provider default
                if provider == 'openai' and settings_model.lower() in ollama_models:
                    logger.warning(
                        f"Model '{settings_model}' from settings is an Ollama model, but provider is 'openai'. "
                        f"Using provider default '{default_model}' instead. "
                        f"Set WAGTAIL_RAG_MODEL_NAME to 'gpt-4' or 'gpt-3.5-turbo' to use OpenAI."
                    )
                    model_name = default_model
                # If provider is Ollama but model is an OpenAI model, use provider default
                elif provider == 'ollama' and any(settings_model.lower().startswith(m) for m in openai_models):
                    logger.warning(
                        f"Model '{settings_model}' from settings is an OpenAI model, but provider is 'ollama'. "
                        f"Using provider default '{default_model}' instead. "
                        f"Set WAGTAIL_RAG_MODEL_NAME to 'mistral', 'llama2', etc. to use Ollama."
                    )
                    model_name = default_model
                else:
                    # Model seems appropriate for provider, use it
                    model_name = settings_model
            else:
                # No model in settings, use provider default
                model_name = default_model
        else:
            model_name = getattr(settings, 'WAGTAIL_RAG_MODEL_NAME', None)
            if not model_name:
                raise ValueError(f"model_name must be specified for provider '{provider}'")
    else:
        # model_name was explicitly passed - validate it matches the provider
        if provider == 'openai' and model_name.lower() in ollama_models:
            logger.warning(
                f"Model '{model_name}' is an Ollama model, but provider is 'openai'. "
                f"Using provider default '{default_model}' instead."
            )
            model_name = default_model
        elif provider == 'ollama' and any(model_name.lower().startswith(m) for m in openai_models):
            logger.warning(
                f"Model '{model_name}' is an OpenAI model, but provider is 'ollama'. "
                f"Using provider default '{default_model}' instead."
            )
            model_name = default_model
    
    logger.info(f"Initializing LLM with provider='{provider}', model='{model_name}'")
    
    # Ollama (default, local LLM)
    if provider == 'ollama':
        try:
            from langchain_community.llms import Ollama
            return Ollama(model=model_name, **kwargs)
        except ImportError:
            raise ImportError(
                "Ollama not available. Install with: pip install langchain-community ollama"
            )
    
    # OpenAI
    elif provider == 'openai':
        try:
            from langchain_openai import ChatOpenAI
            api_key = kwargs.get('api_key') or getattr(settings, 'OPENAI_API_KEY', None)
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set in settings or passed as api_key")
            if not model_name:
                raise ValueError("model_name must be specified for OpenAI (e.g., 'gpt-4', 'gpt-3.5-turbo')")
            return ChatOpenAI(model=model_name, api_key=api_key, **{k: v for k, v in kwargs.items() if k != 'api_key'})
        except ImportError:
            raise ImportError(
                "OpenAI not available. Install with: pip install langchain-openai"
            )
    
    # Anthropic (Claude)
    elif provider == 'anthropic' or provider == 'claude':
        try:
            from langchain_anthropic import ChatAnthropic
            api_key = kwargs.get('api_key') or getattr(settings, 'ANTHROPIC_API_KEY', None)
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY must be set in settings or passed as api_key")
            if not model_name:
                raise ValueError("model_name must be specified for Anthropic (e.g., 'claude-3-sonnet-20240229')")
            return ChatAnthropic(model=model_name, api_key=api_key, **{k: v for k, v in kwargs.items() if k != 'api_key'})
        except ImportError:
            raise ImportError(
                "Anthropic not available. Install with: pip install langchain-anthropic"
            )
    
    # HuggingFace (local or hosted)
    elif provider == 'huggingface' or provider == 'hf':
        try:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import pipeline
            # For local models
            if 'model_id' not in kwargs:
                kwargs['model_id'] = model_name
            pipe = pipeline("text-generation", model=kwargs['model_id'], **{k: v for k, v in kwargs.items() if k != 'model_id'})
            return HuggingFacePipeline(pipeline=pipe)
        except ImportError:
            try:
                # Try HuggingFace Hub endpoint
                from langchain_community.llms import HuggingFaceEndpoint
                endpoint_url = kwargs.get('endpoint_url') or getattr(settings, 'HUGGINGFACE_ENDPOINT_URL', None)
                api_key = kwargs.get('api_key') or getattr(settings, 'HUGGINGFACE_API_KEY', None)
                return HuggingFaceEndpoint(
                    endpoint_url=endpoint_url or f"https://api-inference.huggingface.co/models/{model_name}",
                    huggingfacehub_api_token=api_key,
                    **{k: v for k, v in kwargs.items() if k not in ['endpoint_url', 'api_key']}
                )
            except ImportError:
                raise ImportError(
                    "HuggingFace not available. Install with: pip install langchain-huggingface transformers"
                )
    
    # Google (Gemini)
    elif provider == 'google' or provider == 'gemini':
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = kwargs.get('api_key') or getattr(settings, 'GOOGLE_API_KEY', None)
            if not api_key:
                raise ValueError("GOOGLE_API_KEY must be set in settings or passed as api_key")
            if not model_name:
                raise ValueError("model_name must be specified for Google (e.g., 'gemini-pro')")
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, **{k: v for k, v in kwargs.items() if k != 'api_key'})
        except ImportError:
            raise ImportError(
                "Google Generative AI not available. Install with: pip install langchain-google-genai"
            )
    
    # Cohere
    elif provider == 'cohere':
        try:
            from langchain_community.llms import Cohere
            api_key = kwargs.get('api_key') or getattr(settings, 'COHERE_API_KEY', None)
            if not api_key:
                raise ValueError("COHERE_API_KEY must be set in settings or passed as api_key")
            if not model_name:
                raise ValueError("model_name must be specified for Cohere (e.g., 'command', 'command-light')")
            return Cohere(model=model_name, cohere_api_key=api_key, **{k: v for k, v in kwargs.items() if k != 'api_key'})
        except ImportError:
            raise ImportError(
                "Cohere not available. Install with: pip install langchain-community cohere"
            )
    
    # Custom LLM (pass a callable that returns an LLM instance)
    elif provider == 'custom':
        custom_llm_func = getattr(settings, 'WAGTAIL_RAG_CUSTOM_LLM_FACTORY', None)
        if not custom_llm_func:
            raise ValueError("WAGTAIL_RAG_CUSTOM_LLM_FACTORY must be set for custom provider")
        if callable(custom_llm_func):
            return custom_llm_func(model_name=model_name, **kwargs)
        else:
            raise ValueError("WAGTAIL_RAG_CUSTOM_LLM_FACTORY must be a callable")
    
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: ollama, openai, anthropic, huggingface, google, cohere, custom"
        )

