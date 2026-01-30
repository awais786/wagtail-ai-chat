"""
API views for RAG chatbot.

This module provides API endpoints for querying the RAG chatbot.
"""
import json
import logging

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .rag_chatbot import get_chatbot

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def rag_chat_api(request):
    """
    API endpoint for RAG chatbot queries.
    
    Supports both GET and POST methods for browser-based access.
    
    GET parameters:
        q: Question (required)
        filter: JSON string for metadata filter (optional, e.g., '{"model": "BreadPage"}')
    
    POST data:
    {
        "question": "What types of bread do you have?",
        "filter": {"model": "BreadPage"},  # optional metadata filter
        "llm_kwargs": {"temperature": 0.7}  # optional LLM-specific parameters
    }
    
    Note: LLM provider and model come from Django settings (WAGTAIL_RAG_LLM_PROVIDER, WAGTAIL_RAG_MODEL_NAME)
    
    Returns:
    {
        "answer": "...",
        "sources": [...]
    }
    """
    try:
        # Handle both GET and POST
        if request.method == 'GET':
            question = request.GET.get('q', '').strip()
            filter_str = request.GET.get('filter', '')
            metadata_filter = None
            if filter_str:
                try:
                    metadata_filter = json.loads(filter_str)
                except json.JSONDecodeError:
                    metadata_filter = None
            llm_kwargs = {}
        else:  # POST
            data = json.loads(request.body)
            question = data.get('question', '').strip()
            metadata_filter = data.get('filter')  # Optional metadata filter
            llm_kwargs = data.get('llm_kwargs', {})  # Optional LLM-specific kwargs
        
        if not question:
            return JsonResponse({'error': 'Question is required. Use "q" parameter for GET or "question" for POST.'}, status=400)
        
        # Use settings for LLM provider and model - no need to pass them in request
        llm_provider = getattr(settings, 'WAGTAIL_RAG_LLM_PROVIDER', 'ollama')
        llm_model = getattr(settings, 'WAGTAIL_RAG_MODEL_NAME', None) or 'default'

        logger.info(
            "wagtail_rag.chat API called | question=%r | llm=%s/%s",
            question,
            llm_provider,
            llm_model,
        )

        chatbot = get_chatbot(
            metadata_filter=metadata_filter,
            llm_kwargs=llm_kwargs
        )
        # This calls the LLM under the hood (via the RAG pipeline)
        result = chatbot.query(question)
        
        return JsonResponse(result)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def rag_chatbox_widget(request):
    """
    Serve the RAG chatbox widget HTML page.
    """
    return render(request, 'wagtail_rag/chatbox.html')
