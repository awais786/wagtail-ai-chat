"""
API views for RAG chatbot.

This module provides API endpoints for querying the RAG chatbot.
"""
import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .rag_chatbot import get_chatbot


@csrf_exempt
@require_http_methods(["POST"])
def rag_chat_api(request):
    """
    API endpoint for RAG chatbot queries.
    
    Expected POST data:
    {
        "question": "What types of bread do you have?",
        "model": "mistral",  # optional, defaults to mistral
        "filter": {"model": "BreadPage"}  # optional metadata filter
    }
    
    Returns:
    {
        "answer": "...",
        "sources": [...]
    }
    """
    try:
        data = json.loads(request.body)
        question = data.get('question', '')
        model_name = data.get('model', None)  # Use None to get from settings
        metadata_filter = data.get('filter')  # Optional metadata filter
        
        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)
        
        chatbot = get_chatbot(
            model_name=model_name,
            metadata_filter=metadata_filter
        )
        result = chatbot.query(question)
        
        return JsonResponse(result)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

