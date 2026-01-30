"""
API views for RAG chatbot.

This module provides API endpoints for querying the RAG chatbot.
"""
import json
import logging
import re
from typing import Optional

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .rag_chatbot import get_chatbot

logger = logging.getLogger(__name__)

# Max POST body size (1MB) to avoid DoS from huge payloads
MAX_REQUEST_BODY_SIZE = getattr(settings, "WAGTAIL_RAG_MAX_REQUEST_BODY_SIZE", 1024 * 1024)

# Max question length to prevent abuse
MAX_QUESTION_LENGTH = getattr(settings, "WAGTAIL_RAG_MAX_QUESTION_LENGTH", 1000)


def validate_question(question: str) -> tuple[bool, Optional[str]]:
    """
    Validate user question input.
    
    Args:
        question: User question string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not question or not question.strip():
        return False, "Question cannot be empty"
    
    if len(question) > MAX_QUESTION_LENGTH:
        return False, f"Question too long (max {MAX_QUESTION_LENGTH} characters)"
    
    # Check for suspicious patterns (basic protection)
    # Note: LLM providers have their own prompt injection protection
    suspicious_patterns = [
        r'<script[^>]*>',  # Script tags
        r'javascript:',     # JavaScript URLs
        r'on\w+\s*=',      # Event handlers
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            logger.warning(f"Suspicious pattern detected in question: {pattern}")
            # Don't reject, just log - LLM will handle safely
    
    return True, None



@csrf_exempt
@require_http_methods(["GET", "POST"])
def rag_chat_api(request: HttpRequest) -> JsonResponse:
    """
    API endpoint for RAG chatbot queries.

    CSRF is exempt because this endpoint is intended for programmatic use (e.g. external
    clients, scripts, or non-Django frontends) that do not send Django's CSRF token.
    Protect the endpoint at the network/gateway level (e.g. auth, rate limiting) if needed.

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
        if request.method == "GET":
            question = (request.GET.get("q") or "").strip()
            filter_str = request.GET.get("filter", "")
            metadata_filter: Optional[dict] = None
            if filter_str:
                try:
                    metadata_filter = json.loads(filter_str)
                except json.JSONDecodeError:
                    metadata_filter = None
            llm_kwargs: dict = {}
        else:
            body = request.body
            if len(body) > MAX_REQUEST_BODY_SIZE:
                return JsonResponse(
                    {"error": f"Request body too large (max {MAX_REQUEST_BODY_SIZE} bytes)."},
                    status=413,
                )
            if not body or not body.strip():
                return JsonResponse(
                    {"error": "POST body must be non-empty JSON with a 'question' field."},
                    status=400,
                )
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                return JsonResponse({"error": f"Invalid JSON: {e}"}, status=400)
            question = (data.get("question") or "").strip()
            metadata_filter = data.get("filter")
            llm_kwargs = data.get("llm_kwargs") or {}

        # Validate question input
        is_valid, error_msg = validate_question(question)
        if not is_valid:
            return JsonResponse({"error": error_msg}, status=400)

        llm_provider = getattr(settings, "WAGTAIL_RAG_LLM_PROVIDER", "ollama")
        llm_model = getattr(settings, "WAGTAIL_RAG_MODEL_NAME", None) or "default"

        logger.info(
            "wagtail_rag.chat API called | question=%r | llm=%s/%s",
            question[:200],
            llm_provider,
            llm_model,
        )

        chatbot = get_chatbot(metadata_filter=metadata_filter, llm_kwargs=llm_kwargs)
        result = chatbot.query(question)
        return JsonResponse(result)

    except ValueError as e:
        # Configuration or input validation errors
        logger.warning("RAG chat API validation error: %s", str(e))
        return JsonResponse({"error": f"Invalid input: {str(e)}"}, status=400)
    except ImportError as e:
        # Missing dependencies
        logger.error("RAG chat API dependency error: %s", str(e))
        return JsonResponse({"error": "Service configuration error. Please contact support."}, status=500)
    except Exception:
        # Unexpected errors - log details but return generic message
        logger.exception("RAG chat API unexpected error")
        return JsonResponse({"error": "An unexpected error occurred. Please try again later."}, status=500)


def rag_chatbox_widget(request: HttpRequest) -> HttpResponse:
    """Serve the RAG chatbox widget HTML page."""
    return render(request, "wagtail_rag/chatbox.html")
