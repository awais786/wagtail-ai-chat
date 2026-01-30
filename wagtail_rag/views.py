"""
API views for RAG chatbot.

This module provides API endpoints for querying the RAG chatbot.
"""
import json
import logging
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

        if not question:
            return JsonResponse(
                {"error": "Question is required. Use 'q' for GET or 'question' for POST."},
                status=400,
            )

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

    except Exception:
        logger.exception("RAG chat API error")
        return JsonResponse({"error": "An error occurred processing your request."}, status=500)


def rag_chatbox_widget(request: HttpRequest) -> HttpResponse:
    """Serve the RAG chatbox widget HTML page."""
    return render(request, "wagtail_rag/chatbox.html")
