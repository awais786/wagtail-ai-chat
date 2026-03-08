"""
API views for RAG chatbot.
"""

import json
import logging
import uuid
from typing import Optional

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods

from .chatbot import get_chatbot

logger = logging.getLogger(__name__)

# Whitelist of llm_kwargs keys callers may send.
# Arbitrary kwargs could be used to probe internals or cause unexpected behaviour.
_ALLOWED_LLM_KWARGS = {"temperature", "max_tokens", "top_p", "top_k", "timeout"}


def _settings_int(name: str, default: int) -> int:
    return int(getattr(settings, name, default))


def _validate_metadata_filter(value) -> Optional[dict]:
    """Return value if it is a non-empty dict, otherwise None."""
    if isinstance(value, dict) and value:
        return value
    return None


def _sanitize_llm_kwargs(raw) -> dict:
    """Strip any keys not in _ALLOWED_LLM_KWARGS."""
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k in _ALLOWED_LLM_KWARGS}


@ensure_csrf_cookie
@require_http_methods(["GET", "POST"])
def rag_chat_api(request: HttpRequest) -> JsonResponse:
    """
    Chat API endpoint.

    CSRF protection is enforced for POST requests — the client must include the
    Django CSRF token either as the X-CSRFToken request header or as the
    csrfmiddlewaretoken POST field.  GET requests are CSRF-safe by definition.

    The embedded chatbox widget reads the csrftoken cookie and sends it
    automatically.  External / programmatic clients should first GET any page
    (which sets the cookie via ensure_csrf_cookie) and then mirror the token.

    Protect this endpoint at the network level (auth, rate limiting) if it is
    publicly exposed.

    GET  ?q=<question>[&session_id=<id>][&filter=<json>][&search_only=true]
    POST {"question": "...", "session_id": "...", "filter": {}, "llm_kwargs": {}, "search_only": false}

    Response 200: {"answer": "...", "sources": [...], "session_id": "..."}
    Response 400: {"error": "..."}
    Response 413: {"error": "..."}
    Response 415: {"error": "..."}  POST without Content-Type: application/json
    Response 500: {"error": "..."}
    """
    # Read limits at request time so settings changes take effect without restart.
    max_body = _settings_int("WAGTAIL_RAG_MAX_REQUEST_BODY_SIZE", 1024 * 1024)
    max_q_len = _settings_int("WAGTAIL_RAG_MAX_QUESTION_LENGTH", 0)
    use_history = getattr(settings, "WAGTAIL_RAG_ENABLE_CHAT_HISTORY", True)

    try:
        if request.method == "GET":
            question = (request.GET.get("q") or "").strip()
            session_id = (request.GET.get("session_id") or "").strip() or None
            search_only = (request.GET.get("search_only") or "").lower() in (
                "true",
                "1",
                "yes",
            )
            llm_kwargs: dict = {}

            filter_str = request.GET.get("filter", "")
            metadata_filter: Optional[dict] = None
            if filter_str:
                try:
                    metadata_filter = _validate_metadata_filter(json.loads(filter_str))
                except (json.JSONDecodeError, ValueError):
                    pass  # treat invalid filter as no filter

        else:  # POST
            ct = request.content_type or ""
            if "application/json" not in ct:
                return JsonResponse(
                    {"error": "Content-Type must be application/json."},
                    status=415,
                )
            body = request.body
            if len(body) > max_body:
                return JsonResponse(
                    {"error": f"Request body too large (max {max_body} bytes)."},
                    status=413,
                )
            if not body or not body.strip():
                return JsonResponse(
                    {
                        "error": "POST body must be non-empty JSON with a 'question' field."
                    },
                    status=400,
                )
            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                return JsonResponse({"error": f"Invalid JSON: {exc}"}, status=400)

            if not isinstance(data, dict):
                return JsonResponse(
                    {"error": "POST body must be a JSON object."}, status=400
                )

            question = (data.get("question") or "").strip()
            session_id = (data.get("session_id") or "").strip() or None
            search_only = bool(data.get("search_only", False))
            metadata_filter = _validate_metadata_filter(data.get("filter"))
            llm_kwargs = _sanitize_llm_kwargs(data.get("llm_kwargs"))

        # ── validate question ─────────────────────────────────────────
        if not question:
            return JsonResponse(
                {
                    "error": "Question is required. Use 'q' for GET or 'question' for POST."
                },
                status=400,
            )

        if max_q_len and len(question) > max_q_len:
            return JsonResponse(
                {"error": f"Question too long (max {max_q_len} characters)."},
                status=400,
            )

        # ── session ───────────────────────────────────────────────────
        if use_history and not session_id:
            session_id = uuid.uuid4().hex

        # ── query ─────────────────────────────────────────────────────
        logger.info(
            "rag_chat_api | question=%r | search_only=%s | session=%s",
            question[:200],
            search_only,
            session_id,
        )

        chatbot = get_chatbot(
            metadata_filter=metadata_filter,
            llm_kwargs=llm_kwargs if llm_kwargs else {},
        )
        result = chatbot.query(question, session_id=session_id, search_only=search_only)

        if use_history and session_id:
            result["session_id"] = session_id

        return JsonResponse(result)

    except Exception:
        logger.exception("RAG chat API error")
        return JsonResponse(
            {"error": "An error occurred processing your request."},
            status=500,
        )


@ensure_csrf_cookie
def rag_chatbox_widget(request: HttpRequest) -> HttpResponse:
    """Serve the RAG chatbox widget as a standalone page (for testing).

    @ensure_csrf_cookie guarantees the csrftoken cookie is set on this response
    so the embedded JS can read it for subsequent POST requests.
    """
    return render(request, "wagtail_rag/chatbox.html")
