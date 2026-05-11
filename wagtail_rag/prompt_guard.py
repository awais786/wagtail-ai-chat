"""
Prompt injection guard for wagtail_rag.

Provides a lightweight pre-filter that detects common prompt-injection
patterns before the question reaches the LLM.

Design note
-----------
This module is intentionally thin so that it can be replaced or augmented
by a dedicated package (e.g. django-prompt-guard) in the future.

To plug in a custom backend set ``WAGTAIL_RAG_PROMPT_GUARD_BACKEND`` in
Django settings to a dotted-path string pointing at any callable with the
signature::

    def my_guard(question: str) -> GuardResult: ...

Example::

    WAGTAIL_RAG = {
        ...
        "api": {
            "prompt_guard_backend": "myapp.guards.django_prompt_guard_check",
        },
    }

    # or flat setting:
    WAGTAIL_RAG_PROMPT_GUARD_BACKEND = "myapp.guards.django_prompt_guard_check"
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from django.conf import settings as django_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    """Result of a prompt guard check."""

    blocked: bool = False
    reason: Optional[str] = None
    matched_patterns: list[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return not self.blocked


# ---------------------------------------------------------------------------
# Built-in injection patterns
# ---------------------------------------------------------------------------

# Each entry: (compiled_regex, human-readable label)
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "ignore / forget / disregard your instructions / system prompt"
    (
        re.compile(
            r"\b(ignore|forget|disregard|override|bypass)\b.{0,40}"
            r"\b(instructions?|rules?|system[\s_-]*prompt|context|guidelines?|above)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "override-instructions",
    ),
    # "pretend / act as / you are now / roleplay"
    (
        re.compile(
            r"\b(pretend|act\s+as|behave\s+as|you\s+are\s+now|role[\s_-]*play|"
            r"simulate\s+being|impersonate)\b",
            re.IGNORECASE,
        ),
        "role-hijack",
    ),
    # "reveal / show / print your system prompt / instructions"
    (
        re.compile(
            r"\b(reveal|show|print|output|display|leak|expose|tell\s+me|what\s+is)\b"
            r".{0,40}"
            r"\b(system[\s_-]*prompt|your\s+instructions?|your\s+prompt|hidden\s+instructions?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "prompt-exfiltration",
    ),
    # Marker injection: SYSTEM:, [SYSTEM], <|im_start|>system, ###, ---
    (
        re.compile(
            r"(SYSTEM\s*:|<\|im_start\|>\s*system|\[SYSTEM\]|###\s*(SYSTEM|INST|SYS)|"
            r"<<?/?SYS>>?|<<SYS>>|<</SYS>>)",
            re.IGNORECASE,
        ),
        "marker-injection",
    ),
    # Jailbreak keywords
    (
        re.compile(
            r"\b(DAN|jailbreak|developer[\s_-]*mode|god[\s_-]*mode|"
            r"unrestricted[\s_-]*mode|do\s+anything\s+now|no[\s_-]*restrictions)\b",
            re.IGNORECASE,
        ),
        "jailbreak-keyword",
    ),
    # Prompt boundary attempts: "---END---", "=== NEW PROMPT ==="
    (
        re.compile(
            r"(-{3,}|={3,}|\*{3,})\s*(END|NEW[\s_-]*PROMPT|CONTEXT[\s_-]*END|INSTRUCTION)",
            re.IGNORECASE,
        ),
        "boundary-injection",
    ),
    # Instruction override via new line tricks
    (
        re.compile(
            r"\bNew\s+instruction\b|\bActual\s+task\b|\bReal\s+task\b|"
            r"\bHidden\s+instruction\b|\bSecret\s+instruction\b",
            re.IGNORECASE,
        ),
        "instruction-override",
    ),
]

# ---------------------------------------------------------------------------
# Built-in guard
# ---------------------------------------------------------------------------


def _builtin_guard(question: str) -> GuardResult:
    """Check question against built-in injection pattern list."""
    matched = []
    for pattern, label in _PATTERNS:
        if pattern.search(question):
            matched.append(label)

    if matched:
        logger.warning(
            "prompt_guard: injection patterns detected | patterns=%s | question=%r",
            matched,
            question[:200],
        )
        return GuardResult(
            blocked=True,
            reason="Your message contains content that cannot be processed.",
            matched_patterns=matched,
        )

    return GuardResult(blocked=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _get_backend() -> Callable[[str], GuardResult]:
    """
    Return the active guard callable.

    Checks (in order):
    1. ``WAGTAIL_RAG["api"]["prompt_guard_backend"]``
    2. Flat ``WAGTAIL_RAG_PROMPT_GUARD_BACKEND`` setting
    3. Built-in pattern-based guard (default)
    """
    try:
        rag_conf = getattr(django_settings, "WAGTAIL_RAG", {}) or {}
        dotted_path = (
            (rag_conf.get("api") or {}).get("prompt_guard_backend")
            or getattr(django_settings, "WAGTAIL_RAG_PROMPT_GUARD_BACKEND", None)
        )
        if dotted_path:
            from django.utils.module_loading import import_string

            return import_string(dotted_path)
    except Exception:
        pass  # Django not configured or setting missing — fall back to built-in

    return _builtin_guard


def check_question(question: str) -> GuardResult:
    """
    Run the configured guard against *question*.

    Returns a :class:`GuardResult`.  When ``result.blocked`` is ``True``
    the question must NOT be forwarded to the LLM.

    This function never raises — errors are logged and a safe (non-blocking)
    result is returned so a misconfigured guard does not break the chat.
    """
    try:
        backend = _get_backend()
        return backend(question)
    except Exception:
        logger.exception(
            "prompt_guard: unexpected error; allowing question through"
        )
        return GuardResult(blocked=False)
