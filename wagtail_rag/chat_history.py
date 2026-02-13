"""
Server-side chat history with summarization for multi-turn conversations.

Keeps the most recent messages verbatim and summarizes older turns into a
single system message to keep context compact.
"""

from __future__ import annotations

import threading
import uuid
from typing import Callable, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage


SummarizeFn = Callable[[str, str], str]


def _format_messages(messages: List[BaseMessage]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = getattr(msg, "type", "") or "user"
        content = getattr(msg, "content", "") or ""
        if not content:
            continue
        label = (
            "User"
            if role == "human"
            else "Assistant" if role == "ai" else role.capitalize()
        )
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


class SummarizingChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that summarizes older turns."""

    def __init__(self, summarize_fn: SummarizeFn, recent_window: int) -> None:
        self._summarize_fn = summarize_fn
        self._recent_window = max(int(recent_window), 0)
        self._summary: str = ""
        self._messages: List[BaseMessage] = []
        self._lock = threading.Lock()

    @property
    def messages(self) -> List[BaseMessage]:
        """Return history as messages, including a summary system message."""
        with self._lock:
            items: List[BaseMessage] = []
            if self._summary:
                items.append(
                    SystemMessage(
                        content=f"Summary of earlier conversation:\n{self._summary}"
                    )
                )
            items.extend(self._messages)
            return list(items)

    def add_message(self, message: BaseMessage) -> None:
        """Append a message and summarize older turns beyond the recent window."""
        with self._lock:
            self._messages.append(message)
            if self._recent_window <= 0:
                return
            if len(self._messages) <= self._recent_window:
                return
            older = self._messages[: -self._recent_window]
            older_text = _format_messages(older)
            if older_text:
                try:
                    self._summary = self._summarize_fn(self._summary, older_text)
                except Exception:
                    # If summarization fails, keep existing summary and just drop older turns.
                    pass
            self._messages = self._messages[-self._recent_window :]

    def clear(self) -> None:
        """Clear all stored messages and summary."""
        with self._lock:
            self._summary = ""
            self._messages = []


class SummarizingHistoryStore:
    """In-memory store for per-session chat histories."""

    def __init__(self, summarize_fn: SummarizeFn, recent_window: int) -> None:
        self._summarize_fn = summarize_fn
        self._recent_window = recent_window
        self._store: Dict[str, SummarizingChatMessageHistory] = {}
        self._lock = threading.Lock()

    def new_session_id(self) -> str:
        """Create a new session identifier."""
        return uuid.uuid4().hex

    def get_session_history(self, session_id: str) -> SummarizingChatMessageHistory:
        """Return the history for a session, creating it if needed."""
        with self._lock:
            history = self._store.get(session_id)
            if history is None:
                history = SummarizingChatMessageHistory(
                    summarize_fn=self._summarize_fn,
                    recent_window=self._recent_window,
                )
                self._store[session_id] = history
            return history


_HISTORY_STORE: Optional[SummarizingHistoryStore] = None


def get_history_store(
    summarize_fn: SummarizeFn, recent_window: int
) -> SummarizingHistoryStore:
    """Return a singleton history store."""
    global _HISTORY_STORE
    if _HISTORY_STORE is None:
        _HISTORY_STORE = SummarizingHistoryStore(
            summarize_fn=summarize_fn,
            recent_window=recent_window,
        )
    return _HISTORY_STORE


__all__ = [
    "SummarizingChatMessageHistory",
    "SummarizingHistoryStore",
    "get_history_store",
]
