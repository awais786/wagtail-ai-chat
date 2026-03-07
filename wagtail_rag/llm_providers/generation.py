"""
LLM Generation Module for RAG Chatbot.

This module handles LLM (Large Language Model) generation functionality,
including prompt construction and answer generation with proper context handling.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from django.conf import settings

from wagtail_rag.chat_history import get_history_store

logger = logging.getLogger(__name__)

# Import LangChain components with simplified detection
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage
    from langchain_core.language_models.chat_models import BaseChatModel

    LCEL_AVAILABLE = True
except ImportError:
    # Set to None for graceful degradation
    ChatPromptTemplate = MessagesPlaceholder = RunnablePassthrough = None
    RunnableWithMessageHistory = StrOutputParser = HumanMessage = BaseChatModel = None
    LCEL_AVAILABLE = False

try:
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    LEGACY_AVAILABLE = True
except ImportError:
    PromptTemplate = RetrievalQA = None
    LEGACY_AVAILABLE = False


class LLMGenerator:
    """
    Handles LLM-based answer generation for RAG chatbot.

    This class encapsulates all LLM generation logic including:
    - Prompt template construction
    - QA chain setup (LCEL or legacy)
    - Answer generation with context
    """

    def __init__(self, llm: Any, retriever: Optional[Any] = None):
        """Initialize the generator.

        Args:
            llm: A callable or LangChain LLM object.
            retriever: Optional retriever used by QA chains.
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt_template_str = self._get_prompt_template()
        self.system_prompt_str = self._get_system_prompt()
        self.history_enabled = getattr(
            settings, "WAGTAIL_RAG_ENABLE_CHAT_HISTORY", True
        )
        self.history_recent_window = int(
            getattr(settings, "WAGTAIL_RAG_CHAT_HISTORY_RECENT_MESSAGES", 6)
        )
        self.history_store = (
            get_history_store(self._summarize_history, self.history_recent_window)
            if self.history_enabled
            else None
        )
        self.qa_chain = self._create_qa_chain()
        self.history_chain = self._create_history_chain()

    def _get_prompt_template(self) -> str:
        """Return prompt template optimized for structured content."""
        return getattr(
            settings,
            "WAGTAIL_RAG_PROMPT_TEMPLATE",
            """You are an expert assistant helping users find information from website content. Analyze the provided context carefully and provide accurate, helpful answers.

**Guidelines:**
1. Use ONLY information from the context provided below
2. When citing information, reference the source page title or section
3. For structured content (like FAQ sections, numbered lists, or procedures), maintain the original organization
4. If content has sections or categories, organize your response accordingly
5. If you cannot find relevant information in the context, clearly state: "I don't have enough information in the available content to answer that question."
6. For multi-part questions, address each part systematically
7. Preserve important formatting like bullet points, numbers, or hierarchical structure when relevant

**Context:**
{context}

**Question:**
{question}

**Answer:**""",
        )

    def _get_system_prompt(self) -> str:
        """System prompt optimized for structured content handling."""
        return getattr(
            settings,
            "WAGTAIL_RAG_SYSTEM_PROMPT",
            """You are a knowledgeable assistant for a website. Answer questions using ONLY the provided context. Maintain the structure and organization of the source content when relevant. Always cite sources and clearly state if information is not available.""",
        )

    def _create_qa_chain(self) -> Optional[Any]:
        """Attempt to construct a QA chain using available LangChain interfaces.

        Returns the constructed chain or None if no retriever or supported chain is available.
        """
        if not self.retriever:
            return None

        if LCEL_AVAILABLE:
            try:
                return self._create_lcel_chain()
            except Exception:
                logger.exception(
                    "Failed to create LCEL QA chain; falling back to legacy if available"
                )

        if LEGACY_AVAILABLE:
            try:
                return self._create_legacy_chain()
            except Exception:
                logger.exception("Failed to create legacy RetrievalQA chain")

        return None

    def _create_history_chain(self) -> Optional[Any]:
        """Create a chat chain wrapped with RunnableWithMessageHistory."""
        if not (
            LCEL_AVAILABLE
            and ChatPromptTemplate
            and MessagesPlaceholder
            and StrOutputParser
            and RunnableWithMessageHistory
            and self.history_store
        ):
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_str),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()  # type: ignore[operator]
        return RunnableWithMessageHistory(  # type: ignore[call-arg]
            chain,
            self.history_store.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _create_lcel_chain(self) -> Any:
        """Create an LCEL-style chain using langchain_core primitives.

        This will raise ImportError if required components are missing.
        """
        if not (ChatPromptTemplate and RunnablePassthrough and StrOutputParser):
            raise ImportError("LCEL primitives are not available")

        prompt = ChatPromptTemplate.from_template(self.prompt_template_str)  # type: ignore[attr-defined]

        def format_docs(
            docs: List[Any],
        ) -> str:  # simple formatter used in runnable composition
            return "\n\n".join(getattr(d, "page_content", "") for d in docs)

        # Compose runnables: retriever -> format_docs -> prompt -> llm -> parser
        return (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()  # type: ignore[call-arg]
        )

    def _create_legacy_chain(self) -> Any:
        """Create a legacy RetrievalQA chain (langchain < 0.1 style)."""
        if not (PromptTemplate and RetrievalQA):
            raise ImportError("Legacy RetrievalQA components are not available")

        prompt = PromptTemplate(template=self.prompt_template_str, input_variables=["context", "question"])  # type: ignore[call-arg]
        return RetrievalQA.from_chain_type(  # type: ignore[attr-defined]
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def _get_context_from_docs(self, docs: List[Any]) -> str:
        """Build context string from docs, optionally truncated by WAGTAIL_RAG_MAX_CONTEXT_CHARS."""
        context = "\n\n".join(getattr(d, "page_content", "") for d in docs)
        max_context_chars = int(getattr(settings, "WAGTAIL_RAG_MAX_CONTEXT_CHARS", 0))
        if max_context_chars and len(context) > max_context_chars:
            if max_context_chars <= 3:
                return context[:max_context_chars]
            context = context[: max_context_chars - 3] + "..."
        return context

    def _is_chat_model(self) -> bool:
        """Detect if the LLM is a chat model based on class name and available methods."""
        if LCEL_AVAILABLE and BaseChatModel and isinstance(self.llm, BaseChatModel):
            return True
        return (
            hasattr(self.llm, "__class__") and "Chat" in self.llm.__class__.__name__
        ) or (hasattr(self.llm, "invoke") and not hasattr(self.llm, "generate"))

    def _summarize_history(self, summary: str, new_messages_text: str) -> str:
        """Summarize older history turns to keep context compact."""
        if not new_messages_text.strip():
            return summary

        prompt = (
            "You are summarizing a conversation for future context.\n\n"
            f"Existing summary:\n{summary}\n\n"
            f"New messages:\n{new_messages_text}\n\n"
            "Updated summary (concise, factual, preserve names and decisions):"
        )

        try:
            if self._is_chat_model():
                result = self._invoke_chat_model(prompt)
            else:
                result = self._invoke_regular_llm(prompt)
            return (result or "").strip()
        except Exception:
            logger.exception("History summarization failed; keeping existing summary")
            return summary

    @staticmethod
    def _extract_text_from_result(result: Any) -> str:
        """Extract text content from various LLM response formats."""
        if hasattr(result, "content"):
            return str(result.content)
        if isinstance(result, str):
            return result
        return str(result)

    def _invoke_chat_model(self, prompt_text: str) -> str:
        """Invoke a chat model (ChatOllama, ChatOpenAI, etc.) with messages."""
        # Try with HumanMessage wrapper first
        if LCEL_AVAILABLE and HumanMessage:
            try:
                message = HumanMessage(content=prompt_text)
                result = self.llm.invoke([message])
                return self._extract_text_from_result(result)
            except Exception as e:
                logger.debug("HumanMessage invoke failed: %s", e)

        # Try direct string invoke
        try:
            result = self.llm.invoke(prompt_text)
            return self._extract_text_from_result(result)
        except Exception as e:
            logger.debug("Direct string invoke failed: %s", e)

        # Try dict format as last resort
        try:
            result = self.llm.invoke({"role": "user", "content": prompt_text})
            return self._extract_text_from_result(result)
        except Exception:
            logger.exception("All chat model invocation methods failed")
            raise

    def _invoke_regular_llm(self, prompt_text: str) -> str:
        """Invoke a regular LLM (non-chat) with various methods."""
        # Try direct call
        try:
            return str(self.llm(prompt_text))
        except (TypeError, AttributeError) as e:
            logger.debug("Direct call failed: %s", e)

        # Try invoke method
        try:
            result = self.llm.invoke(prompt_text)
            return self._extract_text_from_result(result)
        except Exception as e:
            logger.debug("Invoke method failed: %s", e)

        # Try generate method (legacy LangChain)
        try:
            return str(self.llm.generate([prompt_text]).generations[0][0].text)
        except Exception:
            logger.exception("All regular LLM invocation methods failed")
            raise

    def generate_answer_with_llm(
        self,
        question: str,
        docs: List[Any],
        session_id: Optional[str] = None,
    ) -> str:
        """Generate an answer by formatting a prompt and calling the raw LLM callable.

        This is used as a fallback when no QA chain is available.
        Handles both chat models (ChatOllama, ChatOpenAI, etc.) and regular LLMs.
        """
        context = self._get_context_from_docs(docs)
        input_text = f"Context:\n{context}\n\nQuestion:\n{question}"
        prompt_text = self.prompt_template_str.format(
            context=context, question=question
        )

        # Detect if this is a chat model (expects messages, not plain strings)
        is_chat_model = self._is_chat_model()

        if is_chat_model and self.history_chain and session_id:
            result = self.history_chain.invoke(
                {"input": input_text},
                config={"configurable": {"session_id": session_id}},
            )
            return self._extract_text_from_result(result)

        if is_chat_model:
            return self._invoke_chat_model(prompt_text)
        return self._invoke_regular_llm(prompt_text)

    def generate_answer(
        self,
        question: str,
        docs: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Main entry point for generating an answer.

        Behavior:
        - If docs are provided, use them directly (bypass chain's retriever) to ensure
          we use the documents from hybrid search (vector + Wagtail).
        - If no docs provided and a QA chain exists, use the chain (which will use its retriever).
        - If no chain exists, docs are required.
        """
        # If docs are provided, use them directly - don't let the chain re-retrieve
        # This ensures we use the documents from our hybrid search (vector + Wagtail)
        if docs:
            logger.debug(f"Using {len(docs)} pre-retrieved documents for LLM context")
            return self.generate_answer_with_llm(
                question,
                docs,
                session_id=session_id,
            )

        # Fallback mode: no chain
        if self.qa_chain is None:
            raise ValueError("docs required when qa_chain is None")

        # We have a chain but no docs - use the chain (it will use its retriever)
        if LCEL_AVAILABLE and hasattr(self.qa_chain, "invoke"):
            # LCEL-style chain handles retrieval/internal flow
            try:
                out = self.qa_chain.invoke(question)
            except Exception:
                logger.exception("LCEL chain invocation failed")
                raise

            # Normalize output to a string
            if isinstance(out, str):
                return out

            if isinstance(out, dict):
                # Try common output keys
                for key in ("output", "text", "result", "answer"):
                    if key in out and isinstance(out[key], str):
                        return out[key]

            # Try object attributes
            for attr in ("text", "output", "content"):
                if hasattr(out, attr):
                    val = getattr(out, attr)
                    if isinstance(val, str):
                        return val

            if out is None:
                raise RuntimeError("LCEL chain produced no output")

            # Last resort: stringify
            return str(out)

        # Legacy RetrievalQA
        # Note: docs check is redundant here since it's already handled above, but kept for safety
        try:
            result = self.qa_chain({"query": question})
            if isinstance(result, dict):
                return result.get("result") or result.get("answer") or ""
            # Best-effort stringify
            return str(result)
        except Exception:
            logger.exception("QA chain invocation failed")
            raise

    def get_source_documents_from_chain(self, question: str) -> List[Any]:
        """Attempt to extract source documents returned by the chain (legacy RetrievalQA path).

        Note: This method is currently unused but kept for potential future use.
        The RAG chatbot uses pre-retrieved documents from hybrid search instead.

        Returns a list (possibly empty) of source Document objects.
        """
        if self.qa_chain is None:
            return []

        # Legacy RetrievalQA returns source_documents in the result dict
        try:
            if not LCEL_AVAILABLE:
                result = self.qa_chain({"query": question})
                return result.get("source_documents", []) or []

            # LCEL chains may not expose source docs in a standard way; try to invoke and inspect
            if hasattr(self.qa_chain, "invoke"):
                out = self.qa_chain.invoke(question)
                if isinstance(out, dict):
                    return (
                        out.get("source_documents", []) or out.get("source", []) or []
                    )
                # Some LCEL runnables return an object with a 'source_documents' attribute
                if hasattr(out, "source_documents"):
                    return getattr(out, "source_documents") or []
                if hasattr(out, "source"):
                    return getattr(out, "source") or []
        except Exception:
            logger.exception("Failed to extract source documents from chain")

        return []


__all__ = ["LLMGenerator"]
