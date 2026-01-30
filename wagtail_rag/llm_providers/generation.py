"""
LLM Generation Module for RAG Chatbot.

This module handles all LLM (Large Language Model) generation functionality,
including prompt construction, chain execution, and answer generation.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from django.conf import settings

logger = logging.getLogger(__name__)

# Detect LangChain edition / available components
try:
    # LCEL-style (newer langchain-core) primitives
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    LCEL_AVAILABLE = True
except Exception:
    ChatPromptTemplate = None
    RunnablePassthrough = None
    StrOutputParser = None
    LCEL_AVAILABLE = False

try:
    from langchain_core.messages import HumanMessage
    HUMAN_MESSAGE_AVAILABLE = True
except Exception:
    HumanMessage = None
    HUMAN_MESSAGE_AVAILABLE = False

try:
    # Legacy LangChain interfaces
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    LEGACY_AVAILABLE = True
except Exception:
    PromptTemplate = None
    RetrievalQA = None
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
        self.qa_chain = self._create_qa_chain()

    def _get_prompt_template(self) -> str:
        """Return prompt template from settings or default."""
        return getattr(
            settings,
            "WAGTAIL_RAG_PROMPT_TEMPLATE",
            """You are a helpful assistant. Use ONLY the following context to answer the question.
If the context does not mention some detail, simply do not talk about that detail.
Do not say things like "the context does not contain" or explain what is missing.

Context:
{context}

Question:
{question}

Answer:""",
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
                logger.exception("Failed to create LCEL QA chain; falling back to legacy if available")

        if LEGACY_AVAILABLE:
            try:
                return self._create_legacy_chain()
            except Exception:
                logger.exception("Failed to create legacy RetrievalQA chain")

        return None

    def _create_lcel_chain(self) -> Any:
        """Create an LCEL-style chain using langchain_core primitives.

        This will raise ImportError if required components are missing.
        """
        if not (ChatPromptTemplate and RunnablePassthrough and StrOutputParser):
            raise ImportError("LCEL primitives are not available")

        prompt = ChatPromptTemplate.from_template(self.prompt_template_str)  # type: ignore[attr-defined]

        def format_docs(docs: List[Any]) -> str:  # simple formatter used in runnable composition
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

    def _format_simple_prompt(self, question: str, docs: List[Any]) -> str:
        """Format a plain-text prompt using retrieved documents (fallback path)."""
        context = "\n\n".join(getattr(d, "page_content", "") for d in docs)
        return self.prompt_template_str.format(context=context, question=question)

    def generate_answer_with_llm(self, question: str, docs: List[Any]) -> str:
        """Generate an answer by formatting a prompt and calling the raw LLM callable.

        This is used as a fallback when no QA chain is available.
        Handles both chat models (ChatOllama, ChatOpenAI, etc.) and regular LLMs.
        """
        prompt_text = self._format_simple_prompt(question, docs)
        
        # Check if this is a chat model (expects messages, not plain strings)
        is_chat_model = (
            hasattr(self.llm, '__class__') and 
            'Chat' in self.llm.__class__.__name__
        ) or (
            hasattr(self.llm, 'invoke') and 
            not hasattr(self.llm, 'generate')
        )
        
        if is_chat_model:
            # Chat models need messages, not plain strings
            try:
                if HUMAN_MESSAGE_AVAILABLE and HumanMessage:
                    message = HumanMessage(content=prompt_text)
                    result = self.llm.invoke([message])
                else:
                    # Fallback: try with string or dict format
                    result = self.llm.invoke(prompt_text)
                
                # Extract text from response (could be AIMessage or string)
                if hasattr(result, 'content'):
                    return result.content
                elif isinstance(result, str):
                    return result
                else:
                    return str(result)
            except Exception:
                # Fallback: try dict format or direct string
                try:
                    result = self.llm.invoke({"role": "user", "content": prompt_text})
                    if hasattr(result, 'content'):
                        return result.content
                    return str(result)
                except Exception:
                    logger.exception("Chat model invocation failed")
                    raise
        else:
            # Regular LLM - try different invocation methods
            try:
                # Try direct call (for callable LLMs)
                return self.llm(prompt_text)
            except TypeError:
                # Try invoke method
                try:
                    result = self.llm.invoke(prompt_text)
                    if isinstance(result, str):
                        return result
                    if hasattr(result, 'content'):
                        return result.content
                    return str(result)
                except Exception:
                    # Try generate method (legacy LangChain)
                    try:
                        return self.llm.generate([prompt_text]).generations[0][0].text
                    except Exception:
                        logger.exception("LLM call failed for simple prompt")
                        raise

    def generate_answer(self, question: str, docs: Optional[List[Any]] = None) -> str:
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
            logger.info(f"Using {len(docs)} pre-retrieved documents for LLM context (bypassing chain retriever)")
            return self.generate_answer_with_llm(question, docs)

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

            # Normalize output to a string when possible
            if isinstance(out, str):
                return out
            if isinstance(out, dict):
                # Common keys that may hold textual output
                for key in ("output", "text", "result", "answer"):
                    if key in out and isinstance(out[key], str):
                        return out[key]
                # Sometimes 'output' is a list or object
                val = out.get("output") or out.get("result") or out.get("answer")
                if isinstance(val, str):
                    return val
                # Best-effort stringify
                return str(out)
            if out is None:
                raise RuntimeError("LCEL chain failed to produce an output")

            # If the chain returned an object with text-like attributes
            if hasattr(out, "text"):
                return str(getattr(out, "text"))
            if hasattr(out, "output"):
                return str(getattr(out, "output"))
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
                    return out.get("source_documents", []) or out.get("source", []) or []
                # Some LCEL runnables return an object with a 'source_documents' attribute
                if hasattr(out, "source_documents"):
                    return getattr(out, "source_documents") or []
                if hasattr(out, "source"):
                    return getattr(out, "source") or []
        except Exception:
            logger.exception("Failed to extract source documents from chain")

        return []


__all__ = ["LLMGenerator"]
