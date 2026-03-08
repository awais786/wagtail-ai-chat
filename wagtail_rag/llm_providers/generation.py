"""
LLM Generation Module for RAG Chatbot.

Handles prompt construction and answer generation with context.
"""

import logging
from typing import Any, Optional

from django.conf import settings

from .chat_history import get_history_store

logger = logging.getLogger(__name__)

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage
    from langchain_core.language_models.chat_models import BaseChatModel

    LCEL_AVAILABLE = True
except ImportError:
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
    """Handles LLM-based answer generation for the RAG chatbot.

    Encapsulates prompt construction, QA chain setup (LCEL or legacy),
    and answer generation with retrieved context.
    """

    def __init__(self, llm: Any, retriever: Optional[Any] = None):
        """Initialize the generator.

        Args:
            llm: A LangChain LLM or chat model instance.
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
        return (
            "You are an assistant that answers questions strictly from the provided context.\n\n"
            "IMPORTANT RULES:\n"
            "- Answer ONLY using information that appears in the context below.\n"
            "- Do NOT use your general knowledge or training data.\n"
            "- If the context does not contain enough information to answer, respond with: "
            '"I don\'t have that information in the available content."\n'
            "- Do not guess or infer beyond what is explicitly stated.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )

    def _get_system_prompt(self) -> str:
        return (
            "You are a website assistant. Answer questions using ONLY the provided context. "
            "Never use your general knowledge. "
            "If the answer is not in the context, say: "
            '"I don\'t have that information in the available content."'
        )

    def _create_qa_chain(self) -> Optional[Any]:
        """Construct a QA chain using available LangChain interfaces."""
        if not self.retriever:
            return None

        if LCEL_AVAILABLE:
            try:
                return self._create_lcel_chain()
            except Exception:
                logger.exception(
                    "Failed to create LCEL QA chain; falling back to legacy"
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
        """Create an LCEL-style chain using langchain_core primitives."""
        if not (ChatPromptTemplate and RunnablePassthrough and StrOutputParser):
            raise ImportError("LCEL primitives are not available")

        prompt = ChatPromptTemplate.from_template(self.prompt_template_str)  # type: ignore[attr-defined]

        def format_docs(docs: list[Any]) -> str:
            return "\n\n".join(getattr(d, "page_content", "") for d in docs)

        return (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()  # type: ignore[call-arg]
        )

    def _create_legacy_chain(self) -> Any:
        """Create a legacy RetrievalQA chain (langchain < 0.1 style)."""
        if not (PromptTemplate and RetrievalQA):
            raise ImportError("Legacy RetrievalQA components are not available")

        prompt = PromptTemplate(  # type: ignore[call-arg]
            template=self.prompt_template_str, input_variables=["context", "question"]
        )
        return RetrievalQA.from_chain_type(  # type: ignore[attr-defined]
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def _get_context_from_docs(self, docs: list[Any]) -> str:
        """Build context string from docs, optionally truncated by WAGTAIL_RAG_MAX_CONTEXT_CHARS."""
        context = "\n\n".join(getattr(d, "page_content", "") for d in docs)
        max_chars = int(getattr(settings, "WAGTAIL_RAG_MAX_CONTEXT_CHARS", 0))
        if max_chars and len(context) > max_chars:
            context = (
                context[: max_chars - 3] + "..."
                if max_chars > 3
                else context[:max_chars]
            )
        return context

    def _is_chat_model(self) -> bool:
        """Detect if the LLM is a chat model."""
        if LCEL_AVAILABLE and BaseChatModel and isinstance(self.llm, BaseChatModel):
            return True
        return "Chat" in self.llm.__class__.__name__ or (
            hasattr(self.llm, "invoke") and not hasattr(self.llm, "generate")
        )

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
        """Invoke a chat model (ChatOllama, ChatOpenAI, etc.) with a HumanMessage."""
        if LCEL_AVAILABLE and HumanMessage:
            try:
                result = self.llm.invoke([HumanMessage(content=prompt_text)])
                return self._extract_text_from_result(result)
            except Exception as e:
                logger.debug("HumanMessage invoke failed: %s", e)

        # Fallback: plain string invoke
        result = self.llm.invoke(prompt_text)
        return self._extract_text_from_result(result)

    def _invoke_regular_llm(self, prompt_text: str) -> str:
        """Invoke a non-chat LLM."""
        try:
            result = self.llm.invoke(prompt_text)
            return self._extract_text_from_result(result)
        except Exception as e:
            logger.debug("Invoke method failed: %s", e)

        # Fallback: legacy generate method
        try:
            return str(self.llm.generate([prompt_text]).generations[0][0].text)
        except Exception:
            logger.exception("All LLM invocation methods failed")
            raise

    def generate_answer_with_llm(
        self,
        question: str,
        docs: list[Any],
        session_id: Optional[str] = None,
    ) -> str:
        """Generate an answer by formatting a prompt and calling the LLM directly.

        Used as a fallback when no QA chain is available, and as the primary
        path when docs are pre-retrieved (hybrid search).
        """
        context = self._get_context_from_docs(docs)
        input_text = f"Context:\n{context}\n\nQuestion:\n{question}"
        prompt_text = self.prompt_template_str.format(
            context=context, question=question
        )

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
        docs: Optional[list[Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Main entry point for generating an answer.

        If docs are provided, uses them directly so hybrid search results
        (vector + Wagtail) are used instead of the chain's retriever.
        If no docs provided, falls back to the QA chain.
        """
        if docs:
            logger.debug(
                "Using %d pre-retrieved document(s) for LLM context", len(docs)
            )
            return self.generate_answer_with_llm(question, docs, session_id=session_id)

        if self.qa_chain is None:
            raise ValueError("docs required when qa_chain is None")

        if LCEL_AVAILABLE and hasattr(self.qa_chain, "invoke"):
            out = self.qa_chain.invoke(question)

            if isinstance(out, str):
                return out
            if isinstance(out, dict):
                for key in ("output", "text", "result", "answer"):
                    if key in out and isinstance(out[key], str):
                        return out[key]
            for attr in ("text", "output", "content"):
                if hasattr(out, attr):
                    val = getattr(out, attr)
                    if isinstance(val, str):
                        return val
            if out is None:
                raise RuntimeError("LCEL chain produced no output")
            return str(out)

        # Legacy RetrievalQA
        result = self.qa_chain({"query": question})
        if isinstance(result, dict):
            return result.get("result") or result.get("answer") or ""
        return str(result)


__all__ = ["LLMGenerator"]
