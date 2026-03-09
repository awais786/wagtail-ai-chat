"""
RAG Chatbot — top-level orchestrator.

Flow for each query:
  1. EmbeddingSearcher retrieves relevant chunks via vector similarity search
     (plus optional Wagtail full-text results when hybrid search is enabled).
  2. LLMGenerator formats a prompt from the retrieved context and calls the LLM.
  3. Returns the answer and source documents.
"""

import logging


try:
    from langchain.retrievers.multi_query import MultiQueryRetriever

    MULTI_QUERY_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.retrievers import MultiQueryRetriever

        MULTI_QUERY_AVAILABLE = True
    except ImportError:
        MULTI_QUERY_AVAILABLE = False
        MultiQueryRetriever = None

from .llm_providers import get_llm, LLMGenerator
from .embeddings import get_embeddings, EmbeddingSearcher
from .llm_providers.generation import (
    LCEL_AVAILABLE as USE_LCEL,
)  # noqa: F401 – re-exported
from .conf import conf
from .content_extraction.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def _is_wagtail_available() -> bool:
    try:
        from wagtail.models import Page  # noqa: F401

        return True
    except Exception:
        return False


class RAGChatBot:
    """RAG Chatbot that retrieves context from the vector store and generates responses."""

    def __init__(
        self,
        collection_name=None,
        model_name=None,
        persist_directory=None,
        metadata_filter=None,
        llm_provider=None,
        llm_kwargs=None,
    ):
        self.collection_name = collection_name or conf.vector_store.collection
        self.persist_directory = persist_directory or conf.vector_store.path
        self.llm_provider = llm_provider or conf.llm.provider
        self.model_name = model_name or conf.llm.model
        self.metadata_filter = metadata_filter or {}
        self.k_value = conf.search.k

        embedding_provider = conf.embedding.provider
        embedding_model = conf.embedding.model

        logger.info(
            "RAGChatBot init: llm=%s/%s  embeddings=%s/%s  collection=%s",
            self.llm_provider,
            self.model_name or "default",
            embedding_provider,
            embedding_model or "default",
            self.collection_name,
        )

        self.embeddings = get_embeddings(
            provider=embedding_provider, model_name=embedding_model
        )
        self.vectorstore = self._create_vectorstore()
        self.llm = get_llm(
            provider=self.llm_provider, model_name=self.model_name, **(llm_kwargs or {})
        )
        self.retriever = self._create_retriever()

        use_hybrid = conf.search.use_hybrid
        wagtail_ok = _is_wagtail_available()
        if use_hybrid and not wagtail_ok:
            logger.warning(
                "WAGTAIL_RAG_USE_HYBRID_SEARCH is True but Wagtail is not available; "
                "falling back to vector-only search."
            )

        self.embedding_searcher = EmbeddingSearcher(
            vectorstore=self.vectorstore,
            retriever=self.retriever,
            k_value=self.k_value,
            use_hybrid_search=use_hybrid and wagtail_ok,
        )
        self.llm_generator = LLMGenerator(llm=self.llm, retriever=self.retriever)

    def _create_vectorstore(self):
        """Load the vector store backend and return the underlying LangChain vectorstore."""
        return get_vector_store(
            path=self.persist_directory,
            collection=self.collection_name,
            embeddings=self.embeddings,
        ).db

    def _create_retriever(self):
        """Create a retriever, optionally wrapped with MultiQueryRetriever."""
        search_kwargs = {"k": self.k_value}
        if self.metadata_filter:
            search_kwargs["filter"] = self.metadata_filter

        base_retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        use_expansion = conf.search.use_query_expansion
        if use_expansion and MULTI_QUERY_AVAILABLE:
            try:
                logger.info("LLM query expansion enabled (MultiQueryRetriever)")
                return MultiQueryRetriever.from_llm(
                    retriever=base_retriever, llm=self.llm, include_original=True
                )
            except Exception:
                logger.warning(
                    "MultiQueryRetriever init failed; falling back to base retriever"
                )
        else:
            logger.info(
                "LLM query expansion %s",
                "disabled" if not use_expansion else "unavailable (install langchain)",
            )

        return base_retriever

    def _format_sources(self, docs) -> list[dict]:
        """Return unique source pages, limited to the top MAX_SOURCES by retrieval rank."""
        MAX_SOURCES = conf.search.max_sources
        seen = set()
        sources = []
        for doc in docs:
            if len(sources) >= MAX_SOURCES:
                break
            page_id = doc.metadata.get("page_id") or doc.metadata.get("title")
            if page_id in seen:
                continue
            seen.add(page_id)
            sources.append(
                {"content": doc.page_content[:200] + "...", "metadata": doc.metadata}
            )
        return sources

    def _build_retrieval_query(self, question: str, session_id: str) -> str:
        """Prepend recent human turns to the question for better retrieval.

        Vague follow-up questions like "do you know the ingredients?" have no
        context for the vector search. Prepending the previous human turn gives
        the retriever enough signal to find the right documents.
        """
        try:
            history = self.llm_generator.history_store.get_session_history(session_id)
            messages = history.messages
            if not messages:
                return question
            human_turns = [
                m.content
                for m in messages
                if getattr(m, "type", "") == "human" and getattr(m, "content", "")
            ]
            if not human_turns:
                return question
            # Use the last human turn as context prefix for retrieval.
            return f"{human_turns[-1]} {question}"
        except Exception:
            return question

    def query(
        self,
        question: str,
        boost_title_matches: bool = True,
        session_id=None,
        search_only: bool = False,
    ) -> dict:
        """Query the RAG chatbot.

        Args:
            question: The user's question.
            boost_title_matches: Prioritize documents whose titles match the query.
            session_id: Optional session ID for chat history.
            search_only: If True, skip LLM generation and return search results only.

        Returns:
            {'answer': str | None, 'sources': list[dict]}
        """
        logger.info("RAG query: %r", question[:200])

        # For follow-up questions, enrich the retrieval query with recent history
        # so vector search finds relevant docs even for vague questions.
        retrieval_query = question
        if session_id and self.llm_generator.history_store:
            retrieval_query = self._build_retrieval_query(question, session_id)
            if retrieval_query != question:
                logger.debug("Retrieval query enriched: %r", retrieval_query)

        docs = self.embedding_searcher.retrieve_with_embeddings(
            retrieval_query, boost_title_matches=boost_title_matches
        )
        logger.info("Retrieved %d document(s)", len(docs))

        if search_only:
            return {"answer": None, "sources": self._format_sources(docs)}

        answer = self.llm_generator.generate_answer(
            question, docs=docs, session_id=session_id
        )
        logger.info(
            "Answer generated (%d chars): %s%s",
            len(answer),
            answer[:200],
            "..." if len(answer) > 200 else "",
        )
        return {"answer": answer, "sources": self._format_sources(docs)}

    def update_filter(self, metadata_filter: dict) -> None:
        """Update the metadata filter used by the retriever."""
        self.metadata_filter = metadata_filter or {}
        search_kwargs = {"k": self.k_value}
        if self.metadata_filter:
            search_kwargs["filter"] = self.metadata_filter
        self.retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)


def get_chatbot(
    collection_name=None,
    model_name=None,
    metadata_filter=None,
    llm_provider=None,
    llm_kwargs=None,
) -> RAGChatBot:
    """Convenience factory for RAGChatBot. All args default to settings."""
    return RAGChatBot(
        collection_name=collection_name,
        model_name=model_name,
        metadata_filter=metadata_filter,
        llm_provider=llm_provider,
        llm_kwargs=llm_kwargs,
    )
