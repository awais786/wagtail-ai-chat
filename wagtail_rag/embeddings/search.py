"""
Embedding Search Module for RAG Chatbot.

Handles all embedding-based similarity search, including vector search,
hybrid search with Wagtail full-text search, and result ranking.
"""

import logging
import re
from collections.abc import Callable
from typing import Any, Optional

from wagtail_rag.conf import conf

try:
    from langchain_core.documents import Document  # type: ignore
except ImportError:
    from langchain.schema import Document  # type: ignore

logger = logging.getLogger(__name__)


class EmbeddingSearcher:
    """
    Handles embedding-based similarity search for the RAG chatbot.

    Search Strategy:
    1. Vector Search (always performed):
       - Uses semantic similarity via embeddings
       - Primary search method

    2. Wagtail Search (optional, only if use_hybrid_search=True):
       - Uses Wagtail's full-text search engine
       - Secondary search that supplements vector results

    Additional Features:
    - Title-based boosting for short queries
    - Conservative re-ranking by title similarity
    - Automatic deduplication between vector and Wagtail results
    """

    def __init__(
        self,
        vectorstore: Any,
        retriever: Any,
        k_value: int,
        use_hybrid_search: bool = True,
    ):
        """
        Initialize the embedding searcher.

        Args:
            vectorstore: Vector store instance (FAISS, ChromaDB, or pgvector)
            retriever: LangChain retriever instance
            k_value: Number of documents to retrieve
            use_hybrid_search: If True, combines vector search + Wagtail full-text search.
        """
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.k_value = k_value
        self.use_hybrid_search = use_hybrid_search

    # --- Static helpers ---

    @staticmethod
    def _strip_html(text: Optional[str]) -> str:
        """Strip HTML tags from a string."""
        if not text:
            return ""
        try:
            from bs4 import BeautifulSoup  # type: ignore

            return BeautifulSoup(text, "html.parser").get_text(
                separator=" ", strip=True
            )
        except Exception as e:
            logger.debug("BeautifulSoup unavailable, stripping HTML with regex: %s", e)
            return " ".join(re.sub(r"<[^>]+>", " ", text).split())

    @staticmethod
    def _fuzzy_match(query_str: str, target_str: str, min_length: int = 3) -> bool:
        """Simple fuzzy check — True if query appears as a substring of target."""
        if not query_str or not target_str or len(query_str) < min_length:
            return False
        return query_str in target_str

    # --- Vector search (primary) ---

    def _get_vector_docs(self, query: str) -> tuple[list[Document], set[str], set[Any]]:
        """Perform vector search (primary search method).

        Returns:
            Tuple of (docs, seen_urls, seen_ids) for downstream deduplication.
        """
        docs: list[Document] = []
        seen_urls: set[str] = set()
        seen_ids: set[Any] = set()

        try:
            docs = self.retriever.invoke(query)
        except Exception as e:
            logger.debug(
                "retriever.invoke failed (%s), falling back to similarity_search()", e
            )
            try:
                docs = self.vectorstore.similarity_search(query, k=self.k_value)
            except Exception as e2:
                logger.error("Vector search failed: %s", e2)

        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            url = meta.get("url", "")
            doc_id = meta.get("page_id") or meta.get("id")
            if url:
                seen_urls.add(url)
            if doc_id:
                seen_ids.add(doc_id)

        return docs, seen_urls, seen_ids

    # --- Wagtail search (secondary, hybrid only) ---

    def _get_wagtail_page_model(self):
        """Lazy import of Wagtail Page model."""
        try:
            from wagtail.models import Page  # type: ignore

            return Page
        except Exception as e:
            logger.debug("Wagtail Page model not available: %s", e)
            return None

    def _get_page_to_documents_function(self) -> Optional[Callable]:
        """Lazy import of page_to_documents_api_extractor."""
        try:
            from wagtail_rag.content_extraction import page_to_documents_api_extractor

            return page_to_documents_api_extractor
        except Exception as e:
            logger.debug("page_to_documents_api_extractor not available: %s", e)
            return None

    def _convert_wagtail_page_to_documents(self, page: Any) -> list[Document]:
        """Convert a Wagtail Page into Documents using the same logic as indexing."""
        try:
            page_to_documents_func = self._get_page_to_documents_function()
            if page_to_documents_func:
                documents = page_to_documents_func(page)
                for doc in documents:
                    doc.metadata["from_wagtail_search"] = True
                return documents

            page_url = getattr(page, "url", "") or getattr(page, "url_path", "")
            return [
                Document(
                    page_content=f"Title: {getattr(page, 'title', '')}",
                    metadata={
                        "title": getattr(page, "title", ""),
                        "url": page_url,
                        "page_id": getattr(page, "id", None),
                        "id": getattr(page, "id", None),
                        "model": page.__class__.__name__,
                        "from_wagtail_search": True,
                    },
                )
            ]
        except Exception as e:
            logger.warning("Failed to convert Wagtail page to documents: %s", e)
            return []

    def _get_wagtail_docs(
        self, query: str, seen_urls: set[str], seen_ids: set[Any]
    ) -> list[Document]:
        """Perform Wagtail full-text search (only called when hybrid search is enabled).

        Returns Documents for pages not already found by vector search.
        """
        if not self.use_hybrid_search:
            return []

        Page = self._get_wagtail_page_model()
        if Page is None:
            return []

        docs: list[Document] = []
        try:
            # Normalize: strip trailing punctuation for cleaner keyword matching.
            normalized = " ".join(query.strip("?.!").split())

            wagtail_results = Page.objects.live().search(normalized)[:10]
            logger.debug("Wagtail search returned %d pages", len(wagtail_results))

            # If normalized query found nothing, try the original.
            if len(wagtail_results) == 0 and normalized != query:
                wagtail_results = Page.objects.live().search(query)[:10]

            if len(wagtail_results) == 0:
                logger.debug(
                    "Wagtail search found 0 pages for query: '%s'. "
                    "Ensure pages are live and the search index is up to date "
                    "(python manage.py update_index).",
                    query,
                )

            skipped = 0
            for page in wagtail_results:
                page_url = getattr(page, "url", "")
                page_id = getattr(page, "id", None)

                if page_url in seen_urls or page_id in seen_ids:
                    skipped += 1
                    continue

                page_docs = self._convert_wagtail_page_to_documents(page)
                if page_docs:
                    docs.extend(page_docs)
                    seen_urls.add(page_url)
                    seen_ids.add(page_id)
                    logger.debug(
                        "Added %d document(s) from Wagtail page ID: %s",
                        len(page_docs),
                        page_id,
                    )

            if skipped:
                logger.debug(
                    "Deduplicated %d/%d Wagtail result(s) already in vector search",
                    skipped,
                    len(wagtail_results),
                )
        except Exception as e:
            logger.warning(
                "Wagtail search failed for query '%s': %s", query, e, exc_info=True
            )

        return docs

    # --- Title boosting and re-ranking ---

    def _boost_title_matches_if_short_query(
        self, query: str, docs: list[Document]
    ) -> list[Document]:
        """For short queries, reorder docs so title matches appear first."""
        query_clean = query.lower().strip("?").strip()
        if len(query_clean.split()) > 2 or len(query_clean) < 4:
            return docs

        title_matches: list[Document] = []
        other_docs: list[Document] = []

        for doc in docs:
            title = (getattr(doc, "metadata", {}) or {}).get("title", "").lower()
            if query_clean in title or title.startswith(query_clean):
                title_matches.append(doc)
            else:
                other_docs.append(doc)

        if title_matches:
            return title_matches + other_docs

        return docs

    def _rerank_by_title_match(
        self, query: str, docs: list[Document]
    ) -> list[Document]:
        """Rerank docs by title similarity to the query."""
        query_lower = query.lower().strip("?").strip()
        query_words = set(query_lower.split())

        scored: list[tuple[float, int, Document]] = []
        for idx, doc in enumerate(docs):
            score = 0.0
            title = (getattr(doc, "metadata", {}) or {}).get("title", "").lower()
            if title:
                if query_lower in title or title in query_lower:
                    score = 1.0
                elif self._fuzzy_match(query_lower, title, min_length=4):
                    score = 0.9
                else:
                    title_words = set(title.split())
                    matches = query_words & title_words
                    if matches:
                        score = len(matches) / len(query_words)
                    for word in query_words:
                        if len(word) >= 4 and word in title:
                            score = max(score, 0.7)
            scored.append((score, idx, doc))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [doc for _, _, doc in scored]

    # --- Public API ---

    def retrieve_with_embeddings(
        self, query: str, boost_title_matches: bool = True
    ) -> list[Document]:
        """Retrieve documents using embedding search, optionally combined with Wagtail search.

        Flow:
        1. Vector search (always)
        2. Wagtail full-text search (if hybrid search enabled)
        3. Combine, deduplicate, and rank results

        Args:
            query: Search query string
            boost_title_matches: Whether to reorder results to surface title matches

        Returns:
            List of Document objects ranked by relevance
        """
        logger.info("Searching for: '%s'", query)
        vector_docs, seen_urls, seen_ids = self._get_vector_docs(query)
        logger.debug("Vector search: %d document(s)", len(vector_docs))

        wagtail_docs: list[Document] = []
        if self.use_hybrid_search:
            wagtail_docs = self._get_wagtail_docs(query, seen_urls, seen_ids)
            logger.debug("Wagtail search: %d additional document(s)", len(wagtail_docs))
            if not vector_docs and not wagtail_docs:
                logger.warning(
                    "No results found for '%s'. "
                    "Ensure the RAG index is built (manage.py build_rag_index) "
                    "and the search index is updated (manage.py update_index).",
                    query,
                )
        else:
            logger.debug("Hybrid search disabled")

        docs = (vector_docs + wagtail_docs)[: self.k_value]
        logger.debug(
            "Combined: %d total (%d vector + %d wagtail)",
            len(vector_docs) + len(wagtail_docs),
            len(vector_docs),
            len(wagtail_docs),
        )

        docs = self._boost_title_matches_if_short_query(query, docs)
        if boost_title_matches and docs:
            docs = self._rerank_by_title_match(query, docs)

        logger.info("Returning %d document(s) for '%s'", len(docs), query)
        return docs

    def _deduplicate_results(
        self, raw_results: list[tuple[Document, float]], k: int
    ) -> list[tuple[Document, float]]:
        """Deduplicate search results by URL/ID, keeping the best score per key."""
        best: dict[tuple[str, Any], tuple[Document, float]] = {}
        for doc, score in raw_results:
            meta = getattr(doc, "metadata", {}) or {}
            key = (meta.get("url") or "", meta.get("page_id") or meta.get("id"))
            if key not in best or score < best[key][1]:
                best[key] = (doc, score)
        return sorted(best.values(), key=lambda pair: pair[1])[:k]

    def _apply_conservative_title_boost(
        self, query: str, results: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        """Apply conservative title boosting for short, specific queries."""
        query_clean = query.lower().strip()
        sig_words = {w for w in query_clean.split() if len(w) >= 3}

        if not (0 < len(sig_words) <= 3):
            return results

        max_score = conf.search.title_boost_max_score
        title_matches: list[dict[str, Any]] = []
        other: list[dict[str, Any]] = []

        for result in results:
            title_words = set(
                (result.get("metadata") or {}).get("title", "").lower().split()
            )
            score_ok = (
                max_score is None or result.get("score", float("inf")) <= max_score
            )
            if sig_words.issubset(title_words) and score_ok:
                title_matches.append(result)
            else:
                other.append(result)

        if title_matches:
            title_matches.sort(key=lambda x: x.get("score", 0))
            return title_matches + other[: max(0, k - len(title_matches))]

        return results

    def search_with_embeddings(
        self,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Lower-level search: vector similarity only, with score and optional metadata filter.

        For normal RAG retrieval (hybrid search + title boosting), use
        retrieve_with_embeddings() instead.
        """
        if k is None:
            k = conf.search.search_k

        if metadata_filter:
            raw_results = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=metadata_filter
            )
        else:
            raw_results = self.vectorstore.similarity_search_with_score(query, k=k)

        deduped = self._deduplicate_results(raw_results, k)
        cleaned: list[dict[str, Any]] = [
            {
                "content": self._strip_html(getattr(doc, "page_content", "") or ""),
                "metadata": getattr(doc, "metadata", {}) or {},
                "score": score,
            }
            for doc, score in deduped
        ]
        return self._apply_conservative_title_boost(query, cleaned, k)
