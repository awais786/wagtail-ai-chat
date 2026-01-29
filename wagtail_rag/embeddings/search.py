"""
Embedding Search Module for RAG Chatbot.

This module handles all embedding-based similarity search functionality,
including vector search, hybrid search with Wagtail, and result ranking.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from django.conf import settings

logger = logging.getLogger(__name__)

# Import Document class for creating document objects
try:
    from langchain_core.documents import Document  # type: ignore
except Exception:
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:
        # Fallback: create a simple Document class
        class Document:  # pragma: no cover - fallback for environments without langchain
            def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
                self.page_content = page_content
                self.metadata = metadata or {}


class EmbeddingSearcher:
    """
    Handles embedding-based similarity search for RAG chatbot.

    Encapsulates:
    - Vector similarity search via vectorstore/retriever
    - Optional hybrid Wagtail full-text search
    - Title-based boosting and conservative re-ranking
    - Deduplication and HTML stripping
    """

    def __init__(self, vectorstore: Any, retriever: Any, k_value: int, use_hybrid_search: bool = True):
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.k_value = k_value
        self.use_hybrid_search = use_hybrid_search

    # --- Internal helpers moved into the class for better encapsulation ---
    def _get_wagtail_page_model(self):
        """Lazy import of Wagtail Page model. Returns None if unavailable."""
        try:
            from wagtail.models import Page  # type: ignore

            return Page
        except Exception:
            return None

    def _get_content_extraction_utils(self) -> Tuple[Optional[Callable], Optional[Callable], Optional[Callable]]:
        """Lazy import content extraction utilities from wagtail_rag.content_extraction."""
        try:
            from wagtail_rag.content_extraction import (
                extract_page_content,
                get_page_url,
                extract_streamfield_text,
            )

            return extract_page_content, get_page_url, extract_streamfield_text
        except Exception:
            return None, None, None

    @staticmethod
    def _strip_html(text: Optional[str]) -> str:
        """Strip HTML tags from a string for cleaner embeddings/search content."""
        if not text:
            return ""
        try:
            from bs4 import BeautifulSoup  # type: ignore

            return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
        except Exception:
            import re

            return " ".join(re.sub(r"<[^>]+>", " ", text).split())

    @staticmethod
    def _fuzzy_match(query_str: str, target_str: str, min_length: int = 3) -> bool:
        """Simple fuzzy check between query and target strings."""
        if not query_str or not target_str or len(query_str) < min_length:
            return False
        if query_str in target_str:
            return True
        if len(target_str) > len(query_str):
            return target_str.endswith(query_str) or query_str in target_str[1:]
        return False

    # --- Document conversion and retrieval methods ---
    def _convert_wagtail_page_to_document(self, page: Any) -> Optional[Document]:
        """Convert a Wagtail Page object into a lightweight Document instance."""
        try:
            extract_page_content, get_page_url_util, extract_streamfield_text = self._get_content_extraction_utils()

            page_url = getattr(page, "url", "")
            if get_page_url_util:
                page_url = get_page_url_util(page) or page_url

            # Extract page content using extraction utilities when available
            page_text = None
            if extract_page_content:
                page_text = extract_page_content(page)
            else:
                parts: List[str] = []
                title = getattr(page, "title", "")
                if title:
                    parts.append(f"Title: {title}")

                search_description = getattr(page, "search_description", "")
                if search_description:
                    parts.append(search_description)

                body = getattr(page, "body", None)
                if body:
                    if extract_streamfield_text:
                        streamfield_text = extract_streamfield_text(body)
                        if streamfield_text:
                            parts.append(streamfield_text[:500])
                    else:
                        parts.append(str(body)[:500])

                page_text = "\n\n".join(parts) if parts else None

            if not page_text or not str(page_text).strip():
                return None

            return Document(
                page_content=page_text,
                metadata={
                    "title": getattr(page, "title", ""),
                    "url": page_url,
                    "id": getattr(page, "id", None),
                    "source": f"{getattr(page, '_meta', None).app_label if getattr(page, '_meta', None) else ''}.{page.__class__.__name__}",
                    "model": page.__class__.__name__,
                    "app": getattr(getattr(page, '_meta', None), 'app_label', ''),
                    "from_wagtail_search": True,
                },
            )
        except Exception:
            return None

    def _get_vector_docs(self, query: str) -> Tuple[List[Document], Set[str], Set[Any]]:
        """Get documents from vector/retriever with fallbacks.

        Returns: (docs, seen_urls, seen_ids)
        """
        docs: List[Document] = []
        seen_urls: Set[str] = set()
        seen_ids: Set[Any] = set()

        try:
            docs = self.retriever.invoke(query)  # new API path
        except Exception:
            try:
                docs = self.vectorstore.similarity_search(query, k=self.k_value)
            except Exception:
                docs = []

        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            url = meta.get("url", "")
            doc_id = meta.get("id")
            if url:
                seen_urls.add(url)
            if doc_id:
                seen_ids.add(doc_id)

        return docs, seen_urls, seen_ids

    def _get_wagtail_docs(self, query: str, seen_urls: Set[str], seen_ids: Set[Any]) -> List[Document]:
        """Query Wagtail full-text search and convert results to Documents."""
        if not self.use_hybrid_search:
            return []

        docs: List[Document] = []
        Page = self._get_wagtail_page_model()
        if Page is None:
            return []

        try:
            wagtail_results = Page.objects.live().search(query)[:10]
            for page in wagtail_results:
                page_url = getattr(page, "url", "")
                page_id = getattr(page, "id", None)

                if page_url in seen_urls or page_id in seen_ids:
                    continue

                doc = self._convert_wagtail_page_to_document(page)
                if doc:
                    docs.append(doc)
                    seen_urls.add(page_url)
                    seen_ids.add(page_id)
        except Exception:
            # Swallow exceptions to keep search resilient in non-Wagtail environments
            pass

        return docs

    def _boost_title_matches_if_short_query(self, query: str, docs: List[Document]) -> List[Document]:
        """Prioritize documents whose titles match short queries or have fuzzy matches."""
        query_clean = query.lower().strip('?').strip()
        if len(query_clean.split()) > 2 or len(query_clean) < 4:
            return docs

        try:
            all_results = self.vectorstore.similarity_search_with_score(query, k=min(self.k_value * 3, 30))
            title_matches: List[Tuple[Document, float]] = []

            for doc, score in all_results:
                title = (getattr(doc, "metadata", {}) or {}).get("title", "").lower()
                title_clean = title.replace('#', '').replace(' ', '')

                if query_clean in title or title.startswith(query_clean):
                    title_matches.append((doc, score))
                elif len(query_clean) >= 4:
                    query_chars = list(query_clean)
                    title_chars = list(title_clean)
                    query_idx = 0
                    for char in title_chars:
                        if query_idx < len(query_chars) and char == query_chars[query_idx]:
                            query_idx += 1
                    if query_idx >= len(query_clean) - 1:
                        title_matches.append((doc, score))

            if title_matches:
                title_matches.sort(key=lambda x: x[1])
                title_docs = [doc for doc, _ in title_matches[:3]]
                existing_ids = {id(d) for d in title_docs}
                other_docs = [d for d in docs if id(d) not in existing_ids]
                return title_docs + other_docs[: self.k_value - len(title_docs)]
        except Exception:
            pass

        return docs

    def _rerank_by_title_match(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank docs by title similarity to the query."""
        query_lower = query.lower().strip('?').strip()
        query_words = set(query_lower.split())

        scored_docs: List[Tuple[float, int, Document]] = []
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
                    matches = query_words.intersection(title_words)
                    if matches:
                        score = len(matches) / len(query_words)
                    for query_word in query_words:
                        if len(query_word) >= 4:
                            if query_word in title:
                                score = max(score, 0.7)
                            for title_word in title_words:
                                if len(title_word) >= 4 and self._fuzzy_match(query_word, title_word):
                                    score = max(score, 0.7)

            scored_docs.append((score, idx, doc))

        scored_docs.sort(key=lambda x: (-x[0], x[1]))
        return [doc for _, _, doc in scored_docs]

    def retrieve_with_embeddings(self, query: str, boost_title_matches: bool = True) -> List[Document]:
        """Retrieve documents using embedding search combined with optional Wagtail search."""
        vector_docs, seen_urls, seen_ids = self._get_vector_docs(query)
        wagtail_docs = self._get_wagtail_docs(query, seen_urls, seen_ids)

        all_docs = vector_docs + wagtail_docs
        docs = all_docs[: self.k_value]

        docs = self._boost_title_matches_if_short_query(query, docs)

        if boost_title_matches and docs:
            docs = self._rerank_by_title_match(query, docs)

        return docs

    def _deduplicate_results(self, raw_results: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        """Deduplicate search results by URL/ID and keep best-scoring per key."""
        best_by_key: Dict[Tuple[str, Optional[Any]], Tuple[Document, float]] = {}
        for doc, score in raw_results:
            meta = getattr(doc, "metadata", {}) or {}
            url = meta.get("url") or ""
            doc_id = meta.get("id")
            key = (url, doc_id)
            if key not in best_by_key or score < best_by_key[key][1]:
                best_by_key[key] = (doc, score)

        deduped = list(best_by_key.values())
        return sorted(deduped, key=lambda pair: pair[1])[:k]

    def _apply_conservative_title_boost(self, query: str, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Apply conservative title boosting for short, specific queries."""
        query_clean = query.lower().strip()
        query_words = set(query_clean.split())
        sig_query_words = {w for w in query_words if len(w) >= 3}

        if not (0 < len(sig_query_words) <= 3):
            return results

        title_matches: List[Dict[str, Any]] = []
        other_results: List[Dict[str, Any]] = []

        max_title_boost_score = getattr(settings, "WAGTAIL_RAG_TITLE_BOOST_MAX_SCORE", None)

        for result in results:
            title = (result.get("metadata") or {}).get("title", "").lower()
            title_words = set(title.split())

            has_all_words = sig_query_words.issubset(title_words)

            score_ok = True if max_title_boost_score is None else result.get("score", float("inf")) <= max_title_boost_score

            if has_all_words and score_ok:
                title_matches.append(result)
            else:
                other_results.append(result)

        if title_matches:
            title_matches.sort(key=lambda x: x.get("score", 0))
            return title_matches + other_results[: max(0, k - len(title_matches))]

        return results

    def search_with_embeddings(self, query: str, k: Optional[int] = None, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar content using vectorstore similarity search and return cleaned results."""
        if k is None:
            k = getattr(settings, "WAGTAIL_RAG_SEARCH_K", 10)

        if metadata_filter:
            raw_results = self.vectorstore.similarity_search_with_score(query, k=k, filter=metadata_filter)
        else:
            raw_results = self.vectorstore.similarity_search_with_score(query, k=k)

        deduped = self._deduplicate_results(raw_results, k)

        cleaned_results: List[Dict[str, Any]] = []
        for doc, score in deduped:
            raw_content = getattr(doc, "page_content", "") or ""
            clean_content = self._strip_html(raw_content)
            cleaned_results.append({"content": clean_content, "metadata": getattr(doc, "metadata", {}) or {}, "score": score})

        return self._apply_conservative_title_boost(query, cleaned_results, k)

