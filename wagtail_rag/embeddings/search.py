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

    Search Strategy:
    1. Vector Search (always performed):
       - Uses semantic similarity via embeddings
       - Finds conceptually similar content even without exact keyword matches
       - Primary search method
    
    2. Wagtail Search (optional, only if use_hybrid_search=True):
       - Uses Wagtail's full-text search engine
       - Finds exact keyword matches in page titles/content
       - Secondary search method that supplements vector search
    
    Additional Features:
    - Title-based boosting for short queries
    - Conservative re-ranking by title similarity
    - Automatic deduplication between vector and Wagtail results
    - HTML stripping for cleaner content
    """

    def __init__(self, vectorstore: Any, retriever: Any, k_value: int, use_hybrid_search: bool = True):
        """
        Initialize the embedding searcher.
        
        Args:
            vectorstore: ChromaDB vectorstore instance
            retriever: LangChain retriever instance
            k_value: Number of documents to retrieve
            use_hybrid_search: If True, combines vector search + Wagtail search.
                             If False, uses only vector search.
        """
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.k_value = k_value
        self.use_hybrid_search = use_hybrid_search

    # --- Static helper methods ---
    @staticmethod
    def _strip_html(text: Optional[str]) -> str:
        """Strip HTML tags from a string for cleaner embeddings/search content."""
        if not text:
            return ""
        try:
            from bs4 import BeautifulSoup  # type: ignore

            return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
        except Exception as e:
            import re

            logger.debug("BeautifulSoup unavailable, stripping HTML with regex: %s", e)
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

    # --- Vector Search Methods (Primary Search) ---
    def _get_vector_docs(self, query: str) -> Tuple[List[Document], Set[str], Set[Any]]:
        """
        Perform vector search using embeddings (primary search method).
        
        This is always executed first. It uses semantic similarity to find
        documents that are conceptually similar to the query, even if they
        don't contain exact keyword matches.
        
        Args:
            query: Search query string
            
        Returns:
            Tuple of:
            - docs: List of Document objects from vector search
            - seen_urls: Set of URLs found (for deduplication)
            - seen_ids: Set of page IDs found (for deduplication)
        """
        docs: List[Document] = []
        seen_urls: Set[str] = set()
        seen_ids: Set[Any] = set()

        # Try modern retriever API first (supports MultiQueryRetriever)
        try:
            logger.debug("Using retriever.invoke() for vector search")
            docs = self.retriever.invoke(query)
        except Exception as e:
            logger.debug("retriever.invoke failed (%s), falling back to vectorstore.similarity_search()", e)
            try:
                docs = self.vectorstore.similarity_search(query, k=self.k_value)
            except Exception as e2:
                logger.error("Vector search failed: %s", e2)
                docs = []

        # Track URLs and IDs for deduplication (used when combining with Wagtail results)
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            url = meta.get("url", "")
            doc_id = meta.get("page_id") or meta.get("id")
            if url:
                seen_urls.add(url)
            if doc_id:
                seen_ids.add(doc_id)

        return docs, seen_urls, seen_ids

    # --- Wagtail Search Methods (Secondary Search, Only if Hybrid Enabled) ---
    def _get_wagtail_page_model(self):
        """Lazy import of Wagtail Page model. Returns None if unavailable."""
        try:
            from wagtail.models import Page  # type: ignore

            return Page
        except Exception as e:
            logger.debug("Wagtail Page model not available: %s", e)
            return None

    def _get_page_to_documents_function(self) -> Optional[Callable]:
        """Lazy import page_to_documents function from wagtail_rag.content_extraction."""
        try:
            from wagtail_rag.content_extraction import page_to_documents_api_extractor
            return page_to_documents_api_extractor
        except Exception as e:
            logger.debug("page_to_documents_api_extractor not available: %s", e)
            return None

    def _convert_wagtail_page_to_documents(self, page: Any) -> List[Document]:
        """
        Convert a Wagtail Page object into Document objects using the same logic as indexing.
        
        This ensures consistency between indexed documents and search-time documents.
        Uses api_fields_extractor for intelligent field-based extraction.
        """
        try:
            page_to_documents_func = self._get_page_to_documents_function()
            
            if page_to_documents_func:
                # Use the same document conversion logic as indexing
                # This creates multiple documents with proper metadata
                documents = page_to_documents_func(page)
                
                # Mark these documents as coming from Wagtail search
                for doc in documents:
                    doc.metadata["from_wagtail_search"] = True
                
                return documents
            else:
                # Fallback: create a simple document if page_to_documents is not available
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

    def _normalize_query_for_wagtail(self, query: str) -> str:
        """
        Normalize query for Wagtail search to improve matching.
        
        Wagtail search works better with clean keywords. This method:
        - Removes question marks and other punctuation
        - Cleans up whitespace
        
        Args:
            query: Original search query
            
        Returns:
            Normalized query string for Wagtail search
        """
        # Remove question marks and common punctuation
        normalized = query.strip('?').strip('!').strip('.')
        
        # Clean up extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized.strip()

    def _get_wagtail_docs(self, query: str, seen_urls: Set[str], seen_ids: Set[Any]) -> List[Document]:
        """
        Perform Wagtail full-text search (secondary search method, only if hybrid search enabled).
        
        This method is only called when use_hybrid_search=True. It uses Wagtail's
        built-in search engine to find pages with keyword matches.
        
        Args:
            query: Search query string
            seen_urls: URLs already found in vector search (for deduplication)
            seen_ids: Page IDs already found in vector search (for deduplication)
            
        Returns:
            List of Document objects from Wagtail search (excluding duplicates from vector search)
        """
        # Early return if hybrid search is disabled
        if not self.use_hybrid_search:
            return []

        docs: List[Document] = []
        Page = self._get_wagtail_page_model()
        if Page is None:
            return []

        try:
            # Normalize query for better Wagtail search matching
            normalized_query = self._normalize_query_for_wagtail(query)
            if normalized_query != query:
                logger.debug(f"Normalized Wagtail query: '{query}' -> '{normalized_query}'")
            
            # Use Wagtail's full-text search (PostgreSQL or Elasticsearch)
            wagtail_results = Page.objects.live().search(normalized_query)[:10]
            logger.debug(f"Wagtail search returned {len(wagtail_results)} pages")
            
            # If normalized query found nothing, try original
            if len(wagtail_results) == 0 and normalized_query != query:
                logger.debug(f"Trying original query after normalized query found 0 results")
                wagtail_results = Page.objects.live().search(query)[:10]
                
            if len(wagtail_results) == 0:
                logger.debug(
                    f"Wagtail search found 0 pages for query: '{query}'. "
                    f"Check: 1) Content is indexed, 2) Pages are live, 3) Search index is updated (python manage.py update_index)"
                )
            
            skipped_count = 0
            for page in wagtail_results:
                page_url = getattr(page, "url", "")
                page_id = getattr(page, "id", None)

                # Skip pages already found in vector search (deduplication)
                if page_url in seen_urls or page_id in seen_ids:
                    skipped_count += 1
                    logger.debug(f"Skipping duplicate page: {page_url} (ID: {page_id})")
                    continue

                # Convert Wagtail Page to Document objects
                page_docs = self._convert_wagtail_page_to_documents(page)
                if page_docs:
                    docs.extend(page_docs)
                    seen_urls.add(page_url)
                    seen_ids.add(page_id)
                    logger.debug(f"Added {len(page_docs)} document(s) from page ID: {page_id}")
            
            if skipped_count > 0:
                logger.debug(
                    f"Deduplicated {skipped_count}/{len(wagtail_results)} Wagtail results (already in vector search)"
                )
        except Exception as e:
            # Log exception but keep search resilient in non-Wagtail environments
            logger.warning(f"Wagtail search failed for query '{query}': {e}", exc_info=True)

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
        except Exception as e:
            logger.debug("Title boost for short query failed: %s", e)

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
        """
        Retrieve documents using embedding search, optionally combined with Wagtail search.
        
        Flow:
        1. Always perform vector search (semantic similarity via embeddings)
        2. If hybrid search is enabled, add Wagtail full-text search results
        3. Combine, deduplicate, and rank results
        
        Args:
            query: Search query string
            boost_title_matches: Whether to boost documents with matching titles
            
        Returns:
            List of Document objects ranked by relevance
        """
        # Step 1: Always perform vector search (primary search method)
        logger.info(f"Searching for: '{query}'")
        vector_docs, seen_urls, seen_ids = self._get_vector_docs(query)
        logger.debug(f"Vector search: {len(vector_docs)} documents")
        
        # Step 2: Optionally add Wagtail search results (only if hybrid search is enabled)
        wagtail_docs = []
        if self.use_hybrid_search:
            wagtail_docs = self._get_wagtail_docs(query, seen_urls, seen_ids)
            logger.debug(f"Wagtail search: {len(wagtail_docs)} additional documents")
            
            # Log if no results at all
            if len(wagtail_docs) == 0 and len(vector_docs) == 0:
                logger.warning(
                    f"No results found for '{query}'. "
                    f"Check: 1) Content is indexed (python manage.py build_rag_index), "
                    f"2) Search index is updated (python manage.py update_index)"
                )
        else:
            logger.debug(f"Hybrid search disabled")
        
        # Step 3: Combine and limit results
        all_docs = vector_docs + wagtail_docs
        docs = all_docs[: self.k_value]
        logger.debug(f"Combined: {len(all_docs)} total ({len(vector_docs)} vector + {len(wagtail_docs)} wagtail)")

        # Step 4: Boost title matches for short queries
        docs = self._boost_title_matches_if_short_query(query, docs)

        # Step 5: Re-rank by title similarity if enabled
        if boost_title_matches and docs:
            docs = self._rerank_by_title_match(query, docs)

        logger.info(f"Returning {len(docs)} document(s) for '{query}'")
        return docs

    def _deduplicate_results(self, raw_results: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        """Deduplicate search results by URL/ID and keep best-scoring per key."""
        best_by_key: Dict[Tuple[str, Optional[Any]], Tuple[Document, float]] = {}
        for doc, score in raw_results:
            meta = getattr(doc, "metadata", {}) or {}
            url = meta.get("url") or ""
            doc_id = meta.get("page_id") or meta.get("id")
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
        """
        Lower-level search: vector similarity only, with score and optional metadata filter.

        This is an internal utility for callers that need raw scored results and filtering.
        For normal RAG retrieval (including hybrid search and title boosting), use
        retrieve_with_embeddings() instead.
        """
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

