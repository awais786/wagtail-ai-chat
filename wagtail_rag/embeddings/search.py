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
            logger.warning(f"  → Using retriever.invoke() (may use MultiQueryRetriever with LLM query expansion)")
            docs = self.retriever.invoke(query)
        except Exception:
            # Fallback to direct vectorstore search
            try:
                logger.warning(f"  → Fallback: Using direct vectorstore.similarity_search()")
                docs = self.vectorstore.similarity_search(query, k=self.k_value)
            except Exception:
                logger.warning(f"  → Error: Vector search failed, returning empty results")
                docs = []

        # Track URLs and IDs for deduplication (used when combining with Wagtail results)
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            url = meta.get("url", "")
            doc_id = meta.get("id")
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

    def _normalize_query_for_wagtail(self, query: str) -> str:
        """
        Normalize query for Wagtail search to improve matching.
        
        Wagtail search works better with clean keywords. This method:
        - Removes question marks and other punctuation
        - Handles common question patterns
        - Extracts key terms from natural language queries
        
        Args:
            query: Original search query
            
        Returns:
            Normalized query string for Wagtail search
        """
        import re
        
        # Remove question marks and common punctuation
        normalized = query.strip('?').strip('!').strip('.')
        
        # Handle common question patterns
        question_patterns = [
            (r'^what\s+is\s+', ''),
            (r'^what\s+are\s+', ''),
            (r'^what\s+', ''),
            (r'^where\s+is\s+', ''),
            (r'^where\s+are\s+', ''),
            (r'^where\s+', ''),
            (r'^how\s+to\s+', ''),
            (r'^how\s+do\s+', ''),
            (r'^how\s+', ''),
            (r'^some\s+thing\s+to\s+', ''),  # "some thing to eat?" -> "eat"
            (r'^something\s+to\s+', ''),      # "something to eat?" -> "eat"
            (r'^tell\s+me\s+about\s+', ''),
            (r'^i\s+want\s+to\s+', ''),
            (r'^i\s+need\s+', ''),
        ]
        
        for pattern, replacement in question_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        normalized = ' '.join(normalized.split())
        
        # If query becomes too short or empty, use original (cleaned)
        if len(normalized.strip()) < 2:
            normalized = query.strip('?').strip('!').strip('.')
        
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
                logger.info(f"Normalized Wagtail query: '{query}' -> '{normalized_query}'")
            
            # Use Wagtail's full-text search (PostgreSQL or Elasticsearch)
            # Try normalized query first, fallback to original if needed
            wagtail_results = Page.objects.live().search(normalized_query)[:10]
            logger.info(f"Wagtail search returned {len(wagtail_results)} pages for query: '{query}'")
            
            if len(wagtail_results) == 0:
                # Try original query if normalized query found nothing
                if normalized_query != query:
                    logger.info(f"Trying original query '{query}' after normalized query '{normalized_query}' found 0 results")
                    wagtail_results = Page.objects.live().search(query)[:10]
                
                if len(wagtail_results) == 0:
                    logger.warning(
                        f"⚠️  Wagtail search found 0 pages for query: '{query}' (normalized: '{normalized_query}'). "
                        f"This might mean:\n"
                        f"  1. No pages contain this keyword in their searchable content\n"
                        f"  2. Wagtail search index needs to be updated (run: python manage.py update_index)\n"
                        f"  3. Pages might not be published/live\n"
                        f"  4. Query might be too vague or use words not in your content"
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

                # Convert Wagtail Page to Document format
                doc = self._convert_wagtail_page_to_document(page)
                if doc:
                    docs.append(doc)
                    # Track to avoid duplicates in future iterations
                    seen_urls.add(page_url)
                    seen_ids.add(page_id)
                    logger.debug(f"Added Wagtail page to results: {page_url} (ID: {page_id})")
            
            if skipped_count > 0:
                logger.info(
                    f"ℹ️  Deduplication: Skipped {skipped_count} out of {len(wagtail_results)} Wagtail pages "
                    f"for query: '{query}' because they were already found in vector search. "
                    f"This is expected behavior - preventing duplicate results."
                )
                
            if skipped_count == len(wagtail_results) and len(wagtail_results) > 0:
                logger.info(
                    f"ℹ️  All {len(wagtail_results)} Wagtail search results were duplicates of vector search results "
                    f"for query: '{query}'. This means:\n"
                    f"  ✓ Wagtail search is working correctly\n"
                    f"  ✓ Vector search already found all relevant pages\n"
                    f"  ✓ No duplicate results will be returned (good!)\n"
                    f"  ℹ️  This is normal when your content is well-indexed in the vector database"
                )
        except Exception as e:
            # Log exception but keep search resilient in non-Wagtail environments
            logger.warning(f"⚠️  Wagtail search failed for query '{query}': {e}", exc_info=True)

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
        logger.warning(f"🔍 STEP 1: Vector Search - Starting vector search for query: '{query}'")
        vector_docs, seen_urls, seen_ids = self._get_vector_docs(query)
        logger.warning(f"✅ STEP 1: Vector Search - Found {len(vector_docs)} documents for query: '{query}'")
        
        # Step 2: Optionally add Wagtail search results (only if hybrid search is enabled)
        wagtail_docs = []
        if self.use_hybrid_search:
            logger.warning(f"🔍 STEP 2: Hybrid Search (Wagtail) - Starting Wagtail search for query: '{query}'")
            wagtail_docs = self._get_wagtail_docs(query, seen_urls, seen_ids)
            logger.warning(f"✅ STEP 2: Hybrid Search (Wagtail) - Found {len(wagtail_docs)} additional documents (after deduplication) for query: '{query}'")
            
            # Warn if Wagtail search didn't add any new results
            if len(wagtail_docs) == 0 and len(vector_docs) > 0:
                logger.info(
                    f"ℹ️  Hybrid search enabled for query: '{query}'. "
                    f"Vector search found {len(vector_docs)} documents. "
                    f"Wagtail search found 0 additional unique documents (all were duplicates or no matches). "
                    f"This is normal - it means your vector search is comprehensive and already found all relevant pages."
                )
            elif len(wagtail_docs) == 0 and len(vector_docs) == 0:
                logger.warning(
                    f"⚠️  Both vector and Wagtail search returned 0 results for query: '{query}'. "
                    f"Consider:\n"
                    f"  1. Checking if content is indexed (run: python manage.py build_rag_index)\n"
                    f"  2. Updating Wagtail search index (run: python manage.py update_index)\n"
                    f"  3. Verifying the query matches your content"
                )
        else:
            logger.warning(f"⏭️  STEP 2: Hybrid Search (Wagtail) - Skipped (hybrid search disabled)")
        # If use_hybrid_search is False, wagtail_docs will be empty list

        # Step 3: Combine results (vector search + optional Wagtail search)
        all_docs = vector_docs + wagtail_docs
        logger.warning(f"📊 STEP 2 Summary: Total documents after combining: {len(all_docs)} (vector: {len(vector_docs)}, wagtail: {len(wagtail_docs)})")
        
        # Info message if results are the same as vector-only (this is actually good - means comprehensive indexing)
        if self.use_hybrid_search and len(wagtail_docs) == 0 and len(all_docs) == len(vector_docs) and len(vector_docs) > 0:
            logger.info(
                f"ℹ️  Hybrid search results for query: '{query}': "
                f"Returning {len(all_docs)} documents from vector search. "
                f"Wagtail search verified these results (no additional unique pages found). "
                f"This indicates your vector database has comprehensive coverage of your content."
            )
        
        # Step 4: Limit to top K documents
        docs = all_docs[: self.k_value]

        # Step 5: Boost title matches for short queries (e.g., "Bread")
        docs = self._boost_title_matches_if_short_query(query, docs)

        # Step 6: Re-rank by title similarity if enabled
        if boost_title_matches and docs:
            docs = self._rerank_by_title_match(query, docs)

        logger.warning(f"✅ Search Complete: Returning {len(docs)} documents for query: '{query}'")
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

