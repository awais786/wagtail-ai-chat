"""
Embedding Search Module for RAG Chatbot.

This module handles all embedding-based similarity search functionality,
including vector search, hybrid search with Wagtail, and result ranking.
"""
import logging

from django.conf import settings

logger = logging.getLogger(__name__)

# Import Document class for creating document objects
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        # Fallback: create a simple Document class
        class Document:
            def __init__(self, page_content, metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}


def _get_wagtail_page():
    """Get Wagtail Page model (lazy import to avoid AppRegistryNotReady)."""
    try:
        from wagtail.models import Page
        return Page
    except (ImportError, Exception):
        return None


def _get_content_extraction_utils():
    """Get content extraction utilities (lazy import)."""
    try:
        from wagtail_rag.content_extraction import extract_page_content, get_page_url, extract_streamfield_text
        return extract_page_content, get_page_url, extract_streamfield_text
    except ImportError:
        return None, None, None


def _strip_html(text):
    """Strip HTML tags from a string for cleaner embeddings/search content."""
    if not text:
        return ""

    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    except ImportError:
        import re
        # Fallback: remove tags and compress whitespace
        return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def _fuzzy_match(query_str, target_str, min_length=3):
    """Check if query fuzzy matches target (handles simple typos)."""
    if not query_str or not target_str or len(query_str) < min_length:
        return False

    # Direct substring match
    if query_str in target_str:
        return True

    # Check if query matches with one character off (e.g., "andama" matches "anadama")
    if len(target_str) > len(query_str):
        return target_str.endswith(query_str) or query_str in target_str[1:]

    return False


class EmbeddingSearcher:
    """
    Handles embedding-based similarity search for RAG chatbot.
    
    This class encapsulates all embedding search logic including:
    - Vector similarity search via ChromaDB
    - Hybrid search with Wagtail full-text search
    - Title-based boosting and ranking
    - Result deduplication and cleaning
    """
    
    def __init__(self, vectorstore, retriever, k_value, use_hybrid_search=True):
        """
        Initialize the embedding searcher.
        
        Args:
            vectorstore: ChromaDB vectorstore instance
            retriever: LangChain retriever instance
            k_value: Default number of results to retrieve
            use_hybrid_search: Whether to enable Wagtail hybrid search
        """
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.k_value = k_value
        self.use_hybrid_search = use_hybrid_search
    
    def _convert_wagtail_page_to_document(self, page):
        """
        Convert a Wagtail page to a LangChain Document.

        Args:
            page: Wagtail Page instance

        Returns:
            Document object or None if conversion fails
        """
        try:
            extract_page_content, get_page_url_util, extract_streamfield_text = _get_content_extraction_utils()

            # Get page URL
            page_url = page.url if hasattr(page, 'url') else ''
            if get_page_url_util:
                page_url = get_page_url_util(page) or page_url

            # Extract page content
            page_text = None
            if extract_page_content:
                page_text = extract_page_content(page)
            else:
                # Fallback extraction
                parts = []
                if hasattr(page, 'title'):
                    parts.append(f"Title: {page.title}")
                if hasattr(page, 'search_description') and page.search_description:
                    parts.append(page.search_description)
                if hasattr(page, 'body'):
                    body = getattr(page, 'body', None)
                    if body:
                        if extract_streamfield_text:
                            streamfield_text = extract_streamfield_text(body)
                            if streamfield_text:
                                parts.append(streamfield_text[:500])
                        else:
                            parts.append(str(body)[:500])
                page_text = '\n\n'.join(parts) if parts else None

            if not page_text or not page_text.strip():
                return None

            return Document(
                page_content=page_text,
                metadata={
                    'title': page.title if hasattr(page, 'title') else '',
                    'url': page_url,
                    'id': page.id,
                    'source': f"{page._meta.app_label}.{page.__class__.__name__}",
                    'model': page.__class__.__name__,
                    'app': page._meta.app_label,
                    'from_wagtail_search': True
                }
            )
        except Exception:
            return None

    def _get_vector_docs(self, query):
        """
        Get documents from ChromaDB vector search.

        Args:
            query: Search query string

        Returns:
            Tuple of (documents_list, seen_urls_set, seen_ids_set)
        """
        docs = []
        seen_urls = set()
        seen_ids = set()

        try:
            # New LangChain API uses invoke()
            docs = self.retriever.invoke(query)
        except Exception:
            # Fallback to similarity_search
            try:
                docs = self.vectorstore.similarity_search(query, k=self.k_value)
            except Exception:
                pass

        # Track seen URLs/IDs
        for doc in docs:
            url = doc.metadata.get('url', '')
            doc_id = doc.metadata.get('id')
            if url:
                seen_urls.add(url)
            if doc_id:
                seen_ids.add(doc_id)

        return docs, seen_urls, seen_ids

    def _get_wagtail_docs(self, query, seen_urls, seen_ids):
        """
        Get documents from Wagtail full-text search.

        Args:
            query: Search query string
            seen_urls: Set of URLs already seen
            seen_ids: Set of IDs already seen

        Returns:
            List of Document objects from Wagtail search
        """
        if not self.use_hybrid_search:
            return []

        docs = []
        try:
            Page = _get_wagtail_page()
            if Page is None:
                return []

            wagtail_results = Page.objects.live().search(query)[:10]

            for page in wagtail_results:
                page_url = page.url if hasattr(page, 'url') else ''
                page_id = page.id

                # Skip if already seen
                if page_url in seen_urls or page_id in seen_ids:
                    continue

                doc = self._convert_wagtail_page_to_document(page)
                if doc:
                    docs.append(doc)
                    seen_urls.add(page_url)
                    seen_ids.add(page_id)
        except Exception:
            pass

        return docs

    def _boost_title_matches_if_short_query(self, query, docs):
        """Try to find title matches for short queries (handles typos)."""
        query_clean = query.lower().strip('?').strip()
        if len(query_clean.split()) > 2 or len(query_clean) < 4:
            return docs

        try:
            # Get more results to find title matches
            all_results = self.vectorstore.similarity_search_with_score(query, k=min(self.k_value * 3, 30))
            title_matches = []

            for doc, score in all_results:
                title = doc.metadata.get('title', '').lower()
                title_clean = title.replace('#', '').replace(' ', '')

                # Direct substring matches
                if query_clean in title or title.startswith(query_clean):
                    title_matches.append((doc, score))
                # Character-based fuzzy matching (handles typos like "andama" -> "anadama")
                elif len(query_clean) >= 4:
                    query_chars = list(query_clean)
                    title_chars = list(title_clean)
                    # Check if query chars appear in order in title (allowing one missing char)
                    query_idx = 0
                    for char in title_chars:
                        if query_idx < len(query_chars) and char == query_chars[query_idx]:
                            query_idx += 1
                    # If most chars match in order, consider it a match
                    if query_idx >= len(query_clean) - 1:  # Allow 1 char difference
                        title_matches.append((doc, score))

            # If we found title matches, prioritize them
            if title_matches:
                # Sort by score (lower is better for similarity_search_with_score)
                title_matches.sort(key=lambda x: x[1])
                # Convert to Document list and prepend to results
                title_docs = [doc for doc, _ in title_matches[:3]]
                # Combine: title matches first, then other docs
                existing_ids = {id(d) for d in title_docs}
                other_docs = [d for d in docs if id(d) not in existing_ids]
                return title_docs + other_docs[:self.k_value - len(title_docs)]
        except Exception:
            pass

        return docs

    def _rerank_by_title_match(self, query, docs):
        """Rerank documents by title similarity to query."""
        query_lower = query.lower().strip('?').strip()
        query_words = set(query_lower.split())

        # Score documents by title match, preserving original index
        scored_docs = []
        for idx, doc in enumerate(docs):
            score = 0
            title = doc.metadata.get('title', '').lower()
            if title:
                # Exact title match (highest priority)
                if query_lower in title or title in query_lower:
                    score = 1.0
                # Fuzzy match for typos (e.g., "andama" -> "anadama")
                elif _fuzzy_match(query_lower, title, min_length=4):
                    score = 0.9
                else:
                    # Check for word matches
                    title_words = set(title.split())
                    matches = query_words.intersection(title_words)
                    if matches:
                        score = len(matches) / len(query_words)

                    # Check for substring matches in individual words
                    for query_word in query_words:
                        if len(query_word) >= 4:  # Only for words 4+ chars
                            if query_word in title:
                                score = max(score, 0.7)
                            # Check fuzzy match in title words
                            for title_word in title_words:
                                if len(title_word) >= 4 and _fuzzy_match(query_word, title_word):
                                    score = max(score, 0.7)

            scored_docs.append((score, idx, doc))

        # Sort by title match score (descending), then by original index for ties
        scored_docs.sort(key=lambda x: (-x[0], x[1]))
        return [doc for _, _, doc in scored_docs]

    def retrieve_with_embeddings(self, query, boost_title_matches=True):
        """
        Retrieve documents using embedding search.

        This method performs embedding-based similarity search and combines
        results from multiple sources (ChromaDB vector search and optional
        Wagtail full-text search).

        Args:
            query: Search query string
            boost_title_matches: If True, prioritize documents whose titles
                look similar to the query (helps with short queries like bread names)

        Returns:
            List of Document objects retrieved via embedding search
        """
        # 1. Get documents from ChromaDB vector search
        vector_docs, seen_urls, seen_ids = self._get_vector_docs(query)

        # 2. Get documents from Wagtail search (hybrid search)
        wagtail_docs = self._get_wagtail_docs(query, seen_urls, seen_ids)

        # 3. Combine and limit results
        all_docs = vector_docs + wagtail_docs
        docs = all_docs[:self.k_value]

        # 4. For short queries, try to find title matches
        docs = self._boost_title_matches_if_short_query(query, docs)

        # 5. Boost title matches if enabled
        if boost_title_matches and docs:
            docs = self._rerank_by_title_match(query, docs)

        return docs

    def _deduplicate_results(self, raw_results, k):
        """Deduplicate search results by URL/ID."""
        best_by_key = {}
        for doc, score in raw_results:
            meta = doc.metadata or {}
            url = meta.get("url") or ""
            doc_id = meta.get("id")
            key = (url, doc_id)
            # For similarity_search_with_score, lower score = more similar
            if key not in best_by_key or score < best_by_key[key][1]:
                best_by_key[key] = (doc, score)

        # Keep at most k unique pages
        deduped = list(best_by_key.values())
        return sorted(deduped, key=lambda pair: pair[1])[:k]

    def _apply_conservative_title_boost(self, query, results, k):
        """Apply conservative title boosting for short, specific queries."""
        query_clean = query.lower().strip()
        query_words = set(query_clean.split())
        # Consider only "significant" query words (length >= 3)
        sig_query_words = {w for w in query_words if len(w) >= 3}
        
        # Only try title boosting when we have a very short, specific query
        if not (0 < len(sig_query_words) <= 3):
            return results

        title_matches = []
        other_results = []

        # Optional safety: only boost results that are already reasonably close
        max_title_boost_score = getattr(settings, "WAGTAIL_RAG_TITLE_BOOST_MAX_SCORE", None)

        for result in results:
            title = result["metadata"].get("title", "").lower()
            title_words = set(title.split())

            # Require that ALL significant query words appear in the title
            has_all_words = sig_query_words.issubset(title_words)

            # Optionally require the embedding score to be "good enough"
            score_ok = (
                True
                if max_title_boost_score is None
                else result["score"] <= max_title_boost_score
            )

            if has_all_words and score_ok:
                title_matches.append(result)
            else:
                other_results.append(result)

        # Return title matches first, then others (but only if we have title matches)
        if title_matches:
            title_matches.sort(key=lambda x: x["score"])
            return title_matches + other_results[:max(0, k - len(title_matches))]

        return results

    def search_with_embeddings(self, query, k=None, metadata_filter=None):
        """
        Search for similar content using embedding search (without generating an LLM response).

        This method performs embedding-based similarity search to find relevant documents
        but does NOT call the LLM to generate an answer.

        Args:
            query: Search query string
            k: Number of results to return (default: from settings or 10)
            metadata_filter: Optional filter dict (e.g., {'model': 'BreadPage'})

        Returns:
            List[dict] where each item is:
              {
                  "content": plain-text snippet (HTML stripped),
                  "metadata": original metadata dict stored with the chunk,
                  "score": similarity score from Chroma
              }
        """
        if k is None:
            k = getattr(settings, 'WAGTAIL_RAG_SEARCH_K', 10)

        # Apply filter if provided
        if metadata_filter:
            raw_results = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=metadata_filter
            )
        else:
            raw_results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Deduplicate by URL/ID so we don't show the same page multiple times
        deduped = self._deduplicate_results(raw_results, k)

        # Clean HTML from results
        cleaned_results = []
        for doc, score in deduped:
            raw_content = getattr(doc, "page_content", "") or ""
            clean_content = _strip_html(raw_content)
            cleaned_results.append(
                {
                    "content": clean_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
            )

        # Apply conservative title-based re-ranking for short, specific queries
        return self._apply_conservative_title_boost(query, cleaned_results, k)

