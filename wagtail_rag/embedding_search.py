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
        from .content_extraction import extract_page_content, get_page_url, extract_streamfield_text
        return extract_page_content, get_page_url, extract_streamfield_text
    except ImportError:
        return None, None, None


def _strip_html(text):
    """
    Strip HTML tags from a string for cleaner embeddings/search content.
    
    Used when returning search results so that `content` is human‑readable
    and does not include raw HTML from RichText / StreamField rendering.
    """
    if not text:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        import re
        # Very simple fallback: remove tags and compress whitespace
        no_tags = re.sub(r"<[^>]+>", " ", text)
        return " ".join(no_tags.split())


def _fuzzy_match(query_str, target_str, min_length=3):
    """Check if query is a substring of target (handles typos)"""
    if not query_str or not target_str:
        return False
    # Direct substring match
    if query_str in target_str:
        return True
    # Check if query is close to start of target (handles missing first char)
    if len(query_str) >= min_length and len(target_str) > len(query_str):
        # Check if query matches end of target (e.g., "andama" matches "anadama")
        if target_str.endswith(query_str) or query_str in target_str[1:]:
            return True
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
        # Collect documents from multiple sources
        all_docs = []
        seen_urls = set()
        seen_ids = set()
        
        # 1. Get documents from ChromaDB vector search
        try:
            # New LangChain API uses invoke() - returns list of Document objects
            vector_docs = self.retriever.invoke(query)
            all_docs.extend(vector_docs)
            
            # Track seen URLs/IDs
            for doc in vector_docs:
                url = doc.metadata.get('url', '')
                doc_id = doc.metadata.get('id')
                if url:
                    seen_urls.add(url)
                if doc_id:
                    seen_ids.add(doc_id)
        except Exception as e:
            # Fallback if retriever fails
            try:
                vector_docs = self.vectorstore.similarity_search(query, k=self.k_value)
                all_docs.extend(vector_docs)
                
                # Track seen URLs/IDs from fallback results
                for doc in vector_docs:
                    url = doc.metadata.get('url', '')
                    doc_id = doc.metadata.get('id')
                    if url:
                        seen_urls.add(url)
                    if doc_id:
                        seen_ids.add(doc_id)
            except Exception:
                pass
        
        # 2. Get documents from Wagtail search (hybrid search)
        if self.use_hybrid_search:
            try:
                Page = _get_wagtail_page()
                if Page is None:
                    raise ImportError("Wagtail not available")
                wagtail_results = Page.objects.live().search(query)
                # Convert Wagtail search results to Document format
                for page in wagtail_results[:10]:  # Limit to top 10 from Wagtail
                    page_url = page.url if hasattr(page, 'url') else ''
                    page_id = page.id
                    
                    # Skip if already seen
                    if page_url in seen_urls or page_id in seen_ids:
                        continue
                    
                    # Extract text from page using content extraction utilities
                    try:
                        # Try to use content extraction utilities if available
                        extract_page_content, get_page_url_util, extract_streamfield_text = _get_content_extraction_utils()
                        
                        if extract_page_content:
                            # Use utility function for comprehensive extraction
                            page_text = extract_page_content(page)
                            if page_text:
                                # Use utility for URL if available
                                if get_page_url_util:
                                    page_url = get_page_url_util(page) or page_url
                        else:
                            # Fallback to simple extraction
                            page_text = ""
                            if hasattr(page, 'title'):
                                page_text = f"Title: {page.title}\n\n"
                            if hasattr(page, 'search_description') and page.search_description:
                                page_text += f"{page.search_description}\n\n"
                            if hasattr(page, 'body'):
                                # Extract text from StreamField if available
                                body = getattr(page, 'body', None)
                                if body:
                                    # Try to use utility for StreamField extraction
                                    if extract_streamfield_text:
                                        streamfield_text = extract_streamfield_text(body)
                                        if streamfield_text:
                                            page_text += streamfield_text[:500]  # Limit length
                                    else:
                                        # Simple text extraction fallback
                                        page_text += str(body)[:500]
                        
                        if page_text and page_text.strip():
                            doc = Document(
                                page_content=page_text,
                                metadata={
                                    'title': page.title if hasattr(page, 'title') else '',
                                    'url': page_url,
                                    'id': page_id,
                                    'source': f"{page._meta.app_label}.{page.__class__.__name__}",
                                    'model': page.__class__.__name__,
                                    'app': page._meta.app_label,
                                    'from_wagtail_search': True
                                }
                            )
                            all_docs.append(doc)
                            seen_urls.add(page_url)
                            seen_ids.add(page_id)
                    except Exception:
                        continue
            except Exception:
                pass  # Wagtail search not available or failed
        
        # Use combined results and limit to k_value
        docs = all_docs
        if len(docs) > self.k_value:
            docs = docs[:self.k_value]
        
        # For short queries (likely bread names), also try direct metadata search
        query_clean = query.lower().strip('?').strip()
        if len(query_clean.split()) <= 2 and len(query_clean) >= 4:
            try:
                # Get more results to find title matches
                all_results = self.vectorstore.similarity_search_with_score(query, k=min(self.k_value * 3, 30))
                # Check if any have matching titles using character-based similarity
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
                    docs = title_docs + other_docs[:self.k_value - len(title_docs)]
            except Exception:
                pass  # Continue with normal results
        
        # Boost title matches if enabled
        if boost_title_matches and docs:
            # Extract potential title from query (simple heuristic)
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
            docs = [doc for _, _, doc in scored_docs]
        
        return docs
    
    def search_with_embeddings(self, query, k=None, metadata_filter=None):
        """
        Search for similar content using embedding search (without generating an LLM response).
        
        This method performs embedding-based similarity search to find relevant documents
        but does NOT call the LLM to generate an answer.
        
        Args:
            query: Search query string.
            k: Number of results to return (default: from settings or 10).
            metadata_filter: Optional filter dict (e.g., {'model': 'BreadPage'})
            
        Returns:
            List[dict] where each item is:
              {
                  "content": plain‑text snippet (HTML stripped),
                  "metadata": original metadata dict stored with the chunk,
                  "score": similarity score from Chroma
              }

        Notes:
            This method does NOT call the LLM. It only:
              1. Uses the embeddings + Chroma to find similar chunks (embedding search).
              2. Deduplicates results so each page (URL/ID) appears at most once.
              3. Strips HTML from the stored chunk text for cleaner display.
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
        deduped = sorted(deduped, key=lambda pair: pair[1])[:k]

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
        
        # Optional title-based re‑ranking for short, specific queries
        # This is intentionally conservative to avoid pulling in many irrelevant matches.
        query_clean = query.lower().strip()
        query_words = set(query_clean.split())
        # Consider only "significant" query words (length >= 3)
        sig_query_words = {w for w in query_words if len(w) >= 3}
        
        # Only try title boosting when we have a very short, specific query
        if 0 < len(sig_query_words) <= 3:
            title_matches = []
            other_results = []
            
            # Optional safety: only boost results that are already reasonably close
            # You can override this in Django settings, e.g. 0.8 for cosine distance.
            max_title_boost_score = getattr(
                settings,
                "WAGTAIL_RAG_TITLE_BOOST_MAX_SCORE",
                None,  # disabled by default to stay backend‑agnostic
            )
            
            for result in cleaned_results:
                title = result["metadata"].get("title", "").lower()
                title_words = set(title.split())
                
                # Require that ALL significant query words appear in the title
                # (e.g. "multigrain bread" -> title must contain both "multigrain" and "bread")
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
                # Sort title matches by score (lower is better)
                title_matches.sort(key=lambda x: x["score"])
                return title_matches + other_results[:max(0, k - len(title_matches))]
        
        return cleaned_results

