"""
RAG Chatbot using LangChain and ChromaDB.

This module provides a RAG (Retrieval-Augmented Generation) chatbot
that uses the indexed content from Wagtail pages.
"""
import os
import logging

from django.conf import settings

# Setup logging
logger = logging.getLogger(__name__)

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    raise ImportError(
        "Could not import LangChain components. "
        "Please install langchain-community: pip install langchain-community"
    )

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError(
            "Could not import HuggingFaceEmbeddings. "
            "Please install langchain-huggingface or langchain-community"
        )

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    USE_LCEL = True
except ImportError:
    ChatPromptTemplate = None
    RunnablePassthrough = None
    StrOutputParser = None
    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA
        USE_LCEL = False
    except ImportError:
        # Fallback - use simple pattern
        PromptTemplate = None
        RetrievalQA = None
        USE_LCEL = None

# Try to import MultiQueryRetriever
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

# Import Wagtail for hybrid search
try:
    from wagtail.models import Page
    WAGTAIL_AVAILABLE = True
except ImportError:
    WAGTAIL_AVAILABLE = False
    Page = None

# Import LLM provider factory
from .llm_providers import get_llm


class RAGChatBot:
    """RAG Chatbot that retrieves context from ChromaDB and generates responses."""
    
    def __init__(
        self,
        collection_name=None,
        model_name=None,
        persist_directory=None,
        metadata_filter=None,
        llm_provider=None,
        llm_kwargs=None
    ):
        """
        Initialize the RAG Chatbot.
        
        Args:
            collection_name: Name of the ChromaDB collection (default: from settings)
            model_name: Name of the LLM model (default: from settings or 'mistral')
            persist_directory: Path to ChromaDB persistence directory (default: from settings)
            metadata_filter: Dict to filter by metadata (e.g., {'model': 'BreadPage'}) (default: None)
            llm_provider: LLM provider type ('ollama', 'openai', 'anthropic', etc.) (default: from settings or 'ollama')
            llm_kwargs: Additional kwargs to pass to LLM initialization (default: None)
        """
        # Get collection name from settings or use default
        if collection_name is None:
            collection_name = getattr(
                settings,
                'WAGTAIL_RAG_COLLECTION_NAME',
                'wagtail_rag'
            )
        
        # Get model name from settings or use default
        if model_name is None:
            model_name = getattr(
                settings,
                'WAGTAIL_RAG_MODEL_NAME',
                'mistral'  # Default to mistral (change to llama2 if preferred)
            )
        
        # Get persist directory from settings or use default
        if persist_directory is None:
            persist_directory = getattr(
                settings,
                'WAGTAIL_RAG_CHROMA_PATH',
                os.path.join(settings.BASE_DIR, 'chroma_db')
            )
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.metadata_filter = metadata_filter or {}
        
        # Initialize embeddings
        embedding_model = getattr(
            settings,
            'WAGTAIL_RAG_EMBEDDING_MODEL',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        
        # Load vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Initialize LLM using the factory function
        llm_kwargs = llm_kwargs or {}
        self.llm = get_llm(
            provider=llm_provider,
            model_name=self.model_name,
            **llm_kwargs
        )
        
        # Create base retriever with configurable parameters
        k_value = getattr(settings, 'WAGTAIL_RAG_RETRIEVE_K', 8)
        
        # Build search kwargs with optional metadata filtering
        search_kwargs = {"k": k_value}
        if self.metadata_filter:
            search_kwargs["filter"] = self.metadata_filter
        
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )
        
        # Use MultiQueryRetriever if available and enabled
        use_multi_query = getattr(settings, 'WAGTAIL_RAG_USE_MULTI_QUERY', True) and MULTI_QUERY_AVAILABLE
        if use_multi_query:
            try:
                self.retriever = MultiQueryRetriever.from_llm(
                    retriever=base_retriever,
                    llm=self.llm
                )
            except Exception:
                # Fallback to base retriever if MultiQuery fails
                self.retriever = base_retriever
        else:
            self.retriever = base_retriever
        
        # Store k value for limiting results
        self.k_value = k_value
        
        # Enable hybrid search (Wagtail + ChromaDB)
        self.use_hybrid_search = getattr(settings, 'WAGTAIL_RAG_USE_HYBRID_SEARCH', True) and WAGTAIL_AVAILABLE
        
        # Create prompt template
        prompt_template_str = getattr(
            settings,
            'WAGTAIL_RAG_PROMPT_TEMPLATE',
            """You are a helpful assistant. Use the following pieces of context from the website to answer the question accurately.
Pay attention to titles, bread types, and other metadata in the context to provide accurate information.

If the context contains information that directly answers the question, use it. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """
        )
        
        # Create QA chain based on available LangChain version
        if USE_LCEL:
            # Use LCEL pattern (LangChain 0.1+)
            prompt = ChatPromptTemplate.from_template(prompt_template_str)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.qa_chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        elif USE_LCEL is False:
            # Use old RetrievalQA pattern
            prompt = PromptTemplate(
                template=prompt_template_str,
                input_variables=["context", "question"]
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
        else:
            # Simple fallback - just use retriever + LLM
            self.qa_chain = None
    
    def query(self, question, boost_title_matches=True):
        """
        Query the RAG chatbot with a question.
        
        Args:
            question: The user's question
            boost_title_matches: If True, prioritize documents with matching titles (default: True)
            
        Returns:
            dict with 'answer' and 'source_documents'
        """
        # Helper to get documents from retriever (handles different LangChain versions)
        def get_docs(query):
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
                    wagtail_results = Page.objects.live().search(query)
                    # Convert Wagtail search results to Document format
                    for page in wagtail_results[:10]:  # Limit to top 10 from Wagtail
                        page_url = page.url if hasattr(page, 'url') else ''
                        page_id = page.id
                        
                        # Skip if already seen
                        if page_url in seen_urls or page_id in seen_ids:
                            continue
                        
                        # Extract text from page (Document imported at top of file)
                        try:
                            # Try to get text content
                            page_text = ""
                            if hasattr(page, 'title'):
                                page_text = f"Title: {page.title}\n\n"
                            if hasattr(page, 'search_description') and page.search_description:
                                page_text += f"{page.search_description}\n\n"
                            if hasattr(page, 'body'):
                                # Extract text from StreamField if available
                                body = getattr(page, 'body', None)
                                if body:
                                    # Simple text extraction
                                    page_text += str(body)[:500]  # Limit length
                            
                            if page_text.strip():
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
                
                # Helper function for fuzzy substring matching
                def fuzzy_match(query_str, target_str, min_length=3):
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
                        elif fuzzy_match(query_lower, title, min_length=4):
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
                                        if len(title_word) >= 4 and fuzzy_match(query_word, title_word):
                                            score = max(score, 0.7)
                    
                    scored_docs.append((score, idx, doc))
                
                # Sort by title match score (descending), then by original index for ties
                scored_docs.sort(key=lambda x: (-x[0], x[1]))
                docs = [doc for _, _, doc in scored_docs]
            
            return docs
        
        if self.qa_chain is None:
            # Simple fallback implementation
            docs = get_docs(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt_text = f"""Use the following context to answer the question.
If you don't know the answer, just say that you don't know.

Context: {context}

Question: {question}

Answer: """
            answer = self.llm(prompt_text)
            return {
                'answer': answer,
                'sources': [
                    {
                        'content': doc.page_content[:200] + '...',
                        'metadata': doc.metadata
                    }
                    for doc in docs
                ]
            }
        
        if USE_LCEL:
            # LCEL pattern
            answer = self.qa_chain.invoke(question)
            # Get source documents
            docs = get_docs(question)
            return {
                'answer': answer,
                'sources': [
                    {
                        'content': doc.page_content[:200] + '...',
                        'metadata': doc.metadata
                    }
                    for doc in docs
                ]
            }
        else:
            # Old RetrievalQA pattern
            result = self.qa_chain({"query": question})
            return {
                'answer': result['result'],
                'sources': [
                    {
                        'content': doc.page_content[:200] + '...',
                        'metadata': doc.metadata
                    }
                    for doc in result['source_documents']
                ]
            }
    
    def update_filter(self, metadata_filter):
        """
        Update the metadata filter for the retriever.
        
        Args:
            metadata_filter: Dict to filter by metadata (e.g., {'model': 'BreadPage'})
        """
        self.metadata_filter = metadata_filter or {}
        k_value = getattr(settings, 'WAGTAIL_RAG_RETRIEVE_K', 8)
        search_kwargs = {"k": k_value}
        if self.metadata_filter:
            search_kwargs["filter"] = self.metadata_filter
        self.retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def search_similar(self, query, k=None, metadata_filter=None):
        """
        Search for similar content without generating a response.
        
        Args:
            query: Search query
            k: Number of results to return (default: from settings or 10)
            metadata_filter: Optional filter dict (e.g., {'model': 'BreadPage'})
            
        Returns:
            List of similar documents with metadata
        """
        if k is None:
            k = getattr(settings, 'WAGTAIL_RAG_SEARCH_K', 10)
        
        # Apply filter if provided
        if metadata_filter:
            docs = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=metadata_filter
            )
        else:
            docs = self.vectorstore.similarity_search_with_score(query, k=k)
        
        return [
            {
                'content': doc[0].page_content,
                'metadata': doc[0].metadata,
                'score': doc[1]
            }
            for doc in docs
        ]


def get_chatbot(collection_name=None, model_name=None, metadata_filter=None, llm_provider=None, llm_kwargs=None):
    """
    Convenience function to get a RAG chatbot instance.
    
    Args:
        collection_name: Name of the ChromaDB collection (default: from settings)
        model_name: Name of the LLM model (default: from settings)
        metadata_filter: Dict to filter by metadata (e.g., {'model': 'BreadPage'}) (default: None)
        llm_provider: LLM provider type ('ollama', 'openai', 'anthropic', etc.) (default: from settings)
        llm_kwargs: Additional kwargs to pass to LLM initialization (default: None)
        
    Returns:
        RAGChatBot instance
    """
    return RAGChatBot(
        collection_name=collection_name,
        model_name=model_name,
        metadata_filter=metadata_filter,
        llm_provider=llm_provider,
        llm_kwargs=llm_kwargs
    )

