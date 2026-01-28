"""
RAG Chatbot using LangChain and ChromaDB.

This module provides a RAG (Retrieval-Augmented Generation) chatbot
that uses the indexed content from Wagtail pages.

The chatbot orchestrates two main components:
1. Embedding Search (embedding_search.py) - retrieves relevant documents
2. LLM Generation (llm_generation.py) - generates answers using LLM
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

# Import provider factories
from .llm_providers import get_llm
from .embedding_providers import get_embeddings

# Import separated modules
from .embedding_search import EmbeddingSearcher
from .llm_generation import LLMGenerator, USE_LCEL


def _is_wagtail_available():
    """Check if Wagtail is available (lazy import to avoid AppRegistryNotReady)."""
    try:
        from wagtail.models import Page
        return True
    except (ImportError, Exception):
        return False


class RAGChatBot:
    """
    RAG Chatbot that retrieves context from ChromaDB and generates responses.

    High‑level flow for a question:
      1. Embed the query and retrieve the most relevant chunks from ChromaDB
         (plus optional Wagtail full‑text results).
      2. Concatenate those chunks into a single context string.
      3. Call the configured LLM with a prompt that contains {context, question}.
      4. Return the LLM's answer together with the source documents.
    """
    
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
        
        # Get persist directory from settings or use default
        if persist_directory is None:
            persist_directory = getattr(
                settings,
                'WAGTAIL_RAG_CHROMA_PATH',
                os.path.join(settings.BASE_DIR, 'chroma_db')
            )
        
        # Get LLM provider from settings if not provided (needed to determine model default)
        if llm_provider is None:
            llm_provider = getattr(settings, 'WAGTAIL_RAG_LLM_PROVIDER', 'ollama')
        
        # Get model name from settings or use provider-specific default
        # Don't set a default here - let get_llm() handle provider-specific defaults
        if model_name is None:
            model_name = getattr(settings, 'WAGTAIL_RAG_MODEL_NAME', None)
            # If still None, get_llm() will use provider-specific defaults
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_name = model_name  # May be None, get_llm() will set appropriate default
        self.metadata_filter = metadata_filter or {}
        
        # Get embedding provider from settings if not provided
        embedding_provider = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_PROVIDER', 'huggingface')
        embedding_model = getattr(
            settings,
            'WAGTAIL_RAG_EMBEDDING_MODEL',
            None  # Will use provider default
        )
        
        # Log the configuration being used
        logger.info(f"RAGChatBot initialized with:")
        logger.info(f"  - Collection: {collection_name}")
        logger.info(f"  - LLM Provider: {llm_provider}")
        logger.info(f"  - LLM Model: {self.model_name}")
        logger.info(f"  - Embedding Provider: {embedding_provider}")
        logger.info(f"  - Embedding Model: {embedding_model or 'default for provider'}")
        
        # Initialize embeddings using the factory function
        self.embeddings = get_embeddings(
            provider=embedding_provider,
            model_name=embedding_model
        )
        
        # Load vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Initialize LLM using the factory function
        # Pass model_name=None if not set, so get_llm() can use provider-specific defaults
        llm_kwargs = llm_kwargs or {}
        self.llm_provider = llm_provider  # Store for reference
        self.llm = get_llm(
            provider=llm_provider,
            model_name=self.model_name,  # May be None, get_llm() will use provider default
            **llm_kwargs
        )
        # Get the actual model name that was used (in case get_llm() set a default)
        # We can't easily get it back from the LLM object, so we'll log what we know
        actual_model = self.model_name or f"default for {llm_provider}"
        logger.info(f"LLM initialized successfully: {llm_provider} / {actual_model}")
        # Update self.model_name for reference (though it may still be None)
        # The actual model is determined by get_llm() internally
        
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
        use_hybrid_search = getattr(settings, 'WAGTAIL_RAG_USE_HYBRID_SEARCH', True) and _is_wagtail_available()
        
        # Initialize embedding searcher
        self.embedding_searcher = EmbeddingSearcher(
            vectorstore=self.vectorstore,
            retriever=self.retriever,
            k_value=k_value,
            use_hybrid_search=use_hybrid_search
        )
        
        # Initialize LLM generator
        self.llm_generator = LLMGenerator(
            llm=self.llm,
            retriever=self.retriever
        )
        
        # Store qa_chain reference for backward compatibility
        self.qa_chain = self.llm_generator.qa_chain
    
    def query(self, question, boost_title_matches=True):
        """
        Query the RAG chatbot with a question.
        
        This method performs a two-step process:
        1. Embedding search: Retrieves relevant documents using embedding-based similarity search
        2. LLM generation: Generates an answer using the LLM with the retrieved context
        
        Args:
            question: The user's natural‑language question.
            boost_title_matches: If True, prioritize documents whose titles
                look similar to the query (helps with short queries like bread names).
            
        Returns:
            A dict with:
              - 'answer': string generated by the LLM (from LLM generation step)
              - 'sources': list of { 'content', 'metadata' } for retrieved docs (from embedding search step)

        Notes:
            The LLM generation step:
              * In the LCEL path: via `self.qa_chain.invoke(question)` which
                executes the chain: retriever → prompt → self.llm → parser.
              * In the fallback path: via a direct `self.llm(prompt_text)` call.
        """
        # Step 1: Retrieve documents using embedding search
        docs = self.embedding_searcher.retrieve_with_embeddings(question, boost_title_matches=boost_title_matches)
        
        # Step 2: Generate answer using LLM with retrieved context
        if self.qa_chain is None:
            # Simple fallback implementation
            answer = self.llm_generator.generate_answer_with_llm(question, docs)
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
            # LCEL pattern - chain handles retrieval + LLM generation internally
            # But we use our custom retrieved docs (with hybrid search, title boosting) for sources
            answer = self.qa_chain.invoke(question)
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
            # Old RetrievalQA pattern - chain handles both embedding search and LLM generation
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
        return self.embedding_searcher.search_with_embeddings(query, k=k, metadata_filter=metadata_filter)
    
    # Backward compatibility alias
    def search_similar(self, query, k=None, metadata_filter=None):
        """
        Deprecated: Use search_with_embeddings() instead.
        
        This method is kept for backward compatibility and will be removed in a future version.
        """
        return self.search_with_embeddings(query, k=k, metadata_filter=metadata_filter)


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
