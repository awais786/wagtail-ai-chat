"""
RAG Chatbot using LangChain and vector stores (FAISS/ChromaDB).

This module provides a RAG (Retrieval-Augmented Generation) chatbot
that uses the indexed content from Wagtail pages.

The chatbot orchestrates two main components:
1. Embedding Search (embeddings/search.py) - retrieves relevant documents using vector similarity
2. LLM Generation (llm_providers/generation.py) - generates answers using LLM
"""
import os
import logging

from django.conf import settings

# Setup logging
logger = logging.getLogger(__name__)

# FAISS is imported in _create_vectorstore when needed

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
from .embeddings import get_embeddings

# Import separated modules
from .llm_providers import LLMGenerator
from .embeddings import EmbeddingSearcher
from .llm_providers.generation import LCEL_AVAILABLE as USE_LCEL


def _is_wagtail_available():
    """Check if Wagtail is available (lazy import to avoid AppRegistryNotReady)."""
    try:
        from wagtail.models import Page
        return True
    except (ImportError, Exception):
        return False


class RAGChatBot:
    """
    RAG Chatbot that retrieves context from vector store (FAISS or ChromaDB) and generates responses.

    High‑level flow for a question:
      1. Embed the query and retrieve the most relevant chunks from vector store
         (plus optional Wagtail full‑text results via hybrid search).
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
            collection_name: Name of the vector store collection/index (default: from settings)
            model_name: Name of the LLM model (default: from settings)
            persist_directory: Path to vector store directory (default: from settings)
            metadata_filter: Dict to filter by metadata (e.g., {'model': 'BreadPage'})
            llm_provider: LLM provider type ('ollama', 'openai', 'anthropic', etc.)
            llm_kwargs: Additional kwargs to pass to LLM initialization
        """
        # Initialize configuration
        self.collection_name = collection_name or getattr(settings, 'WAGTAIL_RAG_COLLECTION_NAME', 'wagtail_rag')
        self.persist_directory = persist_directory or getattr(
            settings,
            'WAGTAIL_RAG_CHROMA_PATH',
            os.path.join(settings.BASE_DIR, 'chroma_db')
        )
        self.llm_provider = llm_provider or getattr(settings, 'WAGTAIL_RAG_LLM_PROVIDER', 'ollama')
        self.model_name = model_name or getattr(settings, 'WAGTAIL_RAG_MODEL_NAME', None)
        self.metadata_filter = metadata_filter or {}
        self.k_value = getattr(settings, 'WAGTAIL_RAG_RETRIEVE_K', 8)

        # Get embedding configuration
        embedding_provider = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_PROVIDER', 'huggingface')
        embedding_model = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_MODEL', None)

        # Log configuration
        self._log_configuration(embedding_provider, embedding_model)

        # Initialize components
        self.embeddings = get_embeddings(provider=embedding_provider, model_name=embedding_model)
        self.vectorstore = self._create_vectorstore()
        self.llm = self._create_llm(llm_kwargs or {})
        self.retriever = self._create_retriever()

        # Initialize searcher and generator
        # Check if hybrid search (vector + Wagtail) is enabled
        use_hybrid_search_setting = getattr(settings, 'WAGTAIL_RAG_USE_HYBRID_SEARCH', True)
        use_hybrid_search = use_hybrid_search_setting and _is_wagtail_available()
        
        if use_hybrid_search_setting and not _is_wagtail_available():
            logger.warning("WAGTAIL_RAG_USE_HYBRID_SEARCH is True but Wagtail is not available. Using vector-only search.")
        elif not use_hybrid_search_setting:
            logger.info("WAGTAIL_RAG_USE_HYBRID_SEARCH is False. Using vector-only search (Wagtail search disabled).")
        
        self.embedding_searcher = EmbeddingSearcher(
            vectorstore=self.vectorstore,
            retriever=self.retriever,
            k_value=self.k_value,
            use_hybrid_search=use_hybrid_search
        )
        self.llm_generator = LLMGenerator(llm=self.llm, retriever=self.retriever)
        self.qa_chain = self.llm_generator.qa_chain

    def _log_configuration(self, embedding_provider, embedding_model):
        """Log the chatbot configuration."""
        use_hybrid = getattr(settings, 'WAGTAIL_RAG_USE_HYBRID_SEARCH', True)
        search_mode = "Hybrid (vector + Wagtail)" if use_hybrid else "Vector-only (Wagtail disabled)"
        
        logger.info(f"RAGChatBot initialized with:")
        logger.info(f"  - Collection: {self.collection_name}")
        logger.info(f"  - LLM Provider: {self.llm_provider}")
        logger.info(f"  - LLM Model: {self.model_name or 'default for provider'}")
        logger.info(f"  - Embedding Provider: {embedding_provider}")
        logger.info(f"  - Embedding Model: {embedding_model or 'default for provider'}")
        logger.info(f"  - Search Mode: {search_mode}")

    def _create_vectorstore(self):
        """Create and return the vectorstore (ChromaDB or FAISS)."""
        backend = getattr(settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss").lower()
        
        if backend == "chroma":
            try:
                from langchain_community.vectorstores import Chroma
            except ImportError:
                raise ImportError(
                    "ChromaDB backend requires chromadb. Install with: pip install chromadb"
                )
            return Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
        elif backend == "faiss":
            try:
                from langchain_community.vectorstores import FAISS
                import faiss
                from langchain_community.docstore.in_memory import InMemoryDocstore
            except ImportError:
                raise ImportError(
                    "FAISS backend requires faiss-cpu. Install with: pip install faiss-cpu"
                )
            
            # Try to load existing index
            index_path = os.path.join(self.persist_directory, f"{self.collection_name}.faiss")
            if os.path.exists(index_path):
                try:
                    return FAISS.load_local(
                        folder_path=self.persist_directory,
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                except Exception as e:
                    logger.warning("Could not load existing FAISS index, creating new one: %s", e)
            
            # Create new FAISS index if it doesn't exist
            test_embedding = self.embeddings.embed_query("test")
            dimension = len(test_embedding)
            index = faiss.IndexFlatL2(dimension)
            return FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'chroma' or 'faiss'")

    def _create_llm(self, llm_kwargs):
        """Create and return the LLM instance."""
        llm = get_llm(provider=self.llm_provider, model_name=self.model_name, **llm_kwargs)
        logger.info(f"LLM initialized successfully: {self.llm_provider} / {self.model_name or 'default'}")
        return llm

    def _create_retriever(self):
        """Create and return the retriever instance."""
        # Build search kwargs with optional metadata filtering
        search_kwargs = {"k": self.k_value}
        if self.metadata_filter:
            search_kwargs["filter"] = self.metadata_filter
        
        base_retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        # Use MultiQueryRetriever if available and enabled (LLM query expansion)
        # Check new setting name first, fallback to old name for backward compatibility
        use_llm_query_expansion = getattr(settings, 'WAGTAIL_RAG_USE_LLM_QUERY_EXPANSION', None)
        if use_llm_query_expansion is None:
            # Backward compatibility: check old setting name
            use_llm_query_expansion = getattr(settings, 'WAGTAIL_RAG_USE_MULTI_QUERY', True)
            if use_llm_query_expansion:
                logger.warning(
                    "WAGTAIL_RAG_USE_MULTI_QUERY is deprecated. "
                    "Please use WAGTAIL_RAG_USE_LLM_QUERY_EXPANSION instead."
                )
        
        use_multi_query = use_llm_query_expansion and MULTI_QUERY_AVAILABLE
        if use_multi_query:
            logger.info(f"LLM Query Expansion enabled - using MultiQueryRetriever for query expansion")
            try:
                return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.llm, include_original=True)
            except Exception:
                logger.warning("Failed to initialize MultiQueryRetriever, falling back to base retriever")
                pass
        else:
            if not MULTI_QUERY_AVAILABLE:
                logger.info("LLM Query Expansion disabled - MultiQueryRetriever not available")
            else:
                logger.info("LLM Query Expansion disabled - using base retriever only")

        return base_retriever

    def _format_sources(self, docs):
        """Format document sources for the response."""
        return [
            {
                'content': doc.page_content[:200] + '...',
                'metadata': doc.metadata
            }
            for doc in docs
        ]

    def query(self, question, boost_title_matches=True, chat_history=None):
        """
        Query the RAG chatbot with a question.
        
        This method performs a two-step process:
        1. Embedding search: Retrieves relevant documents using embedding-based similarity search
        2. LLM generation: Generates an answer using the LLM with the retrieved context
        
        Args:
            question: The user's natural-language question
            boost_title_matches: If True, prioritize documents whose titles
                look similar to the query (helps with short queries like bread names)

        Returns:
            A dict with:
              - 'answer': string generated by the LLM
              - 'sources': list of { 'content', 'metadata' } for retrieved docs
        """
        # Step 1: Retrieve documents using embedding search
        logger.info(f"Starting RAG pipeline for question: '{question}'")
        docs = self.embedding_searcher.retrieve_with_embeddings(
            question,
            boost_title_matches=boost_title_matches,
        )
        logger.info(f"Retrieved {len(docs)} documents for LLM context")
        
        # Step 2: Generate answer using LLM with retrieved context
        logger.info(f"Generating answer using {self.llm_provider}/{self.model_name or 'default'} for question: '{question}'")
        logger.debug(f"Using {len(docs)} retrieved documents as context for LLM")
        
        # Always use the retrieved documents - don't let the chain re-retrieve
        # This ensures we use the documents from our hybrid search (vector + Wagtail)
        answer = self.llm_generator.generate_answer(question, docs=docs, history=chat_history)
        
        logger.info(f"Answer generated successfully")
        logger.info(f"LLM Result: {answer[:200] if isinstance(answer, str) else str(answer)[:200]}{'...' if (len(answer) if isinstance(answer, str) else len(str(answer))) > 200 else ''}")
        return {'answer': answer, 'sources': self._format_sources(docs)}

    def update_filter(self, metadata_filter):
        """
        Update the metadata filter for the retriever.
        
        Args:
            metadata_filter: Dict to filter by metadata (e.g., {'model': 'BreadPage'})
        """
        self.metadata_filter = metadata_filter or {}
        search_kwargs = {"k": self.k_value}
        if self.metadata_filter:
            search_kwargs["filter"] = self.metadata_filter
        self.retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    


def get_chatbot(collection_name=None, model_name=None, metadata_filter=None, llm_provider=None, llm_kwargs=None):
    """
    Convenience function to get a RAG chatbot instance.
    
    Args:
        collection_name: Name of the vector store collection/index (default: from settings)
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
