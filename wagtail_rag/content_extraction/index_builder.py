"""
RAG index building and management.

This module provides the main entry point for building the RAG index,
orchestrating page selection, document extraction, and vector store operations.
It handles both ChromaDB and FAISS backends for storing embeddings.
"""

import hashlib
import logging
import os
from typing import Optional, Iterable, Callable, List, Tuple
from django.conf import settings
from django.apps import apps
from wagtail.models import Page
from .page_to_documents import wagtail_page_to_documents

# Set up logger at module level
logger = logging.getLogger(__name__)

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from wagtail_rag.embeddings import get_embeddings
except ImportError:
    # Fallback for older installations
    try:
        from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "Could not import embeddings. Please install wagtail-rag[embeddings] or langchain-huggingface"
            )
    
    def get_embeddings(provider=None, model_name=None, **kwargs):
        """Fallback embedding function."""
        if provider and provider.lower() not in ("huggingface", "hf", None):
            raise ValueError(f"Only HuggingFace embeddings available in fallback mode. Requested: {provider}")
        return _HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            **kwargs,
        )


# Type alias for an optional output function (e.g. management command stdout.write)
WriteFn = Optional[Callable[[str], None]]


def _write(out: WriteFn, message: str) -> None:
    """Helper to write to stdout if provided."""
    if out is not None:
        out(message)


# ============================================================================
# Model and Page Selection
# ============================================================================

def get_page_models(include: Optional[Iterable[str]] = None, exclude: Optional[Iterable[str]] = None) -> List[type]:
    """
    Get all Wagtail Page models dynamically, filtered by include/exclude lists.
    
    Args:
        include: Optional list of model names to include (e.g., ['breads.BreadPage'])
        exclude: Optional list of model names to exclude (e.g., ['wagtailcore.Page'])
    
    Returns:
        List of Page model classes
    """
    exclude = set(exclude or [])
    models = []

    for app_config in apps.get_app_configs():
        for model in app_config.get_models():
            if issubclass(model, Page) and model != Page:
                name = f"{model._meta.app_label}.{model.__name__}"
                if include and name not in include:
                    continue
                if name in exclude:
                    continue
                models.append(model)
    
    return models


def get_live_pages(model: type, page_id: Optional[int] = None):
    """
    Get live pages for a given model, optionally filtered by page_id.
    
    Args:
        model: Wagtail Page model class
        page_id: Optional page ID to filter to a single page
    
    Returns:
        QuerySet of live pages
    """
    qs = model.objects.live()
    if page_id:
        return qs.filter(id=page_id)
    return qs


# ============================================================================
# Vector Store Wrapper (ChromaDB or FAISS)
# ============================================================================

class ChromaStore:
    """
    Wrapper around vector store (ChromaDB or FAISS) with simplified operations.
    
    Backend is selected via WAGTAIL_RAG_VECTOR_STORE_BACKEND setting.
    
    This class handles:
    - Collection/index initialization
    - Document upsertion with deterministic IDs
    - Page-level deletion
    - Collection reset
    """
    
    def __init__(self, *, path: str, collection: str, embeddings, backend: Optional[str] = None):
        """
        Initialize vector store (ChromaDB or FAISS).
        
        Args:
            path: Directory path for persistence
            collection: Collection/index name
            embeddings: Embedding function/model
            backend: "chroma" or "faiss" (default: from settings)
        """
        if backend is None:
            backend = getattr(settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss").lower()
        
        self.backend = backend
        self.path = path
        self.collection = collection
        
        os.makedirs(path, exist_ok=True)
        
        if backend == "chroma":
            try:
                from langchain_community.vectorstores import Chroma
            except ImportError:
                raise ImportError(
                    "ChromaDB backend requires chromadb. Install with: pip install chromadb"
                )
            self.db = Chroma(
                persist_directory=path,
                collection_name=collection,
                embedding_function=embeddings,
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
            index_path = os.path.join(path, f"{collection}.faiss")
            if os.path.exists(index_path):
                try:
                    self.db = FAISS.load_local(
                        folder_path=path,
                        embeddings=embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    return
                except Exception as e:
                    # Log warning when falling back to creating a new index
                    logger.warning(
                        "Failed to load existing FAISS index from '%s'; creating new empty index. Error: %s",
                        index_path,
                        str(e),
                    )
            
            # Create new FAISS index
            test_embedding = embeddings.embed_query("test")
            dimension = len(test_embedding)
            index = faiss.IndexFlatL2(dimension)
            self.db = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'chroma' or 'faiss'")

    def reset(self):
        """Delete the entire collection/index."""
        if self.backend == "chroma":
            try:
                self.db.delete_collection()
            except Exception:
                # ChromaDB collection may not exist yet, ignore errors
                pass
        elif self.backend == "faiss":
            # Delete FAISS files and reinitialize
            index_path = os.path.join(self.path, f"{self.collection}.faiss")
            pkl_path = os.path.join(self.path, f"{self.collection}.pkl")
            for file_path in [index_path, pkl_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            # Reinitialize
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore
            test_embedding = self.db.embedding_function.embed_query("test")
            dimension = len(test_embedding)
            self.db.index = faiss.IndexFlatL2(dimension)
            self.db.docstore = InMemoryDocstore()
            self.db.index_to_docstore_id = {}

    def delete_page(self, page_id: int):
        """
        Delete all chunks for a specific page.
        
        Args:
            page_id: The page ID to delete chunks for
        """
        try:
            if self.backend == "chroma":
                # ChromaDB: Get all documents with their IDs and metadata
                data = self.db.get()
                if not data or not data.get("ids"):
                    return
                
                ids = []
                metadatas = data.get("metadatas", [])
                for i, doc_id in enumerate(data.get("ids", [])):
                    if i < len(metadatas):
                        metadata = metadatas[i]
                        if metadata and (metadata.get("page_id") == page_id or metadata.get("id") == page_id):
                            ids.append(doc_id)
                
                if ids:
                    self.db.delete(ids=ids)
            
            elif self.backend == "faiss":
                # FAISS: Find all document IDs for this page by checking index_to_docstore_id
                ids_to_delete = []
                
                if hasattr(self.db, 'index_to_docstore_id') and hasattr(self.db, 'docstore'):
                    # Iterate through all stored document IDs
                    for doc_id in list(self.db.index_to_docstore_id.values()):
                        try:
                            stored_doc = self.db.docstore.search(doc_id)
                            if isinstance(stored_doc, Document):
                                stored_meta = stored_doc.metadata
                                if stored_meta and (stored_meta.get("page_id") == page_id or stored_meta.get("id") == page_id):
                                    ids_to_delete.append(doc_id)
                        except (KeyError, AttributeError):
                            # Document might have been deleted already, skip
                            continue
                
                if ids_to_delete:
                    self.db.delete(ids=ids_to_delete)
                    # Save after deletion
                    self.db.save_local(self.path, index_name=self.collection)
        except Exception as e:
            # Log error but don't fail - allow upsert to handle it
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error deleting page {page_id}: {e}")

    def upsert(self, documents: List):
        """
        Upsert documents into the store with deterministic IDs.
        
        This method ensures that documents with the same IDs are replaced,
        preventing duplicates when re-indexing.
        
        Args:
            documents: List of Document objects to upsert
        """
        if not documents:
            return
        
        ids = []

        # Generate deterministic IDs based on page_id, section, and chunk index
        for doc in documents:
            page_id = doc.metadata.get("page_id") or doc.metadata.get("id", "unknown")
            section = doc.metadata.get("section", "body")
            chunk_index = doc.metadata.get("chunk_index")
            
            # chunk_index should always be set (0 for title/intro, 0+ for body chunks)
            # If not set, default to 0
            if chunk_index is None:
                chunk_index = 0
            
            # Generate deterministic ID: {page_id}_{section}_{chunk_index}
            doc_id = f"{page_id}_{section}_{chunk_index}"
            ids.append(doc_id)
            
            # Also store the ID in metadata for reference
            doc.metadata["doc_id"] = doc_id

        # Add documents with upsert behavior
        if self.backend == "chroma":
            # ChromaDB handles upsert automatically when IDs match
            self.db.add_documents(documents, ids=ids)
        elif self.backend == "faiss":
            # FAISS: delete existing IDs first (simulate upsert)
            existing_ids = []
            if hasattr(self.db, 'index_to_docstore_id'):
                # Check which IDs already exist
                existing_ids = [doc_id for doc_id in ids if doc_id in self.db.index_to_docstore_id.values()]
            
            # Delete existing documents with these IDs
            if existing_ids:
                try:
                    self.db.delete(ids=existing_ids)
                except Exception as e:
                    # Log deletion failures for diagnostics
                    logger.warning(
                        "FAISS delete failed for IDs %s during upsert (will proceed with add). Error: %s",
                        existing_ids,
                        str(e),
                    )
            
            # Add new documents (will overwrite if IDs match after delete)
            self.db.add_documents(documents, ids=ids)
            # Save FAISS to disk
            self.db.save_local(self.path, index_name=self.collection)


# ============================================================================
# Main Index Building Logic
# ============================================================================

def _parse_model_fields_shorthand(model_names: Optional[Iterable[str]]) -> Tuple[List[str], Optional[List[str]]]:
    """
    Parse convenience syntax "app.Model:*" in model_names.
    
    Example:
        ["breads.BreadPage", "locations.LocationPage:*"]
    becomes:
        (["breads.BreadPage", "locations.LocationPage"], ["locations.LocationPage:*"])
    
    Returns:
        Tuple of (cleaned_model_names, auto_model_fields)
    """
    if not model_names:
        return [], None
    
    cleaned_model_names = []
    auto_model_fields = []
    
    for name in model_names:
        if isinstance(name, str) and name.endswith(":*"):
            base = name.split(":", 1)[0]
            cleaned_model_names.append(base)
            auto_model_fields.append(f"{base}:*")
        else:
            cleaned_model_names.append(name)
    
    return cleaned_model_names, auto_model_fields if auto_model_fields else None


def build_rag_index(
    *,
    model_names: Optional[Iterable[str]] = None,
    exclude_models: Optional[Iterable[str]] = None,
    page_id: Optional[int] = None,
    reset_only: bool = False,
    stdout: WriteFn = None,
) -> None:
    """
    High-level API to build the RAG index.
    
    This is the main entry point for building the RAG index. It orchestrates:
    1. Model/page selection
    2. Document extraction
    3. Vector store operations
    
    Args:
        model_names: Optional list of model names to include (e.g., ['breads.BreadPage'])
        exclude_models: Optional list of model names to exclude
        page_id: Optional page ID to re-index a single page
        reset_only: If True, only reset the collection without indexing
        stdout: Optional output function for logging
    """
    # Resolve configuration from settings
    if model_names is None:
        model_names = getattr(settings, "WAGTAIL_RAG_MODELS", None)
    
    # Parse shorthand syntax "app.Model:*"
    if model_names:
        model_names, _ = _parse_model_fields_shorthand(model_names)
    
    # Get exclude models from settings if not provided
    if exclude_models is None:
        exclude_models = getattr(
            settings,
            "WAGTAIL_RAG_EXCLUDE_MODELS",
            ["wagtailcore.Page", "wagtailcore.Site"],
        )
    
    # Get embedding configuration
    embedding_provider = getattr(
        settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface"
    )
    embedding_model = getattr(settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None)
    
    # Get vector store configuration
    vector_store_backend = getattr(settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss")
    persist_directory = getattr(
        settings,
        "WAGTAIL_RAG_CHROMA_PATH",  # Path for vector store (works for both ChromaDB and FAISS)
        os.path.join(settings.BASE_DIR, "chroma_db"),
    )
    collection_name = getattr(
        settings,
        "WAGTAIL_RAG_COLLECTION_NAME",
        "wagtail_rag",
    )
    
    # Initialize embeddings
    _write(stdout, "Initializing embeddings...")
    embeddings = get_embeddings(
        provider=embedding_provider,
        model_name=embedding_model,
    )
    
    # Initialize store (ChromaDB or FAISS based on settings)
    _write(stdout, f"Using {vector_store_backend.upper()} vector store backend...")
    store = ChromaStore(
        path=persist_directory,
        collection=collection_name,
        embeddings=embeddings,
        backend=vector_store_backend,
    )
    
    # Handle reset-only mode
    if reset_only:
        _write(stdout, "Resetting collection (reset-only mode)...")
        store.reset()
        _write(stdout, f'Collection "{collection_name}" cleared successfully')
        _write(stdout, "Reset-only complete. No indexing performed.")
        return
    
    # Get page models
    models = get_page_models(include=model_names, exclude=exclude_models)
    if not models:
        _write(stdout, "No page models found to index.")
        return
    
    _write(stdout, f"Found {len(models)} page model(s) to index:")
    for model in models:
        _write(stdout, f"  - {model._meta.app_label}.{model.__name__}")
    
    if exclude_models:
        _write(stdout, f"Excluding models: {', '.join(exclude_models)}")
    
    if page_id:
        _write(stdout, f"Re-indexing specific page ID: {page_id}")
    
    _write(stdout, "Starting RAG index building...")
    
    total_documents = 0
    
    # Process each model
    for model in models:
        model_name = f"{model._meta.app_label}.{model.__name__}"
        live_pages = get_live_pages(model, page_id)
        
        count = live_pages.count()
        if count == 0:
            if not page_id:
                _write(stdout, f"  No live pages found for {model_name}")
            continue
        
        if not page_id:
            _write(stdout, f"  Indexing {count} pages from {model_name}...")
        else:
            _write(stdout, f"  Re-indexing page ID {page_id} from {model_name}...")
        
        # Process each page
        for page in live_pages:
            try:
                # Extract documents from page
                documents = wagtail_page_to_documents(
                    page,
                    chunk_size=getattr(settings, 'WAGTAIL_RAG_CHUNK_SIZE', 500),
                    chunk_overlap=getattr(settings, 'WAGTAIL_RAG_CHUNK_OVERLAP', 75),
                    stdout=stdout,
                )
                
                if not documents:
                    continue
                
                # Add model-level metadata to each document
                for doc in documents:
                    doc.metadata.update({
                        "source": model_name,
                        "model": model.__name__,
                        "app": model._meta.app_label,
                    })
                
                # Always delete old chunks before re-indexing to prevent duplicates
                # This ensures clean re-indexing whether running full index or single page
                store.delete_page(page.id)
                
                # Upsert documents (will overwrite any remaining chunks with same IDs)
                store.upsert(documents)
                total_documents += len(documents)
                
                # Log page extraction summary
                separator = "=" * 80
                _write(stdout, separator)
                _write(
                    stdout,
                    f'Page: {model.__name__} (ID: {page.id}) - "{page.title}" - {len(documents)} documents',
                )
                _write(stdout, "-" * 80)
                
            except Exception as e:
                page_id_str = str(getattr(page, 'id', 'unknown'))
                _write(
                    stdout,
                    f"  Error extracting documents from {model_name} (ID: {page_id_str}): {e}",
                )
                import traceback
                if stdout:
                    _write(stdout, f"  Traceback: {traceback.format_exc()}")
    
    _write(
        stdout,
        f'Successfully indexed {total_documents} documents in {vector_store_backend.upper()} "{collection_name}"',
    )
    _write(stdout, f"Vector store saved to: {persist_directory}")
