"""
RAG index building logic.

This module provides the main entry point for building the RAG index,
orchestrating page selection, document extraction, and vector store operations.
"""

import os
from typing import Optional, Iterable, Callable, List
from django.conf import settings
from django.apps import apps
from wagtail.models import Page
from langchain_community.vectorstores import Chroma

from .extractors import wagtail_page_to_documents

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
# ChromaDB Store Wrapper
# ============================================================================

class ChromaStore:
    """
    Wrapper around ChromaDB vector store with simplified operations.
    
    This class handles:
    - Collection initialization
    - Document upsertion with deterministic IDs
    - Page-level deletion
    - Collection reset
    """
    
    def __init__(self, *, path: str, collection: str, embeddings):
        """
        Initialize ChromaDB store.
        
        Args:
            path: Directory path for ChromaDB persistence
            collection: Collection name
            embeddings: Embedding function/model
        """
        self.db = Chroma(
            persist_directory=path,
            collection_name=collection,
            embedding_function=embeddings,
        )
        self.path = path
        self.collection = collection

    def reset(self):
        """Delete the entire collection."""
        try:
            self.db.delete_collection()
        except Exception:
            # Collection may not exist, which is fine
            pass

    def delete_page(self, page_id: int):
        """
        Delete all chunks for a specific page.
        
        Args:
            page_id: The page ID to delete chunks for
        """
        try:
            data = self.db.get(include=["ids", "metadatas"])
            if not data or not data.get("ids"):
                return
            
            ids = []
            for i, metadata in enumerate(data.get("metadatas", [])):
                if metadata and (metadata.get("page_id") == page_id or metadata.get("id") == page_id):
                    if i < len(data["ids"]):
                        ids.append(data["ids"][i])
            
            if ids:
                self.db.delete(ids=ids)
        except Exception:
            # If deletion fails, continue (page may not exist in store)
            pass

    def upsert(self, documents: List):
        """
        Upsert documents into the store with deterministic IDs.
        
        Args:
            documents: List of Document objects to upsert
        """
        if not documents:
            return
        
        ids = []
        counters = {}

        for doc in documents:
            # Generate deterministic ID based on page_id and section
            page_id = doc.metadata.get("page_id") or doc.metadata.get("id", "unknown")
            section = doc.metadata.get("section", "body")
            key = f"{page_id}_{section}"
            
            idx = counters.get(key, 0)
            counters[key] = idx + 1
            ids.append(f"{key}_{idx}")

        # Use add_documents which handles both new and existing documents
        self.db.add_documents(documents, ids=ids)


# ============================================================================
# Main Index Building Logic
# ============================================================================

def _parse_model_fields_shorthand(model_names: Optional[Iterable[str]]) -> tuple[list[str], Optional[list[str]]]:
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
    
    # Get ChromaDB configuration
    persist_directory = getattr(
        settings,
        "WAGTAIL_RAG_CHROMA_PATH",
        os.path.join(settings.BASE_DIR, "chroma_db"),
    )
    collection_name = getattr(
        settings,
        "WAGTAIL_RAG_COLLECTION_NAME",
        "wagtail_rag",
    )
    
    # Ensure persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize embeddings
    _write(stdout, "Initializing embeddings...")
    embeddings = get_embeddings(
        provider=embedding_provider,
        model_name=embedding_model,
    )
    
    # Initialize store
    store = ChromaStore(
        path=persist_directory,
        collection=collection_name,
        embeddings=embeddings,
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
        f'Successfully indexed {total_documents} documents in ChromaDB collection "{collection_name}"',
    )
    _write(stdout, f"Vector store saved to: {persist_directory}")
