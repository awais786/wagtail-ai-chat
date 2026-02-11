"""
RAG index building and management.

This module provides the main entry point for building the RAG index,
orchestrating page selection, document extraction, and vector store operations.
It handles both ChromaDB and FAISS backends for storing embeddings.
"""

import os
from typing import Optional, Iterable, Callable, List, Set
from django.conf import settings
from django.apps import apps
from wagtail.models import Page
from .page_to_documents import wagtail_page_to_documents

try:
    from .api_extractor import page_to_documents_api_extractor
except ImportError:
    page_to_documents_api_extractor = None

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from wagtail_rag.embeddings import get_embeddings
except ImportError:
    # Fallback: use HuggingFace embeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    
    def get_embeddings(provider=None, model_name=None, **kwargs):
        return HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2", **kwargs
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


def get_content_field_names(model: type) -> List[str]:
    """Get content-bearing field names (StreamField, RichTextField, TextField, CharField)."""
    from django.db import models
    from wagtail.fields import RichTextField, StreamField

    skip = {
        "id", "pk", "path", "depth", "numchild", "url_path", "draft_title",
        "live", "has_unpublished_changes", "search_description", "go_live_at", 
        "expire_at", "first_published_at", "last_published_at", "latest_revision_id", 
        "content_type_id", "translation_key", "locale_id",
    }
    
    content_names = []
    for f in model._meta.get_fields():
        name = getattr(f, "name", None)
        if not name or name in skip:
            continue
        if getattr(f, "remote_field", None) or getattr(f, "many_to_many", False):
            continue
        
        # Check if it's a content field
        if isinstance(f, (RichTextField, StreamField)):
            content_names.append(name)
        elif isinstance(f, models.TextField):
            content_names.append(name)
        elif isinstance(f, models.CharField) and f.max_length >= 100:
            content_names.append(name)
    
    return content_names


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
                except Exception:
                    pass
            
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
                pass
        elif self.backend == "faiss":
            # Delete files
            for ext in [".faiss", ".pkl"]:
                file_path = os.path.join(self.path, f"{self.collection}{ext}")
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Reinitialize
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore
            dimension = len(self.db.embedding_function.embed_query("test"))
            self.db.index = faiss.IndexFlatL2(dimension)
            self.db.docstore = InMemoryDocstore()
            self.db.index_to_docstore_id = {}

    def delete_page(self, page_id: int):
        """Delete all chunks for a specific page."""
        try:
            if self.backend == "chroma":
                data = self.db.get()
                if not data or not data.get("ids"):
                    return
                
                ids = [
                    doc_id for i, doc_id in enumerate(data.get("ids", []))
                    if i < len(data.get("metadatas", [])) 
                    and data["metadatas"][i].get("page_id") == page_id
                ]
                if ids:
                    self.db.delete(ids=ids)
            
            elif self.backend == "faiss":
                if not (hasattr(self.db, 'index_to_docstore_id') and hasattr(self.db, 'docstore')):
                    return
                
                ids_to_delete = []
                for doc_id in list(self.db.index_to_docstore_id.values()):
                    try:
                        doc = self.db.docstore.search(doc_id)
                        if isinstance(doc, Document) and doc.metadata.get("page_id") == page_id:
                            ids_to_delete.append(doc_id)
                    except (KeyError, AttributeError):
                        continue
                
                if ids_to_delete:
                    self.db.delete(ids=ids_to_delete)
                    self.db.save_local(self.path, index_name=self.collection)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error deleting page {page_id}: {e}")

    def page_is_current(self, page_id: int, last_published_at: Optional[str] = None) -> bool:
        """
        Return True if the page already exists in the index and (optionally) matches last_published_at.
        """
        try:
            if self.backend == "chroma":
                try:
                    data = self.db.get(where={"page_id": page_id})
                except Exception:
                    # If the filtered query fails, avoid fetching the full index for performance reasons.
                    # In this case, treat the page as not current.
                    return False
                metadatas = data.get("metadatas", []) if data else []
                found = False
                for meta in metadatas:
                    if meta.get("page_id") != page_id:
                        continue
                    found = True
                    if last_published_at is None:
                        return True
                    if meta.get("last_published_at") != last_published_at:
                        return False
                return found
            
            if self.backend == "faiss":
                if not (hasattr(self.db, 'index_to_docstore_id') and hasattr(self.db, 'docstore')):
                    return False
                found = False
                for doc_id in list(self.db.index_to_docstore_id.values()):
                    try:
                        doc = self.db.docstore.search(doc_id)
                        if not isinstance(doc, Document):
                            continue
                        if doc.metadata.get("page_id") != page_id:
                            continue
                        found = True
                        if last_published_at is None:
                            return True
                        if doc.metadata.get("last_published_at") != last_published_at:
                            return False
                    except (KeyError, AttributeError):
                        continue
                return found
        except Exception:
            return False

    def delete_pages_not_in(self, live_ids: Set[int], source: Optional[str] = None) -> int:
        """
        Delete all indexed pages not in live_ids (optionally scoped to a model source).
        Returns number of docs deleted (best-effort).
        """
        deleted = 0
        try:
            if self.backend == "chroma":
                try:
                    # Prefer using a filtered query when possible to avoid scanning the entire index.
                    data = self.db.get(where={"source": source}) if source else self.db.get()
                except Exception:
                    # Fallback: some ChromaDB backends/versions may not support `where` filters
                    # for this metadata field or may raise unexpectedly. In that case, we fall
                    # back to fetching all documents and filtering in Python so that index
                    # cleanup still works. This can be inefficient for very large indexes but
                    # is an intentional robustness trade-off to ensure stale pages are removed.
                    data = self.db.get()
                if not data or not data.get("ids"):
                    return 0
                
                ids_to_delete = []
                metadatas = data.get("metadatas", [])
                ids = data.get("ids", [])
                for i, meta in enumerate(metadatas):
                    if i >= len(ids):
                        break
                    if source and meta.get("source") != source:
                        continue
                    page_id = meta.get("page_id")
                    if page_id is None:
                        continue
                    if page_id not in live_ids:
                        ids_to_delete.append(ids[i])
                
                if ids_to_delete:
                    self.db.delete(ids=ids_to_delete)
                    deleted = len(ids_to_delete)
            
            elif self.backend == "faiss":
                if not (hasattr(self.db, 'index_to_docstore_id') and hasattr(self.db, 'docstore')):
                    return 0
                
                ids_to_delete = []
                for doc_id in list(self.db.index_to_docstore_id.values()):
                    try:
                        doc = self.db.docstore.search(doc_id)
                        if not isinstance(doc, Document):
                            continue
                        if source and doc.metadata.get("source") != source:
                            continue
                        page_id = doc.metadata.get("page_id")
                        if page_id is None:
                            continue
                        if page_id not in live_ids:
                            ids_to_delete.append(doc_id)
                    except (KeyError, AttributeError):
                        continue
                
                if ids_to_delete:
                    self.db.delete(ids=ids_to_delete)
                    self.db.save_local(self.path, index_name=self.collection)
                    deleted = len(ids_to_delete)
        except Exception:
            return 0
        
        return deleted

    def upsert(self, documents: List):
        """Upsert documents with deterministic IDs to prevent duplicates."""
        if not documents:
            return
        
        # Generate deterministic IDs: {page_id}_{section}_{chunk_index}
        ids = []
        for doc in documents:
            page_id = doc.metadata.get("page_id", "unknown")
            section = doc.metadata.get("section", "body")
            chunk_index = doc.metadata.get("chunk_index", 0)
            doc_id = f"{page_id}_{section}_{chunk_index}"
            ids.append(doc_id)
            doc.metadata["doc_id"] = doc_id

        # Add documents
        if self.backend == "chroma":
            self.db.add_documents(documents, ids=ids)
        elif self.backend == "faiss":
            # Delete existing IDs first
            if hasattr(self.db, 'index_to_docstore_id'):
                existing = [i for i in ids if i in self.db.index_to_docstore_id.values()]
                if existing:
                    try:
                        self.db.delete(ids=existing)
                    except Exception:
                        pass
            
            self.db.add_documents(documents, ids=ids)
            self.db.save_local(self.path, index_name=self.collection)


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
    Build the RAG index from Wagtail pages.
    
    Args:
        model_names: Model names to include (e.g., ['breads.BreadPage'])
        exclude_models: Model names to exclude
        page_id: Specific page ID to re-index
        reset_only: Only reset the collection without indexing
        stdout: Output function for logging
    """
    # Step 1: Configuration
    _write(stdout, "\n" + "="*80)
    _write(stdout, "STEP 1: Loading Configuration")
    _write(stdout, "="*80)
    
    model_names = model_names or getattr(settings, "WAGTAIL_RAG_MODELS", None)
    exclude_models = exclude_models or getattr(
        settings, "WAGTAIL_RAG_EXCLUDE_MODELS", ["wagtailcore.Page", "wagtailcore.Site"]
    )
    
    # Parse :* shorthand (use all content fields)
    models_with_all_fields = set()
    if model_names:
        model_names, auto_fields = _parse_model_fields_shorthand(model_names)
        if auto_fields:
            models_with_all_fields = {s.split(":", 1)[0] for s in auto_fields}
    
    backend = getattr(settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss")
    collection = getattr(settings, "WAGTAIL_RAG_COLLECTION_NAME", "wagtail_rag")
    path = getattr(
        settings, "WAGTAIL_RAG_CHROMA_PATH", 
        os.path.join(settings.BASE_DIR, "chroma_db")
    )
    
    _write(stdout, f"✓ Vector store: {backend.upper()}")
    _write(stdout, f"✓ Collection: {collection}")
    _write(stdout, f"✓ Storage path: {path}")
    
    # Initialize embeddings
    _write(stdout, "\nInitializing embeddings...")
    embeddings = get_embeddings(
        provider=getattr(settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface"),
        model_name=getattr(settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None),
    )
    _write(stdout, "✓ Embeddings loaded")
    
    # Initialize vector store
    _write(stdout, "Initializing vector store...")
    store = ChromaStore(path=path, collection=collection, embeddings=embeddings, backend=backend)
    _write(stdout, "✓ Vector store ready")
    
    # Handle reset-only mode
    if reset_only:
        _write(stdout, "\nResetting collection...")
        store.reset()
        _write(stdout, f'✓ Collection "{collection}" cleared')
        return
    
    # Step 2: Extract Documents
    _write(stdout, "\n" + "="*80)
    _write(stdout, "STEP 2: Extracting Documents from Pages")
    _write(stdout, "="*80)
    
    # Get page models to index
    models = get_page_models(include=model_names, exclude=exclude_models)
    if not models:
        _write(stdout, "✗ No page models found")
        return
    
    _write(stdout, f"✓ Found {len(models)} model(s): {', '.join(f'{m._meta.app_label}.{m.__name__}' for m in models)}")
    if page_id:
        _write(stdout, f"→ Re-indexing specific page ID: {page_id}")
    
    total_docs = 0
    total_pages = 0
    use_api = getattr(settings, "WAGTAIL_RAG_USE_API_EXTRACTOR", True) and page_to_documents_api_extractor
    _write(stdout, f"→ Using {'API' if use_api else 'chunked'} extractor")
    
    # Process each model
    _write(stdout, "")
    for model_idx, model in enumerate(models, 1):
        model_name = f"{model._meta.app_label}.{model.__name__}"
        pages = get_live_pages(model, page_id)
        count = pages.count()
        live_ids: Set[int] = set(pages.values_list("id", flat=True))
        
        if count == 0:
            prune_deleted = getattr(settings, "WAGTAIL_RAG_PRUNE_DELETED", True)
            if not page_id and prune_deleted:
                deleted = store.delete_pages_not_in(live_ids, source=model_name)
                if deleted:
                    _write(stdout, f"  ✓ Pruned {deleted} stale document(s)")
            continue
        
        _write(stdout, f"\n[{model_idx}/{len(models)}] Processing {model_name}:")
        _write(stdout, f"  → Found {count} live page(s)")
        
        # Check if model uses :* (all fields)
        field_names = None
        if model_name in models_with_all_fields:
            field_names = get_content_field_names(model)
            _write(stdout, f"  → Extracting fields: {', '.join(field_names)}")
        
        # Extract documents from each page
        for page_idx, page in enumerate(pages, 1):
            try:
                _write(stdout, f"  [{page_idx}/{count}] {page.title} (ID: {page.id})")
                
                skip_if_indexed = getattr(settings, "WAGTAIL_RAG_SKIP_IF_INDEXED", True)
                last_published_at = None
                if getattr(page, "last_published_at", None):
                    last_published_at = page.last_published_at.isoformat()
                
                if not page_id and skip_if_indexed and store.page_is_current(page.id, last_published_at):
                    _write(stdout, "    ↷ Skipping (already indexed and up-to-date)")
                    continue
                
                # Try API extractor first, fallback to chunked
                docs = None
                if use_api:
                    docs = page_to_documents_api_extractor(page, stdout=stdout)
                if not docs:
                    docs = wagtail_page_to_documents(
                        page, 
                        chunk_size=getattr(settings, 'WAGTAIL_RAG_CHUNK_SIZE', 500),
                        chunk_overlap=getattr(settings, 'WAGTAIL_RAG_CHUNK_OVERLAP', 75),
                        stdout=stdout,
                        streamfield_field_names=field_names,
                    )

                if not docs:
                    _write(stdout, "      ⚠ No documents")
                    continue
                
                # Add model metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": model_name,
                        "model": model.__name__,
                        "app": model._meta.app_label,
                    })
                
                # Save to vector store
                store.delete_page(page.id)
                store.upsert(docs)
                _write(stdout, f"      ✓ Saved {len(docs)} document(s)")

                total_docs += len(docs)
                total_pages += 1
                
            except Exception as e:
                _write(stdout, f"      ✗ Error: {e}")
                logger.exception(f"Error indexing page {page.id}")
                if stdout:
                    import traceback
                    _write(stdout, f"      {traceback.format_exc()}")
        
        prune_deleted = getattr(settings, "WAGTAIL_RAG_PRUNE_DELETED", True)
        if not page_id and prune_deleted:
            deleted = store.delete_pages_not_in(live_ids, source=model_name)
            if deleted:
                _write(stdout, f"  ✓ Pruned {deleted} stale document(s)")

    # Step 3: Summary
    _write(stdout, "\n" + "="*80)
    _write(stdout, "STEP 3: Indexing Complete")
    _write(stdout, "="*80)
    _write(stdout, f"✓ Processed {total_pages} page(s)")
    _write(stdout, f"✓ Created {total_docs} document(s)")
    _write(stdout, f"✓ Saved to {backend.upper()} collection: {collection}")
    _write(stdout, f"✓ Storage location: {path}")
    _write(stdout, "="*80 + "\n")
