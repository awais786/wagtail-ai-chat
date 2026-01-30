"""Build RAG index: page selection, document extraction, vector store (Chroma/FAISS)."""

import logging
import os
import traceback
from typing import Optional, Iterable, Callable, List

from django.apps import apps
from django.conf import settings
from wagtail.models import Page

from .extractors import wagtail_page_to_documents

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from wagtail_rag.embeddings import get_embeddings
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "Could not import embeddings. Install wagtail-rag[embeddings] or langchain-huggingface"
            )

    def get_embeddings(provider=None, model_name=None, **kwargs):
        if provider and provider.lower() not in ("huggingface", "hf", None):
            raise ValueError(f"Only HuggingFace embeddings in fallback. Requested: {provider}")
        return _HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            **kwargs,
        )


logger = logging.getLogger(__name__)
WriteFn = Optional[Callable[[str], None]]


def _write(out: WriteFn, message: str) -> None:
    if out is not None:
        out(message)


def get_page_models(include: Optional[Iterable[str]] = None, exclude: Optional[Iterable[str]] = None) -> List[type]:
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
    qs = model.objects.live()
    if page_id:
        return qs.filter(id=page_id)
    return qs


class ChromaStore:
    """Vector store wrapper (Chroma or FAISS). Backend via WAGTAIL_RAG_VECTOR_STORE_BACKEND."""

    def __init__(self, *, path: str, collection: str, embeddings, backend: Optional[str] = None):
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
        if self.backend == "chroma":
            try:
                self.db.delete_collection()
            except Exception:
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
            logger.warning("Error deleting page %s: %s", page_id, e)

    def upsert(self, documents: List):
        if not documents:
            return
        
        ids = []
        counters = {}

        for doc in documents:
            page_id = doc.metadata.get("page_id") or doc.metadata.get("id", "unknown")
            section = doc.metadata.get("section", "body")
            chunk_index = doc.metadata.get("chunk_index")
            if chunk_index is None:
                chunk_index = 0
            doc_id = f"{page_id}_{section}_{chunk_index}"
            ids.append(doc_id)
            
            # Also store the ID in metadata for reference
            doc.metadata["doc_id"] = doc_id

        if self.backend == "chroma":
            self.db.add_documents(documents, ids=ids)
        elif self.backend == "faiss":
            existing_ids = []
            if hasattr(self.db, "index_to_docstore_id"):
                existing_ids = [i for i in ids if i in self.db.index_to_docstore_id.values()]
            if existing_ids:
                try:
                    self.db.delete(ids=existing_ids)
                except Exception:
                    # If delete fails, continue - add_documents will handle duplicates
                    pass
            
            self.db.add_documents(documents, ids=ids)
            self.db.save_local(self.path, index_name=self.collection)


def _parse_model_fields_shorthand(model_names: Optional[Iterable[str]]) -> tuple[list[str], Optional[list[str]]]:
    """Parse "app.Model:*" in model_names; return (cleaned_names, auto_fields or None)."""
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
    if model_names is None:
        model_names = getattr(settings, "WAGTAIL_RAG_MODELS", None)
    
    if model_names:
        model_names, _ = _parse_model_fields_shorthand(model_names)
    if exclude_models is None:
        exclude_models = getattr(
            settings,
            "WAGTAIL_RAG_EXCLUDE_MODELS",
            ["wagtailcore.Page", "wagtailcore.Site"],
        )
    embedding_provider = getattr(
        settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface"
    )
    embedding_model = getattr(settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None)
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
    _write(stdout, "Initializing embeddings...")
    embeddings = get_embeddings(
        provider=embedding_provider,
        model_name=embedding_model,
    )
    _write(stdout, f"Using {vector_store_backend.upper()} vector store backend...")
    store = ChromaStore(
        path=persist_directory,
        collection=collection_name,
        embeddings=embeddings,
        backend=vector_store_backend,
    )
    if reset_only:
        _write(stdout, "Resetting collection (reset-only mode)...")
        store.reset()
        _write(stdout, f'Collection "{collection_name}" cleared successfully')
        _write(stdout, "Reset-only complete. No indexing performed.")
        return
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
        for page in live_pages:
            try:
                documents = wagtail_page_to_documents(
                    page,
                    chunk_size=getattr(settings, 'WAGTAIL_RAG_CHUNK_SIZE', 500),
                    chunk_overlap=getattr(settings, 'WAGTAIL_RAG_CHUNK_OVERLAP', 75),
                    stdout=stdout,
                )
                
                if not documents:
                    continue
                for doc in documents:
                    doc.metadata.update({
                        "source": model_name,
                        "model": model.__name__,
                        "app": model._meta.app_label,
                    })
                store.delete_page(page.id)
                store.upsert(documents)
                total_documents += len(documents)
                separator = "=" * 80
                _write(stdout, separator)
                _write(
                    stdout,
                    f'Page: {model.__name__} (ID: {page.id}) - "{page.title}" - {len(documents)} documents',
                )
                _write(stdout, "-" * 80)
                
            except Exception as e:
                page_id_str = str(getattr(page, "id", "unknown"))
                _write(stdout, f"  Error extracting from {model_name} (ID: {page_id_str}): {e}")
                if stdout:
                    _write(stdout, f"  Traceback: {traceback.format_exc()}")
    
    _write(
        stdout,
        f'Successfully indexed {total_documents} documents in {vector_store_backend.upper()} "{collection_name}"',
    )
    _write(stdout, f"Vector store saved to: {persist_directory}")
