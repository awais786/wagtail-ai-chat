"""
RAG index building and management.

This module provides the main entry point for building the RAG index,
orchestrating page selection, document extraction, and vector store operations.
It handles ChromaDB, FAISS, and pgvector backends for storing embeddings.
"""

import logging
import os
import traceback
from typing import Optional, Iterable, Callable, List, Set, Tuple
from django.conf import settings
from django.apps import apps
from wagtail.models import Page

logger = logging.getLogger(__name__)

try:
    from .api_fields_extractor import page_to_documents_api_extractor
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
STEP_SEPARATOR = "=" * 80
DEFAULT_EMBEDDING_BATCH_SIZE = 100


def _write(out: WriteFn, message: str) -> None:
    """Helper to write to stdout if provided."""
    if out is not None:
        out(message)


def _write_step_header(stdout: WriteFn, title: str) -> None:
    """Write a standardized step header."""
    _write(stdout, "\n" + STEP_SEPARATOR)
    _write(stdout, title)
    _write(stdout, STEP_SEPARATOR)


def _get_fields_to_attempt(page) -> Tuple[List[str], str]:
    """Return candidate fields and source label for extraction logging."""
    fields_to_show: List[str] = []
    extraction_source = "default fields"

    if hasattr(page, "api_fields"):
        api_field_names = [f.name for f in page.api_fields if hasattr(f, "name")]
        if api_field_names:
            extraction_source = "model api_fields"
            fields_to_show = api_field_names

    if not fields_to_show:
        fields_to_show = getattr(
            settings,
            "WAGTAIL_RAG_DEFAULT_FIELDS",
            ["introduction", "body", "content", "backstory"],
        )

    return fields_to_show, extraction_source


def _is_page_current(store: "ChromaStore", page, page_id: Optional[int]) -> bool:
    """Check whether a page can be skipped because it is already indexed and current."""
    skip_if_indexed = getattr(settings, "WAGTAIL_RAG_SKIP_IF_INDEXED", True)
    if page_id or not skip_if_indexed:
        return False

    last_published_at = None
    if getattr(page, "last_published_at", None):
        last_published_at = page.last_published_at.isoformat()
    return store.page_is_current(page.id, last_published_at)


def _prune_deleted_documents_for_model(
    *,
    store: "ChromaStore",
    live_ids: Set[int],
    model_name: str,
    page_id: Optional[int],
    stdout: WriteFn,
) -> None:
    """Delete stale documents for pages no longer live."""
    prune_deleted = getattr(settings, "WAGTAIL_RAG_PRUNE_DELETED", True)
    if page_id or not prune_deleted:
        return
    deleted = store.delete_pages_not_in(live_ids, source=model_name)
    if deleted:
        _write(stdout, f"  Pruned {deleted} stale document(s)")


def _upsert_in_batches(
    store: "ChromaStore",
    documents: List,
    batch_size: int,
    stdout: WriteFn,
) -> None:
    """Upsert *documents* into *store* in slices of *batch_size*.

    Batching across pages means the embedding model (local or API) receives a
    larger input list per call, which:
    - Reduces HTTP round-trips for API providers (OpenAI, etc.)
    - Improves GPU/CPU utilisation for local models (HuggingFace, Ollama)

    FAISS is saved once after all batches are written, not after every slice.
    """
    if not documents:
        return

    total = len(documents)
    batches = range(0, total, batch_size)
    for i in batches:
        batch = documents[i : i + batch_size]
        _write(
            stdout,
            f"  Embedding batch {i // batch_size + 1}/{len(batches)}: "
            f"{len(batch)} document(s) ...",
        )
        store.upsert(batch, save=False)

    # Single save at the end — avoids writing the FAISS index file N times.
    store.save()
    _write(stdout, f"  Saved {total} document(s) to vector store")


# ============================================================================
# Model and Page Selection
# ============================================================================


def get_page_models(
    include: Optional[Iterable[str]] = None, exclude: Optional[Iterable[str]] = None
) -> List[type]:
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
        "id",
        "pk",
        "path",
        "depth",
        "numchild",
        "url_path",
        "draft_title",
        "live",
        "has_unpublished_changes",
        "search_description",
        "go_live_at",
        "expire_at",
        "first_published_at",
        "last_published_at",
        "latest_revision_id",
        "content_type_id",
        "translation_key",
        "locale_id",
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
# Vector Store Wrapper (ChromaDB, FAISS, or pgvector)
# ============================================================================


def _pgvector_connection_string() -> str:
    """Return a SQLAlchemy connection string for pgvector.

    Checks ``WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING`` first.  If not set,
    derives one from ``settings.DATABASES['default']``, which must be a
    PostgreSQL database (ENGINE contains 'postgresql' or 'postgis').

    Raises ``ValueError`` if no valid PostgreSQL configuration can be found.
    """
    explicit = getattr(settings, "WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING", None)
    if explicit:
        return explicit

    db = settings.DATABASES.get("default", {})
    engine = db.get("ENGINE", "")
    if "postgresql" not in engine and "postgis" not in engine:
        raise ValueError(
            "WAGTAIL_RAG_VECTOR_STORE_BACKEND='pgvector' requires a PostgreSQL database. "
            "Set WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING explicitly, or configure "
            "DATABASES['default'] to use a PostgreSQL ENGINE."
        )

    host = db.get("HOST") or "localhost"
    port = db.get("PORT") or 5432
    name = db.get("NAME", "")
    user = db.get("USER", "")
    password = db.get("PASSWORD", "")

    if password:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    return f"postgresql+psycopg2://{user}@{host}:{port}/{name}"


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

    def __init__(
        self, *, path: str, collection: str, embeddings, backend: Optional[str] = None
    ):
        """Initialize vector store (ChromaDB, FAISS, or pgvector).

        Args:
            path: Directory path for persistence (unused for pgvector).
            collection: Collection/index name.
            embeddings: LangChain embedding function/model.
            backend: ``"chroma"``, ``"faiss"``, or ``"pgvector"``
                     (default: from ``WAGTAIL_RAG_VECTOR_STORE_BACKEND``).
        """
        if backend is None:
            backend = getattr(
                settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss"
            ).lower()

        self.backend = backend
        self.path = path
        self.collection = collection

        if backend == "chroma":
            os.makedirs(path, exist_ok=True)
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
            os.makedirs(path, exist_ok=True)
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

        elif backend == "pgvector":
            try:
                from langchain_community.vectorstores import PGVector
                from sqlalchemy import create_engine
            except ImportError:
                raise ImportError(
                    "pgvector backend requires psycopg2 and sqlalchemy. "
                    "Install with: pip install 'wagtail-rag[pgvector]'"
                )

            conn_string = _pgvector_connection_string()
            self._pg_engine = create_engine(conn_string)
            self.db = PGVector(
                connection_string=conn_string,
                collection_name=collection,
                embedding_function=embeddings,
                pre_delete_collection=False,
            )

        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. Use 'chroma', 'faiss', or 'pgvector'."
            )

    def _pg_collection_id(self) -> Optional[str]:
        """Return the UUID of the PGVector collection scoped to this store.

        All raw SQL operations against ``langchain_pg_embedding`` must be
        scoped by this UUID so they only touch documents from this collection.
        Returns ``None`` and logs a warning when the collection row is missing.
        """
        try:
            from sqlalchemy import text as sa_text

            with self._pg_engine.connect() as conn:
                row = conn.execute(
                    sa_text(
                        "SELECT uuid FROM langchain_pg_collection WHERE name = :name"
                    ),
                    {"name": self.collection},
                ).fetchone()
                return str(row[0]) if row else None
        except Exception:
            logger.warning(
                "Could not retrieve pgvector collection ID for %r", self.collection
            )
            return None

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
        elif self.backend == "pgvector":
            try:
                self.db.delete_collection()
                self.db.create_collection()
            except Exception as e:
                logger.warning("Error resetting pgvector collection: %s", e)

    def delete_page(self, page_id: int):
        """Delete all chunks for a specific page."""
        try:
            if self.backend == "chroma":
                data = self.db.get()
                if not data or not data.get("ids"):
                    return

                ids = [
                    doc_id
                    for i, doc_id in enumerate(data.get("ids", []))
                    if i < len(data.get("metadatas", []))
                    and data["metadatas"][i].get("page_id") == page_id
                ]
                if ids:
                    self.db.delete(ids=ids)

            elif self.backend == "faiss":
                if not (
                    hasattr(self.db, "index_to_docstore_id")
                    and hasattr(self.db, "docstore")
                ):
                    return

                ids_to_delete = []
                for doc_id in list(self.db.index_to_docstore_id.values()):
                    try:
                        doc = self.db.docstore.search(doc_id)
                        if (
                            isinstance(doc, Document)
                            and doc.metadata.get("page_id") == page_id
                        ):
                            ids_to_delete.append(doc_id)
                    except (KeyError, AttributeError):
                        continue

                if ids_to_delete:
                    self.db.delete(ids=ids_to_delete)
                    self.db.save_local(self.path, index_name=self.collection)

            elif self.backend == "pgvector":
                cid = self._pg_collection_id()
                if not cid:
                    return
                from sqlalchemy import text as sa_text

                with self._pg_engine.begin() as conn:
                    conn.execute(
                        sa_text(
                            "DELETE FROM langchain_pg_embedding "
                            "WHERE collection_id = :cid "
                            "AND cmetadata->>'page_id' = :pid"
                        ),
                        {"cid": cid, "pid": str(page_id)},
                    )

        except Exception as e:
            logger.warning("Error deleting page %s: %s", page_id, e)

    def page_is_current(
        self, page_id: int, last_published_at: Optional[str] = None
    ) -> bool:
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
                if not (
                    hasattr(self.db, "index_to_docstore_id")
                    and hasattr(self.db, "docstore")
                ):
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

            if self.backend == "pgvector":
                cid = self._pg_collection_id()
                if not cid:
                    return False
                from sqlalchemy import text as sa_text

                with self._pg_engine.connect() as conn:
                    row = conn.execute(
                        sa_text(
                            "SELECT cmetadata->>'last_published_at' "
                            "FROM langchain_pg_embedding "
                            "WHERE collection_id = :cid "
                            "AND cmetadata->>'page_id' = :pid "
                            "LIMIT 1"
                        ),
                        {"cid": cid, "pid": str(page_id)},
                    ).fetchone()

                if row is None:
                    return False  # page not indexed yet
                if last_published_at is None:
                    return True
                return row[0] == last_published_at

        except Exception:
            return False

    def delete_pages_not_in(
        self, live_ids: Set[int], source: Optional[str] = None
    ) -> int:
        """
        Delete all indexed pages not in live_ids (optionally scoped to a model source).
        Returns number of docs deleted (best-effort).
        """
        deleted = 0
        try:
            if self.backend == "chroma":
                try:
                    # Prefer using a filtered query when possible to avoid scanning the entire index.
                    data = (
                        self.db.get(where={"source": source})
                        if source
                        else self.db.get()
                    )
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
                if not (
                    hasattr(self.db, "index_to_docstore_id")
                    and hasattr(self.db, "docstore")
                ):
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

            elif self.backend == "pgvector":
                cid = self._pg_collection_id()
                if not cid:
                    return 0
                from sqlalchemy import text as sa_text

                # page_id is stored as a JSON number; ->> returns it as text.
                live_id_strings = [str(i) for i in live_ids]

                with self._pg_engine.begin() as conn:
                    if source:
                        result = conn.execute(
                            sa_text(
                                "DELETE FROM langchain_pg_embedding "
                                "WHERE collection_id = :cid "
                                "AND cmetadata->>'source' = :source "
                                "AND NOT (cmetadata->>'page_id' = ANY(:live_ids))"
                            ),
                            {"cid": cid, "source": source, "live_ids": live_id_strings},
                        )
                    else:
                        result = conn.execute(
                            sa_text(
                                "DELETE FROM langchain_pg_embedding "
                                "WHERE collection_id = :cid "
                                "AND NOT (cmetadata->>'page_id' = ANY(:live_ids))"
                            ),
                            {"cid": cid, "live_ids": live_id_strings},
                        )
                    deleted = result.rowcount

        except Exception:
            return 0

        return deleted

    def save(self) -> None:
        """Persist the vector store to disk.

        - **FAISS**: writes the index and docstore pickle files.
        - **ChromaDB**: no-op — ChromaDB auto-persists on every write.
        - **pgvector**: no-op — PostgreSQL commits are transactional and durable.

        Call this explicitly after a bulk indexing run to avoid writing the
        FAISS index file after every single page.
        """
        if self.backend == "faiss":
            self.db.save_local(self.path, index_name=self.collection)

    def upsert(self, documents: List, save: bool = True) -> None:
        """Upsert documents with deterministic IDs to prevent duplicates.

        Args:
            documents: LangChain Document objects to add/update.
            save: If True (default), persist the FAISS index to disk immediately.
                  Pass False during bulk indexing and call save() once at the end
                  to avoid writing the index file after every page.
        """
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
            if hasattr(self.db, "index_to_docstore_id"):
                existing = [
                    i for i in ids if i in self.db.index_to_docstore_id.values()
                ]
                if existing:
                    try:
                        self.db.delete(ids=existing)
                    except Exception:
                        pass

            self.db.add_documents(documents, ids=ids)
            if save:
                self.db.save_local(self.path, index_name=self.collection)

        elif self.backend == "pgvector":
            # langchain_pg_embedding has no unique constraint on custom_id, so
            # calling add_documents twice with the same IDs creates duplicates.
            # Delete any existing rows for these custom_ids first.
            cid = self._pg_collection_id()
            if cid and ids:
                from sqlalchemy import text as sa_text

                with self._pg_engine.begin() as conn:
                    conn.execute(
                        sa_text(
                            "DELETE FROM langchain_pg_embedding "
                            "WHERE collection_id = :cid "
                            "AND custom_id = ANY(:ids)"
                        ),
                        {"cid": cid, "ids": ids},
                    )
            self.db.add_documents(documents, ids=ids)
            # pgvector persists to PostgreSQL automatically; save() is a no-op.


# ============================================================================
# Main Index Building Logic
# ============================================================================


def _parse_model_fields_shorthand(
    model_names: Optional[Iterable[str]],
) -> tuple[list[str], Optional[list[str]]]:
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
    _write_step_header(stdout, "STEP 1: Loading Configuration")

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
        settings,
        "WAGTAIL_RAG_CHROMA_PATH",
        os.path.join(settings.BASE_DIR, "chroma_db"),
    )

    _write(stdout, f"Vector store: {backend.upper()}")
    _write(stdout, f"Collection: {collection}")
    _write(stdout, f"Storage path: {path}")

    # Initialize embeddings
    _write(stdout, "\nInitializing embeddings...")
    embeddings = get_embeddings(
        provider=getattr(settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface"),
        model_name=getattr(settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None),
    )
    _write(stdout, "Embeddings loaded")

    # Initialize vector store
    _write(stdout, "Initializing vector store...")
    store = ChromaStore(
        path=path, collection=collection, embeddings=embeddings, backend=backend
    )
    _write(stdout, "Vector store ready")

    # Handle reset-only mode
    if reset_only:
        _write(stdout, "\nResetting collection...")
        store.reset()
        _write(stdout, f'Collection "{collection}" cleared')
        return

    # Step 2: Extract Documents
    _write_step_header(stdout, "STEP 2: Extracting Documents from Pages")

    # Get page models to index
    models = get_page_models(include=model_names, exclude=exclude_models)
    if not models:
        _write(stdout, "ERROR: No page models found")
        return

    _write(
        stdout,
        f"Found {len(models)} model(s): {', '.join(f'{m._meta.app_label}.{m.__name__}' for m in models)}",
    )
    if page_id:
        _write(stdout, f"Re-indexing specific page ID: {page_id}")

    total_docs = 0
    total_pages = 0
    batch_size = getattr(
        settings, "WAGTAIL_RAG_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE
    )

    if not page_to_documents_api_extractor:
        _write(stdout, "ERROR: api_fields_extractor not available")
        return

    _write(stdout, "Using api_fields_extractor")

    # Process each model
    _write(stdout, "")
    for model_idx, model in enumerate(models, 1):
        model_name = f"{model._meta.app_label}.{model.__name__}"
        pages = get_live_pages(model, page_id)
        count = pages.count()
        live_ids: Set[int] = set(pages.values_list("id", flat=True))

        if count == 0:
            _prune_deleted_documents_for_model(
                store=store,
                live_ids=live_ids,
                model_name=model_name,
                page_id=page_id,
                stdout=stdout,
            )
            continue

        _write(stdout, f"\n[{model_idx}/{len(models)}] Processing {model_name}:")
        _write(stdout, f"  Found {count} live page(s)")

        # If model uses :* shorthand, show discovered content fields for visibility.
        if model_name in models_with_all_fields:
            model_field_names = get_content_field_names(model)
            _write(stdout, f"  Extracting fields: {', '.join(model_field_names)}")

        # Collect documents from all pages in this model before embedding.
        # Batching across pages means one embedding API call per batch_size
        # documents rather than one call per page, which is significantly
        # faster for both API providers and local models.
        pending_docs: List = []

        for page_idx, page in enumerate(pages, 1):
            try:
                _write(stdout, f"  [{page_idx}/{count}] {page.title} (ID: {page.id})")

                if _is_page_current(store=store, page=page, page_id=page_id):
                    _write(stdout, "    -> Skipping (already indexed and up-to-date)")
                    continue

                # Show what will be attempted
                fields_to_show, extraction_source = _get_fields_to_attempt(page)
                _write(
                    stdout,
                    f"      Fields to extract: {', '.join(fields_to_show)} (from {extraction_source})",
                )

                # Extract documents
                _write(stdout, "      Extracting content...")
                docs = page_to_documents_api_extractor(page, stdout=stdout)

                if not docs:
                    _write(
                        stdout, "      WARNING: No content extracted (all fields empty)"
                    )
                    continue

                # Show what was actually indexed
                field_info = docs[0].metadata.get("field_source", "")
                extracted_fields = docs[0].metadata.get("extracted_fields", "title")

                # Compare what was tried vs what was actually indexed
                indexed_list = [f.strip() for f in extracted_fields.split(",")]
                not_indexed = [f for f in fields_to_show if f not in indexed_list]

                _write(stdout, f"      Indexed: {extracted_fields}")
                if field_info:
                    _write(stdout, f"      Source: {field_info}")
                if not_indexed:
                    _write(
                        stdout,
                        f"      Skipped: {', '.join(not_indexed)} (empty or missing)",
                    )
                _write(stdout, f"      Queued {len(docs)} document(s) for embedding")

                # Add model metadata
                for doc in docs:
                    doc.metadata.update(
                        {
                            "source": model_name,
                            "model": model.__name__,
                            "app": model._meta.app_label,
                            "extractor": "api_fields_extractor",
                        }
                    )

                # Delete old chunks for this page immediately so re-indexing
                # a single page never leaves stale chunks behind.
                store.delete_page(page.id)
                pending_docs.extend(docs)

                total_docs += len(docs)
                total_pages += 1

            except Exception as e:
                _write(stdout, f"      ERROR: {e}")
                logger.exception(f"Error indexing page {page.id}")
                if stdout:
                    _write(stdout, f"      {traceback.format_exc()}")

        # Embed and store all collected documents in batches, then save once.
        if pending_docs:
            _write(
                stdout,
                f"\n  Embedding {len(pending_docs)} document(s) "
                f"in batch(es) of {batch_size} ...",
            )
            _upsert_in_batches(store, pending_docs, batch_size, stdout)

        _prune_deleted_documents_for_model(
            store=store,
            live_ids=live_ids,
            model_name=model_name,
            page_id=page_id,
            stdout=stdout,
        )

    # Step 3: Summary
    _write_step_header(stdout, "STEP 3: Indexing Complete")
    _write(stdout, f"Processed {total_pages} page(s)")
    _write(stdout, f"Created {total_docs} document(s)")
    _write(stdout, f"Saved to {backend.upper()} collection: {collection}")
    _write(stdout, f"Storage location: {path}")
    _write(stdout, STEP_SEPARATOR + "\n")
