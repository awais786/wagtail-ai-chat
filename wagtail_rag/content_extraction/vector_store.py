"""
Vector store backends for RAG index storage.

Each backend is a self-contained class implementing BaseVectorStore.
Use get_vector_store() to instantiate the configured backend.

Supported backends (WAGTAIL_RAG_VECTOR_STORE_BACKEND):
  - 'faiss'    — local file-based index (default)
  - 'chroma'   — ChromaDB
  - 'pgvector' — PostgreSQL with pgvector extension
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from django.conf import settings

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


# ============================================================================
# Abstract base
# ============================================================================


class BaseVectorStore(ABC):
    """Common interface for all vector store backends."""

    @abstractmethod
    def reset(self) -> None:
        """Wipe the entire collection/index."""

    @abstractmethod
    def save(self) -> None:
        """Persist to disk (no-op for backends that auto-commit)."""

    @abstractmethod
    def upsert(self, documents: list, save: bool = True) -> None:
        """Add or replace documents using deterministic IDs."""

    @abstractmethod
    def delete_page(self, page_id: int) -> None:
        """Delete all chunks belonging to a page."""

    @abstractmethod
    def page_is_current(
        self, page_id: int, last_published_at: Optional[str] = None
    ) -> bool:
        """Return True if the page is indexed and up to date."""

    @abstractmethod
    def delete_pages_not_in(
        self, live_ids: set[int], source: Optional[str] = None
    ) -> int:
        """Delete chunks for pages not in live_ids. Returns count deleted."""

    # ------------------------------------------------------------------
    # Shared helper — deterministic chunk IDs used by all backends
    # ------------------------------------------------------------------

    @staticmethod
    def _make_ids(documents: list) -> list[str]:
        ids = []
        for doc in documents:
            doc_id = (
                f"{doc.metadata.get('page_id', 'unknown')}"
                f"_{doc.metadata.get('section', 'body')}"
                f"_{doc.metadata.get('chunk_index', 0)}"
            )
            ids.append(doc_id)
            doc.metadata["doc_id"] = doc_id
        return ids


# ============================================================================
# FAISS backend
# ============================================================================


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed vector store (local files, no server required)."""

    def __init__(self, *, path: str, collection: str, embeddings):
        try:
            from langchain_community.vectorstores import FAISS
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore
        except ImportError:
            raise ImportError(
                "FAISS backend requires faiss-cpu. "
                "Install with: pip install 'wagtail-rag[faiss]'"
            )

        self.path = path
        self.collection = collection
        self._FAISS = FAISS
        os.makedirs(path, exist_ok=True)

        index_path = os.path.join(path, f"{collection}.faiss")
        if os.path.exists(index_path):
            try:
                self.db = FAISS.load_local(
                    folder_path=path,
                    embeddings=embeddings,
                    index_name=collection,
                    allow_dangerous_deserialization=True,
                )
                return
            except Exception as exc:
                logger.warning("Could not load FAISS index, creating new one: %s", exc)

        dim = len(embeddings.embed_query("test"))
        self.db = FAISS(
            embedding_function=embeddings,
            index=faiss.IndexFlatL2(dim),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def save(self) -> None:
        self.db.save_local(self.path, index_name=self.collection)

    def reset(self) -> None:
        for ext in (".faiss", ".pkl"):
            fp = os.path.join(self.path, f"{self.collection}{ext}")
            if os.path.exists(fp):
                os.remove(fp)
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore

        dim = len(self.db.embedding_function.embed_query("test"))
        self.db.index = faiss.IndexFlatL2(dim)
        self.db.docstore = InMemoryDocstore()
        self.db.index_to_docstore_id = {}

    def upsert(self, documents: list, save: bool = True) -> None:
        if not documents:
            return
        ids = self._make_ids(documents)
        if hasattr(self.db, "index_to_docstore_id"):
            existing = set(self.db.index_to_docstore_id.values())
            stale = [i for i in ids if i in existing]
            if stale:
                try:
                    self.db.delete(ids=stale)
                except Exception:
                    pass
        self.db.add_documents(documents, ids=ids)
        if save:
            self.save()

    def _iter_docs(self):
        """Yield (doc_id, Document) for every stored document."""
        for doc_id in list(self.db.index_to_docstore_id.values()):
            try:
                doc = self.db.docstore.search(doc_id)
                if isinstance(doc, Document):
                    yield doc_id, doc
            except (KeyError, AttributeError):
                continue

    def delete_page(self, page_id: int) -> None:
        try:
            ids = [
                doc_id
                for doc_id, doc in self._iter_docs()
                if doc.metadata.get("page_id") == page_id
            ]
            if ids:
                self.db.delete(ids=ids)
                self.save()
        except Exception as exc:
            logger.warning("Error deleting page %s: %s", page_id, exc)

    def page_is_current(
        self, page_id: int, last_published_at: Optional[str] = None
    ) -> bool:
        try:
            for _, doc in self._iter_docs():
                if doc.metadata.get("page_id") != page_id:
                    continue
                if last_published_at is None:
                    return True
                return doc.metadata.get("last_published_at") == last_published_at
        except Exception:
            pass
        return False

    def delete_pages_not_in(
        self, live_ids: set[int], source: Optional[str] = None
    ) -> int:
        try:
            ids = [
                doc_id
                for doc_id, doc in self._iter_docs()
                if (not source or doc.metadata.get("source") == source)
                and doc.metadata.get("page_id") is not None
                and doc.metadata.get("page_id") not in live_ids
            ]
            if ids:
                self.db.delete(ids=ids)
                self.save()
            return len(ids)
        except Exception:
            return 0


# ============================================================================
# ChromaDB backend
# ============================================================================


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector store."""

    def __init__(self, *, path: str, collection: str, embeddings):
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            raise ImportError(
                "ChromaDB backend requires chromadb. "
                "Install with: pip install 'wagtail-rag[chroma]'"
            )
        os.makedirs(path, exist_ok=True)
        self.db = Chroma(
            persist_directory=path,
            collection_name=collection,
            embedding_function=embeddings,
        )
        self._source_filter_supported = True

    def reset(self) -> None:
        try:
            self.db.delete_collection()
        except Exception:
            pass

    def save(self) -> None:
        pass  # ChromaDB auto-persists on every write

    def upsert(self, documents: list, save: bool = True) -> None:
        if not documents:
            return
        self.db.add_documents(documents, ids=self._make_ids(documents))

    def delete_page(self, page_id: int) -> None:
        try:
            data = self.db.get()
            if not data or not data.get("ids"):
                return
            ids = [
                doc_id
                for i, doc_id in enumerate(data["ids"])
                if i < len(data.get("metadatas", []))
                and data["metadatas"][i].get("page_id") == page_id
            ]
            if ids:
                self.db.delete(ids=ids)
        except Exception as exc:
            logger.warning("Error deleting page %s: %s", page_id, exc)

    def page_is_current(
        self, page_id: int, last_published_at: Optional[str] = None
    ) -> bool:
        try:
            data = self.db.get(where={"page_id": page_id})
        except Exception:
            return False
        for meta in (data.get("metadatas") or []):
            if meta.get("page_id") != page_id:
                continue
            if last_published_at is None:
                return True
            return meta.get("last_published_at") == last_published_at
        return False

    def delete_pages_not_in(
        self, live_ids: set[int], source: Optional[str] = None
    ) -> int:
        try:
            try:
                data = self.db.get(where={"source": source}) if source else self.db.get()
            except Exception:
                data = self.db.get()

            if not data or not data.get("ids"):
                return 0

            ids_to_delete = [
                data["ids"][i]
                for i, meta in enumerate(data.get("metadatas", []))
                if i < len(data["ids"])
                and (not source or meta.get("source") == source)
                and meta.get("page_id") is not None
                and meta.get("page_id") not in live_ids
            ]
            if ids_to_delete:
                self.db.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception:
            return 0


# ============================================================================
# pgvector backend
# ============================================================================


def _pgvector_connection_string() -> str:
    """Build a SQLAlchemy connection string for pgvector.

    Reads WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING first; falls back to
    deriving one from DATABASES['default'] (must be PostgreSQL).
    """
    explicit = getattr(settings, "WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING", None)
    if explicit:
        return explicit

    db = settings.DATABASES.get("default", {})
    engine = db.get("ENGINE", "")
    if "postgresql" not in engine and "postgis" not in engine:
        raise ValueError(
            "WAGTAIL_RAG_VECTOR_STORE_BACKEND='pgvector' requires PostgreSQL. "
            "Set WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING or use a PostgreSQL ENGINE."
        )
    host = db.get("HOST") or "localhost"
    port = db.get("PORT") or 5432
    name, user, password = db.get("NAME", ""), db.get("USER", ""), db.get("PASSWORD", "")
    if password:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    return f"postgresql+psycopg2://{user}@{host}:{port}/{name}"


class PgVectorStore(BaseVectorStore):
    """pgvector-backed vector store (PostgreSQL)."""

    def __init__(self, *, collection: str, embeddings, path: str = ""):
        try:
            from langchain_community.vectorstores import PGVector
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError(
                "pgvector backend requires psycopg2 and sqlalchemy. "
                "Install with: pip install 'wagtail-rag[pgvector]'"
            )
        conn_string = _pgvector_connection_string()
        self._engine = create_engine(conn_string)
        self._collection = collection
        self.db = PGVector(
            connection_string=conn_string,
            collection_name=collection,
            embedding_function=embeddings,
            pre_delete_collection=False,
        )

    def _collection_id(self) -> Optional[str]:
        """Return the UUID of this collection row, or None on failure."""
        try:
            from sqlalchemy import text as sql

            with self._engine.connect() as conn:
                row = conn.execute(
                    sql("SELECT uuid FROM langchain_pg_collection WHERE name = :n"),
                    {"n": self._collection},
                ).fetchone()
                return str(row[0]) if row else None
        except Exception:
            logger.warning("Could not fetch pgvector collection ID for %r", self._collection)
            return None

    def reset(self) -> None:
        try:
            self.db.delete_collection()
            self.db.create_collection()
        except Exception as exc:
            logger.warning("Error resetting pgvector collection: %s", exc)

    def save(self) -> None:
        pass  # PostgreSQL commits are transactional

    def upsert(self, documents: list, save: bool = True) -> None:
        if not documents:
            return
        ids = self._make_ids(documents)
        cid = self._collection_id()
        if cid:
            from sqlalchemy import text as sql

            with self._engine.begin() as conn:
                conn.execute(
                    sql(
                        "DELETE FROM langchain_pg_embedding "
                        "WHERE collection_id = :cid AND custom_id = ANY(:ids)"
                    ),
                    {"cid": cid, "ids": ids},
                )
        self.db.add_documents(documents, ids=ids)

    def delete_page(self, page_id: int) -> None:
        cid = self._collection_id()
        if not cid:
            return
        try:
            from sqlalchemy import text as sql

            with self._engine.begin() as conn:
                conn.execute(
                    sql(
                        "DELETE FROM langchain_pg_embedding "
                        "WHERE collection_id = :cid AND cmetadata->>'page_id' = :pid"
                    ),
                    {"cid": cid, "pid": str(page_id)},
                )
        except Exception as exc:
            logger.warning("Error deleting page %s: %s", page_id, exc)

    def page_is_current(
        self, page_id: int, last_published_at: Optional[str] = None
    ) -> bool:
        cid = self._collection_id()
        if not cid:
            return False
        try:
            from sqlalchemy import text as sql

            with self._engine.connect() as conn:
                row = conn.execute(
                    sql(
                        "SELECT cmetadata->>'last_published_at' "
                        "FROM langchain_pg_embedding "
                        "WHERE collection_id = :cid AND cmetadata->>'page_id' = :pid "
                        "LIMIT 1"
                    ),
                    {"cid": cid, "pid": str(page_id)},
                ).fetchone()
            if row is None:
                return False
            if last_published_at is None:
                return True
            return row[0] == last_published_at
        except Exception:
            return False

    def delete_pages_not_in(
        self, live_ids: set[int], source: Optional[str] = None
    ) -> int:
        cid = self._collection_id()
        if not cid:
            return 0
        try:
            from sqlalchemy import text as sql

            live_id_strings = [str(i) for i in live_ids]
            with self._engine.begin() as conn:
                if source:
                    result = conn.execute(
                        sql(
                            "DELETE FROM langchain_pg_embedding "
                            "WHERE collection_id = :cid "
                            "AND cmetadata->>'source' = :source "
                            "AND NOT (cmetadata->>'page_id' = ANY(:ids))"
                        ),
                        {"cid": cid, "source": source, "ids": live_id_strings},
                    )
                else:
                    result = conn.execute(
                        sql(
                            "DELETE FROM langchain_pg_embedding "
                            "WHERE collection_id = :cid "
                            "AND NOT (cmetadata->>'page_id' = ANY(:ids))"
                        ),
                        {"cid": cid, "ids": live_id_strings},
                    )
            return result.rowcount
        except Exception:
            return 0


# ============================================================================
# Factory
# ============================================================================

_BACKENDS: dict[str, type[BaseVectorStore]] = {
    "faiss": FAISSVectorStore,
    "chroma": ChromaVectorStore,
    "pgvector": PgVectorStore,
}


def get_vector_store(
    *,
    path: str,
    collection: str,
    embeddings,
    backend: Optional[str] = None,
) -> BaseVectorStore:
    """Instantiate and return the configured vector store backend.

    Args:
        path: Directory for file-based backends (FAISS, ChromaDB).
        collection: Collection / index name.
        embeddings: LangChain embeddings instance.
        backend: 'faiss', 'chroma', or 'pgvector'.
                 Defaults to WAGTAIL_RAG_VECTOR_STORE_BACKEND setting.
    """
    backend = (backend or getattr(settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss")).lower()
    cls = _BACKENDS.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown backend: {backend!r}. Choose from: {', '.join(_BACKENDS)}"
        )
    return cls(path=path, collection=collection, embeddings=embeddings)


# Backward-compat aliases
VectorStore = get_vector_store
ChromaStore = get_vector_store
