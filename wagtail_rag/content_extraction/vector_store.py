"""
Vector store wrapper for RAG index storage.

Supports ChromaDB, FAISS, and pgvector backends, selected via
WAGTAIL_RAG_VECTOR_STORE_BACKEND. The VectorStore class provides a
unified interface for upsert, delete, and persistence operations
regardless of which backend is configured.
"""

import logging
import os
from typing import List, Optional, Set

from django.conf import settings

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


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


class VectorStore:
    """
    Unified wrapper around FAISS, ChromaDB, and pgvector.

    Backend is selected via WAGTAIL_RAG_VECTOR_STORE_BACKEND setting
    ('faiss', 'chroma', or 'pgvector'). Provides a consistent interface
    for upsert, delete, reset, and persistence regardless of backend.
    """

    def __init__(
        self, *, path: str, collection: str, embeddings, backend: Optional[str] = None
    ):
        """
        Args:
            path: Directory path for persistence (unused for pgvector).
            collection: Collection/index name.
            embeddings: LangChain embedding function/model.
            backend: 'chroma', 'faiss', or 'pgvector'
                     (default: WAGTAIL_RAG_VECTOR_STORE_BACKEND setting).
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
                except Exception:
                    pass

            # Create new FAISS index
            test_embedding = embeddings.embed_query("test")
            index = faiss.IndexFlatL2(len(test_embedding))
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
        """Return the UUID of the pgvector collection, or None on failure."""
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

    def reset(self) -> None:
        """Delete the entire collection/index."""
        if self.backend == "chroma":
            try:
                self.db.delete_collection()
            except Exception:
                pass

        elif self.backend == "faiss":
            for ext in (".faiss", ".pkl"):
                file_path = os.path.join(self.path, f"{self.collection}{ext}")
                if os.path.exists(file_path):
                    os.remove(file_path)
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

    def save(self) -> None:
        """Persist the index to disk (FAISS only; other backends auto-persist)."""
        if self.backend == "faiss":
            self.db.save_local(self.path, index_name=self.collection)

    def upsert(self, documents: List, save: bool = True) -> None:
        """Upsert documents with deterministic IDs to prevent duplicates.

        Args:
            documents: LangChain Document objects to add/update.
            save: Persist FAISS index immediately. Pass False during bulk
                  indexing and call save() once at the end.
        """
        if not documents:
            return

        ids = []
        for doc in documents:
            doc_id = (
                f"{doc.metadata.get('page_id', 'unknown')}"
                f"_{doc.metadata.get('section', 'body')}"
                f"_{doc.metadata.get('chunk_index', 0)}"
            )
            ids.append(doc_id)
            doc.metadata["doc_id"] = doc_id

        if self.backend == "chroma":
            self.db.add_documents(documents, ids=ids)

        elif self.backend == "faiss":
            if hasattr(self.db, "index_to_docstore_id"):
                existing_ids = set(self.db.index_to_docstore_id.values())
                existing = [i for i in ids if i in existing_ids]
                if existing:
                    try:
                        self.db.delete(ids=existing)
                    except Exception:
                        pass
            self.db.add_documents(documents, ids=ids)
            if save:
                self.db.save_local(self.path, index_name=self.collection)

        elif self.backend == "pgvector":
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

    def delete_page(self, page_id: int) -> None:
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
                ids_to_delete = [
                    doc_id
                    for doc_id in self.db.index_to_docstore_id.values()
                    if isinstance(self.db.docstore.search(doc_id), Document)
                    and self.db.docstore.search(doc_id).metadata.get("page_id") == page_id
                ]
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
        """Return True if the page is already indexed and up to date."""
        try:
            if self.backend == "chroma":
                try:
                    data = self.db.get(where={"page_id": page_id})
                except Exception:
                    return False
                found = False
                for meta in (data.get("metadatas") or []):
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
                for doc_id in self.db.index_to_docstore_id.values():
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
                    return False
                if last_published_at is None:
                    return True
                return row[0] == last_published_at

        except Exception:
            return False

        return False

    def delete_pages_not_in(
        self, live_ids: Set[int], source: Optional[str] = None
    ) -> int:
        """Delete indexed pages whose IDs are not in live_ids.

        Optionally scoped to a model source. Returns number of docs deleted.
        """
        deleted = 0
        try:
            if self.backend == "chroma":
                try:
                    data = (
                        self.db.get(where={"source": source})
                        if source
                        else self.db.get()
                    )
                except Exception:
                    data = self.db.get()
                if not data or not data.get("ids"):
                    return 0

                ids = data.get("ids", [])
                metadatas = data.get("metadatas", [])
                ids_to_delete = [
                    ids[i]
                    for i, meta in enumerate(metadatas)
                    if i < len(ids)
                    and (not source or meta.get("source") == source)
                    and meta.get("page_id") is not None
                    and meta.get("page_id") not in live_ids
                ]
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
                        if page_id is not None and page_id not in live_ids:
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


# Backward-compat alias
ChromaStore = VectorStore
