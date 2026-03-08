"""
RAG index building — orchestrates page discovery, extraction, and vector storage.
"""

import logging
from collections.abc import Callable, Iterable
from typing import Optional

from django.apps import apps
from django.conf import settings
from wagtail.models import Page

from .api_fields_extractor import page_to_documents_api_extractor
from .vector_store import VectorStore
from wagtail_rag.embeddings import get_embeddings
from wagtail_rag.conf import conf

logger = logging.getLogger(__name__)

WriteFn = Optional[Callable[[str], None]]
STEP_SEPARATOR = "=" * 80
DEFAULT_EMBEDDING_BATCH_SIZE = 100


# ============================================================================
# Output helpers
# ============================================================================


def _write(out: WriteFn, message: str) -> None:
    if out is not None:
        out(message)


def _write_step_header(stdout: WriteFn, title: str) -> None:
    _write(stdout, "\n" + STEP_SEPARATOR)
    _write(stdout, title)
    _write(stdout, STEP_SEPARATOR)


# ============================================================================
# Page selection
# ============================================================================


def get_page_models(
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> list[type]:
    """Return Wagtail Page subclasses, filtered by include/exclude lists."""
    exclude_set = set(exclude or [])
    models = []
    for app_config in apps.get_app_configs():
        for model in app_config.get_models():
            if issubclass(model, Page) and model != Page:
                name = f"{model._meta.app_label}.{model.__name__}"
                if include and name not in include:
                    continue
                if name in exclude_set:
                    continue
                models.append(model)
    return models


def get_live_pages(model: type, page_id: Optional[int] = None):
    """Return live pages for a model, optionally filtered to a single page_id."""
    qs = model.objects.live()
    return qs.filter(id=page_id) if page_id else qs


# ============================================================================
# Index building helpers
# ============================================================================


def _parse_model_fields_shorthand(model_names: Iterable[str]) -> list[str]:
    """Strip ':*' shorthand from model names (e.g. 'breads.BreadPage:*' → 'breads.BreadPage').

    The ':*' suffix signals auto-discovery of all content fields; the extractor
    handles this automatically so only the cleaned name is needed here.
    """
    return [
        name.split(":", 1)[0] if isinstance(name, str) and name.endswith(":*") else name
        for name in model_names
    ]


def _is_page_current(store: VectorStore, page, page_id: Optional[int]) -> bool:
    """Return True if the page can be skipped (already indexed and up to date)."""
    if page_id or not getattr(settings, "WAGTAIL_RAG_SKIP_IF_INDEXED", True):
        return False
    last_published_at = (
        page.last_published_at.isoformat()
        if getattr(page, "last_published_at", None)
        else None
    )
    return store.page_is_current(page.id, last_published_at)


def _prune_stale(
    *,
    store: VectorStore,
    live_ids: set[int],
    model_name: str,
    page_id: Optional[int],
    stdout: WriteFn,
) -> None:
    """Remove chunks for pages that are no longer live."""
    if page_id or not getattr(settings, "WAGTAIL_RAG_PRUNE_DELETED", True):
        return
    deleted = store.delete_pages_not_in(live_ids, source=model_name)
    if deleted:
        _write(stdout, f"  Pruned {deleted} stale document(s)")


def _upsert_in_batches(
    store: VectorStore,
    documents: list,
    batch_size: int,
    stdout: WriteFn,
) -> None:
    """Embed and upsert documents in batches, saving the index once at the end."""
    if not documents:
        return

    total = len(documents)
    batch_starts = range(0, total, batch_size)
    n_batches = len(batch_starts)

    for i in batch_starts:
        batch = documents[i : i + batch_size]
        _write(
            stdout,
            f"  Embedding batch {i // batch_size + 1}/{n_batches}: "
            f"{len(batch)} document(s) ...",
        )
        store.upsert(batch, save=False)

    store.save()
    _write(stdout, f"  Saved {total} document(s) to vector store")


# ============================================================================
# Main entry point
# ============================================================================


def build_rag_index(
    *,
    model_names: Optional[Iterable[str]] = None,
    exclude_models: Optional[Iterable[str]] = None,
    page_id: Optional[int] = None,
    reset_only: bool = False,
    stdout: WriteFn = None,
) -> None:
    """Build the RAG index from live Wagtail pages.

    Args:
        model_names: Models to index (e.g. ['breads.BreadPage']). Defaults to
                     WAGTAIL_RAG_MODELS. Append ':*' to auto-discover all fields.
        exclude_models: Models to skip. Defaults to WAGTAIL_RAG_EXCLUDE_MODELS.
        page_id: Re-index a single page by ID.
        reset_only: Wipe the collection without re-indexing.
        stdout: Callable for progress output (e.g. management command write).
    """
    _write_step_header(stdout, "STEP 1: Loading Configuration")

    raw_names = list(model_names or getattr(settings, "WAGTAIL_RAG_MODELS", None) or [])
    model_names = _parse_model_fields_shorthand(raw_names)
    exclude_models = list(
        exclude_models
        or getattr(
            settings,
            "WAGTAIL_RAG_EXCLUDE_MODELS",
            ["wagtailcore.Page", "wagtailcore.Site"],
        )
    )

    backend = conf.vector_store.backend
    collection = conf.vector_store.collection
    path = conf.vector_store.path

    _write(stdout, f"Vector store: {backend.upper()}")
    _write(stdout, f"Collection:   {collection}")
    _write(stdout, f"Path:         {path}")

    _write(stdout, "\nInitializing embeddings...")
    embeddings = get_embeddings(
        provider=conf.embedding.provider,
        model_name=conf.embedding.model,
    )
    _write(stdout, "Embeddings loaded")

    _write(stdout, "Initializing vector store...")
    store = VectorStore(
        path=path, collection=collection, embeddings=embeddings, backend=backend
    )
    _write(stdout, "Vector store ready")

    if reset_only:
        _write(stdout, "\nResetting collection...")
        store.reset()
        _write(stdout, f'Collection "{collection}" cleared')
        return

    _write_step_header(stdout, "STEP 2: Extracting Documents from Pages")

    models = get_page_models(include=model_names or None, exclude=exclude_models)
    if not models:
        _write(stdout, "ERROR: No page models found")
        return

    _write(
        stdout,
        f"Found {len(models)} model(s): "
        f"{', '.join(f'{m._meta.app_label}.{m.__name__}' for m in models)}",
    )
    if page_id:
        _write(stdout, f"Re-indexing page ID: {page_id}")

    batch_size = getattr(
        settings, "WAGTAIL_RAG_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE
    )
    total_docs = 0
    total_pages = 0

    for model_idx, model in enumerate(models, 1):
        model_name = f"{model._meta.app_label}.{model.__name__}"
        pages = get_live_pages(model, page_id)
        count = pages.count()
        live_ids: set[int] = set(pages.values_list("id", flat=True))

        if count == 0:
            _prune_stale(
                store=store,
                live_ids=live_ids,
                model_name=model_name,
                page_id=page_id,
                stdout=stdout,
            )
            continue

        _write(stdout, f"\n[{model_idx}/{len(models)}] {model_name}: {count} page(s)")
        pending_docs: list = []

        for page_idx, page in enumerate(pages, 1):
            try:
                _write(stdout, f"  [{page_idx}/{count}] {page.title} (ID: {page.id})")

                if _is_page_current(store=store, page=page, page_id=page_id):
                    _write(stdout, "    -> Skipping (up-to-date)")
                    continue

                docs = page_to_documents_api_extractor(page)

                if not docs:
                    _write(stdout, "    WARNING: No content extracted")
                    continue

                extracted_fields = docs[0].metadata.get("extracted_fields", "title")
                _write(
                    stdout, f"    Indexed: {extracted_fields} → {len(docs)} chunk(s)"
                )

                for doc in docs:
                    doc.metadata.update(
                        {
                            "source": model_name,
                            "model": model.__name__,
                            "app": model._meta.app_label,
                            "extractor": "api_fields_extractor",
                        }
                    )

                store.delete_page(page.id)
                pending_docs.extend(docs)
                total_docs += len(docs)
                total_pages += 1

            except Exception as e:
                _write(stdout, f"    ERROR: {e}")
                logger.exception("Error indexing page %s", page.id)

        if pending_docs:
            _write(
                stdout,
                f"\n  Embedding {len(pending_docs)} chunk(s) in batches of {batch_size}...",
            )
            _upsert_in_batches(store, pending_docs, batch_size, stdout)

        _prune_stale(
            store=store,
            live_ids=live_ids,
            model_name=model_name,
            page_id=page_id,
            stdout=stdout,
        )

    _write_step_header(stdout, "STEP 3: Indexing Complete")
    _write(stdout, f"Pages:     {total_pages}")
    _write(stdout, f"Chunks:    {total_docs}")
    _write(stdout, f"Backend:   {backend.upper()}  collection={collection}")
    _write(stdout, f"Location:  {path}")
    _write(stdout, STEP_SEPARATOR + "\n")
