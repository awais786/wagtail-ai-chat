import os
from typing import Iterable, List, Dict, Tuple, Optional, Any, Callable

from django.apps import apps
from django.conf import settings
from wagtail.models import Page

try:
    from .content_extraction import extract_page_content, get_page_url, wagtail_page_to_documents

    CONTENT_EXTRACTION_AVAILABLE = True
    WAGTAIL_PAGE_TO_DOCS_AVAILABLE = True
except ImportError:
    CONTENT_EXTRACTION_AVAILABLE = False
    WAGTAIL_PAGE_TO_DOCS_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError(
                "Could not import RecursiveCharacterTextSplitter. "
                "Please install langchain-text-splitters: pip install langchain-text-splitters"
            )

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    try:
        from langchain.vectorstores import Chroma
    except ImportError:
        raise ImportError(
            "Could not import Chroma. "
            "Please install langchain-community: pip install langchain-community"
        )

try:
    from wagtail_rag.embeddings import get_embeddings
except ImportError:
    # Fallback for older installations - try multiple import paths
    _fallback_embeddings_available = False
    
    # Try langchain_huggingface first
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        _fallback_embeddings_available = True
        _HuggingFaceEmbeddings = HuggingFaceEmbeddings
    except ImportError:
        # Try langchain_community as fallback
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _fallback_embeddings_available = True
            _HuggingFaceEmbeddings = HuggingFaceEmbeddings
        except ImportError:
            pass
    
    if not _fallback_embeddings_available:
        raise ImportError(
            "Could not import embeddings. "
            "Please install wagtail-rag[embeddings] or langchain-huggingface or langchain-community"
        )
    
    def get_embeddings(provider=None, model_name=None, **kwargs):
        """Fallback embedding function for older installations."""
        if provider and provider.lower() not in ("huggingface", "hf", None):
            raise ValueError(
                f"Only HuggingFace embeddings available in fallback mode. Requested: {provider}"
            )
        return _HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            **kwargs,
        )


# Type alias for an optional output function (e.g. management command stdout.write)
WriteFn = Optional[Callable[[str], None]]


def _write(out: WriteFn, message: str) -> None:
    if out is not None:
        out(message)


def _get_page_models(
    model_names: Optional[Iterable[str]] = None,
    exclude_models: Optional[Iterable[str]] = None,
) -> List[type]:
    """
    Get all Wagtail Page models dynamically, filtered by include/exclude lists.
    """
    page_models: List[type] = []
    exclude_models = set(exclude_models or [])

    for app_config in apps.get_app_configs():
        for model in app_config.get_models():
            if issubclass(model, Page) and model != Page:
                model_name = f"{model._meta.app_label}.{model.__name__}"

                if model_names and model_name not in model_names:
                    continue
                if model_name in exclude_models:
                    continue

                page_models.append(model)

    return page_models


def _extract_page_text(page: Page, important_fields: Optional[List[str]] = None) -> str:
    """
    Extract text content from a Wagtail Page using the centralized content_extraction utility.
    """
    if CONTENT_EXTRACTION_AVAILABLE:
        return extract_page_content(page, important_fields=important_fields)

    # Fallback: Basic extraction if content_extraction is not available
    text_parts: List[str] = [f"Title: {page.title}"]

    if hasattr(page, "search_description") and page.search_description:
        text_parts.append(f"Description: {page.search_description}")

    if hasattr(page, "body"):
        body = getattr(page, "body", None)
        if body:
            text_parts.append(f"Body: {str(body)[:1000]}")

    return "\n\n".join(text_parts)


def _get_page_url_safe(page: Page) -> str:
    """
    Safely get page URL using centralized utility or fallback.
    """
    if CONTENT_EXTRACTION_AVAILABLE:
        return get_page_url(page)

    try:
        return page.url if hasattr(page, "url") else ""
    except Exception:
        return ""


def _parse_model_fields(model_fields_arg: Any) -> Dict[str, List[str]]:
    """
    Parse model-fields configuration into a mapping of model_name -> [field,...].
    """
    if not model_fields_arg:
        return {}
    if isinstance(model_fields_arg, dict):
        return model_fields_arg

    model_fields_map: Dict[str, List[str]] = {}
    for item in model_fields_arg:
        if ":" in item:
            model_name, fields_str = item.split(":", 1)
            fields = [f.strip() for f in fields_str.split(",")]
            model_fields_map[model_name] = fields
    return model_fields_map


EXCLUDED_FIELD_NAMES = {
    # Reverse relations / admin metadata
    "index_entries",
    "specific_workflow_states",
    "workflow_states",
    "revisions",
    "subscribers",
    "wagtail_admin_comments",
    "view_restrictions",
    "group_permissions",
    "aliases",
    "sites_rooted_here",
    # Low-level internals
    "content_type",
    "page_ptr",
    # Wagtail Page base class internals
    "path",
    "depth",
    "translation_key",
    "locale",
    "latest_revision",
    "live",
    "first_published_at",
    "last_published_at",
    "live_revision",
    "draft_title",
    "slug",
    "url_path",
    "owner",
    "latest_revision_created_at",
    "seo_title",
    "search_description",
    "show_in_menus",
    "has_unpublished_changes",
    "go_live_at",
    "expire_at",
    "expired",
    "locked",
    "locked_at",
    "locked_by",
}


def _get_all_content_field_names(model: type) -> List[str]:
    """
    Return a filtered list of "content" field names for a model.
    Used when model fields contain a wildcard '*' for a model.
    """
    from django.db.models import Field

    field_names: List[str] = []
    for f in model._meta.get_fields():
        if getattr(f, "auto_created", False):
            continue
        if not isinstance(f, Field):
            continue
        if f.name in EXCLUDED_FIELD_NAMES:
            continue
        field_names.append(f.name)
    return field_names


def _get_metadata_field_value(value: Any) -> Optional[str]:
    """
    Extract string representation from a field value for metadata.
    """
    if value is None:
        return None

    if hasattr(value, "title") and not isinstance(value, str):
        attr = getattr(value, "title")
        return attr() if callable(attr) else attr
    if hasattr(value, "name") and not isinstance(value, str):
        attr = getattr(value, "name")
        return attr() if callable(attr) else attr
    if hasattr(value, "all"):
        items = value.all()
        if items:
            return ", ".join(
                getattr(item, "name", getattr(item, "title", str(item)))
                for item in items
            )
        return None

    return str(value)


def get_pages_to_index(
    model_names: Optional[Iterable[str]] = None,
    exclude_models: Optional[Iterable[str]] = None,
    model_fields_map: Optional[Dict[str, List[str]]] = None,
    page_id: Optional[int] = None,
    stdout: WriteFn = None,
    use_document_approach: bool = True,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Public helper that collects documents and metadata for pages to index.
    
    Args:
        use_document_approach: If True, uses wagtail_page_to_documents (new approach).
                              If False, uses old text extraction approach.
    
    Returns:
        Tuple of (documents/texts, metadata list)
        - If use_document_approach=True: (List[Document], List[Dict])
        - If use_document_approach=False: (List[str], List[Dict])
    """
    model_fields_map = model_fields_map or {}
    
    # Use new document-based approach by default
    if use_document_approach and WAGTAIL_PAGE_TO_DOCS_AVAILABLE:
        return _get_pages_as_documents(
            model_names=model_names,
            exclude_models=exclude_models,
            model_fields_map=model_fields_map,
            page_id=page_id,
            stdout=stdout,
        )
    
    # Fallback to old text-based approach
    pages: List[str] = []
    metadata: List[Dict[str, Any]] = []

    page_models = _get_page_models(model_names=model_names, exclude_models=exclude_models)
    if not page_models:
        _write(stdout, "No page models found to index.")
        return pages, metadata

    _write(stdout, f"Found {len(page_models)} page model(s) to index:")
    for model in page_models:
        _write(stdout, f"  - {model._meta.app_label}.{model.__name__}")

    for model in page_models:
        model_name = f"{model._meta.app_label}.{model.__name__}"

        raw_model_fields = model_fields_map.get(model_name, [])
        if raw_model_fields == ["*"]:
            try:
                model_important_fields = _get_all_content_field_names(model)
            except Exception:
                model_important_fields = []
        else:
            model_important_fields = raw_model_fields

        if page_id:
            try:
                live_pages = model.objects.filter(id=page_id, live=True)
                if not live_pages.exists():
                    continue
                _write(stdout, f"  Re-indexing page ID {page_id} from {model_name}...")
            except Exception as e:
                _write(stdout, f"  Error finding page {page_id} in {model_name}: {e}")
                continue
        else:
            live_pages = model.objects.live()

        count = live_pages.count()
        if count == 0:
            if not page_id:
                _write(stdout, f"  No live pages found for {model_name}")
            continue

        if not page_id:
            _write(stdout, f"  Indexing {count} pages from {model_name}...")

        for page in live_pages:
            try:
                page_text = _extract_page_text(
                    page,
                    important_fields=model_important_fields or None,
                )
                if not page_text.strip():
                    continue

                pages.append(page_text)
                page_metadata: Dict[str, Any] = {
                    "source": model_name,
                    "model": model.__name__,
                    "app": model._meta.app_label,
                    "title": page.title,
                    "url": _get_page_url_safe(page),
                    "id": page.id,
                    "slug": getattr(page, "slug", "") if hasattr(page, "slug") else "",
                }

                separator = "=" * 80
                _write(stdout, separator)
                _write(
                    stdout,
                    f'Page: {page_metadata["model"]} (ID: {page_metadata["id"]}) '
                    f'- "{page_metadata["title"]}"',
                )
                _write(stdout, "-" * 80)
                _write(stdout, page_text)
                _write(stdout, "")

                if model_important_fields:
                    for field_name in model_important_fields:
                        if hasattr(page, field_name):
                            value = getattr(page, field_name, None)
                            field_value = _get_metadata_field_value(value)
                            if field_value:
                                page_metadata[field_name] = field_value

                metadata.append(page_metadata)
            except Exception as e:
                _write(
                    stdout,
                    f"  Error extracting text from {model_name} (ID: {page.id}): {e}",
                )

    return pages, metadata


def _get_pages_as_documents(
    model_names: Optional[Iterable[str]] = None,
    exclude_models: Optional[Iterable[str]] = None,
    model_fields_map: Optional[Dict[str, List[str]]] = None,
    page_id: Optional[int] = None,
    stdout: WriteFn = None,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Get pages as Document objects using the new wagtail_page_to_documents approach.
    
    Note: model_fields_map is accepted for API consistency but not currently used
    in the document-based approach. The new approach extracts content from standard
    Wagtail fields (title, intro, body, content, etc.) automatically.
    
    Returns:
        Tuple of (List[Document], List[Dict]) where each Document has its own metadata
    """
    from django.conf import settings
    
    # model_fields_map is kept for API consistency but not used in document approach
    # The new wagtail_page_to_documents handles field extraction automatically
    _ = model_fields_map  # Suppress unused variable warning
    documents: List[Any] = []
    all_metadata: List[Dict[str, Any]] = []
    
    # Get chunk size/overlap from settings
    chunk_size = getattr(settings, 'WAGTAIL_RAG_CHUNK_SIZE', 500)
    chunk_overlap = getattr(settings, 'WAGTAIL_RAG_CHUNK_OVERLAP', 75)
    
    page_models = _get_page_models(model_names=model_names, exclude_models=exclude_models)
    if not page_models:
        _write(stdout, "No page models found to index.")
        return documents, all_metadata

    _write(stdout, f"Found {len(page_models)} page model(s) to index:")
    for model in page_models:
        _write(stdout, f"  - {model._meta.app_label}.{model.__name__}")

    for model in page_models:
        model_name = f"{model._meta.app_label}.{model.__name__}"

        if page_id:
            try:
                live_pages = model.objects.filter(id=page_id, live=True)
                if not live_pages.exists():
                    continue
                _write(stdout, f"  Re-indexing page ID {page_id} from {model_name}...")
            except Exception as e:
                _write(stdout, f"  Error finding page {page_id} in {model_name}: {e}")
                continue
        else:
            live_pages = model.objects.live()

        count = live_pages.count()
        if count == 0:
            if not page_id:
                _write(stdout, f"  No live pages found for {model_name}")
            continue

        if not page_id:
            _write(stdout, f"  Indexing {count} pages from {model_name}...")

        for page in live_pages:
            try:
                # Use new document-based approach
                page_docs = wagtail_page_to_documents(
                    page,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    stdout=stdout,  # Pass stdout to print documents
                )
                
                if not page_docs:
                    continue

                # Add model-level metadata to each document
                for doc in page_docs:
                    doc.metadata.update({
                        "source": model_name,
                        "model": model.__name__,
                        "app": model._meta.app_label,
                    })
                    documents.append(doc)
                    all_metadata.append(doc.metadata)

                # Log page extraction summary (detailed document output is handled by wagtail_page_to_documents)
                separator = "=" * 80
                _write(stdout, separator)
                _write(
                    stdout,
                    f'Page: {model.__name__} (ID: {page.id}) - "{page.title}" - {len(page_docs)} documents',
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

    return documents, all_metadata


def build_rag_index(
    *,
    model_names: Optional[Iterable[str]] = None,
    reset_only: bool = False,
    page_id: Optional[int] = None,
    stdout: WriteFn = None,
) -> None:
    """
    High-level API to build the RAG index.

    This is extracted from the management command so it can be reused
    programmatically (e.g. from views, signals, or custom scripts).
    """
    # Resolve model configuration from settings if not explicitly provided
    if model_names is None:
        model_names = getattr(settings, "WAGTAIL_RAG_MODELS", None)

    # Initialize model_fields_arg (will be set from shorthand or settings)
    model_fields_arg: Optional[Any] = None

    # Support convenience syntax "app.Model:*" in WAGTAIL_RAG_MODELS / model_names.
    #
    # Example:
    #   WAGTAIL_RAG_MODELS = [
    #       "breads.BreadPage",
    #       "locations.LocationPage:*",
    #   ]
    #
    # becomes:
    #   model_names      = ["breads.BreadPage", "locations.LocationPage"]
    #   model_fields_arg = ["locations.LocationPage:*"]  (if not already provided)
    auto_model_fields: List[str] = []
    if model_names:
        cleaned_model_names: List[str] = []
        for name in model_names:
            if isinstance(name, str) and name.endswith(":*"):
                base = name.split(":", 1)[0]
                cleaned_model_names.append(base)
                auto_model_fields.append(f"{base}:*")
            else:
                cleaned_model_names.append(name)
        model_names = cleaned_model_names

        # If we have shorthand-generated model fields, use them
        if auto_model_fields:
            model_fields_arg = auto_model_fields

    # If no model_fields_arg from shorthand, try to get from settings
    if model_fields_arg is None:
        model_fields_arg = getattr(settings, "WAGTAIL_RAG_MODEL_FIELDS", None)

    # Resolve other configuration from settings
    # Note: Defaults match the new document-based approach (500/75)
    chunk_size = getattr(settings, "WAGTAIL_RAG_CHUNK_SIZE", 500)
    chunk_overlap = getattr(settings, "WAGTAIL_RAG_CHUNK_OVERLAP", 75)
    collection_name = getattr(
        settings,
        "WAGTAIL_RAG_COLLECTION_NAME",
        "wagtail_rag",
    )

    # Default exclude models from settings
    default_excludes = getattr(
        settings,
        "WAGTAIL_RAG_EXCLUDE_MODELS",
        ["wagtailcore.Page", "wagtailcore.Site"],
    )
    exclude_models: Iterable[str] = default_excludes

    # Parse model-specific fields
    model_fields_map = _parse_model_fields(model_fields_arg or [])

    if page_id:
        _write(stdout, f"Re-indexing specific page ID: {page_id}")

    if model_fields_map:
        _write(stdout, f"Model-specific important fields: {model_fields_map}")

    # Embedding configuration
    embedding_provider = getattr(
        settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface"
    )
    embedding_model = getattr(settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None)

    # If we only want to reset the collection, do it now and exit
    if reset_only:
        _write(stdout, "Resetting collection (reset-only mode)...")
        embeddings = get_embeddings(
            provider=embedding_provider,
            model_name=embedding_model,
        )
        persist_directory = getattr(
            settings,
            "WAGTAIL_RAG_CHROMA_PATH",
            os.path.join(settings.BASE_DIR, "chroma_db"),
        )
        os.makedirs(persist_directory, exist_ok=True)

        try:
            existing_db = Chroma(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
            existing_db.delete_collection()
            _write(stdout, f'Collection "{collection_name}" cleared successfully')
        except Exception as e:
            _write(stdout, f"Collection may not exist: {e}")

        _write(stdout, "Reset-only complete. No indexing performed.")
        return

    _write(stdout, "Starting RAG index building...")

    if model_names:
        _write(
            stdout,
            f"Extracting data from specified models: {', '.join(model_names)}",
        )
    else:
        _write(stdout, "Extracting data from all Page models...")

    if exclude_models:
        _write(stdout, f"Excluding models: {', '.join(exclude_models)}")

    pages_data, metadatas = get_pages_to_index(
        model_names=model_names,
        exclude_models=exclude_models,
        model_fields_map=model_fields_map,
        page_id=page_id,
        stdout=stdout,
        use_document_approach=True,  # Use new document-based approach
    )

    if not pages_data:
        _write(stdout, "No pages found to index.")
        return

    # Check if we got Document objects (new approach) or text strings (old approach)
    is_document_approach = pages_data and hasattr(pages_data[0], 'page_content')
    
    if is_document_approach:
        # New approach: pages_data is List[Document], already chunked
        _write(stdout, f"Extracted {len(pages_data)} documents from pages")
        chunks: List[str] = [doc.page_content for doc in pages_data]
        chunk_metadatas: List[Dict[str, Any]] = [doc.metadata for doc in pages_data]
        
        # Generate deterministic chunk IDs
        chunk_ids: List[str] = []
        chunk_counters: Dict[str, int] = {}  # Track chunk index per page+section
        
        for doc in pages_data:
            meta = doc.metadata
            page_id_val = meta.get("page_id") or meta.get("id")
            app = meta.get("app", "unknown")
            model = meta.get("page_type") or meta.get("model", "unknown")
            section = meta.get("section", "body")
            
            # Create unique key for this page+section combination
            key = f"{app}_{model}_{page_id_val}_{section}"
            chunk_idx = chunk_counters.get(key, 0)
            chunk_counters[key] = chunk_idx + 1
            
            chunk_id = f"{key}_{chunk_idx}"
            chunk_ids.append(chunk_id)
            
            # Update metadata with chunk index
            meta["chunk_index"] = chunk_idx
        
        _write(stdout, f"Using {len(chunks)} documents (already chunked with title context)")
    else:
        # Old approach: pages_data is List[str], need to chunk
        texts = pages_data
        _write(stdout, f"Extracted {len(texts)} pages")

        # Split into chunks
        _write(
            stdout,
            f"Chunking text (size: {chunk_size}, overlap: {chunk_overlap})...",
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        chunks: List[str] = []
        chunk_metadatas: List[Dict[str, Any]] = []
        chunk_ids: List[str] = []

        for text, metadata in zip(texts, metadatas):
            text_chunks = text_splitter.split_text(text)

            prefix_parts = [f"Title: {metadata.get('title', 'Unknown')}"]
            exclude_from_prefix = {
                "source",
                "model",
                "app",
                "id",
                "slug",
                "url",
                "chunk_index",
                "total_chunks",
            }
            for key, value in metadata.items():
                if key not in exclude_from_prefix and value:
                    prefix_parts.append(f"{key.replace('_', ' ').title()}: {value}")

            prefix = " | ".join(prefix_parts) + "\n\n"

            page_id_val = metadata.get("id")
            app = metadata.get("app", "unknown")
            model = metadata.get("model", "unknown")

            for chunk_idx, chunk in enumerate(text_chunks):
                chunk_with_context = prefix + chunk
                chunks.append(chunk_with_context)

                chunk_id = f"{app}_{model}_{page_id_val}_{chunk_idx}"
                chunk_ids.append(chunk_id)

                chunk_metadatas.append(
                    {
                        **metadata,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(text_chunks),
                    }
                )

        _write(stdout, f"Created {len(chunks)} chunks from {len(texts)} pages")

    _write(stdout, "Initializing embeddings...")
    embeddings = get_embeddings(
        provider=embedding_provider,
        model_name=embedding_model,
    )

    persist_directory = getattr(
        settings,
        "WAGTAIL_RAG_CHROMA_PATH",
        os.path.join(settings.BASE_DIR, "chroma_db"),
    )
    os.makedirs(persist_directory, exist_ok=True)

    # Check if collection exists and warn about embedding dimension mismatch
    try:
        existing_db = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        try:
            sample = existing_db.get(limit=1)
            if sample and sample.get("ids"):
                if not reset_only:
                    _write(
                        stdout,
                        (
                            f'\nWARNING: Collection "{collection_name}" already exists.\n'
                            f"   If you changed embedding providers/models, you SHOULD clear\n"
                            f"   the collection first (e.g. via --reset-only).\n"
                            f"   Current embedding: {embedding_provider} / {embedding_model or 'default'}\n"
                        ),
                    )
        except Exception:
            pass
    except Exception:
        existing_db = None

    # Create/update vector store with deterministic IDs
    _write(
        stdout,
        f"Storing chunks in ChromaDB (collection: {collection_name})...",
    )

    try:
        existing_db = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

        try:
            existing_db._collection.count()
        except Exception as e:
            error_msg = str(e)
            if "dimension" in error_msg.lower() or "embedding" in error_msg.lower():
                _write(
                    stdout,
                    (
                        "\nERROR: Embedding dimension mismatch!\n"
                        "   The existing collection was created with a different embedding model.\n"
                        f"   Current embedding: {embedding_provider} / {embedding_model or 'default'}\n"
                        "   You MUST reset the collection first.\n"
                        f"   Error: {error_msg}\n"
                    ),
                )
                return
            else:
                raise

        if page_id:
            _write(stdout, f"Deleting old chunks for page ID {page_id}...")
            try:
                all_chunks = existing_db.get(include=["metadatas"])
                if all_chunks and all_chunks.get("ids"):
                    page_chunk_ids: List[str] = []
                    for idx, chunk_id in enumerate(all_chunks["ids"]):
                        if all_chunks.get("metadatas") and idx < len(
                            all_chunks["metadatas"]
                        ):
                            m = all_chunks["metadatas"][idx]
                            # Check both 'id' (old approach) and 'page_id' (new approach)
                            if m and (m.get("id") == page_id or m.get("page_id") == page_id):
                                page_chunk_ids.append(chunk_id)

                    if page_chunk_ids:
                        existing_db.delete(ids=page_chunk_ids)
                        _write(
                            stdout,
                            f"  Deleted {len(page_chunk_ids)} old chunks",
                        )
            except Exception as e:
                _write(stdout, f"  Could not delete old chunks: {e}")

        _write(stdout, "Updating/adding documents with deterministic IDs...")

        try:
            existing_db.add_texts(
                texts=chunks,
                metadatas=chunk_metadatas,
                ids=chunk_ids,
            )
        except Exception as e:
            error_msg = str(e)
            if "dimension" in error_msg.lower() or "embedding" in error_msg.lower():
                _write(
                    stdout,
                    (
                        "\nERROR: Embedding dimension mismatch!\n"
                        "   The collection was created with a different embedding model.\n"
                        f"   Current embedding: {embedding_provider} / {embedding_model or 'default'}\n"
                        "   You MUST reset the collection first.\n"
                        f"   Error: {error_msg}\n"
                    ),
                )
                return
            else:
                raise
    except Exception as e:
        error_msg = str(e)
        if "dimension" in error_msg.lower() or "embedding" in error_msg.lower():
            _write(
                stdout,
                (
                    "\nERROR: Embedding dimension mismatch!\n"
                    "   The collection was created with a different embedding model.\n"
                    f"   Current embedding: {embedding_provider} / {embedding_model or 'default'}\n"
                    "   You MUST reset the collection first.\n"
                    f"   Error: {error_msg}\n"
                ),
            )
            return

        _write(stdout, "Creating new collection...")
        Chroma.from_texts(
            texts=chunks,
            metadatas=chunk_metadatas,
            ids=chunk_ids,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    _write(
        stdout,
        f'Successfully indexed {len(chunks)} chunks in ChromaDB collection "{collection_name}"',
    )
    _write(stdout, f"Vector store saved to: {persist_directory}")


