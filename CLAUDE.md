# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

`wagtail-rag` is a reusable Django app (distributed as a pip package) that adds a plug-and-play RAG chatbot to any Wagtail CMS site. It is **not** a standalone Django project — it is a library installed into a host Wagtail project.

## Commands

```bash
# Install for development (with test dependencies)
pip install -e ".[test]"

# Run all tests
pytest wagtail_rag/tests/ -v
# or
make test

# Run a single test module
pytest wagtail_rag/tests/test_api_views.py -v

# Run a specific test
pytest wagtail_rag/tests/test_api_views.py::TestClassName::test_method_name -v

# Lint
make lint                  # runs black --check + flake8
black --check wagtail_rag
flake8 wagtail_rag

# RAG management command (run inside a host Wagtail project)
python manage.py rag index                           # build index
python manage.py rag index --clear             # clear index only
python manage.py rag index --page-id 123            # re-index one page
python manage.py rag chat                            # interactive chat
python manage.py rag chat -q "Your question here"   # single question
python manage.py rag test                            # smoke-test the pipeline
```

Tests use `wagtail_rag/tests/settings.py` as the Django settings module (SQLite in-memory, no real LLM needed — providers are mocked).

## Architecture

### Data flow

```
Wagtail Pages
    -> build_rag_index (management command)
    -> api_fields_extractor (content_extraction/)
    -> VectorStore.upsert() [FAISS, ChromaDB, or pgvector]
    -> Vector index on disk

User query (API or CLI)
    -> RAGChatBot.query()
    -> EmbeddingSearcher.retrieve_with_embeddings()
        -> vector similarity search (always)
        -> Wagtail full-text search (if hybrid enabled)
        -> title-boosting + re-ranking
    -> LLMGenerator.generate_answer()
        -> builds prompt with context
        -> calls LLM (Ollama / OpenAI / Anthropic)
    -> JSON response {answer, sources}
```

### Key modules

| Path | Responsibility |
|------|---------------|
| `wagtail_rag/chatbot.py` | `RAGChatBot` class — top-level orchestrator; `get_chatbot()` convenience factory |
| `wagtail_rag/conf.py` | `conf` singleton — centralised typed settings access for all `WAGTAIL_RAG` groups |
| `wagtail_rag/prompt_guard.py` | `check_question()` — pre-LLM prompt injection detection; pluggable via `WAGTAIL_RAG_PROMPT_GUARD_BACKEND` |
| `wagtail_rag/embeddings/search.py` | `EmbeddingSearcher` — vector search, Wagtail hybrid search, title boosting |
| `wagtail_rag/llm_providers/generation.py` | `LLMGenerator` — prompt templates, LCEL/legacy chain setup, LLM invocation |
| `wagtail_rag/llm_providers/chat_history.py` | `SummarizingHistoryStore` — in-memory per-session chat history with LLM summarization of older turns |
| `wagtail_rag/content_extraction/index_builder.py` | `build_rag_index()` orchestration |
| `wagtail_rag/content_extraction/vector_store.py` | `VectorStore` factory + `FAISSVectorStore`, `ChromaVectorStore`, `PgVectorStore` backends |
| `wagtail_rag/content_extraction/api_fields_extractor.py` | Converts a Wagtail `Page` to LangChain `Document` objects via `api_fields` or auto-discovered content fields |
| `wagtail_rag/embeddings/providers.py` | `get_embeddings()` factory — HuggingFace, OpenAI, Sentence Transformers |
| `wagtail_rag/llm_providers/providers.py` | `get_llm()` factory — Ollama, OpenAI, Anthropic |
| `wagtail_rag/views.py` | `rag_chat_api` — CSRF-exempt GET/POST endpoint at `/api/rag/chat/` |
| `wagtail_rag/management/commands/rag.py` | Unified `rag` command with `index`, `chat`, and `test` subcommands |

### Vector store abstraction

`VectorStore` (in `content_extraction/vector_store.py`) is a factory function that returns the configured backend — `FAISSVectorStore`, `ChromaVectorStore`, or `PgVectorStore`. Switching backends is controlled by `WAGTAIL_RAG_VECTOR_STORE_BACKEND` (`"faiss"`, `"chroma"`, or `"pgvector"`).

For pgvector, metadata operations (`delete_page`, `page_is_current`, `delete_pages_not_in`) use raw SQLAlchemy queries against LangChain's internal `langchain_pg_embedding` table (columns: `collection_id`, `custom_id`, `cmetadata` JSONB). If LangChain changes this schema, these queries will need updating. The pgvector connection string is read from `WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING` or derived from `settings.DATABASES['default']`.

### LangChain version handling

The codebase supports both modern LCEL-style chains (`langchain_core`) and legacy `RetrievalQA`. Import try/except blocks at module level detect which is available (`LCEL_AVAILABLE`, `LEGACY_AVAILABLE`). When adding LangChain code, always guard imports the same way and provide a fallback.

### Content extraction priority

When indexing a page, the extractor checks in this order:
1. `page.api_fields` — uses Wagtail's declared API fields if present
2. `WAGTAIL_RAG_DEFAULT_FIELDS` setting — configured field names (default: `["introduction", "body", "content", "backstory"]`)
3. Auto-discovery fallback — scans all `StreamField`, `RichTextField`, `TextField`, long `CharField` on the model (activated by appending `:*` to the model name in `WAGTAIL_RAG_MODELS`)

### Prompt injection guard

`wagtail_rag/prompt_guard.py` runs before every LLM call. `check_question(question)` returns a `GuardResult(blocked, reason, matched_patterns)`. When `blocked=True`, the view returns a 400 and the question is never forwarded to the LLM.

Built-in patterns detected: override-instructions, role-hijack, prompt-exfiltration, marker-injection (SYSTEM:, `<|im_start|>`, etc.), jailbreak-keyword (DAN, etc.), boundary-injection (`---END---`), instruction-override.

To replace the guard, point `WAGTAIL_RAG["api"]["prompt_guard_backend"]` (or flat `WAGTAIL_RAG_PROMPT_GUARD_BACKEND`) at any callable with signature `def my_guard(question: str) -> GuardResult`. The built-in guard is the default.

### Centralised config (`conf.py`)

All settings are accessed through the `conf` singleton (`from wagtail_rag.conf import conf`). Flat `WAGTAIL_RAG_*` keys are read as fallbacks so existing deployments work without changes.

### Settings reference

Full grouped dict config (`conf.py` maps each key):

```python
WAGTAIL_RAG = {
    "embedding": {
        "provider": "openai",            # "openai" | "sentence-transformers" | "huggingface"
        "model":    "text-embedding-3-small",
    },
    "llm": {
        "provider":                "openai",   # "openai" | "ollama" | "anthropic"
        "model":                   "gpt-4o",
        "max_context_chars":       8000,       # 0 = unlimited
        "enable_history":          True,
        "history_recent_messages": 6,
    },
    "vector_store": {
        "backend":    "faiss",           # "faiss" | "chroma" | "pgvector"
        "path":       "/path/to/index",
        "collection": "wagtail_rag",
        # "connection_string": "postgresql+psycopg2://..."  # pgvector only
    },
    "indexing": {
        "chunk_size":      1500,
        "chunk_overlap":   100,
        "batch_size":      100,
        "skip_if_indexed": True,
        "prune_deleted":   True,
        "models": {
            "myapp.BlogPage": ["introduction", "body"],
            "myapp.OtherPage": "*",          # "*" = use Wagtail search_fields
        },
    },
    "search": {
        "k":                   8,    # chunks retrieved per query
        "max_sources":         3,    # unique pages shown as sources
        "use_hybrid":          True, # combine vector + Wagtail full-text search
        "use_query_expansion": True, # MultiQueryRetriever
        "search_k":            10,   # k for score-based search
        "title_boost_max_score": None,
    },
    "api": {
        "max_question_length":    150,     # chars; 0 = unlimited
        "max_request_body_size":  1048576, # bytes (1 MB)
        "rate_limit_per_minute":  0,       # 0 = disabled
        "prompt_guard_backend":   None,    # dotted path to custom guard callable
    },
}
```

Flat fallback keys (legacy, still supported):

| Flat key | Group key |
|---|---|
| `WAGTAIL_RAG_EMBEDDING_PROVIDER` | `embedding.provider` |
| `WAGTAIL_RAG_EMBEDDING_MODEL` | `embedding.model` |
| `WAGTAIL_RAG_LLM_PROVIDER` | `llm.provider` |
| `WAGTAIL_RAG_MODEL_NAME` | `llm.model` |
| `WAGTAIL_RAG_ENABLE_CHAT_HISTORY` | `llm.enable_history` |
| `WAGTAIL_RAG_VECTOR_STORE_BACKEND` | `vector_store.backend` |
| `WAGTAIL_RAG_CHROMA_PATH` | `vector_store.path` |
| `WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING` | `vector_store.connection_string` |
| `WAGTAIL_RAG_COLLECTION_NAME` | `vector_store.collection` |
| `WAGTAIL_RAG_MODELS` | `indexing.models` (all get `"*"`) |
| `WAGTAIL_RAG_USE_HYBRID_SEARCH` | `search.use_hybrid` |
| `WAGTAIL_RAG_USE_LLM_QUERY_EXPANSION` | `search.use_query_expansion` |
| `WAGTAIL_RAG_RETRIEVE_K` | `search.k` |
| `WAGTAIL_RAG_MAX_SOURCES` | `search.max_sources` |
| `WAGTAIL_RAG_MAX_QUESTION_LENGTH` | `api.max_question_length` |
| `WAGTAIL_RAG_RATE_LIMIT_PER_MINUTE` | `api.rate_limit_per_minute` |
| `WAGTAIL_RAG_PROMPT_GUARD_BACKEND` | `api.prompt_guard_backend` |

## Code Style

- PEP 8 formatting enforced via `black` and `flake8`
- Type hints on all functions and classes
- Docstrings on all functions and classes
- Keep views thin — business logic belongs in model methods or dedicated modules, not in views
- Use class-based views where appropriate; the existing `rag_chat_api` is function-based for simplicity but new views should prefer CBVs
- Follow Django REST Framework conventions for any new API endpoints

## Code Review Checklist

When modifying or reviewing code, check for:

- **N+1 queries** — use `select_related`/`prefetch_related` on Wagtail page querysets
- **Wagtail page model correctness** — use `.live()`, `.specific()`, and `StreamField` properly
- **Security** — CSRF exemption is intentional only on `rag_chat_api`; all other views must respect CSRF. Also check for XSS (HTML in responses) and SQL injection (avoid raw queries)
- **Missing migrations** — if any Django model is changed, a migration is required
- **Test coverage** — new logic should be covered in `wagtail_rag/tests/`

## Test files

| File | What it covers |
|------|---------------|
| `wagtail_rag/tests/test_api_views.py` | `rag_chat_api` HTTP endpoint — question validation, rate limiting, guard blocking |
| `wagtail_rag/tests/test_chatbot.py` | `RAGChatBot._format_sources` deduplication and `_build_retrieval_query` history enrichment |
| `wagtail_rag/tests/test_rag_command.py` | Management command subcommands: `rag index`, `rag chat`, `rag test` |

Tests use `wagtail_rag/tests/settings.py` (SQLite in-memory, all LLM/embedding providers mocked — no real API calls).

## Package distribution

- **Package name**: `wagtail-rag` v0.1.1
- **PyPI install**: `pip install wagtail-rag[faiss,openai]` (choose extras by provider)
- **Extras**: `faiss`, `faiss-gpu`, `chroma`, `pgvector`, `huggingface`, `sentence-transformers`, `openai`, `ollama`, `anthropic`, `local` (FAISS+ST+Ollama bundle), `all`, `dev`, `test`
- **GitHub**: https://github.com/awais786/wagtail-ai-chat

## Important Constraints

- The chat API endpoint (`/api/rag/chat/`) is CSRF-exempt by design — external clients need it without CSRF tokens.
- Chat history is stored in-memory (process-local); it is lost on server restart. There is no persistent session storage.
- If the vector index dimension changes (e.g. switching embedding models), the index must be reset before rebuilding: `manage.py rag index --clear` then `manage.py rag index`.
- FAISS does not natively support metadata filtering the way ChromaDB does. Filter operations on FAISS iterate the in-memory docstore.
- `check_question()` in `prompt_guard.py` must never raise — errors are logged and the question is allowed through to avoid breaking the chat on misconfiguration.
- All new settings must be added to both the grouped dict in `conf.py` and the flat-key fallback mapping. Access settings via `conf.*` — never read `django.conf.settings.WAGTAIL_RAG_*` directly in feature code.
