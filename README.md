# Wagtail RAG Chatbot

A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS. Drop `wagtail_rag` into any Wagtail site to get a fully working chatbot backed by your page content — no model-specific configuration required.

## Features

- **Universal**: works with any Wagtail page model; auto-discovers content fields
- **Per-field chunking**: each field (body, introduction, …) is chunked independently with section metadata, so the LLM always knows where a chunk came from
- **Hybrid retrieval**: vector similarity search + optional Wagtail full-text search
- **Multiple vector stores**: FAISS (default), ChromaDB, pgvector (PostgreSQL)
- **Multiple embedding providers**: HuggingFace, Sentence Transformers, OpenAI
- **Multiple LLM providers**: Ollama (local), OpenAI, Anthropic
- **Chat history**: server-side with LLM summarisation of older turns
- **Unified CLI**: one management command covers indexing, chat, and pipeline smoke-testing
- **CSRF protection**: API endpoint enforces Django CSRF on POST requests

## Installation

### 1. Install the package with your chosen provider

```bash
# Local stack (recommended for development): FAISS + Sentence Transformers + Ollama
pip install "wagtail-rag[local] @ git+https://github.com/awais786/wagtail-ai-chat.git"

# OpenAI
pip install "wagtail-rag[openai] @ git+https://github.com/awais786/wagtail-ai-chat.git"

# All providers
pip install "wagtail-rag[all] @ git+https://github.com/awais786/wagtail-ai-chat.git"

# Local checkout (development)
pip install -e ".[local]"
```

Available extras:

| Extra | Installs |
|---|---|
| `faiss` | faiss-cpu |
| `chroma` | chromadb |
| `pgvector` | psycopg2-binary, sqlalchemy |
| `sentence-transformers` | sentence-transformers |
| `huggingface` | langchain-huggingface, sentence-transformers |
| `openai` | langchain-openai |
| `ollama` | ollama |
| `anthropic` | langchain-anthropic |
| `local` | faiss + sentence-transformers + ollama |
| `all` | every provider |

### 2. Add to INSTALLED_APPS

```python
# settings.py
INSTALLED_APPS = [
    # ...
    "wagtail_rag",
]
```

> `wagtail_rag` has no database models — no migration needed.

### 3. Configure settings

Pick one setup block and add it to `settings.py`:

```python
# --- Option A: Local (Ollama + Sentence Transformers + FAISS) ---
WAGTAIL_RAG = {
    "embedding":    {"provider": "sentence-transformers", "model": "all-MiniLM-L6-v2"},
    "llm":          {"provider": "ollama", "model": "mistral"},
    "vector_store": {"backend": "faiss", "path": os.path.join(BASE_DIR, "faiss_index")},
}

# --- Option B: OpenAI ---
WAGTAIL_RAG = {
    "embedding":    {"provider": "openai", "model": "text-embedding-3-small"},
    "llm":          {"provider": "openai", "model": "gpt-4o"},
    "vector_store": {"backend": "faiss", "path": os.path.join(BASE_DIR, "faiss_index")},
}
# Set OPENAI_API_KEY as an environment variable, not here
```

For Option A, make sure Ollama is running: `ollama serve` and `ollama pull mistral`.

### 4. Add URL configuration (for the API endpoint and chatbox)

```python
# urls.py
from django.urls import include, path
from wagtail import urls as wagtail_urls

urlpatterns = [
    path("", include("wagtail_rag.urls")),  # must come before wagtail_urls
    path("", include(wagtail_urls)),
]
```

This exposes:
- `GET/POST /api/rag/chat/` — chat API
- `GET /chatbox/` — standalone widget page (for testing)

### 5. Build the index

```bash
python manage.py rag index
```

### 6. Verify with a test question

```bash
python manage.py rag chat -q "What content is on this site?"
```

Or run the built-in smoke test:

```bash
python manage.py rag test
```

### 7. Add the chatbox widget to your base template (optional)

Add just before `</body>` in your base template:

```django
{% include "wagtail_rag/chatbox.html" %}
```

This renders a floating action button (bottom-right). Clicking it opens the chat panel. The widget reads the Django `csrftoken` cookie and sends it automatically on every request.

---

## Management Command

All RAG operations go through a single command: `manage.py rag <subcommand>`.

### `rag index` — build / reset the vector store

```bash
python manage.py rag index                  # full index build
python manage.py rag index --clear     # wipe collection only
python manage.py rag index --page-id 42     # re-index one page
```

If you change embedding models, always reset first:

```bash
python manage.py rag index --clear
python manage.py rag index
```

### `rag chat` — interactive or single-question chat

```bash
python manage.py rag chat                          # interactive loop
python manage.py rag chat -q "What is sourdough?"  # single question
python manage.py rag chat --search-only            # retrieval only, skip LLM
python manage.py rag chat --no-sources             # hide source list
python manage.py rag chat --no-history             # stateless mode
python manage.py rag chat --filter '{"model":"BreadPage"}'
```

Interactive session commands: `exit`/`quit`, `clear` (new session), `sources on/off`, `Ctrl+C`.

### `rag test` — smoke-test the full pipeline

```bash
python manage.py rag test                                     # built-in questions, full RAG
python manage.py rag test --search-only                       # retrieval only (faster)
python manage.py rag test --questions "Q1" "Q2" "Q3"          # custom questions
python manage.py rag test --filter '{"model":"BlogPage"}'     # scoped to a model
```

Pass criteria: non-empty answer **and** at least 1 source retrieved (full RAG), or at least 1 source (search-only).

Override the default questions via settings:
```python
WAGTAIL_RAG_TEST_QUESTIONS = [
    "What breads do you sell?",
    "Where are you located?",
]
```

---

## Makefile shortcuts

```bash
make install-local   # pip install -e ".[local,test,dev]"
make install-openai  # pip install -e ".[openai,test,dev]"
make install-all     # pip install -e ".[all,test,dev]"

make index           # rag index
make index-reset     # rag index --clear
make index-rebuild   # reset then index
make chat            # rag chat
make test-rag        # rag test (full pipeline)
make test-rag-search # rag test --search-only
make test            # pytest unit tests
make test-cov        # pytest with coverage
make lint            # black + flake8
make format          # black auto-format
make clean           # remove cache and build files
```

---

## Configuration Reference

### Core

```python
WAGTAIL_RAG = {
    "embedding": {
        "provider": "sentence-transformers",  # "sentence-transformers" | "huggingface" | "openai"
        "model":    "all-MiniLM-L6-v2",
    },
    "llm": {
        "provider": "ollama",  # "ollama" | "openai" | "anthropic"
        "model":    "mistral",
    },
    "vector_store": {
        "backend":    "faiss",                           # "faiss" | "chroma" | "pgvector"
        "path":       os.path.join(BASE_DIR, "faiss_index"),
        "collection": "wagtail_rag",
        # "connection_string": "postgresql+psycopg2://..."  # pgvector only
    },
    "indexing": {
        # key   = model to index
        # value = "*" (auto-discover all fields) or ["f1", "f2"] (explicit fields)
        "models": {
            "locations.LocationPage": ["introduction", "body", "address", "lat_long"],
            "breads.BreadPage": "*",
            "blog.BlogPage":    "*",
        },
    },
}
```

### Retrieval

```python
WAGTAIL_RAG_RETRIEVE_K              = 8      # documents returned per query
WAGTAIL_RAG_USE_HYBRID_SEARCH       = True   # vector + Wagtail full-text
WAGTAIL_RAG_USE_LLM_QUERY_EXPANSION = False  # MultiQueryRetriever query expansion
```

### Indexing

Models to index are defined by `indexing.models` inside `WAGTAIL_RAG`.
The flat `WAGTAIL_RAG_MODELS` setting is still accepted as a fallback when `indexing.models` is not set.

```python
WAGTAIL_RAG_EXCLUDE_MODELS = ["wagtailcore.Page"]
WAGTAIL_RAG_CHUNK_SIZE      = 1500
WAGTAIL_RAG_CHUNK_OVERLAP   = 100
WAGTAIL_RAG_SKIP_IF_INDEXED = True  # skip unchanged pages on re-index
WAGTAIL_RAG_PRUNE_DELETED   = True  # remove stale chunks for deleted pages
```

### API and security

```python
WAGTAIL_RAG_MAX_REQUEST_BODY_SIZE    = 1024 * 1024  # 1 MB POST body limit
WAGTAIL_RAG_MAX_QUESTION_LENGTH      = 0            # 0 = no limit
WAGTAIL_RAG_MAX_CONTEXT_CHARS        = 0            # 0 = no limit
WAGTAIL_RAG_ENABLE_CHAT_HISTORY      = True
WAGTAIL_RAG_CHAT_HISTORY_RECENT_MESSAGES = 6
```

### pgvector (PostgreSQL)

Use pgvector when you want the vector index stored in your existing PostgreSQL database rather than on disk.

```python
WAGTAIL_RAG = {
    "embedding": {
        "provider": "openai",
        "model":    "text-embedding-3-small",
    },
    "llm": {
        "provider": "openai",
        "model":    "gpt-4o",
    },
    "vector_store": {
        "backend":           "pgvector",
        "collection":        "wagtail_rag",
        # Explicit connection string — omit to auto-derive from DATABASES['default']
        "connection_string": "postgresql+psycopg2://user:password@localhost:5432/mydb",
    },
}
```

If `connection_string` is omitted, the connection is derived automatically from `DATABASES['default']` (must be a PostgreSQL engine). Install the extra:

```bash
pip install "wagtail-rag[pgvector]"
```

---

## API Endpoints

### `GET /api/rag/chat/`

No CSRF token required for GET. The response also sets the `csrftoken` cookie for subsequent POST requests.

```bash
curl "http://localhost:8000/api/rag/chat/?q=What+types+of+bread+do+you+have?"
```

### `POST /api/rag/chat/`

CSRF token required. Obtain it from the `csrftoken` cookie set by the GET above:

```bash
# Step 1: get the CSRF token
CSRF=$(curl -sc /tmp/jar "http://localhost:8000/api/rag/chat/?q=ping" \
       | python3 -c "import sys,json; print(json.load(sys.stdin).get('answer',''))" 2>/dev/null; \
       grep csrftoken /tmp/jar | awk '{print $NF}')

# Step 2: POST with the token
curl -b /tmp/jar -X POST http://localhost:8000/api/rag/chat/ \
  -H "Content-Type: application/json" \
  -H "X-CSRFToken: $CSRF" \
  -d '{"question": "What types of bread do you have?", "session_id": "abc123"}'
```

Response `200`:
```json
{
  "answer": "We have several types including...",
  "sources": [
    {
      "content": "...",
      "metadata": {"title": "Multigrain Bread", "url": "/breads/multigrain/", "model": "BreadPage", "section": "body"}
    }
  ],
  "session_id": "abc123"
}
```

Error codes: `400` bad request · `403` missing/invalid CSRF token · `413` body too large · `415` wrong Content-Type · `500` server error.

---

## Python API

```python
from wagtail_rag.chatbot import get_chatbot

chatbot = get_chatbot()
result  = chatbot.query("What types of bread do you have?")
print(result["answer"])
print(result["sources"])

# Filter by model
chatbot = get_chatbot(metadata_filter={"model": "BreadPage"})

# Search only (no LLM call)
result = chatbot.query("sourdough", search_only=True)

# Specific provider
chatbot = get_chatbot(llm_provider="openai", model_name="gpt-4o")
```

---

## How It Works

1. **Indexing** (`rag index`): discovers live Wagtail pages → extracts each field independently → chunks with paragraph preservation → prepends `Page: / Section:` header to every chunk → upserts into vector store with deterministic IDs (`{page_id}_{field}_{chunk_index}`). Stale chunks are removed before re-indexing.

2. **Querying** (`rag chat` / API): embeds the question → vector similarity search → optional Wagtail full-text search → deduplicate & title-boost → pass top-k chunks as context to LLM → return answer + sources.

---

## Testing

```bash
# Unit tests
pytest wagtail_rag/tests/ -v

# With coverage
pytest wagtail_rag/tests/ --cov=wagtail_rag --cov-report=term-missing

# Individual modules
pytest wagtail_rag/tests/test_rag_command.py    # unified rag command
pytest wagtail_rag/tests/test_providers.py      # embedding & LLM factories
pytest wagtail_rag/tests/test_extraction.py     # chunking & field extraction
pytest wagtail_rag/tests/test_index_builder.py  # pgvector, batch upsert helpers
pytest wagtail_rag/tests/test_generation.py     # LLM generation
pytest wagtail_rag/tests/test_api_views.py      # REST API + CSRF
pytest wagtail_rag/tests/test_search.py         # hybrid search
```

CI runs on Python 3.11 and 3.12 against Django 4.2 and 5.2.

---

## Troubleshooting

**"Collection expecting embedding with dimension of X, got Y"** — changed embedding model without resetting the index.
```bash
make index-rebuild
```

**"The model X does not exist"** — model name doesn't match the provider. Set the correct model in `WAGTAIL_RAG`:
```python
WAGTAIL_RAG = {
    "llm": {"provider": "openai", "model": "gpt-4o"},
    ...
}
```

**"No pages found to index"** — check pages are published and `WAGTAIL_RAG_MODELS` names are correct (format: `"app.ModelName"`).

**"Connection refused" (Ollama)** — run `ollama serve` first, then `ollama pull mistral`.

**403 on POST** — CSRF token missing. Read the `csrftoken` cookie from a GET response and send it as `X-CSRFToken` header.

**Import errors** — install the required extra: `pip install "wagtail-rag[local]"`.

---

## Requirements

- Python 3.9+
- Django 4.2+
- Wagtail 6.0+
- LangChain (installed automatically)
- At least one provider extra (see Installation)

## License

MIT — see [LICENSE](LICENSE).
