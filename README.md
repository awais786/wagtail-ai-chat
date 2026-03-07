# Wagtail RAG Chatbot

A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS. Drop `wagtail_rag` into any Wagtail site to get a fully working chatbot backed by your page content — no model-specific configuration required.

## Features

- **Universal**: works with any Wagtail page model; auto-discovers content fields
- **Per-field chunking**: each field (body, introduction, …) is chunked independently with section metadata, so the LLM always knows where a chunk came from
- **Hybrid retrieval**: vector similarity search + optional Wagtail full-text search
- **Multiple vector stores**: FAISS (default), ChromaDB, pgvector (PostgreSQL)
- **Multiple embedding providers**: HuggingFace, Sentence Transformers, OpenAI, Ollama
- **Multiple LLM providers**: Ollama (local), OpenAI, Anthropic
- **Chat history**: server-side with LLM summarisation of older turns
- **Unified CLI**: one management command covers indexing, chat, and pipeline smoke-testing

## Installation

### 1. Install the package

```bash
# From GitHub
pip install git+https://github.com/awais786/wagtail-ai-chat.git

# Local checkout
pip install -e .
```

### 2. Add to INSTALLED_APPS

```python
INSTALLED_APPS = [
    # ...
    "wagtail_rag",
]
```

### 3. Install provider extras

```bash
# Local (recommended): Sentence Transformers embeddings + Ollama LLM + FAISS
pip install "wagtail-rag[local]"

# OpenAI for both embeddings and LLM
pip install "wagtail-rag[openai]"

# Anthropic LLM only
pip install "wagtail-rag[anthropic]"

# pgvector backend
pip install "wagtail-rag[pgvector]"

# Everything
pip install "wagtail-rag[all]"
```

| Extra | Installs |
|---|---|
| `faiss` | faiss-cpu |
| `chroma` | chromadb |
| `pgvector` | psycopg2-binary, sqlalchemy |
| `huggingface` | langchain-huggingface, sentence-transformers |
| `sentence-transformers` | sentence-transformers only |
| `openai` | langchain-openai |
| `ollama` | ollama |
| `anthropic` | langchain-anthropic |
| `local` | faiss + sentence-transformers + ollama |
| `all` | every provider |

### 4. Add URL configuration (optional — for the API endpoint)

```python
# urls.py
urlpatterns = [
    path("", include("wagtail_rag.urls")),  # before wagtail_urls
    path("", include(wagtail_urls)),
]
```

This exposes the chat API at `GET/POST /api/rag/chat/`.

### 5. Add the floating chatbox to your base template (optional)

```django
<div id="rag-chatbox-wrapper"
     style="position: fixed; bottom: 1rem; right: 1rem; z-index: 9999;">
    {% include "wagtail_rag/chatbox.html" %}
</div>
```

## Quick Start

### Configure settings

```python
# settings.py

# --- Local setup (Ollama + Sentence Transformers + FAISS) ---
WAGTAIL_RAG_EMBEDDING_PROVIDER = "sentence-transformers"
WAGTAIL_RAG_EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
WAGTAIL_RAG_LLM_PROVIDER       = "ollama"
WAGTAIL_RAG_MODEL_NAME         = "mistral"
WAGTAIL_RAG_VECTOR_STORE_BACKEND = "faiss"
WAGTAIL_RAG_CHROMA_PATH        = os.path.join(BASE_DIR, "faiss_index")

# --- OpenAI setup ---
WAGTAIL_RAG_EMBEDDING_PROVIDER = "openai"
WAGTAIL_RAG_EMBEDDING_MODEL    = "text-embedding-3-small"
WAGTAIL_RAG_LLM_PROVIDER       = "openai"
WAGTAIL_RAG_MODEL_NAME         = "gpt-4o"
OPENAI_API_KEY                 = "sk-..."

# --- pgvector backend ---
WAGTAIL_RAG_VECTOR_STORE_BACKEND = "pgvector"
# Derived automatically from DATABASES['default'] if PostgreSQL,
# or set explicitly:
WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING = "postgresql+psycopg2://user:pass@localhost:5432/mydb"
```

### Build the index

```bash
python manage.py rag index
```

### Chat

```bash
python manage.py rag chat
```

### Smoke-test the pipeline

```bash
python manage.py rag test
```

## Management Command

All RAG operations go through a single command: `manage.py rag <subcommand>`.

### `rag index` — build / reset the vector store

```bash
python manage.py rag index                  # full index build
python manage.py rag index --reset-only     # wipe collection only
python manage.py rag index --page-id 42     # re-index one page
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

Interactive session commands:
- `exit` / `quit` — stop
- `clear` — start a new session
- `sources on` / `sources off` — toggle source display
- `Ctrl+C` — exit

### `rag test` — smoke-test the full pipeline

```bash
python manage.py rag test                                    # built-in questions, full RAG
python manage.py rag test --search-only                      # retrieval only (faster)
python manage.py rag test --questions "Q1" "Q2" "Q3"         # custom questions
python manage.py rag test --filter '{"model":"BlogPage"}'    # scoped to a model
```

Pass criteria:
- **Full RAG**: non-empty answer **and** at least 1 source retrieved → PASS
- **Search-only**: at least 1 source retrieved → PASS

Override the default questions via settings:
```python
WAGTAIL_RAG_TEST_QUESTIONS = [
    "What breads do you sell?",
    "Where are you located?",
]
```

## Makefile shortcuts

```bash
make index           # python manage.py rag index
make index-reset     # python manage.py rag index --reset-only
make index-rebuild   # reset then index
make chat            # python manage.py rag chat
make test-rag        # python manage.py rag test
make test-rag-search # python manage.py rag test --search-only
make test            # pytest unit tests
make test-cov        # pytest with coverage
make lint            # black + flake8
make format          # black auto-format
make clean           # remove __pycache__, *.egg-info, coverage files
```

## Configuration Reference

### Core

```python
WAGTAIL_RAG_VECTOR_STORE_BACKEND = "faiss"        # "faiss" | "chroma" | "pgvector"
WAGTAIL_RAG_COLLECTION_NAME      = "wagtail_rag"
WAGTAIL_RAG_CHROMA_PATH          = os.path.join(BASE_DIR, "faiss_index")

WAGTAIL_RAG_LLM_PROVIDER         = "ollama"       # "ollama" | "openai" | "anthropic"
WAGTAIL_RAG_MODEL_NAME           = "mistral"

WAGTAIL_RAG_EMBEDDING_PROVIDER   = "sentence-transformers"  # "sentence-transformers" | "huggingface" | "openai" | "ollama"
WAGTAIL_RAG_EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
```

### pgvector

```python
WAGTAIL_RAG_VECTOR_STORE_BACKEND           = "pgvector"
# Auto-derived from DATABASES['default'] when using PostgreSQL.
# Set explicitly to override:
WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING     = "postgresql+psycopg2://user:pass@host:5432/db"
```

### Retrieval

```python
WAGTAIL_RAG_RETRIEVE_K               = 8      # documents returned per query
WAGTAIL_RAG_USE_HYBRID_SEARCH        = True   # vector + Wagtail full-text
WAGTAIL_RAG_USE_LLM_QUERY_EXPANSION  = False  # MultiQueryRetriever
```

### Indexing

```python
WAGTAIL_RAG_MODELS = [
    "blog.BlogPage",
    "breads.BreadPage:*",   # :* = use all content fields
]
WAGTAIL_RAG_EXCLUDE_MODELS = ["wagtailcore.Page"]
WAGTAIL_RAG_CHUNK_SIZE    = 1000
WAGTAIL_RAG_CHUNK_OVERLAP = 200
WAGTAIL_RAG_SKIP_IF_INDEXED = True   # skip unchanged pages
WAGTAIL_RAG_PRUNE_DELETED   = True   # remove stale chunks
```

### API and security

```python
WAGTAIL_RAG_MAX_REQUEST_BODY_SIZE    = 1024 * 1024  # 1 MB
WAGTAIL_RAG_MAX_QUESTION_LENGTH      = 0            # 0 = no limit
WAGTAIL_RAG_MAX_CONTEXT_CHARS        = 0            # 0 = no limit
WAGTAIL_RAG_ENABLE_CHAT_HISTORY      = True
WAGTAIL_RAG_CHAT_HISTORY_RECENT_MESSAGES = 6
```

### Provider-specific model overrides

```python
WAGTAIL_RAG_OPENAI_MODEL_NAME         = "gpt-4o"
WAGTAIL_RAG_ANTHROPIC_MODEL_NAME      = "claude-3-5-sonnet-20241022"
WAGTAIL_RAG_OLLAMA_MODEL_NAME         = "mistral"

WAGTAIL_RAG_OPENAI_EMBEDDING_MODEL    = "text-embedding-3-small"
WAGTAIL_RAG_OLLAMA_EMBEDDING_MODEL    = "nomic-embed-text"
WAGTAIL_RAG_HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Custom prompt templates

```python
WAGTAIL_RAG_PROMPT_TEMPLATE = """You are a helpful assistant for a Wagtail CMS website.
Answer ONLY using the context below. Cite the source page when possible.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:"""

WAGTAIL_RAG_SYSTEM_PROMPT = "You are a helpful assistant. Answer using only the provided context."
```

## API Endpoints

### `GET /api/rag/chat/`

```bash
curl "http://localhost:8000/api/rag/chat/?q=What+types+of+bread+do+you+have?"
```

### `POST /api/rag/chat/`

```bash
curl -X POST http://localhost:8000/api/rag/chat/ \
  -H "Content-Type: application/json" \
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
  ]
}
```

Error codes: `400` bad request · `413` body too large · `500` server error.

## Python API

```python
from wagtail_rag.rag_chatbot import get_chatbot

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

## How It Works

1. **Indexing** (`rag index`): discovers live Wagtail pages → extracts each field independently → chunks with paragraph preservation → prepends `Page: / Section:` header to every chunk → upserts into vector store with deterministic IDs (`{page_id}_{field}_{chunk_index}`). Stale chunks are removed before re-indexing.

2. **Querying** (`rag chat` / API): embeds the question → vector similarity search → optional Wagtail full-text search → deduplicate & title-boost → pass top-k chunks as context to LLM → return answer + sources.

## Testing

```bash
# Unit tests
pytest wagtail_rag/tests/ -v

# With coverage
pytest wagtail_rag/tests/ --cov=wagtail_rag --cov-report=term-missing

# Individual modules
pytest wagtail_rag/tests/test_rag_command.py   # unified rag command
pytest wagtail_rag/tests/test_providers.py     # embedding & LLM factories
pytest wagtail_rag/tests/test_extraction.py    # chunking & field extraction
pytest wagtail_rag/tests/test_index_builder.py # pgvector, batch upsert helpers
pytest wagtail_rag/tests/test_generation.py    # LLM generation
pytest wagtail_rag/tests/test_api_views.py     # REST API
pytest wagtail_rag/tests/test_search.py        # hybrid search
```

CI runs on Python 3.11 and 3.12 against Django 4.2 and 5.2.

## Troubleshooting

**"Collection expecting embedding with dimension of X, got Y"** — you changed embedding model without resetting the index.
```bash
make index-rebuild
```

**"The model X does not exist"** — model name doesn't match the provider. Use provider-specific settings:
```python
WAGTAIL_RAG_OPENAI_MODEL_NAME  = "gpt-4o"
WAGTAIL_RAG_OLLAMA_MODEL_NAME  = "mistral"
```

**"No pages found to index"** — check pages are published and `WAGTAIL_RAG_MODELS` names are correct.

**"Connection refused" (Ollama)** — run `ollama serve` first.

**Import errors** — install the relevant extra: `pip install "wagtail-rag[local]"`.

## Requirements

- Python 3.9+
- Django 4.2+
- Wagtail 5.0+
- LangChain (installed automatically)
- At least one provider extra

## License

MIT
