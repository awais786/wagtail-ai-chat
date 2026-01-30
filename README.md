# Wagtail RAG Chatbot

> **Security Notice (Jan 2026)**: This version includes critical security updates for LangChain dependencies. Please upgrade to the latest version or ensure you have `langchain-community>=0.3.27` and `langchain-text-splitters>=0.3.9` installed. See [SECURITY.md](SECURITY.md) for details.

A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS. This Django app provides a complete RAG solution that indexes your Wagtail pages into FAISS or ChromaDB and provides a chatbot interface using LangChain with support for multiple LLM and embedding providers.

## Features

- **Automatic Page Indexing**: Automatically discovers and indexes all Wagtail Page models
- **Intelligent Chunking**: Creates separate documents for title, intro, and body content with title context
- **Hybrid Retrieval**: Combines vector similarity search with Wagtail's full-text search (optional)
- **MultiQuery Retriever**: Uses LangChain's MultiQueryRetriever for query expansion (optional)
- **Title-Based Boosting**: Prioritizes documents with matching titles for better relevance
- **Metadata Filtering**: Filter results by page model, app, or custom metadata
- **Deterministic IDs**: Enables efficient updates and single-page re-indexing
- **Multiple LLM Providers**: Built-in support for **Ollama** (local), **OpenAI**, and **Anthropic** (hosted)
- **Multiple Embedding Providers**: Support for OpenAI, HuggingFace, and Sentence Transformers
- **Multiple Vector Stores**: Support for FAISS (default) and ChromaDB
- **Generic & Reusable**: Works with any Wagtail project without hardcoding model names
- **Configurable via Django Settings**: All options configurable through Django settings

## Installation

### 1. Install the Package

**From GitHub (recommended):**
```bash
pip install git+https://github.com/awais786/wagtail-ai-chat.git
```

**From source (local checkout):**
```bash
cd wagtail_rag
pip install -e .
```

### 2. Add to INSTALLED_APPS

In your Django `settings.py`:
```python
INSTALLED_APPS = [
    # ... other apps
    'wagtail_rag',
]
```

### 3. Install Provider Dependencies

**Install only the providers you need:**

**For local setup (HuggingFace embeddings + Ollama LLM):**
```bash
pip install wagtail-rag[local]
# Or: pip install wagtail-rag[huggingface,ollama]
# Or separately: pip install langchain-huggingface sentence-transformers ollama
```

**For HuggingFace embeddings only:**
```bash
pip install wagtail-rag[huggingface]
# Or: pip install langchain-huggingface sentence-transformers
```

**For OpenAI (embeddings and/or LLM):**
```bash
pip install wagtail-rag[openai]
# Or: pip install langchain-openai
```

**For Anthropic Claude LLM:**
```bash
pip install wagtail-rag[anthropic]
# Or: pip install langchain-anthropic
```

**Install all providers (optional):**
```bash
pip install wagtail-rag[all]
```

**Note:** You can combine multiple providers in one command, e.g., `pip install wagtail-rag[huggingface,openai]`

**Note:** Core dependencies (langchain, etc.) are automatically installed with the package. You need to install at least one vector store backend (FAISS or ChromaDB) and provider-specific dependencies.

### 4. Add URL Configuration (Optional, for API endpoints)

In your main `urls.py` (e.g., `bakerydemo/urls.py`):
```python
# Import wagtail_rag URLs
urlpatterns += [
    path("", include("wagtail_rag.urls")),
]
```

**Important**: Place `wagtail_rag_urls` before `wagtail_urls` so API routes are matched first.

After adding this, the API endpoint will be available at:
- `http://localhost:8000/api/rag/chat/` - Chat endpoint (GET or POST)

### 5. Add the Global Floating Chatbox to Your Templates

To render the bundled chat widget on every page (floating in the bottom-right corner), include this in a base template such as `base.html`:

```django
{# Global RAG chatbox (from wagtail_rag) shown on all pages, floating bottom-right #}
<div id="rag-chatbox-wrapper"
     style="position: fixed; bottom: 1rem; right: 1rem; z-index: 9999;">
    {% include "wagtail_rag/chatbox.html" %}
</div>
```

## Quick Start

### 1. Configure Settings

Add these settings to your Django `settings.py`:

```python
# Example 1: Local LLM (Ollama) + Local Embeddings (HuggingFace)

# Embedding model configuration
WAGTAIL_RAG_EMBEDDING_PROVIDER = "huggingface"   # or "hf"
WAGTAIL_RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM configuration
WAGTAIL_RAG_LLM_PROVIDER = "ollama"
WAGTAIL_RAG_MODEL_NAME = "mistral"


# Example 2: Hosted OpenAI for both LLM and embeddings

# Embedding configuration
WAGTAIL_RAG_EMBEDDING_PROVIDER = "openai"
WAGTAIL_RAG_EMBEDDING_MODEL = "text-embedding-ada-002"

# LLM configuration
WAGTAIL_RAG_LLM_PROVIDER = "openai"
WAGTAIL_RAG_MODEL_NAME = "gpt-4"
OPENAI_API_KEY = "sk-..."  # or configure via environment variable


# Vector Store Configuration (choose FAISS or ChromaDB)
WAGTAIL_RAG_VECTOR_STORE_BACKEND = "faiss"  # or "chroma"
WAGTAIL_RAG_COLLECTION_NAME = "wagtail_rag"
WAGTAIL_RAG_CHROMA_PATH = os.path.join(BASE_DIR, "faiss_index")  # Path for vector store (works for both FAISS and ChromaDB)
```

### 2. Build the Index

```bash
python manage.py build_rag_index
```

### 3. Use the Chatbot

**Via API (GET - Browser-friendly):**
```bash
# Simple GET request (works in browser)
curl "http://localhost:8000/api/rag/chat/?q=What content is available?"
```

**Via API (POST - JSON):**
```bash
curl -X POST http://localhost:8000/api/rag/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What content is available?"}'
```

**Note**: LLM provider and model are automatically read from Django settings (`WAGTAIL_RAG_LLM_PROVIDER` and `WAGTAIL_RAG_MODEL_NAME`). You don't need to pass them in the request.

**Via Python:**
```python
from wagtail_rag.rag_chatbot import get_chatbot

chatbot = get_chatbot()
result = chatbot.query("What content is available?")
print(result['answer'])
```

## Configuration

### Basic Settings

```python
# Vector store backend (faiss or chroma)
WAGTAIL_RAG_VECTOR_STORE_BACKEND = 'faiss'

# Collection/index name
WAGTAIL_RAG_COLLECTION_NAME = 'wagtail_rag'

# Vector store directory (works for both FAISS and ChromaDB)
WAGTAIL_RAG_CHROMA_PATH = os.path.join(BASE_DIR, 'faiss_index')

# Number of documents to retrieve (default: 8)
WAGTAIL_RAG_RETRIEVE_K = 8

# Enable/disable LLM query expansion via MultiQueryRetriever
WAGTAIL_RAG_USE_LLM_QUERY_EXPANSION = True

# Enable/disable hybrid search (default: True)
WAGTAIL_RAG_USE_HYBRID_SEARCH = True
```

### API and Security

```python
# Max POST body size in bytes for the chat API (default: 1MB). Helps prevent DoS from huge payloads.
WAGTAIL_RAG_MAX_REQUEST_BODY_SIZE = 1024 * 1024

# Max question length (default: 1000 characters). Helps prevent abuse.
WAGTAIL_RAG_MAX_QUESTION_LENGTH = 1000
```

**Security Considerations**:

The chat endpoint (`/api/rag/chat/`) is CSRF-exempt so it can be called by external clients, scripts, or non-Django frontends. If the endpoint is public, you **must** protect it:

1. **Authentication**: Add authentication middleware or decorator
2. **Rate Limiting**: Use django-ratelimit or similar
3. **IP Allowlisting**: Restrict to known IP addresses
4. **Network Security**: Use firewall rules, reverse proxy authentication

Example with django-ratelimit:
```python
from django_ratelimit.decorators import ratelimit

@ratelimit(key='ip', rate='10/m', method='POST')
def rag_chat_api(request):
    # ... existing code
```

**See SECURITY.md for complete security guidelines.**

### Model Indexing Configuration

```python
# Models to index (None = index all Page models).
# You can use the shorthand "app.Model:*" here to say:
#   "index this model and treat all its content fields as important".
WAGTAIL_RAG_MODELS = [
    "blog.BlogPage",
    "breads.BreadPage:*",       # index BreadPage, all fields
    "products.ProductPage",     # index ProductPage, standard field extraction
]

# Models to exclude from indexing (always excluded, even if in WAGTAIL_RAG_MODELS)
WAGTAIL_RAG_EXCLUDE_MODELS = [
    "wagtailcore.Page",
    "wagtailcore.Site",
    "wagtailcore.Redirect",
]

# Text chunking configuration
WAGTAIL_RAG_CHUNK_SIZE = 1000  # Size of each text chunk
WAGTAIL_RAG_CHUNK_OVERLAP = 200  # Overlap between chunks
```

### Custom Prompt Template (Optional)

```python
WAGTAIL_RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following pieces of context from the website to answer the question accurately.

Context: {context}

Question: {question}

Answer: """
```

## Usage

### Building the RAG Index

Index your Wagtail pages (uses your Django settings):

```bash
python manage.py build_rag_index
```

Common variations:

```bash
# Only reset/clear the collection without indexing (no documents added)
python manage.py build_rag_index --reset-only

# Re-index a single page by ID (useful after editing one page)
python manage.py build_rag_index --page-id 123
```

Model selection is controlled by Django settings (`WAGTAIL_RAG_MODELS` and `WAGTAIL_RAG_EXCLUDE_MODELS`), not by CLI flags. To rebuild from scratch, clear first then index:

```bash
python manage.py build_rag_index --reset-only
python manage.py build_rag_index
```

### Using the Chatbot in Python

```python
from wagtail_rag.rag_chatbot import get_chatbot

# Get chatbot instance (uses settings defaults)
chatbot = get_chatbot()

# Query the chatbot (this calls the LLM under the hood)
result = chatbot.query("What types of bread do you have?")
print(result['answer'])
print(result['sources'])

# Filter by model
chatbot_filtered = get_chatbot(metadata_filter={'model': 'BreadPage'})
result = chatbot_filtered.query("Tell me about multigrain bread")

# Use different provider/model
chatbot_openai = get_chatbot(
    llm_provider='openai',
    model_name='gpt-4',
    llm_kwargs={'temperature': 0.7}
)
result = chatbot_openai.query("What content do you have?")
```

### Using the API Endpoints

#### Chat API (`/api/rag/chat/`)

The chat API supports both GET and POST methods. LLM provider and model are automatically read from Django settings.

**GET Request (Browser-friendly):**
```bash
# Simple query
curl "http://localhost:8000/api/rag/chat/?q=What types of bread do you have?"

# With metadata filter (JSON string)
curl "http://localhost:8000/api/rag/chat/?q=Tell me about multigrain bread&filter=%7B%22model%22%3A%22BreadPage%22%7D"
# Or in browser: http://localhost:8000/api/rag/chat/?q=bread&filter={"model":"BreadPage"}
```

**POST Request (JSON):**
```bash
# Basic query
curl -X POST http://localhost:8000/api/rag/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What types of bread do you have?"
  }'

# With LLM parameters (temperature, etc.)
curl -X POST http://localhost:8000/api/rag/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What types of bread do you have?",
    "llm_kwargs": {"temperature": 0.7}
  }'

# With metadata filter
curl -X POST http://localhost:8000/api/rag/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Tell me about multigrain bread",
    "filter": {"model": "BreadPage"}
  }'
```

**Response (200):**
```json
{
  "answer": "We have several types of bread including...",
  "sources": [
    {
      "content": "...",
      "metadata": {
        "title": "Multigrain Bread",
        "url": "/breads/multigrain/",
        "model": "BreadPage"
      }
    }
  ]
}
```

**Error responses:**
- **400** – Missing or invalid `question`, empty POST body, or invalid JSON (e.g. `{"error": "Invalid JSON: ..."}`).
- **413** – Request body larger than `WAGTAIL_RAG_MAX_REQUEST_BODY_SIZE` (default 1MB).
- **500** – Server error; message is generic; details are logged on the server.

## How It Works

1. **Indexing**: When you run `python manage.py build_rag_index`:
   - The command delegates to the shared index builder (`wagtail_rag.content_extraction.index_builder.build_rag_index`).
   - The builder discovers Wagtail Page models from settings (`WAGTAIL_RAG_MODELS` or all page types), gets live pages, and for each page calls `wagtail_page_to_documents()` to turn it into LangChain Document objects (title, intro, and chunked body with title context).
   - Each document gets metadata (page_id, page_type, slug, url, etc.); the builder adds model-level metadata (source, model, app) and upserts chunks into the vector store (FAISS or ChromaDB) with deterministic IDs. Old chunks for a page are removed before re-indexing so updates stay consistent.

2. **Querying**: The chatbot:
   - Uses embedding-based similarity search to find relevant document chunks
   - Optionally combines with Wagtail's full-text search (hybrid search)
   - Uses MultiQueryRetriever for query expansion (if enabled)
   - Combines and deduplicates results
   - Boosts documents with matching titles (handles typos)
   - Passes context to LLM for answer generation

## Architecture

- **Vector Store**: FAISS or ChromaDB for storing embeddings
- **Embeddings**: Multiple providers supported (HuggingFace, OpenAI, Sentence Transformers)
- **LLM**: Multiple providers supported (Ollama, OpenAI, Anthropic; extensible to others)
- **Framework**: LangChain for orchestration
- **Retrieval**: Hybrid search (vector similarity + optional Wagtail full-text)
- **Document Processing**: Intelligent chunking with title context for better semantic understanding

## Troubleshooting

### "Collection expecting embedding with dimension of X, got Y"

This error occurs when you change embedding providers or models without resetting the index. Different embedding models produce vectors of different dimensions.

**Solution:** Clear the index, then rebuild:
```bash
python manage.py build_rag_index --reset-only
python manage.py build_rag_index
```

### "The model 'X' does not exist or you do not have access to it"

This error occurs when you specify a model name that doesn't match the provider (e.g., using an Ollama model name with OpenAI provider).

**Solution**: The system will automatically detect this and use the provider's default model. Make sure your settings match:
```python
WAGTAIL_RAG_LLM_PROVIDER = 'openai'
WAGTAIL_RAG_MODEL_NAME = 'gpt-4'  # Not 'mistral' (Ollama model)
```

### "No pages found to index"

- Make sure you have published pages in Wagtail admin
- Check pages are live (not draft)
- Verify your `WAGTAIL_RAG_MODELS` setting includes the correct model names

### "Connection refused" (Ollama)

- Make sure `ollama serve` is running
- Test with: `ollama list`

### "Could not import LangChain" or "HuggingFace embeddings not available"

- Core dependencies are installed automatically with the package
- Install provider-specific dependencies based on your configuration:
  - **Local setup**: `pip install wagtail-rag[local]` (HuggingFace + Ollama)
  - **HuggingFace**: `pip install wagtail-rag[huggingface]`
  - **OpenAI**: `pip install wagtail-rag[openai]`
  - **Ollama**: `pip install wagtail-rag[ollama]`
  - **All providers**: `pip install wagtail-rag[all]`

## Requirements

**Core Requirements** (installed automatically):
- Python 3.8+
- Django 3.2+
- Wagtail 4.0+
- FAISS
- LangChain (core packages)

**Provider Requirements** (install only what you need):

**Embedding Providers:**
- **HuggingFace** (default): `pip install wagtail-rag[huggingface]` or `pip install langchain-huggingface sentence-transformers`
- **OpenAI**: `pip install wagtail-rag[openai]` or `pip install langchain-openai`

**LLM Providers:**
- **Ollama** (local): `pip install wagtail-rag[ollama]` or `pip install ollama`
- **OpenAI**: `pip install wagtail-rag[openai]` or `pip install langchain-openai`
- **Anthropic**: `pip install wagtail-rag[anthropic]` or `pip install langchain-anthropic`

**Common combinations:**
- **Local setup**: `pip install wagtail-rag[local]` (HuggingFace embeddings + Ollama LLM)
- **OpenAI setup**: `pip install wagtail-rag[openai]` (OpenAI embeddings + LLM)

**Install all providers at once:**
```bash
pip install wagtail-rag[all]
```

## Example Configuration

All configuration is done via Django settings. See the "Configuration" section above for all available settings.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
