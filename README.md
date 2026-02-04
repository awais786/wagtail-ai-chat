# Wagtail RAG Chatbot

A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS. This Django app provides a complete RAG solution that indexes your Wagtail pages into FAISS or ChromaDB and provides a chatbot interface using LangChain with support for multiple LLM and embedding providers.

## Features

### Universal Compatibility
- 🔌 **Plug-and-Play**: Works with **any Wagtail site** - no model-specific configuration required
- 🎯 **Smart Field Detection**: Automatically discovers and extracts content from any page model
- 🏗️ **Zero Assumptions**: No hardcoded field names or page type checks - works with custom models out-of-the-box
- ⚙️ **Fully Configurable**: Customize field patterns and extraction behavior via Django settings (optional)

### Content Extraction
- 📝 **Adaptive Chunking**: Small pages stay whole, large pages are intelligently chunked
- 🌳 **Multi-StreamField Support**: Handles multiple StreamFields per page (e.g., `body` + `backstory`)
- 📄 **Rich Content**: Extracts from StreamField, RichTextField, TextField, and all standard Django fields
- 🔗 **Relationships**: Includes ForeignKey and ManyToMany field values in metadata
- 💬 **Natural Language**: Creates narrative-style documents for better LLM comprehension

### RAG & Search
- 🔍 **Hybrid Retrieval**: Combines vector similarity search with Wagtail's full-text search (optional)
- 🎭 **MultiQuery Retriever**: Uses LangChain's MultiQueryRetriever for query expansion (optional)
- 🏆 **Title-Based Boosting**: Prioritizes documents with matching titles for better relevance
- 🏷️ **Metadata Filtering**: Filter results by page model, app, or custom metadata
- 🆔 **Deterministic IDs**: Enables efficient updates and single-page re-indexing

### Provider Flexibility
- 🤖 **Multiple LLM Providers**: Built-in support for **Ollama** (local), **OpenAI**, and **Anthropic** (hosted)
- 🧠 **Multiple Embedding Providers**: Support for OpenAI, HuggingFace, and Sentence Transformers
- 💾 **Multiple Vector Stores**: Support for FAISS (default) and ChromaDB

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
```

The chat endpoint (`/api/rag/chat/`) is CSRF-exempt so it can be called by external clients, scripts, or non-Django frontends. If the endpoint is public, protect it at the network or gateway level (e.g. authentication, rate limiting, or IP allowlisting).

### Model Indexing Configuration

```python
# Models to index (None = index all Page models).
# Use "app.Model:*" to treat all content fields as important for that model:
#   the default (chunked) extractor will use every content-bearing field
#   (StreamField, RichTextField, TextField, long CharField) on the model.
# Without ":*", only the default field list (body, content, backstory, instructions) is used.
WAGTAIL_RAG_MODELS = [
    "blog.BlogPage",
    "breads.BreadPage:*",       # index BreadPage using all content fields
    "products.ProductPage",     # index ProductPage with default field list only
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
# Index build behavior
# Skip re-indexing pages that are already indexed and unchanged (default: True)
WAGTAIL_RAG_SKIP_IF_INDEXED = True
# Remove stale documents for pages that have been deleted (default: True)
WAGTAIL_RAG_PRUNE_DELETED = True
# Use new extractor by default (SmartWagtailExtractor: adaptive chunking for any page)
# Default True. Set False to use the original chunked extractor (page_to_documents) only.
WAGTAIL_RAG_USE_NEW_EXTRACTOR = True
# New extractor: chunk only when content exceeds this many characters (default 2000)
WAGTAIL_RAG_NEW_EXTRACTOR_SIZE_THRESHOLD = 2000
# New extractor: chunk large pages by section (title, intro, body, metadata) when True
WAGTAIL_RAG_NEW_EXTRACTOR_CHUNK_BY_SECTION = True
```

### Advanced Extractor Configuration (Optional)

The new extractor (`SmartWagtailExtractor`) is **fully generic** and works with **any Wagtail site** without model-specific assumptions. It automatically detects field types and extracts content accordingly.

**Default behavior works for most sites.** These settings are only needed for customization:

```python
# Maximum items to extract from ManyToMany fields (default: 10)
WAGTAIL_RAG_MAX_METADATA_ITEMS = 10

# Maximum length for text fields in metadata before truncation (default: 500)
WAGTAIL_RAG_MAX_TEXT_FIELD_LENGTH = 500

# Custom field name patterns for introduction/description fields
# (default: ["introduction", "intro", "description", "summary", "excerpt", "lead", "standfirst"])
WAGTAIL_RAG_INTRO_PATTERNS = ["introduction", "intro", "description", "excerpt"]

# Custom field name patterns for main body/content fields
# (default: ["body", "content", "main_content", "text", "streamfield", "page_body"])
WAGTAIL_RAG_BODY_PATTERNS = ["body", "content", "main_content"]

# Additional system fields to skip during extraction (optional)
# Wagtail system fields are already skipped by default
WAGTAIL_RAG_SKIP_FIELDS = {"custom_internal_field", "temp_data"}
```

**How the extractor works generically:**

1. **Field Discovery**: Automatically finds intro, body, and metadata fields by field type and naming patterns
2. **StreamFields**: Extracts ALL StreamFields (including non-body ones like `backstory` on RecipePage)
3. **RichText**: Extracts plain text from RichTextField objects
4. **Relationships**: Includes ForeignKey and ManyToMany field values
5. **Narrative Structure**: Creates natural-language documents for better LLM understanding
6. **No Model Assumptions**: Works with BlogPage, ProductPage, LocationPage, or any custom model

**Example**: For a custom `EventPage` with fields `event_date`, `venue`, and `program` (StreamField), the extractor will automatically:
- Extract `event_date` and `venue` as metadata
- Extract `program` StreamField as body content
- Create a document like: "Summer Festival. The event date is 2024-07-15. The venue is Central Park. [program content]"

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

### Comparing extractors (new vs chunked)

To compare the new extractor (SmartWagtailExtractor, default) with the original chunked extractor on a single page:

```bash
python manage.py compare_extractors --page-id 123
```

Optional: write full comparison to JSON for diffing:

```bash
python manage.py compare_extractors --page-id 123 --output /tmp/compare.json
```

The new extractor is used by default. Set `WAGTAIL_RAG_USE_NEW_EXTRACTOR = False` in settings to use only the chunked extractor.

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
   - The builder discovers Wagtail Page models from settings (`WAGTAIL_RAG_MODELS` or all page types), gets live pages, and for each page tries the **new extractor** (SmartWagtailExtractor) first by default. It converts pages to LangChain Document objects with adaptive chunking (small pages → one doc, large pages → chunked by section). If the new extractor returns nothing, it falls back to the **chunked extractor** (`wagtail_page_to_documents`: title, intro, chunked body with title context).
   - Each document gets metadata (page_id, page_type, slug, url, section, chunk_index, doc_id, etc.); the builder adds model-level metadata (source, model, app) and upserts chunks into the vector store (FAISS or ChromaDB) with deterministic IDs. Old chunks for a page are removed before re-indexing so updates stay consistent.

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
