# Wagtail RAG Chatbot

A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS. This Django app provides a complete RAG solution that indexes your Wagtail pages into ChromaDB and provides a chatbot interface using LangChain with support for multiple LLM and embedding providers.

## Features

- **Automatic Page Indexing**: Automatically discovers and indexes all Wagtail Page models
- **Hybrid Search**: Combines ChromaDB vector search with Wagtail's built-in full-text search
- **MultiQuery Retriever**: Uses LangChain's MultiQueryRetriever for better query understanding
- **Fuzzy Matching**: Handles typos and partial matches in search queries
- **Metadata Filtering**: Filter search results by page model, app, or custom metadata
- **Deterministic IDs**: Enables efficient updates and single-page re-indexing
- **Multiple LLM Providers**: Built-in support for **Ollama** (local) and **OpenAI** (hosted)
- **Multiple Embedding Providers**: Support for OpenAI, HuggingFace, Cohere, Google, Sentence Transformers, and custom providers
- **Generic & Reusable**: Works with any Wagtail project without hardcoding model names
- **Configurable via Django Settings**: All options configurable through Django settings (no command-line args required)

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

### 3. Install Dependencies

**Core dependencies:**
```bash
pip install langchain langchain-community langchain-text-splitters chromadb beautifulsoup4
```

**LLM Provider dependencies**:
- **Ollama** (default, local): `pip install langchain-community ollama`
- **OpenAI**: `pip install langchain-openai`

**Embedding Provider dependencies**:
- **HuggingFace** (default, local): `pip install langchain-huggingface sentence-transformers`
- **OpenAI**: `pip install langchain-openai`

### 4. Add URL Configuration (Optional, for API endpoints)

In your main `urls.py` (e.g., `bakerydemo/urls.py`):
```python
# Import wagtail_rag URLs
urlpatterns += [
    path("", include("wagtail_rag.urls")),
]


```

**Important**: Place `wagtail_rag_urls` before `wagtail_urls` so API routes are matched first.

After adding this, the API endpoints will be available at:
- `http://localhost:8000/api/rag/chat/` - Chat endpoint (GET or POST)
- `http://localhost:8000/api/rag/search/` - Search endpoint (GET or POST)

### 5. Add the Global Floating Chatbox to Your Templates

To render the bundled chat/search widget on every page (floating in the bottom-right corner), include this in a base template such as `base.html`:

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


# ChromaDB configuration (common to all setups)
WAGTAIL_RAG_COLLECTION_NAME = "wagtail_rag"
WAGTAIL_RAG_CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
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

## Configuration

### Basic Settings

```python
# ChromaDB collection name
WAGTAIL_RAG_COLLECTION_NAME = 'wagtail_rag'

# ChromaDB persistence directory
WAGTAIL_RAG_CHROMA_PATH = os.path.join(BASE_DIR, 'chroma_db')

# Number of documents to retrieve (default: 8)
WAGTAIL_RAG_RETRIEVE_K = 8

# Enable/disable MultiQueryRetriever (default: True)
WAGTAIL_RAG_USE_MULTI_QUERY = True

# Enable/disable hybrid search (default: True)
WAGTAIL_RAG_USE_HYBRID_SEARCH = True
```

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
# Reset (clear) the existing Chroma collection, then rebuild the index
python manage.py build_rag_index --reset

# Only reset/clear the collection without indexing
python manage.py build_rag_index --reset-only

# Re-index a single page by ID (useful after editing one page)
python manage.py build_rag_index --page-id 123

# Limit to specific models
python manage.py build_rag_index --models blog.BlogPage breads.BreadPage

# Exclude specific models
python manage.py build_rag_index --exclude-models blog.DraftPage

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

# Search without generating response (embedding search only, no LLM)
results = chatbot.search_with_embeddings("multigrain bread", k=10)
for result in results:
    print(f"Title: {result['metadata']['title']}")
    print(f"Score: {result['score']}")
    print(f"Content: {result['content'][:200]}...")
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

**Response:**
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

#### Search API (`/api/rag/search/`)

The search API performs semantic search without generating an AI response. Useful for finding relevant content.

**GET Request:**
```bash
# Basic search
curl "http://localhost:8000/api/rag/search/?q=sourdough bread"

# With number of results
curl "http://localhost:8000/api/rag/search/?q=sourdough bread&k=5"

# With metadata filter
curl "http://localhost:8000/api/rag/search/?q=bread&filter=%7B%22model%22%3A%22BreadPage%22%7D"
```

**POST Request:**
```bash
curl -X POST http://localhost:8000/api/rag/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sourdough bread",
    "k": 5,
    "filter": {"model": "BreadPage"}
  }'
```

**Response:**
```json
{
  "query": "sourdough bread",
  "results": [
    {
      "content": "...",
      "metadata": {
        "title": "Sourdough Bread Recipe",
        "url": "/breads/sourdough/",
        "model": "BreadPage"
      },
      "score": 0.8234
    }
  ],
  "count": 5
}
```

## How It Works

1. **Indexing**: The `build_rag_index` command:
   - Discovers all Wagtail Page models dynamically
   - Extracts text from pages (title, body, StreamField, RichTextField, etc.)
   - Chunks the text using RecursiveCharacterTextSplitter
   - Generates embeddings using the configured embedding provider
   - Stores chunks in ChromaDB with deterministic IDs

2. **Querying**: The chatbot:
   - Uses MultiQueryRetriever to generate query variations
   - Searches ChromaDB for similar chunks
   - Optionally searches Wagtail's full-text index (hybrid search)
   - Combines and deduplicates results
   - Boosts documents with matching titles (handles typos)
   - Passes context to LLM for answer generation

## Architecture

- **Vector Store**: ChromaDB for storing embeddings
- **Embeddings**: Multiple providers supported (HuggingFace, OpenAI, Cohere, Google, Sentence Transformers, Custom)
- **LLM**: Multiple providers supported (Ollama, OpenAI; extensible to others)
- **Framework**: LangChain for orchestration
- **Search**: Hybrid search (ChromaDB + Wagtail full-text)

## Troubleshooting

### "Collection expecting embedding with dimension of X, got Y"

This error occurs when you change embedding providers/models without resetting the index. Different embedding models produce vectors of different dimensions.

**Solution:**
```bash
python manage.py build_rag_index --reset
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

### "Could not import LangChain"

- Make sure you installed the dependencies from `wagtail_rag/requirements.txt`, e.g.:
  - `pip install -r wagtail_rag/requirements.txt`

## Requirements

- Python 3.8+
- Django 3.2+
- Wagtail 4.0+
- ChromaDB
- LangChain

**LLM Provider Requirements**:
- **Ollama**: `pip install langchain-community ollama` (for local LLM)
- **OpenAI**: `pip install langchain-openai`

**Embedding Provider Requirements** (install based on your choice):
- **HuggingFace**: `pip install langchain-huggingface sentence-transformers`
- **OpenAI**: `pip install langchain-openai`
- **Cohere**: `pip install langchain-community cohere`
- **Google**: `pip install langchain-google-genai`
- **Sentence Transformers**: `pip install sentence-transformers`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
