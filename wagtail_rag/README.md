# Wagtail RAG Chatbot

A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS. This Django app provides a complete RAG solution that indexes your Wagtail pages into ChromaDB and provides a chatbot interface using LangChain with support for multiple LLM and embedding providers.

## Features

- **Automatic Page Indexing**: Automatically discovers and indexes all Wagtail Page models
- **Hybrid Search**: Combines ChromaDB vector search with Wagtail's built-in full-text search
- **MultiQuery Retriever**: Uses LangChain's MultiQueryRetriever for better query understanding
- **Fuzzy Matching**: Handles typos and partial matches in search queries
- **Metadata Filtering**: Filter search results by page model, app, or custom metadata
- **Deterministic IDs**: Enables efficient updates and single-page re-indexing
- **Multiple LLM Providers**: Support for OpenAI, Anthropic, Ollama, Google, HuggingFace, Cohere, and custom providers
- **Multiple Embedding Providers**: Support for OpenAI, HuggingFace, Cohere, Google, Sentence Transformers, and custom providers
- **Generic & Reusable**: Works with any Wagtail project without hardcoding model names
- **Configurable via Django Settings**: All options configurable through Django settings (no command-line args required)

## Installation

### 1. Install the Package

**From source:**
```bash
cd wagtail_rag
pip install -e .
```

**Or copy the `wagtail_rag` directory to your project**

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

**LLM Provider dependencies** (install based on your choice):
- **Ollama** (default, local): `pip install langchain-community ollama`
- **OpenAI**: `pip install langchain-openai`
- **Anthropic**: `pip install langchain-anthropic`
- **Google**: `pip install langchain-google-genai`
- **HuggingFace**: `pip install langchain-huggingface transformers`
- **Cohere**: `pip install langchain-community cohere`

**Embedding Provider dependencies** (install based on your choice):
- **HuggingFace** (default, local): `pip install langchain-huggingface sentence-transformers`
- **OpenAI**: `pip install langchain-openai`
- **Cohere**: `pip install langchain-community cohere`
- **Google**: `pip install langchain-google-genai`
- **Sentence Transformers**: `pip install sentence-transformers`

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
# LLM Configuration
WAGTAIL_RAG_LLM_PROVIDER = 'openai'  # or 'ollama', 'anthropic', 'google', etc.
WAGTAIL_RAG_MODEL_NAME = 'gpt-4'  # Provider-specific model name
OPENAI_API_KEY = 'sk-...'  # Required for OpenAI (or set as environment variable)

# Embedding Configuration
WAGTAIL_RAG_EMBEDDING_PROVIDER = 'openai'  # or 'huggingface', 'cohere', etc.
WAGTAIL_RAG_EMBEDDING_MODEL = 'text-embedding-ada-002'  # Provider-specific model

# ChromaDB Configuration
WAGTAIL_RAG_COLLECTION_NAME = 'wagtail_rag'
WAGTAIL_RAG_CHROMA_PATH = os.path.join(BASE_DIR, 'chroma_db')
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
# Models to index (None = index all Page models)
WAGTAIL_RAG_MODELS = [
    'blog.BlogPage',
    'breads.BreadPage',
    'products.ProductPage',
]

# Models to exclude from indexing
WAGTAIL_RAG_EXCLUDE_MODELS = [
    'wagtailcore.Page',
    'wagtailcore.Site',
    'wagtailcore.Redirect',
]

# Model-specific important fields
WAGTAIL_RAG_MODEL_FIELDS = [
    'breads.BreadPage:bread_type,origin',
    'blog.BlogPage:author,date_published',
    'products.ProductPage:price,sku',
]

# Text chunking configuration
WAGTAIL_RAG_CHUNK_SIZE = 1000  # Size of each text chunk
WAGTAIL_RAG_CHUNK_OVERLAP = 200  # Overlap between chunks
```

### Custom Prompt Template

```python
WAGTAIL_RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following pieces of context from the website to answer the question accurately.

Context: {context}

Question: {question}

Answer: """
```

## LLM Provider Configuration

### Quick Reference

| Provider | Setting | Default Model | Install Command | API Key Required |
|----------|---------|--------------|-----------------|------------------|
| **Ollama** | `'ollama'` | `'mistral'` | `pip install langchain-community ollama` | No |
| **OpenAI** | `'openai'` | `'gpt-4'` | `pip install langchain-openai` | Yes |
| **Anthropic** | `'anthropic'` | `'claude-3-sonnet-20240229'` | `pip install langchain-anthropic` | Yes |
| **Google** | `'google'` | `'gemini-pro'` | `pip install langchain-google-genai` | Yes |
| **HuggingFace** | `'huggingface'` | None (must specify) | `pip install langchain-huggingface transformers` | Optional |
| **Cohere** | `'cohere'` | `'command'` | `pip install langchain-community cohere` | Yes |

### Ollama (Default - Local LLM)

**Best for**: Local development, privacy-sensitive applications, no API costs

```python
WAGTAIL_RAG_LLM_PROVIDER = 'ollama'
WAGTAIL_RAG_MODEL_NAME = 'mistral'  # or 'llama2', 'phi', 'gemma'
```

**Setup:**
```bash
# Install Ollama
brew install ollama  # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull mistral
```

**Pros**: Free, local, private, no API costs  
**Cons**: Requires local resources, slower than cloud APIs

### OpenAI

**Best for**: Production applications, best quality, fast responses

```python
WAGTAIL_RAG_LLM_PROVIDER = 'openai'
WAGTAIL_RAG_MODEL_NAME = 'gpt-4'  # or 'gpt-3.5-turbo', 'gpt-4-turbo-preview'
OPENAI_API_KEY = 'sk-...'  # or set environment variable
```

**Available Models:**
- `gpt-4` - Most capable (default)
- `gpt-4-turbo-preview` - Latest GPT-4
- `gpt-3.5-turbo` - Faster, cheaper

**Pros**: Best quality, fast, reliable  
**Cons**: API costs, requires internet

### Anthropic (Claude)

**Best for**: Long context, high-quality responses

```python
WAGTAIL_RAG_LLM_PROVIDER = 'anthropic'
WAGTAIL_RAG_MODEL_NAME = 'claude-3-sonnet-20240229'
ANTHROPIC_API_KEY = 'sk-ant-...'  # or set environment variable
```

**Available Models:**
- `claude-3-opus-20240229` - Most capable
- `claude-3-sonnet-20240229` - Balanced (recommended)
- `claude-3-haiku-20240307` - Fastest, cheapest

**Pros**: Excellent quality, long context windows  
**Cons**: API costs, requires internet

### Google (Gemini)

**Best for**: Cost-effective, good quality

```python
WAGTAIL_RAG_LLM_PROVIDER = 'google'  # or 'gemini'
WAGTAIL_RAG_MODEL_NAME = 'gemini-pro'
GOOGLE_API_KEY = 'your-api-key'  # or set environment variable
```

**Available Models:**
- `gemini-pro` - Text generation
- `gemini-pro-vision` - Multimodal

**Pros**: Good quality, competitive pricing  
**Cons**: API costs, requires internet

### HuggingFace

**Best for**: Open-source models, custom deployments

```python
WAGTAIL_RAG_LLM_PROVIDER = 'huggingface'  # or 'hf'
WAGTAIL_RAG_MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
HUGGINGFACE_API_KEY = 'hf_...'  # Optional, for private models
```

**Available Models**: Any model from HuggingFace Hub

**Pros**: Many open-source options, flexible  
**Cons**: Varies by model, some require local GPU

### Cohere

**Best for**: Fast, reliable API

```python
WAGTAIL_RAG_LLM_PROVIDER = 'cohere'
WAGTAIL_RAG_MODEL_NAME = 'command'  # or 'command-light', 'command-nightly'
COHERE_API_KEY = 'your-api-key'  # or set environment variable
```

**Available Models:**
- `command` - Standard
- `command-light` - Faster, cheaper
- `command-nightly` - Latest features

**Pros**: Fast, reliable  
**Cons**: API costs, requires internet

### Custom LLM

**Best for**: Custom deployments, proprietary models

```python
WAGTAIL_RAG_LLM_PROVIDER = 'custom'

def create_custom_llm(model_name, **kwargs):
    from langchain.llms.base import LLM
    # Return your custom LLM implementation
    return YourCustomLLM(model=model_name, **kwargs)

WAGTAIL_RAG_CUSTOM_LLM_FACTORY = create_custom_llm
```

**Requirements**: Your custom LLM must be LangChain-compatible (implement `BaseLLM` or `BaseChatModel`)

## Embedding Provider Configuration

### Quick Reference

| Provider | Setting | Default Model | Install Command | API Key Required |
|----------|---------|---------------|-----------------|------------------|
| **HuggingFace** | `'huggingface'` | `'sentence-transformers/all-MiniLM-L6-v2'` | `pip install langchain-huggingface` | No |
| **OpenAI** | `'openai'` | `'text-embedding-ada-002'` | `pip install langchain-openai` | Yes |
| **Cohere** | `'cohere'` | `'embed-english-v3.0'` | `pip install langchain-community cohere` | Yes |
| **Google** | `'google'` | `'models/embedding-001'` | `pip install langchain-google-genai` | Yes |
| **Sentence Transformers** | `'sentence-transformers'` | None (must specify) | `pip install sentence-transformers` | No |

### HuggingFace (Default - Local, Free)

**Best for**: Most use cases, local development, no API costs

```python
WAGTAIL_RAG_EMBEDDING_PROVIDER = 'huggingface'  # or 'hf'
WAGTAIL_RAG_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
```

**Popular Models:**
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, good quality (default)
- `sentence-transformers/all-mpnet-base-v2` - Better quality, slower
- `sentence-transformers/all-MiniLM-L12-v2` - Balanced
- `intfloat/e5-large-v2` - High quality

**Pros**: Free, local, private, no API costs, works offline  
**Cons**: Requires local resources, first download takes time

### OpenAI Embeddings

**Best for**: Production, best quality, consistent performance

```python
WAGTAIL_RAG_EMBEDDING_PROVIDER = 'openai'
WAGTAIL_RAG_EMBEDDING_MODEL = 'text-embedding-ada-002'  # or 'text-embedding-3-small'
OPENAI_API_KEY = 'sk-...'  # or set environment variable
```

**Available Models:**
- `text-embedding-ada-002` - Standard (1536 dimensions)
- `text-embedding-3-small` - Newer, smaller (1536 dimensions)
- `text-embedding-3-large` - Newer, larger (3072 dimensions)

**Pros**: Best quality, consistent, fast  
**Cons**: API costs (~$0.0001 per 1K tokens), requires internet

### Cohere Embeddings

**Best for**: Multilingual support, good quality

```python
WAGTAIL_RAG_EMBEDDING_PROVIDER = 'cohere'
WAGTAIL_RAG_EMBEDDING_MODEL = 'embed-english-v3.0'  # or 'embed-multilingual-v3.0'
COHERE_API_KEY = 'your-api-key'  # or set environment variable
```

**Available Models:**
- `embed-english-v3.0` - English
- `embed-multilingual-v3.0` - Multilingual
- `embed-english-light-v3.0` - Faster, cheaper

**Pros**: Good multilingual support, reliable  
**Cons**: API costs, requires internet

### Google Embeddings

**Best for**: Google ecosystem integration

```python
WAGTAIL_RAG_EMBEDDING_PROVIDER = 'google'  # or 'gemini'
WAGTAIL_RAG_EMBEDDING_MODEL = 'models/embedding-001'
GOOGLE_API_KEY = 'your-api-key'  # or set environment variable
```

**Pros**: Good quality, competitive pricing  
**Cons**: API costs, requires internet

### Custom Embeddings

```python
WAGTAIL_RAG_EMBEDDING_PROVIDER = 'custom'

def create_custom_embeddings(model_name, **kwargs):
    from langchain.embeddings.base import Embeddings
    # Return your custom Embeddings implementation
    return YourCustomEmbeddings(model=model_name, **kwargs)

WAGTAIL_RAG_CUSTOM_EMBEDDING_FACTORY = create_custom_embeddings
```

## Usage

### Building the RAG Index

Index your Wagtail pages:

```bash
python manage.py build_rag_index
```

**Command Options:**

```bash
# Index specific models
python manage.py build_rag_index --models blog.BlogPage breads.BreadPage

# Exclude specific models
python manage.py build_rag_index --exclude-models wagtailcore.Page

# Emphasize important fields globally
python manage.py build_rag_index --important-fields bread_type origin

# Emphasize fields per model
python manage.py build_rag_index --model-fields breads.BreadPage:bread_type,origin

# Custom chunk size and overlap
python manage.py build_rag_index --chunk-size 1500 --chunk-overlap 300

# Reset collection before indexing
python manage.py build_rag_index --reset

# Re-index a specific page
python manage.py build_rag_index --page-id 123
```

**Note**: Command-line arguments override Django settings. If you configure everything in settings, you can just run `python manage.py build_rag_index` without arguments.

### Using the Chatbot in Python

```python
from wagtail_rag.rag_chatbot import get_chatbot

# Get chatbot instance (uses settings defaults)
chatbot = get_chatbot()

# Query the chatbot
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

# Search without generating response
results = chatbot.search_similar("multigrain bread", k=10)
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
- **LLM**: Multiple providers supported (Ollama, OpenAI, Anthropic, HuggingFace, Google, Cohere, Custom)
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

- Install required packages: `pip install langchain langchain-community langchain-text-splitters`
- Install provider-specific packages based on your configuration

## Requirements

- Python 3.8+
- Django 3.2+
- Wagtail 4.0+
- ChromaDB
- LangChain

**LLM Provider Requirements** (install based on your choice):
- **Ollama**: `pip install langchain-community ollama` (for local LLM)
- **OpenAI**: `pip install langchain-openai`
- **Anthropic**: `pip install langchain-anthropic`
- **Google**: `pip install langchain-google-genai`
- **HuggingFace**: `pip install langchain-huggingface transformers`
- **Cohere**: `pip install langchain-community cohere`

**Embedding Provider Requirements** (install based on your choice):
- **HuggingFace**: `pip install langchain-huggingface sentence-transformers`
- **OpenAI**: `pip install langchain-openai`
- **Cohere**: `pip install langchain-community cohere`
- **Google**: `pip install langchain-google-genai`
- **Sentence Transformers**: `pip install sentence-transformers`

## Cost Considerations

| Provider | Cost per 1M tokens (input) | Notes |
|----------|----------------------------|-------|
| **Ollama** | Free | Local only, requires hardware |
| **OpenAI GPT-4** | ~$30 | Most capable |
| **OpenAI GPT-3.5** | ~$0.50 | Fast, cost-effective |
| **Anthropic Claude** | ~$15 | Excellent quality |
| **Google Gemini** | ~$0.50 | Competitive pricing |
| **Cohere** | ~$15 | Fast responses |
| **HuggingFace** | Varies | Free for public models |

**Embedding Costs:**
- **HuggingFace**: Free (local)
- **OpenAI**: ~$0.0001 per 1K tokens
- **Cohere**: ~$0.0001 per 1K tokens
- **Google**: Varies

*Prices are approximate and subject to change. Check provider websites for current pricing.*

## Recommendations

### Development/Testing
- **LLM**: Use Ollama (free, local)
- **Embeddings**: Use HuggingFace (free, local)

### Production (Budget)
- **LLM**: Use OpenAI GPT-3.5 or Google Gemini
- **Embeddings**: Use HuggingFace or OpenAI `text-embedding-3-small`

### Production (Quality)
- **LLM**: Use OpenAI GPT-4 or Anthropic Claude
- **Embeddings**: Use OpenAI `text-embedding-ada-002` or `text-embedding-3-large`

### Privacy-Sensitive
- **LLM**: Use Ollama or self-hosted HuggingFace models
- **Embeddings**: Use HuggingFace (local)

### Multilingual
- **Embeddings**: Use Cohere `embed-multilingual-v3.0`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
