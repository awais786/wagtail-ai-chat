# Wagtail RAG

A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS. This Django app provides a complete RAG solution that indexes your Wagtail pages into ChromaDB and provides a chatbot interface using LangChain and Ollama.

## Features

- **Automatic Page Indexing**: Automatically discovers and indexes all Wagtail Page models
- **Hybrid Search**: Combines ChromaDB vector search with Wagtail's built-in full-text search
- **MultiQuery Retriever**: Uses LangChain's MultiQueryRetriever for better query understanding
- **Fuzzy Matching**: Handles typos and partial matches in search queries
- **Metadata Filtering**: Filter search results by page model, app, or custom metadata
- **Deterministic IDs**: Enables efficient updates and single-page re-indexing
- **Generic & Reusable**: Works with any Wagtail project without hardcoding model names

## Installation

1. **Install the package** (or copy the `wagtail_rag` directory to your project):

```bash
pip install wagtail-rag
```

Or if installing from source:

```bash
cd wagtail_rag
pip install -e .
```

2. **Add to INSTALLED_APPS** in your Django settings:

```python
INSTALLED_APPS = [
    # ... other apps
    'wagtail_rag',
]
```

3. **Install dependencies**:

```bash
pip install langchain langchain-community langchain-text-splitters langchain-huggingface chromadb beautifulsoup4 ollama
```

4. **Install and start Ollama** (for local LLM):

```bash
# Install Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (e.g., mistral)
ollama pull mistral
```

5. **Add URL configuration** (optional, for API endpoint):

```python
# In your main urls.py
urlpatterns = [
    # ... other patterns
    path('', include('wagtail_rag.urls')),
]
```

## Configuration

Add these settings to your Django `settings.py`:

```python
# ChromaDB collection name
WAGTAIL_RAG_COLLECTION_NAME = 'wagtail_rag'

# ChromaDB persistence directory
WAGTAIL_RAG_CHROMA_PATH = os.path.join(BASE_DIR, 'chroma_db')

# LLM Provider (default: 'ollama')
# Options: 'ollama', 'openai', 'anthropic', 'huggingface', 'google', 'cohere', 'custom'
WAGTAIL_RAG_LLM_PROVIDER = 'ollama'

# LLM model name (default: 'mistral')
# Model name depends on provider:
#   - Ollama: 'mistral', 'llama2', 'phi', 'gemma', etc.
#   - OpenAI: 'gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo-preview', etc.
#   - Anthropic: 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'
#   - Google: 'gemini-pro', 'gemini-pro-vision'
#   - HuggingFace: model ID from HuggingFace Hub
WAGTAIL_RAG_MODEL_NAME = 'mistral'

# API Keys for cloud providers (set in environment variables or here)
# OPENAI_API_KEY = 'your-openai-key'
# ANTHROPIC_API_KEY = 'your-anthropic-key'
# GOOGLE_API_KEY = 'your-google-key'
# COHERE_API_KEY = 'your-cohere-key'
# HUGGINGFACE_API_KEY = 'your-hf-key'

# Embedding model (default: 'sentence-transformers/all-MiniLM-L6-v2')
WAGTAIL_RAG_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Number of documents to retrieve (default: 8)
WAGTAIL_RAG_RETRIEVE_K = 8

# Enable/disable MultiQueryRetriever (default: True)
WAGTAIL_RAG_USE_MULTI_QUERY = True

# Enable/disable hybrid search (default: True)
WAGTAIL_RAG_USE_HYBRID_SEARCH = True

# Models to exclude from indexing (default: ['wagtailcore.Page', 'wagtailcore.Site'])
WAGTAIL_RAG_EXCLUDE_MODELS = ['wagtailcore.Page', 'wagtailcore.Site']

# Custom prompt template (optional)
WAGTAIL_RAG_PROMPT_TEMPLATE = """You are a helpful assistant...
"""
```

## Usage

### 1. Build the RAG Index

Index your Wagtail pages:

```bash
python manage.py build_rag_index
```

**Options:**

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

### 2. Use the Chatbot

**In Python code:**

```python
from wagtail_rag.rag_chatbot import get_chatbot

# Get chatbot instance
chatbot = get_chatbot()

# Query the chatbot
result = chatbot.query("What types of bread do you have?")
print(result['answer'])
print(result['sources'])

# Filter by model
chatbot_filtered = get_chatbot(metadata_filter={'model': 'BreadPage'})
result = chatbot_filtered.query("Tell me about multigrain bread")
```

**Via API endpoint:**

```bash
curl -X POST http://localhost:8000/api/rag/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What types of bread do you have?",
    "model": "mistral",
    "filter": {"model": "BreadPage"}
  }'
```

**Search without generating response:**

```python
from wagtail_rag.rag_chatbot import get_chatbot

chatbot = get_chatbot()
results = chatbot.search_similar("multigrain bread", k=10)

for result in results:
    print(f"Title: {result['metadata']['title']}")
    print(f"Score: {result['score']}")
    print(f"Content: {result['content'][:200]}...")
```

## How It Works

1. **Indexing**: The `build_rag_index` command:
   - Discovers all Wagtail Page models dynamically
   - Extracts text from pages (title, body, StreamField, RichTextField, etc.)
   - Chunks the text using RecursiveCharacterTextSplitter
   - Generates embeddings using HuggingFace models
   - Stores chunks in ChromaDB with deterministic IDs

2. **Querying**: The chatbot:
   - Uses MultiQueryRetriever to generate query variations
   - Searches ChromaDB for similar chunks
   - Optionally searches Wagtail's full-text index (hybrid search)
   - Combines and deduplicates results
   - Boosts documents with matching titles (handles typos)
   - Passes context to LLM (Ollama) for answer generation

## Architecture

- **Vector Store**: ChromaDB for storing embeddings
- **Embeddings**: HuggingFace sentence transformers
- **LLM**: Multiple providers supported (Ollama, OpenAI, Anthropic, HuggingFace, Google, Cohere, or custom)
- **Framework**: LangChain for orchestration
- **Search**: Hybrid search (ChromaDB + Wagtail full-text)

## LLM Provider Configuration

### Ollama (Default - Local LLM)

```python
WAGTAIL_RAG_LLM_PROVIDER = 'ollama'
WAGTAIL_RAG_MODEL_NAME = 'mistral'  # or 'llama2', 'phi', 'gemma', etc.

# Install: pip install langchain-community ollama
# Setup: ollama serve && ollama pull mistral
```

### OpenAI

```python
WAGTAIL_RAG_LLM_PROVIDER = 'openai'
WAGTAIL_RAG_MODEL_NAME = 'gpt-4'  # or 'gpt-3.5-turbo', 'gpt-4-turbo-preview'
OPENAI_API_KEY = 'your-api-key'  # or set OPENAI_API_KEY environment variable

# Install: pip install langchain-openai
```

### Anthropic (Claude)

```python
WAGTAIL_RAG_LLM_PROVIDER = 'anthropic'
WAGTAIL_RAG_MODEL_NAME = 'claude-3-sonnet-20240229'  # or 'claude-3-opus-20240229', 'claude-3-haiku-20240307'
ANTHROPIC_API_KEY = 'your-api-key'  # or set ANTHROPIC_API_KEY environment variable

# Install: pip install langchain-anthropic
```

### Google (Gemini)

```python
WAGTAIL_RAG_LLM_PROVIDER = 'google'  # or 'gemini'
WAGTAIL_RAG_MODEL_NAME = 'gemini-pro'
GOOGLE_API_KEY = 'your-api-key'  # or set GOOGLE_API_KEY environment variable

# Install: pip install langchain-google-genai
```

### HuggingFace

```python
WAGTAIL_RAG_LLM_PROVIDER = 'huggingface'  # or 'hf'
WAGTAIL_RAG_MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'  # or any HF model ID
HUGGINGFACE_API_KEY = 'your-api-key'  # Optional, for private models

# Install: pip install langchain-huggingface transformers
```

### Cohere

```python
WAGTAIL_RAG_LLM_PROVIDER = 'cohere'
WAGTAIL_RAG_MODEL_NAME = 'command'  # or 'command-light', 'command-nightly'
COHERE_API_KEY = 'your-api-key'  # or set COHERE_API_KEY environment variable

# Install: pip install langchain-community cohere
```

### Custom LLM

```python
WAGTAIL_RAG_LLM_PROVIDER = 'custom'
WAGTAIL_RAG_CUSTOM_LLM_FACTORY = lambda model_name, **kwargs: YourCustomLLM(model=model_name, **kwargs)

# YourCustomLLM should be a LangChain-compatible LLM/ChatModel
```

### Using Different Providers in Code

```python
from wagtail_rag.rag_chatbot import get_chatbot

# Use OpenAI
chatbot = get_chatbot(llm_provider='openai', model_name='gpt-4')

# Use Anthropic
chatbot = get_chatbot(llm_provider='anthropic', model_name='claude-3-sonnet-20240229')

# Use with custom kwargs
chatbot = get_chatbot(
    llm_provider='openai',
    model_name='gpt-4',
    llm_kwargs={'temperature': 0.7, 'max_tokens': 1000}
)
```

### Using Different Providers via API

```bash
curl -X POST http://localhost:8000/api/rag/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What types of bread do you have?",
    "provider": "openai",
    "model": "gpt-4",
    "llm_kwargs": {"temperature": 0.7}
  }'
```

## Requirements

- Python 3.8+
- Django 3.2+
- Wagtail 4.0+
- ChromaDB
- LangChain
- HuggingFace embeddings

**LLM Provider Requirements** (install based on your choice):
- **Ollama**: `pip install langchain-community ollama` (for local LLM)
- **OpenAI**: `pip install langchain-openai`
- **Anthropic**: `pip install langchain-anthropic`
- **Google**: `pip install langchain-google-genai`
- **HuggingFace**: `pip install langchain-huggingface transformers`
- **Cohere**: `pip install langchain-community cohere`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

