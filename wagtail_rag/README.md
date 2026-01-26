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

# LLM model name (default: 'mistral')
WAGTAIL_RAG_MODEL_NAME = 'mistral'

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
- **LLM**: Ollama (local LLM, supports Mistral, Llama2, etc.)
- **Framework**: LangChain for orchestration
- **Search**: Hybrid search (ChromaDB + Wagtail full-text)

## Requirements

- Python 3.8+
- Django 3.2+
- Wagtail 4.0+
- Ollama (for local LLM)
- ChromaDB
- LangChain
- HuggingFace embeddings

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

