# Quick Start Guide

## Minimal Setup (5 minutes)

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Wagtail RAG and dependencies
pip install -e .
pip install langchain langchain-community langchain-text-splitters langchain-huggingface chromadb beautifulsoup4 ollama
```

### 2. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama (keep this running)
ollama serve

# In another terminal, pull a model
ollama pull mistral
```

### 3. Add to Your Wagtail Project

In `settings.py`:

```python
INSTALLED_APPS = [
    # ... existing apps
    'wagtail_rag',
]

# Optional: Add these settings
WAGTAIL_RAG_COLLECTION_NAME = 'wagtail_rag'
WAGTAIL_RAG_MODEL_NAME = 'mistral'
```

In `urls.py`:

```python
urlpatterns = [
    # ... existing patterns
    path('', include('wagtail_rag.urls')),
]
```

### 4. Build Index

```bash
python manage.py build_rag_index
```

### 5. Test

```bash
# Option 1: Python shell
python manage.py shell
>>> from wagtail_rag.rag_chatbot import get_chatbot
>>> chatbot = get_chatbot()
>>> result = chatbot.query("What content do you have?")
>>> print(result['answer'])

# Option 2: Test script
python manage.py shell < test_rag.py

# Option 3: API
curl -X POST http://localhost:8000/api/rag/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What content do you have?"}'
```

## Common Issues

**"No pages found to index"**
- Make sure you have published pages in Wagtail admin
- Check pages are live (not draft)

**"Connection refused" (Ollama)**
- Make sure `ollama serve` is running
- Test with: `ollama list`

**"Could not import LangChain"**
- Run: `pip install langchain langchain-community langchain-text-splitters langchain-huggingface`

**ChromaDB errors**
- Delete `chroma_db` folder and rebuild: `python manage.py build_rag_index --reset`

