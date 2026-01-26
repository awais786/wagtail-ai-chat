# Wagtail RAG Chatbot

A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS. This Django app provides a complete RAG solution that indexes your Wagtail pages into ChromaDB and provides a chatbot interface using LangChain and Ollama.

## Quick Start - Local Testing

### Prerequisites

- Python 3.8+
- Django 3.2+
- Wagtail 4.0+
- Ollama (for local LLM)

### Step 1: Set Up a Test Wagtail Project

If you don't have a Wagtail project yet, create one:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Wagtail and create a project
pip install wagtail
wagtail start myproject
cd myproject

# Run migrations and create superuser
python manage.py migrate
python manage.py createsuperuser
```

### Step 2: Install Wagtail RAG

#### Option A: Install from Source (Recommended for Testing)

```bash
# Navigate to the wagtail-ai-chat directory
cd /path/to/wagtail-ai-chat

# Install in development mode
pip install -e .
```

#### Option B: Copy to Your Project

```bash
# Copy the wagtail_rag directory to your Django project
cp -r wagtail_rag /path/to/your/wagtail/project/
```

### Step 3: Configure Django Settings

Add to your `settings/base.py` (or `settings.py`):

```python
INSTALLED_APPS = [
    # ... other apps
    'wagtail_rag',
]

# Wagtail RAG Configuration
WAGTAIL_RAG_COLLECTION_NAME = 'wagtail_rag'
WAGTAIL_RAG_CHROMA_PATH = os.path.join(BASE_DIR, 'chroma_db')
WAGTAIL_RAG_MODEL_NAME = 'mistral'  # or 'llama2', 'phi', etc.
WAGTAIL_RAG_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
WAGTAIL_RAG_RETRIEVE_K = 8
WAGTAIL_RAG_USE_MULTI_QUERY = True
WAGTAIL_RAG_USE_HYBRID_SEARCH = True
```

### Step 4: Install Dependencies

```bash
pip install -r wagtail_rag/requirements.txt
# Or install individually:
pip install langchain langchain-community langchain-text-splitters langchain-huggingface chromadb beautifulsoup4 ollama
```

### Step 5: Set Up Ollama

```bash
# Install Ollama
# macOS:
brew install ollama

# Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download

# Start Ollama service (in a separate terminal)
ollama serve

# Pull a model (in another terminal)
ollama pull mistral
# Or use a smaller model for testing:
ollama pull phi  # Smaller, faster model
```

### Step 6: Add URL Configuration

In your main `urls.py`:

```python
from django.urls import include, path

urlpatterns = [
    # ... other patterns
    path('', include('wagtail_rag.urls')),
]
```

### Step 7: Create Some Test Content

1. Start your Django server:
```bash
python manage.py runserver
```

2. Go to `http://localhost:8000/admin/` and log in
3. Create some pages with content (e.g., blog posts, articles, etc.)
4. Make sure pages are published (live)

### Step 8: Build the RAG Index

```bash
# Index all pages
python manage.py build_rag_index

# Or index specific models
python manage.py build_rag_index --models blog.BlogPage

# Reset and re-index everything
python manage.py build_rag_index --reset
```

### Step 9: Test the Chatbot

#### Option A: Using Python Shell

```bash
python manage.py shell
```

```python
from wagtail_rag.rag_chatbot import get_chatbot

# Get chatbot instance
chatbot = get_chatbot()

# Query the chatbot
result = chatbot.query("What content do you have?")
print("Answer:", result['answer'])
print("\nSources:")
for source in result['sources']:
    print(f"- {source['metadata'].get('title', 'Unknown')}")
    print(f"  URL: {source['metadata'].get('url', 'N/A')}")
```

#### Option B: Using API Endpoint

1. Make sure your server is running: `python manage.py runserver`

2. Test with curl:
```bash
curl -X POST http://localhost:8000/api/rag/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What content do you have?",
    "model": "mistral"
  }'
```

3. Or use Python requests:
```python
import requests

response = requests.post(
    'http://localhost:8000/api/rag/chat/',
    json={
        'question': 'What content do you have?',
        'model': 'mistral'
    }
)
print(response.json())
```

#### Option C: Test Search Functionality

```python
from wagtail_rag.rag_chatbot import get_chatbot

chatbot = get_chatbot()
results = chatbot.search_similar("your search query", k=5)

for result in results:
    print(f"Title: {result['metadata']['title']}")
    print(f"Score: {result['score']}")
    print(f"Content: {result['content'][:200]}...")
    print("---")
```

## Troubleshooting

### Issue: "Could not import LangChain components"

**Solution**: Install LangChain dependencies:
```bash
pip install langchain langchain-community langchain-text-splitters langchain-huggingface
```

### Issue: "Connection refused" when using Ollama

**Solution**: Make sure Ollama is running:
```bash
# Check if Ollama is running
ollama list

# If not, start it
ollama serve
```

### Issue: "No pages found to index"

**Solution**: 
- Make sure you have published (live) pages in your Wagtail site
- Check that pages are not excluded in settings: `WAGTAIL_RAG_EXCLUDE_MODELS`
- Try indexing specific models: `python manage.py build_rag_index --models your_app.YourPageModel`

### Issue: ChromaDB errors

**Solution**: 
- Delete the `chroma_db` directory and rebuild: `python manage.py build_rag_index --reset`
- Make sure you have write permissions in the project directory

### Issue: Embedding model download fails

**Solution**: The first run will download the embedding model (~80MB). Make sure you have internet connection and sufficient disk space.

## Testing Checklist

- [ ] Wagtail project created and running
- [ ] Wagtail RAG installed and added to INSTALLED_APPS
- [ ] All dependencies installed
- [ ] Ollama installed and running
- [ ] Ollama model pulled (mistral, phi, etc.)
- [ ] Test pages created and published in Wagtail
- [ ] RAG index built successfully
- [ ] Chatbot query works in Python shell
- [ ] API endpoint responds correctly

## Example Test Script

Create a file `test_rag.py` in your project root:

```python
#!/usr/bin/env python
"""
Quick test script for Wagtail RAG chatbot.
Run with: python manage.py shell < test_rag.py
Or: python test_rag.py (if Django is configured)
"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

from wagtail_rag.rag_chatbot import get_chatbot

def test_chatbot():
    print("Testing Wagtail RAG Chatbot...")
    print("=" * 50)
    
    try:
        chatbot = get_chatbot()
        print("✓ Chatbot initialized successfully")
        
        # Test query
        question = "What content is available?"
        print(f"\nQuestion: {question}")
        print("-" * 50)
        
        result = chatbot.query(question)
        print(f"Answer: {result['answer']}")
        print(f"\nFound {len(result['sources'])} sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['metadata'].get('title', 'Unknown')}")
            print(f"   URL: {source['metadata'].get('url', 'N/A')}")
        
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_chatbot()
```

Run it with:
```bash
python manage.py shell < test_rag.py
```

## Next Steps

- Read the full documentation in `wagtail_rag/README.md`
- Customize the prompt template in settings
- Configure field emphasis for better search results
- Set up metadata filtering for specific use cases

## License

MIT License - see [LICENSE](LICENSE) file for details.

