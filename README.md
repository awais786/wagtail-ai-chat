# Wagtail RAG Chatbot

A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS.

## Quick Start

### 1. Install

```bash
cd wagtail_rag
pip install -e .
```

### 2. Add to INSTALLED_APPS

```python
INSTALLED_APPS = [
    # ... other apps
    'wagtail_rag',
]
```

### 3. Configure Settings

```python
# LLM Configuration
WAGTAIL_RAG_LLM_PROVIDER = 'openai'  # or 'ollama', 'anthropic', etc.
WAGTAIL_RAG_MODEL_NAME = 'gpt-4'
OPENAI_API_KEY = 'sk-...'  # or set as environment variable

# Embedding Configuration
WAGTAIL_RAG_EMBEDDING_PROVIDER = 'openai'
WAGTAIL_RAG_EMBEDDING_MODEL = 'text-embedding-ada-002'
```

### 4. Build Index

```bash
python manage.py build_rag_index
```

### 5. Use the Chatbot

```python
from wagtail_rag.rag_chatbot import get_chatbot

chatbot = get_chatbot()
result = chatbot.query("What content is available?")
print(result['answer'])
```

## Full Documentation

For complete documentation, configuration options, and examples, see:

**[wagtail_rag/README.md](wagtail_rag/README.md)**

The main README includes:
- Complete installation instructions
- All configuration options
- LLM provider setup (OpenAI, Anthropic, Ollama, Google, HuggingFace, Cohere, Custom)
- Embedding provider setup (OpenAI, HuggingFace, Cohere, Google, Sentence Transformers, Custom)
- Usage examples
- Troubleshooting guide
- Cost considerations and recommendations

## Example Configuration

See [CONFIGURATION_EXAMPLE.py](CONFIGURATION_EXAMPLE.py) for a complete example of all available settings.

## License

MIT License
