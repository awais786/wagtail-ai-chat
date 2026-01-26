"""
Wagtail RAG - A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS.

This app provides a complete RAG solution that:
- Indexes Wagtail pages into ChromaDB
- Provides a chatbot interface using LangChain and Ollama
- Supports hybrid search (Wagtail + ChromaDB)
- Includes MultiQueryRetriever for better retrieval
"""

__version__ = '0.1.0'

