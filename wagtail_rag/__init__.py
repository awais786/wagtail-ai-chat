"""
Wagtail RAG - A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS.

This app provides a complete RAG solution that:
- Indexes Wagtail pages into ChromaDB
- Provides a chatbot interface using LangChain and multiple LLM providers
- Supports hybrid search (Wagtail + ChromaDB)
- Includes MultiQueryRetriever for better retrieval
"""

__version__ = '0.1.0'

# Export main functions for easier imports
from .rag_chatbot import RAGChatBot, get_chatbot
from .llm_providers import get_llm

__all__ = ['RAGChatBot', 'get_chatbot', 'get_llm']

