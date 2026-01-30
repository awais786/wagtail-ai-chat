"""
Wagtail RAG - A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS.

This app provides a complete RAG solution that:
- Indexes Wagtail pages into FAISS or ChromaDB
- Provides a chatbot interface using LangChain and multiple LLM providers
- Supports hybrid search (Wagtail + vector search)
- Includes MultiQueryRetriever for better retrieval
"""

__version__ = '0.1.0'

# Note: Direct imports from __init__.py may cause AppRegistryNotReady errors
# during Django startup. Import directly from modules instead:
#   from wagtail_rag.rag_chatbot import RAGChatBot, get_chatbot
#   from wagtail_rag.llm_providers import get_llm
#   from wagtail_rag.embeddings import get_embeddings
#
# BREAKING CHANGE (v0.1.0):
# The embedding_providers module has been renamed to embeddings:
#   OLD: from wagtail_rag.embedding_providers import get_embeddings
#   NEW: from wagtail_rag.embeddings import get_embeddings

__all__ = ['RAGChatBot', 'get_chatbot', 'get_llm', 'get_embeddings']

