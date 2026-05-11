"""
Tests for hybrid search functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

from wagtail_rag.embeddings.search import EmbeddingSearcher


class TestHybridSearch(unittest.TestCase):
    """Test hybrid search functionality."""

    def test_hybrid_search_basic(self):
        """Test basic hybrid search with mocked components."""
        # Mock vector store and retriever
        mock_vector_doc = MagicMock()
        mock_vector_doc.page_content = "Vector content"
        mock_vector_doc.metadata = {"page_id": 1, "title": "Test"}
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_vector_doc]

        # Mock page model
        mock_page = MagicMock()
        mock_page.id = 1
        mock_page.title = "Test Page"
        mock_page.search_description = "Test description"
        mock_page_model = MagicMock()
        mock_page_model.objects.live.return_value.search.return_value = [mock_page]

        searcher = EmbeddingSearcher(
            vectorstore=mock_vector_store,
            retriever=mock_retriever,
            k_value=5,
            use_hybrid_search=True,
        )

        results = searcher.retrieve_with_embeddings(query="test query")

        self.assertIsInstance(results, list)
        # Should have results from at least one source
        self.assertGreater(len(results), 0)

    def test_hybrid_search_disabled(self):
        """Test hybrid search when disabled (vector-only)."""
        mock_vector_doc = MagicMock()
        mock_vector_doc.page_content = "Vector content"
        mock_vector_doc.metadata = {"page_id": 1}
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_vector_doc]

        searcher = EmbeddingSearcher(
            vectorstore=mock_vector_store,
            retriever=mock_retriever,
            k_value=5,
            use_hybrid_search=False,
        )

        results = searcher.retrieve_with_embeddings(query="test query")

        self.assertIsInstance(results, list)
        # Verify retriever was called
        mock_retriever.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
