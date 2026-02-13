"""
Tests for hybrid search functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

from wagtail_rag.embeddings.search import EmbeddingSearcher


class TestHybridSearch(unittest.TestCase):
    """Test hybrid search functionality."""

    @patch("wagtail_rag.embeddings.search.settings")
    def test_hybrid_search_basic(self, mock_settings):
        """Test basic hybrid search with mocked components."""
        mock_settings.WAGTAIL_RAG_VECTOR_WEIGHT = 0.5
        mock_settings.WAGTAIL_RAG_WAGTAIL_WEIGHT = 0.5
        mock_settings.WAGTAIL_RAG_ENABLE_HYBRID_SEARCH = True
        mock_settings.WAGTAIL_RAG_MAX_RESULTS = 5

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

    @patch("wagtail_rag.embeddings.search.settings")
    def test_hybrid_search_disabled(self, mock_settings):
        """Test hybrid search when disabled (vector-only)."""
        mock_settings.WAGTAIL_RAG_ENABLE_HYBRID_SEARCH = False
        mock_settings.WAGTAIL_RAG_MAX_RESULTS = 5

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

    def test_hybrid_search_deduplication(self):
        """Test that duplicate results are properly deduplicated."""
        with patch("wagtail_rag.embeddings.search.settings") as mock_settings:
            mock_settings.WAGTAIL_RAG_VECTOR_WEIGHT = 0.5
            mock_settings.WAGTAIL_RAG_WAGTAIL_WEIGHT = 0.5
            mock_settings.WAGTAIL_RAG_ENABLE_HYBRID_SEARCH = True
            mock_settings.WAGTAIL_RAG_MAX_RESULTS = 10

            # Create duplicate page_id in results
            mock_vector_doc1 = MagicMock()
            mock_vector_doc1.page_content = "Content 1"
            mock_vector_doc1.metadata = {"page_id": 1, "title": "Test"}

            mock_vector_doc2 = MagicMock()
            mock_vector_doc2.page_content = "Content 2"
            mock_vector_doc2.metadata = {"page_id": 1, "title": "Test"}  # Duplicate

            mock_vector_store = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [mock_vector_doc1, mock_vector_doc2]

            mock_page_model = MagicMock()
            mock_page_model.objects.live.return_value.search.return_value = []

            searcher = EmbeddingSearcher(
                vectorstore=mock_vector_store,
                retriever=mock_retriever,
                k_value=10,
                use_hybrid_search=True,
            )

            results = searcher.retrieve_with_embeddings(query="test query")

            # Should deduplicate and only return 1 result
            page_ids = [doc.metadata.get("page_id") for doc in results]
            self.assertEqual(
                len(page_ids),
                len(set(page_ids)),
                "Results should be deduplicated by page_id",
            )


if __name__ == "__main__":
    unittest.main()
