"""
Tests for index building functionality.
"""

import unittest
from unittest.mock import MagicMock, call, patch

from wagtail_rag.content_extraction.index_builder import (
    _parse_model_fields_shorthand,
    _upsert_in_batches,
)
from wagtail_rag.content_extraction.vector_store import _pgvector_connection_string


class TestIndexBuilder(unittest.TestCase):
    """Test index building functions."""

    def test_parse_model_fields_shorthand(self):
        """Test that :* suffix is stripped from model names."""
        names = _parse_model_fields_shorthand(
            ["blog.BlogPage", "breads.BreadPage:*", "locations.LocationPage"]
        )
        self.assertEqual(names, ["blog.BlogPage", "breads.BreadPage", "locations.LocationPage"])

        # No :* — returned unchanged
        names = _parse_model_fields_shorthand(["blog.BlogPage", "breads.BreadPage"])
        self.assertEqual(names, ["blog.BlogPage", "breads.BreadPage"])

        # Empty input
        self.assertEqual(_parse_model_fields_shorthand([]), [])


class TestPgvectorConnectionString(unittest.TestCase):
    """Tests for the pgvector connection string helper."""

    def test_explicit_setting_takes_precedence(self):
        explicit = "postgresql+psycopg2://user:pass@db:5432/mydb"
        mock_settings = MagicMock()
        mock_settings.WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING = explicit

        with patch(
            "wagtail_rag.content_extraction.vector_store.settings", mock_settings
        ):
            result = _pgvector_connection_string()

        self.assertEqual(result, explicit)

    def test_derived_from_database_settings_with_password(self):
        mock_settings = MagicMock()
        mock_settings.WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING = None
        mock_settings.DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": "testdb",
                "USER": "alice",
                "PASSWORD": "secret",
                "HOST": "localhost",
                "PORT": 5432,
            }
        }

        with patch(
            "wagtail_rag.content_extraction.vector_store.settings", mock_settings
        ):
            result = _pgvector_connection_string()

        self.assertIn("alice:secret@localhost:5432/testdb", result)

    def test_raises_for_non_postgres_database(self):
        mock_settings = MagicMock()
        mock_settings.WAGTAIL_RAG_PGVECTOR_CONNECTION_STRING = None
        mock_settings.DATABASES = {
            "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}
        }

        with patch(
            "wagtail_rag.content_extraction.vector_store.settings", mock_settings
        ):
            with self.assertRaises(ValueError) as ctx:
                _pgvector_connection_string()

        self.assertIn("PostgreSQL", str(ctx.exception))


class TestUpsertInBatches(unittest.TestCase):
    """Tests for the cross-page embedding batching helper."""

    def _make_docs(self, n: int):
        docs = []
        for i in range(n):
            doc = MagicMock()
            doc.metadata = {"page_id": i, "section": "body", "chunk_index": 0}
            docs.append(doc)
        return docs

    def test_splits_into_correct_number_of_batches(self):
        """11 docs with batch_size=5 should produce 3 upsert calls (5+5+1)."""
        store = MagicMock()
        docs = self._make_docs(11)

        _upsert_in_batches(store, docs, batch_size=5, stdout=None)

        self.assertEqual(store.upsert.call_count, 3)
        # First two batches have 5 docs, last has 1
        sizes = [len(c.args[0]) for c in store.upsert.call_args_list]
        self.assertEqual(sizes, [5, 5, 1])

    def test_all_upsert_calls_use_save_false(self):
        """save=False must be passed to every upsert call so FAISS is not written per batch."""
        store = MagicMock()
        docs = self._make_docs(3)

        _upsert_in_batches(store, docs, batch_size=10, stdout=None)

        for c in store.upsert.call_args_list:
            self.assertFalse(c.kwargs.get("save", True))

    def test_save_called_exactly_once(self):
        """store.save() must be called once at the end, not per batch."""
        store = MagicMock()
        docs = self._make_docs(20)

        _upsert_in_batches(store, docs, batch_size=7, stdout=None)

        store.save.assert_called_once()

    def test_empty_documents_does_nothing(self):
        store = MagicMock()
        _upsert_in_batches(store, [], batch_size=10, stdout=None)
        store.upsert.assert_not_called()
        store.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
