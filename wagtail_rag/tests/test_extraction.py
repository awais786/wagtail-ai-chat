"""
Tests for content extraction functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

from wagtail_rag.content_extraction.api_fields_extractor import (
    WagtailAPIExtractor,
    DEFAULT_FIELDS,
    SYSTEM_FIELDS,
    CORE_SKIP_FIELDS,
)


class TestWagtailAPIExtractor(unittest.TestCase):
    """Test WagtailAPIExtractor functionality."""

    def test_get_page_url_full_url(self):
        """Test URL extraction with full_url available."""
        page = MagicMock()
        page.full_url = "https://example.com/page/"
        url = WagtailAPIExtractor._get_page_url(page)
        self.assertEqual(url, "https://example.com/page/")

    def test_get_page_url_fallback(self):
        """Test URL extraction falls back to url attribute."""
        page = MagicMock()
        # Simulate full_url not existing by making it raise AttributeError
        type(page).full_url = property(lambda self: (_ for _ in ()).throw(AttributeError()))
        page.url = "/page/"
        url = WagtailAPIExtractor._get_page_url(page)
        self.assertEqual(url, "/page/")

    def test_get_page_url_final_fallback(self):
        """Test URL extraction final fallback to page ID."""
        page = MagicMock()
        # Simulate both full_url and url not existing
        type(page).full_url = property(lambda self: (_ for _ in ()).throw(AttributeError()))
        type(page).url = property(lambda self: (_ for _ in ()).throw(AttributeError()))
        page.id = 123
        url = WagtailAPIExtractor._get_page_url(page)
        self.assertEqual(url, "/page/123/")

    def test_clean_text_strips_html(self):
        """Test HTML stripping in text cleaning."""
        extractor = WagtailAPIExtractor()
        text = "<p>Hello <b>World</b></p>"
        result = extractor._clean_text(text)
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)
        self.assertIn("Hello", result)
        self.assertIn("World", result)

    def test_clean_text_normalizes_whitespace(self):
        """Test whitespace normalization."""
        extractor = WagtailAPIExtractor()
        text = "Hello    \n\n   World"
        result = extractor._clean_text(text)
        self.assertEqual(result, "Hello World")

    def test_clean_text_empty_input(self):
        """Test clean_text handles empty input."""
        extractor = WagtailAPIExtractor()
        self.assertEqual(extractor._clean_text(""), "")
        self.assertEqual(extractor._clean_text(None), "")

    def test_resolve_candidate_fields_with_api_fields(self):
        """Test field resolution when page has api_fields."""
        page = MagicMock()
        field1 = MagicMock()
        field1.name = "body"
        field2 = MagicMock()
        field2.name = "introduction"
        page.api_fields = [field1, field2]

        fields, source = WagtailAPIExtractor._resolve_candidate_fields(page)
        self.assertEqual(fields, ["body", "introduction"])
        self.assertIn("api_fields", source)

    def test_resolve_candidate_fields_defaults(self):
        """Test field resolution falls back to defaults."""
        page = MagicMock()
        page.api_fields = []

        with patch(
            "wagtail_rag.content_extraction.api_fields_extractor.settings"
        ) as mock_settings:
            mock_settings.WAGTAIL_RAG_DEFAULT_FIELDS = DEFAULT_FIELDS
            fields, source = WagtailAPIExtractor._resolve_candidate_fields(page)
            self.assertEqual(fields, DEFAULT_FIELDS)
            self.assertIn("default", source.lower())

    def test_build_metadata(self):
        """Test metadata building for documents."""
        page = MagicMock()
        page.id = 123
        page.title = "Test Page"
        page.slug = "test-page"
        page.full_url = "https://example.com/test/"
        page.last_published_at = None

        metadata = WagtailAPIExtractor._build_metadata(page)

        self.assertEqual(metadata["page_id"], 123)
        self.assertEqual(metadata["title"], "Test Page")
        self.assertEqual(metadata["slug"], "test-page")
        self.assertEqual(metadata["url"], "https://example.com/test/")


class TestConstants(unittest.TestCase):
    """Test that module constants are properly defined."""

    def test_default_fields_defined(self):
        """Test DEFAULT_FIELDS constant."""
        self.assertIsInstance(DEFAULT_FIELDS, list)
        self.assertGreater(len(DEFAULT_FIELDS), 0)
        self.assertIn("body", DEFAULT_FIELDS)

    def test_system_fields_defined(self):
        """Test SYSTEM_FIELDS constant."""
        self.assertIsInstance(SYSTEM_FIELDS, set)
        self.assertGreater(len(SYSTEM_FIELDS), 0)
        self.assertIn("id", SYSTEM_FIELDS)
        self.assertIn("slug", SYSTEM_FIELDS)

    def test_core_skip_fields_subset_of_system(self):
        """Test CORE_SKIP_FIELDS is a subset of SYSTEM_FIELDS."""
        self.assertTrue(CORE_SKIP_FIELDS.issubset(SYSTEM_FIELDS))


if __name__ == "__main__":
    unittest.main()
