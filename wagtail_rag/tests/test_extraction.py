"""
Tests for content extraction functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

from wagtail_rag.content_extraction.api_fields_extractor import (
    WagtailAPIExtractor,
    DEFAULT_FIELDS,
)


class TestWagtailAPIExtractor(unittest.TestCase):
    """Test WagtailAPIExtractor functionality."""

    def test_get_page_url_with_fallbacks(self):
        """Test URL extraction with full_url and fallbacks."""
        extractor = WagtailAPIExtractor

        # Test full_url
        page = MagicMock()
        page.full_url = "https://example.com/page/"
        self.assertEqual(extractor._get_page_url(page), "https://example.com/page/")

        # Test fallback to url
        page = MagicMock()
        page.url = "/page/"
        del page.full_url
        self.assertEqual(extractor._get_page_url(page), "/page/")

        # Test final fallback to page ID
        page = MagicMock()
        page.id = 123
        del page.full_url
        del page.url
        self.assertEqual(extractor._get_page_url(page), "/page/123/")

    def test_clean_text(self):
        """Test HTML stripping and whitespace normalization."""
        extractor = WagtailAPIExtractor()

        # HTML stripping
        text = "<p>Hello <b>World</b></p>"
        result = extractor._clean_text(text)
        self.assertNotIn("<", result)
        self.assertIn("Hello", result)
        self.assertIn("World", result)

        # Whitespace normalization
        text = "Hello    \n\n   World"
        self.assertEqual(extractor._clean_text(text), "Hello World")

        # Empty input
        self.assertEqual(extractor._clean_text(""), "")
        self.assertEqual(extractor._clean_text(None), "")

    def test_resolve_candidate_fields(self):
        """Test field resolution with api_fields and defaults."""
        # With api_fields
        page = MagicMock()
        field1, field2 = MagicMock(), MagicMock()
        field1.name, field2.name = "body", "introduction"
        page.api_fields = [field1, field2]

        fields, source = WagtailAPIExtractor._resolve_candidate_fields(page)
        self.assertEqual(fields, ["body", "introduction"])
        self.assertIn("api_fields", source)

        # Fallback to defaults
        page.api_fields = []
        with patch(
            "wagtail_rag.content_extraction.api_fields_extractor.settings"
        ) as mock_settings:
            mock_settings.WAGTAIL_RAG_DEFAULT_FIELDS = DEFAULT_FIELDS
            fields, source = WagtailAPIExtractor._resolve_candidate_fields(page)
            self.assertEqual(fields, DEFAULT_FIELDS)

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


if __name__ == "__main__":
    unittest.main()
