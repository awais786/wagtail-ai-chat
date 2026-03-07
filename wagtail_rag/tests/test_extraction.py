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

    def test_clean_field_text_preserves_paragraphs(self):
        """_clean_field_text should keep paragraph breaks so the splitter can use them."""
        extractor = WagtailAPIExtractor()

        text = "First paragraph.\n\nSecond paragraph."
        result = extractor._clean_field_text(text)
        self.assertIn("\n\n", result)
        self.assertIn("First paragraph.", result)
        self.assertIn("Second paragraph.", result)

    def test_clean_field_text_converts_html_block_elements(self):
        """Block HTML elements should become newlines, not run-on text."""
        extractor = WagtailAPIExtractor()

        html = "<p>First.</p><p>Second.</p>"
        result = extractor._clean_field_text(html)
        self.assertNotIn("<", result)
        self.assertIn("First.", result)
        self.assertIn("Second.", result)
        # Paragraphs should be separated, not merged
        self.assertNotEqual(result.strip(), "First. Second.")

    def test_extract_page_per_field_chunking(self):
        """Each field should produce its own document(s) with correct section metadata."""
        extractor = WagtailAPIExtractor(size_threshold=10000)

        page = MagicMock()
        page.id = 1
        page.title = "Sourdough Bread"
        page.slug = "sourdough"
        page.full_url = "https://example.com/sourdough/"
        page.last_published_at = None
        page.__class__.__name__ = "BreadPage"

        field1, field2 = MagicMock(), MagicMock()
        field1.name, field2.name = "introduction", "body"
        page.api_fields = [field1, field2]

        with patch.object(extractor, "_extract_field_value") as mock_extract:
            mock_extract.side_effect = lambda p, name: (
                "A short intro." if name == "introduction" else "The body content."
            )
            docs = extractor.extract_page(page)

        sections = [d.metadata["section"] for d in docs]
        self.assertIn("introduction", sections)
        self.assertIn("body", sections)
        # No cross-field bleed — each doc belongs to exactly one field
        for doc in docs:
            self.assertIn(doc.metadata["section"], ("introduction", "body"))

    def test_extract_page_chunk_header(self):
        """Every chunk must start with the page title and section name."""
        extractor = WagtailAPIExtractor(size_threshold=10000)

        page = MagicMock()
        page.id = 2
        page.title = "Multigrain"
        page.slug = "multigrain"
        page.full_url = "https://example.com/multigrain/"
        page.last_published_at = None
        page.__class__.__name__ = "BreadPage"

        field = MagicMock()
        field.name = "body"
        page.api_fields = [field]

        with patch.object(
            extractor, "_extract_field_value", return_value="Some body text."
        ):
            docs = extractor.extract_page(page)

        self.assertEqual(len(docs), 1)
        content = docs[0].page_content
        self.assertIn("Multigrain", content)
        self.assertIn("body", content)

    def test_extract_page_large_field_is_split(self):
        """A field larger than size_threshold should produce multiple chunks."""
        extractor = WagtailAPIExtractor(
            chunk_size=50, chunk_overlap=0, size_threshold=60
        )

        page = MagicMock()
        page.id = 3
        page.title = "Long Page"
        page.slug = "long"
        page.full_url = "https://example.com/long/"
        page.last_published_at = None
        page.__class__.__name__ = "ArticlePage"

        field = MagicMock()
        field.name = "body"
        page.api_fields = [field]

        long_text = "word " * 100  # well over any threshold
        with patch.object(extractor, "_extract_field_value", return_value=long_text):
            docs = extractor.extract_page(page)

        self.assertGreater(len(docs), 1)
        for doc in docs:
            self.assertEqual(doc.metadata["section"], "body")

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
