"""
Tests for index building functionality.
"""

import unittest
from unittest.mock import MagicMock

from wagtail_rag.content_extraction.index_builder import (
    _parse_model_fields_shorthand,
    _get_fields_to_attempt,
)


class TestIndexBuilder(unittest.TestCase):
    """Test index building functions."""

    def test_parse_model_fields_shorthand(self):
        """Test parsing model names with :* suffix and without."""
        # With :* suffix
        model_names = ["blog.BlogPage", "breads.BreadPage:*", "locations.LocationPage"]
        cleaned, auto_fields = _parse_model_fields_shorthand(model_names)
        
        self.assertEqual(len(cleaned), 3)
        self.assertIn("breads.BreadPage", cleaned)
        self.assertIsNotNone(auto_fields)
        self.assertIn("breads.BreadPage:*", auto_fields)
        
        # Without :* suffix
        model_names = ["blog.BlogPage", "breads.BreadPage"]
        cleaned, auto_fields = _parse_model_fields_shorthand(model_names)
        self.assertEqual(len(cleaned), 2)
        self.assertIsNone(auto_fields)
        
        # Empty input
        cleaned, auto_fields = _parse_model_fields_shorthand(None)
        self.assertEqual(cleaned, [])
        self.assertIsNone(auto_fields)

    def test_get_fields_to_attempt(self):
        """Test field extraction with api_fields and defaults."""
        # With api_fields
        page = MagicMock()
        field1, field2 = MagicMock(), MagicMock()
        field1.name, field2.name = "body", "intro"
        page.api_fields = [field1, field2]
        
        fields, source = _get_fields_to_attempt(page)
        self.assertEqual(fields, ["body", "intro"])
        self.assertEqual(source, "model api_fields")
        
        # Fallback to defaults
        page.api_fields = []
        fields, source = _get_fields_to_attempt(page)
        self.assertEqual(fields, ["introduction", "body"])
        self.assertEqual(source, "default fields")


if __name__ == "__main__":
    unittest.main()
