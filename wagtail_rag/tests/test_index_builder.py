"""
Tests for index building functionality.
"""

import unittest
from unittest.mock import MagicMock

from wagtail_rag.content_extraction.index_builder import (
    _parse_model_fields_shorthand,
    _get_fields_to_attempt,
    STEP_SEPARATOR,
)


class TestParseModelFieldsShorthand(unittest.TestCase):
    """Test _parse_model_fields_shorthand function."""

    def test_parse_model_fields_shorthand_with_star(self):
        """Test parsing model names with :* suffix."""
        model_names = ["blog.BlogPage", "breads.BreadPage:*", "locations.LocationPage"]
        cleaned, auto_fields = _parse_model_fields_shorthand(model_names)

        self.assertEqual(len(cleaned), 3)
        self.assertIn("blog.BlogPage", cleaned)
        self.assertIn("breads.BreadPage", cleaned)
        self.assertIn("locations.LocationPage", cleaned)

        self.assertIsNotNone(auto_fields)
        self.assertEqual(len(auto_fields), 1)
        self.assertIn("breads.BreadPage:*", auto_fields)

    def test_parse_model_fields_shorthand_no_star(self):
        """Test parsing model names without :* suffix."""
        model_names = ["blog.BlogPage", "breads.BreadPage"]
        cleaned, auto_fields = _parse_model_fields_shorthand(model_names)

        self.assertEqual(len(cleaned), 2)
        self.assertIsNone(auto_fields)

    def test_parse_model_fields_shorthand_empty(self):
        """Test parsing empty model names."""
        cleaned, auto_fields = _parse_model_fields_shorthand(None)
        self.assertEqual(cleaned, [])
        self.assertIsNone(auto_fields)


class TestGetFieldsToAttempt(unittest.TestCase):
    """Test _get_fields_to_attempt function."""

    def test_get_fields_to_attempt_with_api_fields(self):
        """Test field extraction attempt with api_fields."""
        page = MagicMock()
        field1 = MagicMock()
        field1.name = "body"
        field2 = MagicMock()
        field2.name = "intro"
        page.api_fields = [field1, field2]

        fields, source = _get_fields_to_attempt(page)
        self.assertEqual(fields, ["body", "intro"])
        self.assertEqual(source, "model api_fields")

    def test_get_fields_to_attempt_default(self):
        """Test field extraction attempt falls back to defaults."""
        page = MagicMock()
        page.api_fields = []  # No api_fields

        fields, source = _get_fields_to_attempt(page)
        expected_fields = ["introduction", "body"]
        self.assertEqual(fields, expected_fields)
        self.assertEqual(source, "default fields")


class TestConstants(unittest.TestCase):
    """Test index builder constants."""

    def test_step_separator_defined(self):
        """Test STEP_SEPARATOR constant is defined."""
        self.assertIsInstance(STEP_SEPARATOR, str)
        self.assertEqual(len(STEP_SEPARATOR), 80)


if __name__ == "__main__":
    unittest.main()
