"""
Tests for LLM generation functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

from wagtail_rag.llm_providers.generation import LLMGenerator


class TestLLMGenerator(unittest.TestCase):
    """Test LLM generation functionality."""

    def setUp(self):
        self.mock_llm = MagicMock()
        self.generator = LLMGenerator(llm=self.mock_llm, retriever=None)

    def test_context_building_with_truncation(self):
        """Test context building with and without truncation."""
        doc1, doc2 = MagicMock(), MagicMock()
        doc1.page_content = "First doc content"
        doc2.page_content = "Second doc content"

        with patch("wagtail_rag.llm_providers.generation.settings") as mock_settings:
            # No truncation
            mock_settings.WAGTAIL_RAG_MAX_CONTEXT_CHARS = 0
            context = self.generator._get_context_from_docs([doc1, doc2])
            self.assertIn("First doc content", context)
            self.assertIn("Second doc content", context)

            # With truncation
            doc = MagicMock()
            doc.page_content = "A" * 100
            mock_settings.WAGTAIL_RAG_MAX_CONTEXT_CHARS = 50
            context = self.generator._get_context_from_docs([doc])
            self.assertLessEqual(len(context), 50)
            self.assertTrue(context.endswith("..."))

    def test_extract_text_from_result(self):
        """Test text extraction from different result types."""
        # With content attribute
        result = MagicMock()
        result.content = "Response content"
        self.assertEqual(
            LLMGenerator._extract_text_from_result(result), "Response content"
        )

        # Direct string
        self.assertEqual(
            LLMGenerator._extract_text_from_result("Direct string"), "Direct string"
        )

        # Other types
        self.assertEqual(LLMGenerator._extract_text_from_result(123), "123")

    def test_prompt_templates(self):
        """Test default and custom prompt templates."""
        mock_llm = MagicMock()
        generator = LLMGenerator(llm=mock_llm, retriever=None)

        # Default templates
        self.assertIn("{context}", generator.prompt_template_str)
        self.assertIn("{question}", generator.prompt_template_str)
        self.assertIsNotNone(generator.system_prompt_str)

        # Custom templates
        with patch("wagtail_rag.llm_providers.generation.settings") as mock_settings:
            mock_settings.WAGTAIL_RAG_PROMPT_TEMPLATE = "Custom: {context} {question}"
            mock_settings.WAGTAIL_RAG_SYSTEM_PROMPT = "Custom system prompt"
            mock_settings.WAGTAIL_RAG_ENABLE_CHAT_HISTORY = False

            generator = LLMGenerator(llm=mock_llm, retriever=None)
            self.assertEqual(
                generator.prompt_template_str, "Custom: {context} {question}"
            )
            self.assertEqual(generator.system_prompt_str, "Custom system prompt")


if __name__ == "__main__":
    unittest.main()
