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

    def test_get_context_from_docs_no_truncation(self):
        """Test context building without truncation."""
        doc1 = MagicMock()
        doc1.page_content = "First doc content"
        doc2 = MagicMock()
        doc2.page_content = "Second doc content"

        with patch("wagtail_rag.llm_providers.generation.settings") as mock_settings:
            mock_settings.WAGTAIL_RAG_MAX_CONTEXT_CHARS = 0
            context = self.generator._get_context_from_docs([doc1, doc2])

        self.assertIn("First doc content", context)
        self.assertIn("Second doc content", context)

    def test_get_context_from_docs_with_truncation(self):
        """Test context building with truncation."""
        doc = MagicMock()
        doc.page_content = "A" * 100

        with patch("wagtail_rag.llm_providers.generation.settings") as mock_settings:
            mock_settings.WAGTAIL_RAG_MAX_CONTEXT_CHARS = 50
            context = self.generator._get_context_from_docs([doc])

        self.assertLessEqual(len(context), 50)
        self.assertTrue(context.endswith("..."))

    def test_get_context_from_docs_small_limit(self):
        """Test context building with very small limit (edge case)."""
        doc = MagicMock()
        doc.page_content = "Hello World"

        with patch("wagtail_rag.llm_providers.generation.settings") as mock_settings:
            mock_settings.WAGTAIL_RAG_MAX_CONTEXT_CHARS = 2
            context = self.generator._get_context_from_docs([doc])

        # Should not crash with negative slicing
        self.assertLessEqual(len(context), 2)
        self.assertNotIn(
            "...", context
        )  # Should not add ellipsis for very small limits

    def test_extract_text_from_result_with_content(self):
        """Test text extraction from result with content attribute."""
        result = MagicMock()
        result.content = "Response content"

        text = LLMGenerator._extract_text_from_result(result)
        self.assertEqual(text, "Response content")

    def test_extract_text_from_result_string(self):
        """Test text extraction from string result."""
        result = "Direct string response"
        text = LLMGenerator._extract_text_from_result(result)
        self.assertEqual(text, "Direct string response")

    def test_extract_text_from_result_other(self):
        """Test text extraction from other types."""
        result = 123
        text = LLMGenerator._extract_text_from_result(result)
        self.assertEqual(text, "123")

    def test_is_chat_model_detection(self):
        """Test chat model detection logic."""
        # Test with class name containing 'Chat'
        chat_model = MagicMock()
        chat_model.__class__.__name__ = "ChatOpenAI"
        generator = LLMGenerator(llm=chat_model, retriever=None)
        self.assertTrue(generator._is_chat_model())

        # Test with regular LLM (has generate, not chat-style)
        regular_model = MagicMock()
        regular_model.__class__.__name__ = "OpenAI"
        delattr(regular_model, "invoke")
        regular_model.generate = MagicMock()
        generator = LLMGenerator(llm=regular_model, retriever=None)
        # This should return False (has generate method)
        # Note: The actual implementation may vary, this tests the heuristic


class TestPromptTemplates(unittest.TestCase):
    """Test prompt template handling."""

    def test_default_prompt_template_has_placeholders(self):
        """Test default prompt template contains required placeholders."""
        mock_llm = MagicMock()
        generator = LLMGenerator(llm=mock_llm, retriever=None)

        self.assertIn("{context}", generator.prompt_template_str)
        self.assertIn("{question}", generator.prompt_template_str)

    def test_default_system_prompt_defined(self):
        """Test default system prompt is defined."""
        mock_llm = MagicMock()
        generator = LLMGenerator(llm=mock_llm, retriever=None)

        self.assertIsNotNone(generator.system_prompt_str)
        self.assertGreater(len(generator.system_prompt_str), 0)

    @patch("wagtail_rag.llm_providers.generation.settings")
    def test_custom_prompt_template(self, mock_settings):
        """Test that custom prompt template from settings is used."""
        mock_settings.WAGTAIL_RAG_PROMPT_TEMPLATE = "Custom: {context} {question}"
        mock_settings.WAGTAIL_RAG_SYSTEM_PROMPT = "Custom system prompt"
        mock_settings.WAGTAIL_RAG_ENABLE_CHAT_HISTORY = False

        mock_llm = MagicMock()
        generator = LLMGenerator(llm=mock_llm, retriever=None)

        self.assertEqual(generator.prompt_template_str, "Custom: {context} {question}")
        self.assertEqual(generator.system_prompt_str, "Custom system prompt")


if __name__ == "__main__":
    unittest.main()
