"""
Tests for LLM generation functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

from django.test import override_settings

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

        base_rag = {"llm": {"provider": "ollama", "model": "llama2"}}

        # No truncation
        with override_settings(WAGTAIL_RAG={**base_rag, **{"llm": {**base_rag["llm"], "max_context_chars": 0}}}):
            context = self.generator._get_context_from_docs([doc1, doc2])
        self.assertIn("First doc content", context)
        self.assertIn("Second doc content", context)

        # With truncation
        doc = MagicMock()
        doc.page_content = "A" * 100
        with override_settings(WAGTAIL_RAG={**base_rag, **{"llm": {**base_rag["llm"], "max_context_chars": 50}}}):
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
        """Prompt and system prompt contain required placeholders and security rules."""
        mock_llm = MagicMock()
        generator = LLMGenerator(llm=mock_llm, retriever=None)

        self.assertIn("{context}", generator.prompt_template_str)
        self.assertIn("{question}", generator.prompt_template_str)

        # Injection defence — both prompt and system prompt must warn the model
        self.assertIn("ignore", generator.prompt_template_str.lower())
        self.assertIn("disregard", generator.system_prompt_str.lower())

        # Structural separation — user input wrapped in XML delimiters
        self.assertIn("<context>", generator.prompt_template_str)
        self.assertIn("<question>", generator.prompt_template_str)

    def test_sanitize_output_strips_script_tags(self):
        """_sanitize_output removes script and style blocks from LLM output."""
        gen = LLMGenerator(llm=MagicMock(), retriever=None)

        dirty = 'Safe answer <script>alert("xss")</script> more text'
        clean = gen._sanitize_output(dirty)
        self.assertNotIn("<script>", clean)
        self.assertNotIn("alert(", clean)
        self.assertIn("Safe answer", clean)
        self.assertIn("more text", clean)

        with_style = "Answer <style>body{color:red}</style> end"
        clean2 = gen._sanitize_output(with_style)
        self.assertNotIn("<style>", clean2)
        self.assertIn("Answer", clean2)
        self.assertIn("end", clean2)

    def test_sanitize_output_preserves_prose_with_angle_brackets(self):
        """_sanitize_output does not strip non-script/style angle brackets."""
        gen = LLMGenerator(llm=MagicMock(), retriever=None)

        text = "Use <input type='text'> in your HTML form."
        result = gen._sanitize_output(text)
        self.assertIn("input", result)

    def test_sanitize_output_empty_string(self):
        """_sanitize_output handles empty input gracefully."""
        gen = LLMGenerator(llm=MagicMock(), retriever=None)
        self.assertEqual(gen._sanitize_output(""), "")


if __name__ == "__main__":
    unittest.main()
