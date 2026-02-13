"""
Tests for the chat management command.
"""

import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

from django.core.management import call_command
from django.test import TestCase


class TestChatCommand(TestCase):
    """Test chat management command functionality."""

    @patch("wagtail_rag.management.commands.chat.get_chatbot")
    def test_single_question_mode(self, mock_get_chatbot):
        """Test single question mode with -q flag."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "Test answer", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        out = StringIO()
        call_command("chat", "-q", "test question", "--no-sources", stdout=out)

        # Verify chatbot was called with question
        mock_chatbot.query.assert_called_once()
        self.assertEqual(mock_chatbot.query.call_args[0][0], "test question")
        self.assertIn("Test answer", out.getvalue())

    @patch("wagtail_rag.management.commands.chat.get_chatbot")
    def test_filter_parsing(self, mock_get_chatbot):
        """Test filter parsing for valid and invalid JSON."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "Test", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        # Valid JSON filter
        out = StringIO()
        call_command("chat", "-q", "test", "--filter", '{"model": "BlogPage"}', "--no-sources", stdout=out)
        call_args = mock_get_chatbot.call_args
        self.assertEqual(call_args[1]["metadata_filter"], {"model": "BlogPage"})

        # Invalid JSON filter
        out = StringIO()
        with self.assertRaises(SystemExit):
            call_command("chat", "-q", "test", "--filter", "invalid-json", stdout=out)
        
        # Non-dict JSON filter
        out = StringIO()
        with self.assertRaises(SystemExit):
            call_command("chat", "-q", "test", "--filter", '["not", "a", "dict"]', stdout=out)


if __name__ == "__main__":
    unittest.main()
