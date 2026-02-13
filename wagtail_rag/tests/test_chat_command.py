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
        call_args = mock_chatbot.query.call_args
        self.assertEqual(call_args[0][0], "test question")

        # Verify output contains answer
        output = out.getvalue()
        self.assertIn("Test answer", output)

    @patch("wagtail_rag.management.commands.chat.get_chatbot")
    def test_filter_parsing_valid_json(self, mock_get_chatbot):
        """Test that valid JSON filter is parsed correctly."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "Test", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        out = StringIO()
        call_command(
            "chat",
            "-q",
            "test",
            "--filter",
            '{"model": "BlogPage"}',
            "--no-sources",
            stdout=out,
        )

        # Verify chatbot was initialized with metadata_filter
        call_args = mock_get_chatbot.call_args
        self.assertEqual(call_args[1]["metadata_filter"], {"model": "BlogPage"})

    @patch("wagtail_rag.management.commands.chat.get_chatbot")
    def test_filter_parsing_invalid_json(self, mock_get_chatbot):
        """Test that invalid JSON filter is rejected."""
        out = StringIO()
        with self.assertRaises(SystemExit):
            call_command("chat", "-q", "test", "--filter", "invalid-json", stdout=out)

    @patch("wagtail_rag.management.commands.chat.get_chatbot")
    def test_filter_parsing_non_dict(self, mock_get_chatbot):
        """Test that non-dict JSON filter is rejected."""
        out = StringIO()
        with self.assertRaises(SystemExit):
            call_command(
                "chat", "-q", "test", "--filter", '["not", "a", "dict"]', stdout=out
            )


class TestChatCommandHelpers(unittest.TestCase):
    """Test chat command helper functions."""

    def test_exit_commands_constant(self):
        """Test EXIT_COMMANDS constant contains expected values."""
        from wagtail_rag.management.commands.chat import EXIT_COMMANDS

        self.assertIn("exit", EXIT_COMMANDS)
        self.assertIn("quit", EXIT_COMMANDS)

    def test_sources_usage_constant(self):
        """Test SOURCES_USAGE constant is defined."""
        from wagtail_rag.management.commands.chat import SOURCES_USAGE

        self.assertIn("sources", SOURCES_USAGE.lower())


if __name__ == "__main__":
    unittest.main()
