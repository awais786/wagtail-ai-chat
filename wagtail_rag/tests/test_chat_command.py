"""
Tests for the 'rag chat' subcommand (formerly the standalone 'chat' command).

These tests exercise the chat path of the unified management command.
For full coverage of all three subcommands see test_rag_command.py.
"""

import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase


class TestRagChatCompat(TestCase):
    """Regression tests kept from the old standalone chat command."""

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_single_question_mode(self, mock_get_chatbot):
        """chat subcommand with -q shows the answer."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "Test answer", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        out = StringIO()
        call_command("rag", "chat", "-q", "test question", "--no-sources", stdout=out)

        mock_chatbot.query.assert_called_once()
        self.assertEqual(mock_chatbot.query.call_args[0][0], "test question")
        self.assertIn("Test answer", out.getvalue())

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_filter_parsing(self, mock_get_chatbot):
        """Valid JSON filter is forwarded; invalid JSON raises SystemExit."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "Test", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        # Valid filter
        out = StringIO()
        call_command(
            "rag",
            "chat",
            "-q",
            "test",
            "--filter",
            '{"model": "BlogPage"}',
            "--no-sources",
            stdout=out,
        )
        self.assertEqual(
            mock_get_chatbot.call_args[1]["metadata_filter"], {"model": "BlogPage"}
        )

        # Invalid JSON
        with self.assertRaises(SystemExit):
            call_command(
                "rag",
                "chat",
                "-q",
                "test",
                "--filter",
                "invalid-json",
                stdout=StringIO(),
            )

        # Non-dict JSON
        with self.assertRaises(SystemExit):
            call_command(
                "rag",
                "chat",
                "-q",
                "test",
                "--filter",
                '["not","a","dict"]',
                stdout=StringIO(),
            )


if __name__ == "__main__":
    unittest.main()
