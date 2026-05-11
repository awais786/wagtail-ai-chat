"""
Tests for the unified 'rag' management command (index / chat / test subcommands).
"""

import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase


def _make_chatbot(answer="Test answer", sources=None):
    """Return a mock chatbot whose .query() returns a canned result."""
    if sources is None:
        sources = [
            {"content": "snippet", "metadata": {"title": "Page A", "section": "body"}}
        ]
    chatbot = MagicMock()
    chatbot.query.return_value = {"answer": answer, "sources": sources}
    return chatbot


# ---------------------------------------------------------------------------
# rag index
# ---------------------------------------------------------------------------


class TestRagIndexSubcommand(TestCase):
    """Tests for: python manage.py rag index"""

    @patch("wagtail_rag.management.commands.rag.build_rag_index")
    def test_default_call(self, mock_build):
        """Default call: reset_only=False, page_id=None, stdout callable forwarded."""
        call_command("rag", "index", stdout=StringIO())
        mock_build.assert_called_once()
        kwargs = mock_build.call_args[1]
        self.assertFalse(kwargs["reset_only"])
        self.assertIsNone(kwargs["page_id"])
        self.assertTrue(callable(kwargs["stdout"]))

    @patch("wagtail_rag.management.commands.rag.build_rag_index")
    def test_clear_flag(self, mock_build):
        """--clear is forwarded correctly."""
        call_command("rag", "index", "--clear", stdout=StringIO())
        self.assertTrue(mock_build.call_args[1]["reset_only"])

    @patch("wagtail_rag.management.commands.rag.build_rag_index")
    def test_page_id_flag(self, mock_build):
        """--page-id is forwarded as an integer."""
        call_command("rag", "index", "--page-id", "42", stdout=StringIO())
        self.assertEqual(mock_build.call_args[1]["page_id"], 42)


# ---------------------------------------------------------------------------
# rag chat
# ---------------------------------------------------------------------------


class TestRagChatSubcommand(TestCase):
    """Tests for: python manage.py rag chat"""

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_single_question(self, mock_get_chatbot):
        """-q sends the question to chatbot and prints the answer."""
        mock_get_chatbot.return_value = _make_chatbot(answer="Bread is baked.")
        out = StringIO()
        call_command("rag", "chat", "-q", "What is bread?", "--no-sources", stdout=out)
        self.assertIn("Bread is baked.", out.getvalue())
        self.assertEqual(
            mock_get_chatbot.return_value.query.call_args[0][0], "What is bread?"
        )

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_search_only_flag_forwarded(self, mock_get_chatbot):
        """--search-only is passed as search_only=True to chatbot.query()."""
        chatbot = _make_chatbot()
        mock_get_chatbot.return_value = chatbot
        call_command(
            "rag",
            "chat",
            "-q",
            "Hi",
            "--search-only",
            "--no-sources",
            stdout=StringIO(),
        )
        self.assertTrue(chatbot.query.call_args[1]["search_only"])

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_valid_filter_forwarded(self, mock_get_chatbot):
        """--filter JSON is parsed and passed as metadata_filter to get_chatbot()."""
        mock_get_chatbot.return_value = _make_chatbot()
        call_command(
            "rag",
            "chat",
            "-q",
            "Hi",
            "--filter",
            '{"model": "BlogPage"}',
            "--no-sources",
            stdout=StringIO(),
        )
        self.assertEqual(
            mock_get_chatbot.call_args[1]["metadata_filter"], {"model": "BlogPage"}
        )

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_invalid_filter_exits(self, mock_get_chatbot):
        """Bad --filter (invalid JSON or non-dict) raises SystemExit."""
        mock_get_chatbot.return_value = _make_chatbot()
        with self.assertRaises(SystemExit):
            call_command(
                "rag", "chat", "-q", "Hi", "--filter", "not-json", stdout=StringIO()
            )
        with self.assertRaises(SystemExit):
            call_command(
                "rag", "chat", "-q", "Hi", "--filter", '["a","b"]', stdout=StringIO()
            )

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_sources_visibility(self, mock_get_chatbot):
        """Sources shown by default; hidden with --no-sources."""
        mock_get_chatbot.return_value = _make_chatbot(
            sources=[{"content": "t", "metadata": {"title": "MyPage"}}]
        )
        out = StringIO()
        call_command("rag", "chat", "-q", "Hi", stdout=out)
        self.assertIn("MyPage", out.getvalue())

        out2 = StringIO()
        call_command("rag", "chat", "-q", "Hi", "--no-sources", stdout=out2)
        self.assertNotIn("MyPage", out2.getvalue())

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_chatbot_init_error(self, mock_get_chatbot):
        """If get_chatbot() raises, an error message is printed (no crash)."""
        mock_get_chatbot.side_effect = RuntimeError("no model found")
        out = StringIO()
        call_command("rag", "chat", "-q", "Hi", stdout=out)
        self.assertIn("Failed to initialize chatbot", out.getvalue())


# ---------------------------------------------------------------------------
# rag test
# ---------------------------------------------------------------------------


class TestRagTestSubcommand(TestCase):
    """Tests for: python manage.py rag test"""

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_pass_and_fail_summary(self, mock_get_chatbot):
        """Correct pass/total count for all-pass and mixed results."""
        # All pass
        mock_get_chatbot.return_value = _make_chatbot()
        out = StringIO()
        call_command("rag", "test", "--questions", "Q1", "Q2", stdout=out)
        self.assertIn("2/2 passed", out.getvalue())

        # Mixed: Q1 passes, Q2 fails (no sources)
        chatbot = MagicMock()
        chatbot.query.side_effect = [
            {"answer": "ok", "sources": [{"content": "x", "metadata": {"title": "P"}}]},
            {"answer": "ok", "sources": []},
        ]
        mock_get_chatbot.return_value = chatbot
        out2 = StringIO()
        call_command("rag", "test", "--questions", "Q1", "Q2", stdout=out2)
        self.assertIn("1/2 passed", out2.getvalue())

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_fail_conditions(self, mock_get_chatbot):
        """Empty answer or zero sources each count as FAIL."""
        for result in [
            {"answer": "", "sources": [{"content": "x", "metadata": {"title": "P"}}]},
            {"answer": "ok", "sources": []},
        ]:
            chatbot = MagicMock()
            chatbot.query.return_value = result
            mock_get_chatbot.return_value = chatbot
            out = StringIO()
            call_command("rag", "test", "--questions", "Q1", stdout=out)
            self.assertIn("0/1 passed", out.getvalue())

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_search_only_mode(self, mock_get_chatbot):
        """search-only: PASS needs sources only; no sources → FAIL."""
        chatbot = MagicMock()
        chatbot.query.side_effect = [
            {"answer": None, "sources": [{"content": "x", "metadata": {"title": "P"}}]},
            {"answer": None, "sources": []},
        ]
        mock_get_chatbot.return_value = chatbot
        out = StringIO()
        call_command(
            "rag", "test", "--questions", "Q1", "Q2", "--search-only", stdout=out
        )
        self.assertIn("1/2 passed", out.getvalue())

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_query_exception_is_fail(self, mock_get_chatbot):
        """Exception from chatbot.query() counts as failed."""
        chatbot = MagicMock()
        chatbot.query.side_effect = RuntimeError("timeout")
        mock_get_chatbot.return_value = chatbot
        out = StringIO()
        call_command("rag", "test", "--questions", "Q1", stdout=out)
        self.assertIn("0/1 passed", out.getvalue())

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_chatbot_init_failure_aborts(self, mock_get_chatbot):
        """If chatbot init fails, an error is printed and nothing is queried."""
        mock_get_chatbot.side_effect = RuntimeError("no embedding model")
        out = StringIO()
        call_command("rag", "test", "--questions", "Q1", stdout=out)
        self.assertIn("FAILED to initialize chatbot", out.getvalue())

    @patch("wagtail_rag.management.commands.rag.get_chatbot")
    def test_filter_forwarded(self, mock_get_chatbot):
        """--filter is parsed and passed as metadata_filter to get_chatbot()."""
        mock_get_chatbot.return_value = _make_chatbot()
        call_command(
            "rag",
            "test",
            "--questions",
            "Q1",
            "--filter",
            '{"model": "BreadPage"}',
            stdout=StringIO(),
        )
        self.assertEqual(
            mock_get_chatbot.call_args[1]["metadata_filter"], {"model": "BreadPage"}
        )


if __name__ == "__main__":
    unittest.main()
