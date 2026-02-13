"""
Tests for API views.
"""

import json
from unittest.mock import MagicMock, patch

from django.test import TestCase, RequestFactory
from django.conf import settings

from wagtail_rag.views import chat_api


class TestChatAPI(TestCase):
    """Test chat API view."""

    def setUp(self):
        self.factory = RequestFactory()

    @patch("wagtail_rag.views.get_chatbot")
    def test_chat_api_valid_request(self, mock_get_chatbot):
        """Test chat API with valid request."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {
            "answer": "Test answer",
            "sources": [{"title": "Test Page", "url": "/test/"}],
        }
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps({"question": "test question"}),
            content_type="application/json",
        )

        response = chat_api(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["answer"], "Test answer")
        self.assertIn("sources", data)
        self.assertEqual(len(data["sources"]), 1)

    def test_chat_api_missing_question(self):
        """Test chat API rejects request without question."""
        request = self.factory.post(
            "/api/rag/chat/", data=json.dumps({}), content_type="application/json"
        )

        response = chat_api(request)

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn("error", data)

    def test_chat_api_empty_question(self):
        """Test chat API rejects empty question."""
        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps({"question": "   "}),
            content_type="application/json",
        )

        response = chat_api(request)

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn("error", data)

    def test_chat_api_invalid_json(self):
        """Test chat API handles invalid JSON."""
        request = self.factory.post(
            "/api/rag/chat/", data="invalid json", content_type="application/json"
        )

        response = chat_api(request)

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn("error", data)

    def test_chat_api_get_method_not_allowed(self):
        """Test chat API rejects GET requests."""
        request = self.factory.get("/api/rag/chat/")
        response = chat_api(request)

        self.assertEqual(response.status_code, 405)

    @patch("wagtail_rag.views.get_chatbot")
    def test_chat_api_with_session_id(self, mock_get_chatbot):
        """Test chat API with session_id for history."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "Test answer", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps(
                {"question": "test question", "session_id": "test-session-123"}
            ),
            content_type="application/json",
        )

        response = chat_api(request)

        self.assertEqual(response.status_code, 200)
        # Verify session_id was passed to query
        call_kwargs = mock_chatbot.query.call_args[1]
        self.assertEqual(call_kwargs.get("session_id"), "test-session-123")

    @patch("wagtail_rag.views.get_chatbot")
    def test_chat_api_with_metadata_filter(self, mock_get_chatbot):
        """Test chat API with metadata filter."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "Test answer", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps(
                {"question": "test question", "metadata_filter": {"model": "BlogPage"}}
            ),
            content_type="application/json",
        )

        response = chat_api(request)

        self.assertEqual(response.status_code, 200)
        # Verify metadata_filter was used when creating chatbot
        call_kwargs = mock_get_chatbot.call_args[1]
        self.assertEqual(call_kwargs.get("metadata_filter"), {"model": "BlogPage"})

    @patch("wagtail_rag.views.get_chatbot")
    def test_chat_api_handles_chatbot_exception(self, mock_get_chatbot):
        """Test chat API handles exceptions from chatbot."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.side_effect = Exception("Test error")
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps({"question": "test question"}),
            content_type="application/json",
        )

        response = chat_api(request)

        self.assertEqual(response.status_code, 500)
        data = json.loads(response.content)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
