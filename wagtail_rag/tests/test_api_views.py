"""
Tests for API views.
"""

import json
from unittest.mock import MagicMock, patch

from django.test import TestCase, RequestFactory

from wagtail_rag.views import rag_chat_api


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

        response = rag_chat_api(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["answer"], "Test answer")
        self.assertIn("sources", data)

    def test_chat_api_validation(self):
        """Test chat API validates input correctly."""
        # Missing question
        request = self.factory.post(
            "/api/rag/chat/", data=json.dumps({}), content_type="application/json"
        )
        response = rag_chat_api(request)
        self.assertEqual(response.status_code, 400)

        # Empty question
        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps({"question": "   "}),
            content_type="application/json",
        )
        response = rag_chat_api(request)
        self.assertEqual(response.status_code, 400)

        # Invalid JSON
        request = self.factory.post(
            "/api/rag/chat/", data="invalid json", content_type="application/json"
        )
        response = rag_chat_api(request)
        self.assertEqual(response.status_code, 400)

    @patch("wagtail_rag.views.get_chatbot")
    def test_chat_api_with_session_and_filter(self, mock_get_chatbot):
        """Test chat API with session_id and metadata filter."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "Test answer", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps(
                {
                    "question": "test question",
                    "session_id": "test-session-123",
                    "filter": {"model": "BlogPage"},
                }
            ),
            content_type="application/json",
        )

        response = rag_chat_api(request)

        self.assertEqual(response.status_code, 200)
        # Verify session_id passed to query
        call_kwargs = mock_chatbot.query.call_args[1]
        self.assertEqual(call_kwargs.get("session_id"), "test-session-123")
        # Verify metadata_filter passed to chatbot
        chatbot_kwargs = mock_get_chatbot.call_args[1]
        self.assertEqual(chatbot_kwargs.get("metadata_filter"), {"model": "BlogPage"})

    @patch("wagtail_rag.views.get_chatbot")
    def test_chat_api_error_handling(self, mock_get_chatbot):
        """Test chat API handles exceptions from chatbot."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.side_effect = Exception("Test error")
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps({"question": "test question"}),
            content_type="application/json",
        )

        response = rag_chat_api(request)

        self.assertEqual(response.status_code, 500)
        data = json.loads(response.content)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
