"""
Tests for API views.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from django.test import TestCase, RequestFactory, Client

from wagtail_rag.views import rag_chat_api


class TestChatAPI(TestCase):
    """Test chat API view using RequestFactory (bypasses middleware, including CSRF)."""

    def setUp(self):
        self.factory = RequestFactory()

    @patch("wagtail_rag.views.get_chatbot")
    def test_valid_post_returns_answer(self, mock_get_chatbot):
        """Valid POST with question returns 200 and answer."""
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

    def test_missing_question_returns_400(self):
        """POST without question field returns 400."""
        request = self.factory.post(
            "/api/rag/chat/", data=json.dumps({}), content_type="application/json"
        )
        self.assertEqual(rag_chat_api(request).status_code, 400)

    def test_blank_question_returns_400(self):
        """POST with whitespace-only question returns 400."""
        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps({"question": "   "}),
            content_type="application/json",
        )
        self.assertEqual(rag_chat_api(request).status_code, 400)

    def test_invalid_json_returns_400(self):
        """POST with non-JSON body returns 400."""
        request = self.factory.post(
            "/api/rag/chat/", data="not json", content_type="application/json"
        )
        self.assertEqual(rag_chat_api(request).status_code, 400)

    def test_json_array_body_returns_400(self):
        """POST body that is a JSON array (not object) returns 400."""
        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps(["a", "b"]),
            content_type="application/json",
        )
        self.assertEqual(rag_chat_api(request).status_code, 400)

    def test_body_too_large_returns_413(self):
        """POST body exceeding limit returns 413."""
        big = json.dumps({"question": "hi", "extra": "x" * (1024 * 1024 + 1)})
        request = self.factory.post(
            "/api/rag/chat/", data=big, content_type="application/json"
        )
        self.assertEqual(rag_chat_api(request).status_code, 413)

    @patch("wagtail_rag.views.get_chatbot")
    def test_session_and_filter_forwarded(self, mock_get_chatbot):
        """session_id and valid dict filter are forwarded correctly."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "ok", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps(
                {
                    "question": "test",
                    "session_id": "sess-abc",
                    "filter": {"model": "BlogPage"},
                }
            ),
            content_type="application/json",
        )
        response = rag_chat_api(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_chatbot.query.call_args[1]["session_id"], "sess-abc")
        self.assertEqual(
            mock_get_chatbot.call_args[1]["metadata_filter"], {"model": "BlogPage"}
        )

    @patch("wagtail_rag.views.get_chatbot")
    def test_llm_kwargs_sanitised(self, mock_get_chatbot):
        """Only whitelisted llm_kwargs keys are forwarded; others are stripped."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "ok", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps(
                {
                    "question": "hi",
                    "llm_kwargs": {
                        "temperature": 0.5,  # allowed
                        "evil_param": "hack",  # should be stripped
                    },
                }
            ),
            content_type="application/json",
        )
        rag_chat_api(request)

        forwarded = mock_get_chatbot.call_args[1]["llm_kwargs"]
        self.assertIn("temperature", forwarded)
        self.assertNotIn("evil_param", forwarded)

    @patch("wagtail_rag.views.get_chatbot")
    def test_non_dict_filter_ignored(self, mock_get_chatbot):
        """A filter value that is not a dict is silently ignored (treated as no filter)."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "ok", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps({"question": "hi", "filter": ["not", "a", "dict"]}),
            content_type="application/json",
        )
        rag_chat_api(request)

        self.assertIsNone(mock_get_chatbot.call_args[1]["metadata_filter"])

    @patch("wagtail_rag.views.get_chatbot")
    def test_chatbot_exception_returns_500(self, mock_get_chatbot):
        """Unhandled exception from chatbot returns 500 with generic message."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.side_effect = Exception("boom")
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.post(
            "/api/rag/chat/",
            data=json.dumps({"question": "test"}),
            content_type="application/json",
        )
        response = rag_chat_api(request)

        self.assertEqual(response.status_code, 500)
        self.assertIn("error", json.loads(response.content))

    @patch("wagtail_rag.views.get_chatbot")
    def test_get_request_accepted(self, mock_get_chatbot):
        """GET with ?q= parameter returns 200."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "ok", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        request = self.factory.get("/api/rag/chat/", {"q": "hello"})
        response = rag_chat_api(request)
        self.assertEqual(response.status_code, 200)


class TestChatAPICSRF(TestCase):
    """Integration tests verifying CSRF enforcement via the Django test Client."""

    @patch("wagtail_rag.views.get_chatbot")
    def test_post_without_csrf_token_rejected(self, mock_get_chatbot):
        """POST without CSRF token is rejected with 403 when enforcement is active."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "ok", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        # enforce_csrf_checks=True simulates a real browser request without the token
        client = Client(enforce_csrf_checks=True)
        response = client.post(
            "/api/rag/chat/",
            data=json.dumps({"question": "hello"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

    @patch("wagtail_rag.views.get_chatbot")
    def test_post_with_csrf_token_accepted(self, mock_get_chatbot):
        """POST with valid CSRF token is accepted."""
        mock_chatbot = MagicMock()
        mock_chatbot.query.return_value = {"answer": "ok", "sources": []}
        mock_get_chatbot.return_value = mock_chatbot

        client = Client(enforce_csrf_checks=True)
        # Fetch CSRF token by making a GET first (ensure_csrf_cookie sets the cookie)
        client.get("/api/rag/chat/", {"q": "warmup"})
        csrf_token = client.cookies.get("csrftoken")
        self.assertIsNotNone(csrf_token, "csrftoken cookie should be set after GET")

        response = client.post(
            "/api/rag/chat/",
            data=json.dumps({"question": "hello"}),
            content_type="application/json",
            HTTP_X_CSRFTOKEN=csrf_token.value,
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
