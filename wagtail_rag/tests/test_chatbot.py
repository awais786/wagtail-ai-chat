"""
Tests for RAGChatBot — source deduplication and history-enriched retrieval.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def _make_doc(page_id, title, content="some content"):
    doc = MagicMock(spec=Document)
    doc.page_content = content
    doc.metadata = {"page_id": page_id, "title": title, "section": "body"}
    return doc


class TestFormatSources(unittest.TestCase):
    """RAGChatBot._format_sources deduplicates by page_id."""

    def _get_chatbot(self):
        with patch("wagtail_rag.chatbot.get_embeddings"), patch(
            "wagtail_rag.chatbot.get_llm"
        ), patch("wagtail_rag.chatbot.RAGChatBot._create_vectorstore"), patch(
            "wagtail_rag.chatbot.RAGChatBot._create_retriever"
        ), patch(
            "wagtail_rag.chatbot.EmbeddingSearcher"
        ), patch(
            "wagtail_rag.chatbot.LLMGenerator"
        ):
            from wagtail_rag.chatbot import RAGChatBot

            return RAGChatBot.__new__(RAGChatBot)

    def test_deduplicates_same_page(self):
        bot = self._get_chatbot()
        docs = [
            _make_doc(1, "Sourdough", "chunk 1"),
            _make_doc(1, "Sourdough", "chunk 2"),
            _make_doc(1, "Sourdough", "chunk 3"),
        ]
        sources = bot._format_sources(docs)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["metadata"]["title"], "Sourdough")

    def test_keeps_distinct_pages(self):
        bot = self._get_chatbot()
        docs = [
            _make_doc(1, "Sourdough", "chunk 1"),
            _make_doc(2, "Cornbread", "chunk 1"),
            _make_doc(1, "Sourdough", "chunk 2"),
        ]
        sources = bot._format_sources(docs)
        self.assertEqual(len(sources), 2)
        titles = {s["metadata"]["title"] for s in sources}
        self.assertEqual(titles, {"Sourdough", "Cornbread"})

    def test_limits_to_max_sources(self):
        bot = self._get_chatbot()
        docs = [_make_doc(i, f"Page {i}") for i in range(10)]
        sources = bot._format_sources(docs)
        self.assertLessEqual(len(sources), 3)

    def test_empty_docs(self):
        bot = self._get_chatbot()
        self.assertEqual(bot._format_sources([]), [])

    def test_fallback_to_title_when_no_page_id(self):
        bot = self._get_chatbot()
        doc1 = MagicMock(spec=Document)
        doc1.page_content = "text"
        doc1.metadata = {"title": "Rye Bread"}
        doc2 = MagicMock(spec=Document)
        doc2.page_content = "text"
        doc2.metadata = {"title": "Rye Bread"}
        sources = bot._format_sources([doc1, doc2])
        self.assertEqual(len(sources), 1)


class TestBuildRetrievalQuery(unittest.TestCase):
    """RAGChatBot._build_retrieval_query enriches vague follow-up questions."""

    def _get_chatbot(self):
        with patch("wagtail_rag.chatbot.get_embeddings"), patch(
            "wagtail_rag.chatbot.get_llm"
        ), patch("wagtail_rag.chatbot.RAGChatBot._create_vectorstore"), patch(
            "wagtail_rag.chatbot.RAGChatBot._create_retriever"
        ), patch(
            "wagtail_rag.chatbot.EmbeddingSearcher"
        ), patch(
            "wagtail_rag.chatbot.LLMGenerator"
        ):
            from wagtail_rag.chatbot import RAGChatBot

            bot = RAGChatBot.__new__(RAGChatBot)
            bot.llm_generator = MagicMock()
            return bot

    def test_returns_raw_question_when_no_history(self):
        bot = self._get_chatbot()
        mock_history = MagicMock()
        mock_history.messages = []
        bot.llm_generator.history_store.get_session_history.return_value = mock_history

        result = bot._build_retrieval_query("do you know the ingredients?", "session-1")
        self.assertEqual(result, "do you know the ingredients?")

    def test_prepends_last_human_turn(self):
        bot = self._get_chatbot()
        human_msg = MagicMock()
        human_msg.type = "human"
        human_msg.content = "What is Southern Cornbread?"
        ai_msg = MagicMock()
        ai_msg.type = "ai"
        ai_msg.content = "Southern Cornbread is a hearty variety..."

        mock_history = MagicMock()
        mock_history.messages = [human_msg, ai_msg]
        bot.llm_generator.history_store.get_session_history.return_value = mock_history

        result = bot._build_retrieval_query("do you know the ingredients?", "session-1")
        self.assertIn("What is Southern Cornbread?", result)
        self.assertIn("do you know the ingredients?", result)

    def test_returns_question_on_exception(self):
        bot = self._get_chatbot()
        bot.llm_generator.history_store.get_session_history.side_effect = Exception(
            "fail"
        )

        result = bot._build_retrieval_query("some question", "session-1")
        self.assertEqual(result, "some question")

    def test_returns_question_when_no_history_store(self):
        bot = self._get_chatbot()
        bot.llm_generator.history_store = None

        from wagtail_rag.chatbot import RAGChatBot

        result = RAGChatBot._build_retrieval_query(bot, "some question", "session-1")
        self.assertEqual(result, "some question")


class TestPromptIsStrict(unittest.TestCase):
    """Prompt must instruct the LLM not to use general knowledge."""

    def test_prompt_forbids_general_knowledge(self):
        with patch("wagtail_rag.chatbot.get_embeddings"), patch(
            "wagtail_rag.chatbot.get_llm"
        ), patch("wagtail_rag.chatbot.RAGChatBot._create_vectorstore"), patch(
            "wagtail_rag.chatbot.RAGChatBot._create_retriever"
        ), patch(
            "wagtail_rag.chatbot.EmbeddingSearcher"
        ), patch(
            "wagtail_rag.chatbot.LLMGenerator"
        ):
            from wagtail_rag.llm_providers.generation import LLMGenerator

            gen = LLMGenerator(llm=MagicMock(), retriever=None)

        prompt = gen.prompt_template_str.lower()
        self.assertIn("only", prompt)
        self.assertIn("{context}", gen.prompt_template_str)
        self.assertIn("{question}", gen.prompt_template_str)
        # Injection defence present in prompt
        self.assertIn("ignore", prompt)

        system = gen.system_prompt_str.lower()
        self.assertIn("only", system)
        # Injection defence present in system prompt
        self.assertIn("disregard", system)


if __name__ == "__main__":
    unittest.main()
