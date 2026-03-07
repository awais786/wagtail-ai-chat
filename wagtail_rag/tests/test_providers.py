"""
Tests for embedding and LLM provider factories.
"""

import unittest
from unittest.mock import MagicMock, patch

from wagtail_rag.embeddings.providers import (
    EmbeddingProviderFactory,
    SentenceTransformerProvider,
    PROVIDER_DEFAULTS,
)
from wagtail_rag.llm_providers.providers import LLMProviderFactory


class TestEmbeddingProviderFactory(unittest.TestCase):
    """Test embedding provider factory functionality."""

    def setUp(self):
        self.mock_settings = MagicMock()
        self.factory = EmbeddingProviderFactory(self.mock_settings)

    def test_model_name_resolution_priority(self):
        """Test model name resolution priority: explicit > provider-specific > global > default."""
        # Explicit takes precedence
        result = self.factory._resolve_model_name("openai", "custom-model")
        self.assertEqual(result, "custom-model")

        # Provider-specific setting
        self.mock_settings.WAGTAIL_RAG_EMBEDDING_MODEL = "global-model"
        self.mock_settings.WAGTAIL_RAG_OPENAI_EMBEDDING_MODEL = "openai-specific"
        result = self.factory._resolve_setting_model_name("openai")
        self.assertEqual(result, "openai-specific")

        # Incompatible model falls back to default
        self.mock_settings.WAGTAIL_RAG_EMBEDDING_MODEL = "incompatible-model"
        self.mock_settings.WAGTAIL_RAG_OPENAI_EMBEDDING_MODEL = None
        result = self.factory._resolve_model_name("openai", None)
        self.assertEqual(result, PROVIDER_DEFAULTS["openai"])

    def test_model_compatibility_detection(self):
        """Test model compatibility detection for different providers."""
        # OpenAI models
        self.assertTrue(
            self.factory._is_model_compatible("openai", "text-embedding-3-small")
        )
        self.assertFalse(
            self.factory._is_model_compatible(
                "openai", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        # HuggingFace models (repo-style names required)
        self.assertTrue(
            self.factory._is_model_compatible(
                "huggingface", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        self.assertFalse(
            self.factory._is_model_compatible("huggingface", "text-embedding-3-small")
        )

        # sentence-transformers accepts both short and repo-style names
        self.assertTrue(
            self.factory._is_model_compatible(
                "sentence-transformers", "all-MiniLM-L6-v2"
            )
        )
        self.assertTrue(
            self.factory._is_model_compatible(
                "sentence-transformers", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        self.assertFalse(
            self.factory._is_model_compatible(
                "sentence-transformers", "text-embedding-3-small"
            )
        )

    def test_sentence_transformer_provider_is_distinct_from_huggingface(self):
        """sentence-transformers key must map to SentenceTransformerProvider, not HuggingFaceProvider."""
        from wagtail_rag.embeddings.providers import HuggingFaceProvider

        st_cls = EmbeddingProviderFactory.PROVIDER_MAP["sentence-transformers"]
        hf_cls = EmbeddingProviderFactory.PROVIDER_MAP["huggingface"]
        self.assertIs(st_cls, SentenceTransformerProvider)
        self.assertIsNot(st_cls, hf_cls)

    def test_get_embeddings_reads_settings_at_call_time(self):
        """get_embeddings() must not capture settings at import time (no module singleton)."""
        from wagtail_rag.embeddings.providers import get_embeddings

        mock_embedding = MagicMock()
        with patch(
            "wagtail_rag.embeddings.providers.EmbeddingProviderFactory.get",
            return_value=mock_embedding,
        ) as mock_get:
            result = get_embeddings(
                provider="openai", model_name="text-embedding-3-small"
            )
            mock_get.assert_called_once_with(
                provider="openai", model_name="text-embedding-3-small"
            )
            self.assertIs(result, mock_embedding)

    def test_invalid_provider_error(self):
        """Test that invalid provider raises clear error."""
        with self.assertRaises(ValueError) as context:
            self.factory.get(provider="invalid-provider")
        self.assertIn("Unknown embedding provider", str(context.exception))


class TestLLMProviderFactory(unittest.TestCase):
    """Test LLM provider factory functionality."""

    def setUp(self):
        self.mock_settings = MagicMock()
        self.factory = LLMProviderFactory(self.mock_settings)

    def test_model_name_resolution(self):
        """Test model name resolution for LLM providers."""
        # Explicit precedence
        result = self.factory._resolve_model_name("openai", "custom-model")
        self.assertEqual(result, "custom-model")

        # Provider-specific setting
        self.mock_settings.WAGTAIL_RAG_MODEL_NAME = "global-model"
        self.mock_settings.WAGTAIL_RAG_OPENAI_MODEL_NAME = "openai-specific"
        result = self.factory._resolve_setting_model_name("openai")
        self.assertEqual(result, "openai-specific")

    def test_llm_compatibility_checks(self):
        """Test LLM model compatibility for all providers."""
        # OpenAI
        self.assertTrue(self.factory._is_model_compatible("openai", "gpt-4"))
        self.assertFalse(self.factory._is_model_compatible("openai", "claude-3-sonnet"))

        # Anthropic
        self.assertTrue(
            self.factory._is_model_compatible("anthropic", "claude-3-sonnet-20240229")
        )
        self.assertFalse(self.factory._is_model_compatible("anthropic", "gpt-4"))

        # Ollama
        self.assertTrue(self.factory._is_model_compatible("ollama", "mistral"))
        self.assertFalse(self.factory._is_model_compatible("ollama", "gpt-4"))

    def test_invalid_llm_provider_error(self):
        """Test that invalid LLM provider raises clear error."""
        with self.assertRaises(ValueError) as context:
            self.factory.get(provider="invalid-llm-provider")
        self.assertIn("Unknown LLM provider", str(context.exception))


if __name__ == "__main__":
    unittest.main()
