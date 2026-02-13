"""
Tests for embedding and LLM provider factories.
"""

import unittest
from unittest.mock import MagicMock, patch

from wagtail_rag.embeddings.providers import EmbeddingProviderFactory, PROVIDER_DEFAULTS
from wagtail_rag.llm_providers.providers import LLMProviderFactory


class TestEmbeddingProviderFactory(unittest.TestCase):
    """Test embedding provider factory functionality."""

    def setUp(self):
        self.mock_settings = MagicMock()
        self.factory = EmbeddingProviderFactory(self.mock_settings)

    def test_resolve_model_name_explicit(self):
        """Test that explicit model_name takes precedence."""
        result = self.factory._resolve_model_name("openai", "custom-model")
        self.assertEqual(result, "custom-model")

    def test_resolve_model_name_global_setting(self):
        """Test fallback to global WAGTAIL_RAG_EMBEDDING_MODEL setting."""
        self.mock_settings.WAGTAIL_RAG_EMBEDDING_MODEL = "text-embedding-ada-002"
        self.mock_settings.WAGTAIL_RAG_OPENAI_EMBEDDING_MODEL = None
        result = self.factory._resolve_model_name("openai", None)
        # Should use setting if compatible
        self.assertEqual(result, "text-embedding-ada-002")

    def test_resolve_model_name_provider_specific(self):
        """Test provider-specific setting takes precedence over global."""
        self.mock_settings.WAGTAIL_RAG_EMBEDDING_MODEL = "global-model"
        self.mock_settings.WAGTAIL_RAG_OPENAI_EMBEDDING_MODEL = "openai-specific"
        result = self.factory._resolve_setting_model_name("openai")
        self.assertEqual(result, "openai-specific")

    def test_resolve_model_name_incompatible_fallback(self):
        """Test that incompatible model in settings falls back to provider default."""
        self.mock_settings.WAGTAIL_RAG_EMBEDDING_MODEL = "incompatible-model"
        self.mock_settings.WAGTAIL_RAG_OPENAI_EMBEDDING_MODEL = None
        result = self.factory._resolve_model_name("openai", None)
        # Should fall back to provider default when model is incompatible
        self.assertEqual(result, PROVIDER_DEFAULTS["openai"])

    def test_resolve_model_name_default(self):
        """Test fallback to provider default."""
        self.mock_settings.WAGTAIL_RAG_EMBEDDING_MODEL = None
        self.mock_settings.WAGTAIL_RAG_OPENAI_EMBEDDING_MODEL = None
        result = self.factory._resolve_model_name("openai", None)
        self.assertEqual(result, PROVIDER_DEFAULTS["openai"])

    def test_compatibility_check_openai_model(self):
        """Test OpenAI embedding model compatibility detection."""
        self.assertTrue(
            self.factory._is_model_compatible("openai", "text-embedding-3-small")
        )
        self.assertTrue(
            self.factory._is_model_compatible("openai", "text-embedding-ada-002")
        )
        self.assertFalse(
            self.factory._is_model_compatible(
                "openai", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    def test_compatibility_check_huggingface_model(self):
        """Test HuggingFace embedding model compatibility detection."""
        self.assertTrue(
            self.factory._is_model_compatible(
                "huggingface", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        self.assertFalse(
            self.factory._is_model_compatible("huggingface", "text-embedding-3-small")
        )

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises clear error."""
        with self.assertRaises(ValueError) as context:
            self.factory.get(provider="invalid-provider")
        self.assertIn("Unknown embedding provider", str(context.exception))


class TestLLMProviderFactory(unittest.TestCase):
    """Test LLM provider factory functionality."""

    def setUp(self):
        self.mock_settings = MagicMock()
        self.factory = LLMProviderFactory(self.mock_settings)

    def test_resolve_model_name_explicit(self):
        """Test that explicit model_name takes precedence."""
        result = self.factory._resolve_model_name("openai", "custom-model")
        self.assertEqual(result, "custom-model")

    def test_provider_specific_setting_priority(self):
        """Test provider-specific setting takes precedence."""
        self.mock_settings.WAGTAIL_RAG_MODEL_NAME = "global-model"
        self.mock_settings.WAGTAIL_RAG_OPENAI_MODEL_NAME = "openai-specific"
        result = self.factory._resolve_setting_model_name("openai")
        self.assertEqual(result, "openai-specific")

    def test_compatibility_check_openai(self):
        """Test OpenAI model compatibility detection."""
        self.assertTrue(self.factory._is_model_compatible("openai", "gpt-4"))
        self.assertTrue(self.factory._is_model_compatible("openai", "gpt-3.5-turbo"))
        self.assertFalse(self.factory._is_model_compatible("openai", "claude-3-sonnet"))

    def test_compatibility_check_anthropic(self):
        """Test Anthropic model compatibility detection."""
        self.assertTrue(
            self.factory._is_model_compatible("anthropic", "claude-3-sonnet-20240229")
        )
        self.assertFalse(self.factory._is_model_compatible("anthropic", "gpt-4"))

    def test_compatibility_check_ollama(self):
        """Test Ollama model compatibility detection."""
        self.assertTrue(self.factory._is_model_compatible("ollama", "mistral"))
        self.assertTrue(self.factory._is_model_compatible("ollama", "llama2"))
        self.assertFalse(self.factory._is_model_compatible("ollama", "gpt-4"))
        self.assertFalse(self.factory._is_model_compatible("ollama", "claude-3-sonnet"))

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises clear error."""
        with self.assertRaises(ValueError) as context:
            self.factory.get(provider="invalid-llm-provider")
        self.assertIn("Unknown LLM provider", str(context.exception))


if __name__ == "__main__":
    unittest.main()
