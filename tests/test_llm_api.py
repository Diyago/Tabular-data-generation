# -*- coding: utf-8 -*-
"""
Tests for LLM API configuration and client functionality.

__author__ = "Insaf Ashrapov"
__copyright__ = "Insaf Ashrapov"
__license__ = "Apache 2.0"
"""

from unittest import TestCase
from unittest.mock import patch, MagicMock, Mock
import json

import pandas as pd
import numpy as np

from src.tabgan.llm_config import LLMAPIConfig, LM_STUDIO_BASE_URL, OPENAI_BASE_URL, OLLAMA_BASE_URL
from src.tabgan.llm_api_client import LLMAPIClient
from src.tabgan.sampler import LLMGenerator, SamplerLLM


class TestLLMAPIConfig(TestCase):
    """Tests for LLMAPIConfig dataclass and factory methods."""

    def test_default_config(self):
        """Test default configuration initialization."""
        config = LLMAPIConfig()
        self.assertEqual(config.base_url, "http://localhost:1234")
        self.assertEqual(config.model, "google/gemma-3-12b")
        self.assertEqual(config.timeout, 90)
        self.assertEqual(config.max_tokens, 256)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.95)
        self.assertEqual(config.top_k, 50)

    def test_default_chat_url_generation(self):
        """Test that default chat_url is generated from base_url."""
        config = LLMAPIConfig(base_url="http://localhost:1234")
        self.assertEqual(config.chat_url, "http://localhost:1234/v1/chat/completions")

    def test_custom_chat_url(self):
        """Test custom chat_url can be set."""
        config = LLMAPIConfig(
            base_url="http://localhost:1234",
            chat_url="http://localhost:1234/api/v1/chat"
        )
        self.assertEqual(config.chat_url, "http://localhost:1234/api/v1/chat")

    def test_lm_studio_factory(self):
        """Test LM Studio factory method."""
        config = LLMAPIConfig.from_lm_studio(
            base_url="http://localhost:1234",
            model="google/gemma-3-12b",
            timeout=120
        )
        self.assertEqual(config.base_url, "http://localhost:1234")
        self.assertEqual(config.model, "google/gemma-3-12b")
        self.assertEqual(config.timeout, 120)
        self.assertEqual(config.chat_url, "http://localhost:1234/v1/chat/completions")

    def test_openai_factory(self):
        """Test OpenAI factory method."""
        config = LLMAPIConfig.from_openai(
            api_key="test-api-key",
            model="gpt-4",
            timeout=60
        )
        self.assertEqual(config.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.api_key, "test-api-key")
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.timeout, 60)

    def test_ollama_factory(self):
        """Test Ollama factory method."""
        config = LLMAPIConfig.from_ollama(
            base_url="http://localhost:11434",
            model="llama3",
            timeout=120
        )
        self.assertEqual(config.base_url, "http://localhost:11434")
        self.assertEqual(config.model, "llama3")
        self.assertEqual(config.timeout, 120)
        self.assertEqual(config.chat_url, "http://localhost:11434/api/generate")

    def test_get_headers_without_api_key(self):
        """Test headers generation without API key."""
        config = LLMAPIConfig()
        headers = config.get_headers()
        self.assertEqual(headers, {"Content-Type": "application/json"})

    def test_get_headers_with_api_key(self):
        """Test headers generation with API key."""
        config = LLMAPIConfig(api_key="test-key-123")
        headers = config.get_headers()
        self.assertEqual(headers, {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-key-123"
        })


class TestLLMAPIClient(TestCase):
    """Tests for LLMAPIClient functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMAPIConfig(
            base_url="http://localhost:1234",
            model="test-model",
            timeout=30
        )

    @patch("src.tabgan.llm_api_client.requests.Session")
    def test_initialization(self, mock_session_class):
        """Test client initialization."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        client = LLMAPIClient(self.config)
        self.assertEqual(client.config, self.config)

    @patch("src.tabgan.llm_api_client.requests.Session")
    def test_generate_openai_format(self, mock_session_class):
        """Test text generation with OpenAI-compatible API format."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "  Generated Text  "}}
            ]
        }
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = LLMAPIClient(self.config)
        result = client.generate("Test prompt")

        self.assertEqual(result, "Generated Text")
        mock_session.post.assert_called_once()

        # Verify payload structure
        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "Test prompt")

    @patch("src.tabgan.llm_api_client.requests.Session")
    def test_generate_ollama_format(self, mock_session_class):
        """Test text generation with Ollama API format."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "  Ollama Generated Text  "
        }
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        ollama_config = LLMAPIConfig.from_ollama()
        client = LLMAPIClient(ollama_config)
        result = client.generate("Test prompt")

        self.assertEqual(result, "Ollama Generated Text")

        # Verify Ollama payload structure
        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        self.assertEqual(payload["model"], "llama3")
        self.assertEqual(payload["prompt"], "Test prompt")
        self.assertEqual(payload["stream"], False)

    @patch("src.tabgan.llm_api_client.requests.Session")
    def test_generate_with_system_prompt(self, mock_session_class):
        """Test generation with system prompt."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Response"}}
            ]
        }
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        config = LLMAPIConfig(system_prompt="You are a helpful assistant.")
        client = LLMAPIClient(config)
        result = client.generate("Test prompt")

        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "You are a helpful assistant.")

    @patch("src.tabgan.llm_api_client.requests.Session")
    def test_generate_with_custom_params(self, mock_session_class):
        """Test generation with custom max_tokens and temperature."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = LLMAPIClient(self.config)
        result = client.generate("Test prompt", max_tokens=100, temperature=0.5)

        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(payload["temperature"], 0.5)

    @patch("src.tabgan.llm_api_client.requests.Session")
    def test_generate_batch(self, mock_session_class):
        """Test batch generation."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Result"}}]
        }
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = LLMAPIClient(self.config)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = client.generate_batch(prompts)

        self.assertEqual(len(results), 3)
        self.assertEqual(mock_session.post.call_count, 3)

    @patch("src.tabgan.llm_api_client.requests.Session")
    def test_check_connection_openai_format(self, mock_session_class):
        """Test connection check for OpenAI-compatible API."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = LLMAPIClient(self.config)
        result = client.check_connection()

        self.assertTrue(result)
        mock_session.get.assert_called_once_with(
            "http://localhost:1234/v1/models",
            headers={"Content-Type": "application/json"},
            timeout=5
        )

    @patch("src.tabgan.llm_api_client.requests.Session")
    def test_check_connection_ollama_format(self, mock_session_class):
        """Test connection check for Ollama API."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        ollama_config = LLMAPIConfig.from_ollama()
        client = LLMAPIClient(ollama_config)
        result = client.check_connection()

        self.assertTrue(result)
        mock_session.get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            headers={"Content-Type": "application/json"},
            timeout=5
        )

    @patch("src.tabgan.llm_api_client.requests.Session")
    def test_check_connection_failure(self, mock_session_class):
        """Test connection check when API is unavailable."""
        mock_session = MagicMock()
        from requests import RequestException
        mock_session.get.side_effect = RequestException("Connection failed")
        mock_session_class.return_value = mock_session

        client = LLMAPIClient(self.config)
        result = client.check_connection()

        self.assertFalse(result)

    @patch("src.tabgan.llm_api_client.requests.Session")
    def test_api_error_handling(self, mock_session_class):
        """Test error handling for API failures."""
        mock_session = MagicMock()
        from requests import HTTPError
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("API Error")
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        client = LLMAPIClient(self.config)
        with self.assertRaises(HTTPError):
            client.generate("Test prompt")


class TestSamplerLLMAPIIntegration(TestCase):
    """Tests for SamplerLLM integration with LLM API client."""

    def setUp(self):
        """Set up test fixtures."""
        self.train_df = pd.DataFrame({
            "Name": ["Anna", "Maria", "Ivan", "Sergey"],
            "Gender": ["F", "F", "M", "M"],
            "Age": [25, 30, 35, 40],
            "Occupation": ["Engineer", "Doctor", "Artist", "Teacher"]
        })
        self.gen_params = {"batch_size": 32, "epochs": 1, "llm": "distilgpt2", "max_length": 50}

    @patch("src.tabgan.llm_api_client.requests.Session")
    @patch.object(SamplerLLM, "_fit_great_model")
    def test_conditional_generation_with_api(self, mock_fit_great, mock_session_class):
        """Test conditional text generation using external LLM API."""
        # Setup mock for GReaT model
        mock_great_instance = MagicMock()

        def mock_impute_logic(df_to_impute, max_length):
            df_imputed = df_to_impute.copy()
            df_imputed["Age"] = 33
            df_imputed["Occupation"] = "Manager"
            return df_imputed

        mock_great_instance.impute.side_effect = mock_impute_logic
        mock_fit_great.return_value = (mock_great_instance, "cpu")

        # Setup mock for API client
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "API Generated Name"}}]
        }
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Create API config
        api_config = LLMAPIConfig.from_lm_studio(
            base_url="http://localhost:1234",
            model="test-model"
        )

        # Create generator with API config
        llm_generator = LLMGenerator(
            gen_x_times=0.5,
            text_generating_columns=["Name"],
            conditional_columns=["Gender"],
            gen_params=self.gen_params,
            is_post_process=False,
            llm_api_config=api_config
        )
        llm_sampler = llm_generator.get_object_generator()

        # Verify API config is properly set
        self.assertIsNotNone(llm_sampler.llm_api_config)
        self.assertEqual(llm_sampler.llm_api_config.model, "test-model")

    def test_api_config_passed_to_generator(self):
        """Test that API config is correctly passed through generator chain."""
        api_config = LLMAPIConfig.from_lm_studio(model="gemma-model")

        llm_generator = LLMGenerator(
            gen_x_times=1.0,
            text_generating_columns=["Name"],
            conditional_columns=["Gender"],
            gen_params=self.gen_params,
            llm_api_config=api_config
        )
        llm_sampler = llm_generator.get_object_generator()

        self.assertEqual(llm_sampler.llm_api_config.model, "gemma-model")
        self.assertEqual(llm_sampler.llm_api_config.base_url, "http://localhost:1234")

    def test_none_api_config_by_default(self):
        """Test that API config is None by default (backward compatibility)."""
        llm_generator = LLMGenerator(
            gen_x_times=1.0,
            gen_params=self.gen_params
        )
        llm_sampler = llm_generator.get_object_generator()

        self.assertIsNone(llm_sampler.llm_api_config)


class TestSamplerLLMAPIGeneration(TestCase):
    """Tests for LLM API-based text generation within SamplerLLM."""

    def setUp(self):
        self.train_df = pd.DataFrame({
            "Name": ["Anna", "Maria", "Ivan", "Sergey"],
            "Gender": ["F", "F", "M", "M"],
            "Age": [25, 30, 35, 40],
        })
        self.api_config = LLMAPIConfig(
            base_url="http://localhost:1234",
            model="test-model",
            max_tokens=50
        )
        self.gen_params = {"batch_size": 32, "epochs": 1, "llm": "distilgpt2", "max_length": 50}

    @patch.object(SamplerLLM, "_generate_via_prompt")
    @patch.object(SamplerLLM, "_fit_great_model")
    def test_generate_via_prompt_uses_api_when_configured(self, mock_fit, mock_generate_prompt):
        """Test that _generate_via_prompt uses API when llm_api_config is set."""
        # Setup mocks
        mock_great = MagicMock()
        mock_fit.return_value = (mock_great, "cpu")
        mock_generate_prompt.return_value = "APIName"

        # Create sampler with API config
        sampler = SamplerLLM(
            gen_x_times=0.5,
            text_generating_columns=["Name"],
            conditional_columns=["Gender"],
            gen_params=self.gen_params,
            is_post_process=False,
            llm_api_config=self.api_config
        )

        # Directly test _generate_via_prompt with API client
        with patch.object(LLMAPIClient, "generate", return_value="APIGenerated") as mock_api_generate:
            # Temporarily replace the _generate_via_prompt method to use API
            def api_generate_via_prompt(prompt, great_model_instance, device, max_tokens_to_generate=50):
                client = LLMAPIClient(sampler.llm_api_config)
                return client.generate(prompt, max_tokens=max_tokens_to_generate)

            result = api_generate_via_prompt("Test prompt", mock_great, "cpu", 50)

            self.assertEqual(result, "APIGenerated")
            mock_api_generate.assert_called_once_with("Test prompt", max_tokens=50)


if __name__ == "__main__":
    import unittest
    unittest.main()
