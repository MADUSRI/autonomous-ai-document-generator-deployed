import unittest
from unittest.mock import MagicMock, patch

import httpx

import app.llm as llm


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.original_provider = llm.LLM_PROVIDER
        self.original_api_key = llm.GROQ_API_KEY
        self.original_model = llm.GROQ_MODEL

    def tearDown(self):
        llm.LLM_PROVIDER = self.original_provider
        llm.GROQ_API_KEY = self.original_api_key
        llm.GROQ_MODEL = self.original_model

    def test_send_llm_request_defaults_to_ollama(self):
        llm.LLM_PROVIDER = "ollama"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "OK"}

        with patch("app.llm.httpx.post", return_value=mock_response) as mock_post:
            result = llm.send_llm_request("hello", model="llama3.2:latest", timeout=10.0)

        self.assertEqual(result, "OK")
        self.assertEqual(mock_post.call_count, 1)
        self.assertIn("11434/api/generate", mock_post.call_args[0][0])

    def test_send_llm_request_uses_groq(self):
        llm.LLM_PROVIDER = "groq"
        llm.GROQ_API_KEY = "secret-key"
        llm.GROQ_MODEL = "llama"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": [{"content": "Hello from Groq"}]}

        with patch("app.llm.httpx.post", return_value=mock_response) as mock_post:
            result = llm.send_llm_request("hello groq", timeout=10.0)

        self.assertEqual(result, "Hello from Groq")
        self.assertEqual(mock_post.call_count, 1)
        call_args, call_kwargs = mock_post.call_args
        self.assertIn("/models/llama/infer", call_args[0])
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(call_kwargs["json"]["input"], "hello groq")

    def test_send_llm_request_uses_openrouter(self):
        llm.LLM_PROVIDER = "openrouter"
        llm.OPENROUTER_API_KEY = "openrouter-secret"
        llm.OPENROUTER_MODEL = "openai/gpt-4o"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from OpenRouter"}}]
        }

        with patch("app.llm.httpx.post", return_value=mock_response) as mock_post:
            result = llm.send_llm_request("hello openrouter", timeout=10.0)

        self.assertEqual(result, "Hello from OpenRouter")
        self.assertEqual(mock_post.call_count, 1)
        call_args, call_kwargs = mock_post.call_args
        self.assertIn("/chat/completions", call_args[0])
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer openrouter-secret")
        self.assertEqual(call_kwargs["json"]["model"], "openai/gpt-4o")
        self.assertEqual(call_kwargs["json"]["messages"][0]["content"], "hello openrouter")

    def test_send_llm_request_raises_when_openrouter_key_missing(self):
        llm.LLM_PROVIDER = "openrouter"
        llm.OPENROUTER_API_KEY = ""

        with self.assertRaises(RuntimeError):
            llm.send_llm_request("hello", timeout=10.0)

    def test_send_llm_request_falls_back_from_ollama_to_openrouter_on_timeout(self):
        llm.LLM_PROVIDER = "ollama"
        llm.OLLAMA_URL = "http://localhost:11434"
        llm.OPENROUTER_API_KEY = "openrouter-secret"
        llm.OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
        llm.OPENROUTER_MODEL = "openai/gpt-4o"

        mock_timeout = httpx.ReadTimeout("timed out")
        openrouter_response = MagicMock()
        openrouter_response.status_code = 200
        openrouter_response.json.return_value = {
            "choices": [{"message": {"content": "Fallback response from OpenRouter"}}]
        }

        with patch("app.llm.httpx.post", side_effect=[mock_timeout, openrouter_response]) as mock_post:
            result = llm.send_llm_request("hello", timeout=10.0)

        self.assertEqual(result, "Fallback response from OpenRouter")
        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("/api/generate", mock_post.call_args_list[0][0][0])
        self.assertIn("/chat/completions", mock_post.call_args_list[1][0][0])


if __name__ == "__main__":
    unittest.main()
