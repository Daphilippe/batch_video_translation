import pytest

from modules.providers.llama_provider import LlamaCPPProvider, LLMProviderError


class TestLLMProviderError:
    def test_error_is_exception(self):
        """LLMProviderError should be a proper Exception subclass."""
        error = LLMProviderError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"


class TestLlamaCPPProvider:
    def test_init_default_url(self):
        provider = LlamaCPPProvider()
        assert provider.url == "http://127.0.0.1:8080/v1/chat/completions"
        assert provider.name == "Local LLM"

    def test_init_custom_url(self):
        provider = LlamaCPPProvider(url="http://localhost:9090")
        assert provider.url == "http://localhost:9090/v1/chat/completions"

    def test_ask_raises_on_connection_error(self):
        """Calling ask with unreachable server should raise LLMProviderError."""
        provider = LlamaCPPProvider(url="http://127.0.0.1:1")
        with pytest.raises(LLMProviderError, match="Connection error"):
            provider.ask("system", "prompt")


# ── Mocked HTTP responses ────────────────────────────────────────────

from unittest.mock import MagicMock, patch  # noqa: E402


class TestLlamaCPPProviderAsk:
    """Test ask() with mocked HTTP responses (no real server)."""

    @patch("modules.providers.llama_provider.requests.post")
    def test_successful_ask(self, mock_post):
        """Valid LLM response is returned."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Translated text"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        provider = LlamaCPPProvider()
        result = provider.ask("system instructions", "translate this")

        assert result == "Translated text"
        mock_post.assert_called_once()

    @patch("modules.providers.llama_provider.requests.post")
    def test_empty_response_raises(self, mock_post):
        """Empty response from LLM raises LLMProviderError."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        provider = LlamaCPPProvider()
        with pytest.raises(LLMProviderError, match="empty response"):
            provider.ask("system", "prompt")

    @patch("modules.providers.llama_provider.requests.post")
    def test_whitespace_only_response_raises(self, mock_post):
        """Whitespace-only response raises LLMProviderError."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "   \n  "}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        provider = LlamaCPPProvider()
        with pytest.raises(LLMProviderError, match="empty response"):
            provider.ask("system", "prompt")

    @patch("modules.providers.llama_provider.requests.post")
    def test_invalid_json_format_raises(self, mock_post):
        """Missing 'choices' key raises LLMProviderError."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "bad request"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        provider = LlamaCPPProvider()
        with pytest.raises(LLMProviderError, match="Invalid response format"):
            provider.ask("system", "prompt")

    @patch("modules.providers.llama_provider.requests.post")
    def test_http_error_raises(self, mock_post):
        """HTTP error status raises LLMProviderError."""
        import requests

        mock_post.side_effect = requests.exceptions.HTTPError("500 Server Error")

        provider = LlamaCPPProvider()
        with pytest.raises(LLMProviderError, match="Connection error"):
            provider.ask("system", "prompt")
