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
