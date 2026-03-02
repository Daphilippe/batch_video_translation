from modules.llm_translator import LLMTranslator
from modules.providers.base_provider import LLMProvider
from modules.providers.llama_provider import LLMProviderError


class MockProvider(LLMProvider):
    """Test double for LLMProvider."""
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.name = "MockLLM"

    def ask(self, content: str, prompt: str) -> str:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            if isinstance(resp, Exception):
                raise resp
            return resp
        self.call_count += 1
        return ""


class TestLLMTranslatorLogic:
    def test_translate_logic_single_chunk(self, tmp_path):
        """Should translate a simple SRT correctly."""
        translated_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour le monde\n"
        provider = MockProvider(responses=[translated_srt])

        config = {"source_lang": "English", "target_lang": "French", "chunk_size": 50}
        translator = LLMTranslator(
            input_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            provider=provider,
            config=config,
        )

        source = "1\n00:00:01,000 --> 00:00:03,000\nHello world\n"
        result = translator.translate_logic(source)
        assert "Bonjour le monde" in result

    def test_translate_logic_provider_error_keeps_source(self, tmp_path):
        """When provider raises, the source chunk should be kept."""
        provider = MockProvider(responses=[LLMProviderError("Server down")])

        config = {"source_lang": "English", "target_lang": "French", "chunk_size": 50}
        translator = LLMTranslator(
            input_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            provider=provider,
            config=config,
        )

        source = "1\n00:00:01,000 --> 00:00:03,000\nHello world\n"
        result = translator.translate_logic(source)
        # Should fallback to source text
        assert "Hello world" in result

    def test_translate_logic_empty_response_keeps_source(self, tmp_path):
        """When provider returns unparseable content, source chunk should be kept."""
        provider = MockProvider(responses=[""])

        config = {"source_lang": "English", "target_lang": "French", "chunk_size": 50}
        translator = LLMTranslator(
            input_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            provider=provider,
            config=config,
        )

        source = "1\n00:00:01,000 --> 00:00:03,000\nHello world\n"
        result = translator.translate_logic(source)
        assert "Hello world" in result

    def test_chunk_splitting(self, tmp_path):
        """Should split blocks into chunks of the configured size."""
        config = {"source_lang": "English", "target_lang": "French", "chunk_size": 2}
        provider = MockProvider(responses=[
            "1\n00:00:01,000 --> 00:00:02,000\nA\n\n2\n00:00:03,000 --> 00:00:04,000\nB\n",
            "3\n00:00:05,000 --> 00:00:06,000\nC\n",
        ])

        translator = LLMTranslator(
            input_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            provider=provider,
            config=config,
        )

        source = (
            "1\n00:00:01,000 --> 00:00:02,000\nA\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nB\n\n"
            "3\n00:00:05,000 --> 00:00:06,000\nC\n"
        )
        result = translator.translate_logic(source)  # noqa: F841
        assert provider.call_count == 2  # 2 chunks: [A,B] and [C]
