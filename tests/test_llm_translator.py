from helpers import MockProvider
from modules.llm_translator import LLMTranslator
from modules.providers.base_provider import LLMProviderError


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
        config = {"source_lang": "English", "target_lang": "French", "chunk_size": 2, "chunk_delay": 0}
        provider = MockProvider(
            responses=[
                "1\n00:00:01,000 --> 00:00:02,000\nX\n\n2\n00:00:03,000 --> 00:00:04,000\nY\n",
                "3\n00:00:05,000 --> 00:00:06,000\nZ\n",
            ]
        )

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


# ── Checkpoint / mid-file recovery ───────────────────────────────────

SRT_3_BLOCKS = (
    "1\n00:00:01,000 --> 00:00:02,000\nAlpha\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nBravo\n\n"
    "3\n00:00:05,000 --> 00:00:06,000\nCharlie\n"
)


class TestCheckpointRecovery:
    """Chunk-level checkpoint save and resume after interruption."""

    def test_checkpoint_saved_after_each_chunk(self, tmp_path):
        """A .partial.srt file is written after every chunk."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "test.srt").write_text(SRT_3_BLOCKS, encoding="utf-8")

        provider = MockProvider(
            responses=[
                "1\n00:00:01,000 --> 00:00:02,000\nUn\n",
                "2\n00:00:03,000 --> 00:00:04,000\nDeux\n",
                "3\n00:00:05,000 --> 00:00:06,000\nTrois\n",
            ]
        )
        config = {"source_lang": "English", "target_lang": "French", "chunk_size": 1, "chunk_delay": 0}
        translator = LLMTranslator(str(input_dir), str(output_dir), provider, config)

        translator.run()

        # Final output should exist, checkpoint should be cleaned up
        assert (output_dir / "test.srt").exists()
        assert not (output_dir / "test.partial.srt").exists()
        assert provider.call_count == 3

    def test_checkpoint_preserved_on_crash(self, tmp_path):
        """If provider raises mid-run, checkpoint persists for resume."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "test.srt").write_text(SRT_3_BLOCKS, encoding="utf-8")

        # Chunk 1 succeeds, chunk 2 raises a fatal error
        from modules.providers.base_provider import LLMProviderError
        provider = MockProvider(
            responses=[
                "1\n00:00:01,000 --> 00:00:02,000\nUn\n",
                LLMProviderError("LLM crashed"),
                "3\n00:00:05,000 --> 00:00:06,000\nTrois\n",
            ]
        )
        config = {"source_lang": "English", "target_lang": "French", "chunk_size": 1, "chunk_delay": 0}
        translator = LLMTranslator(str(input_dir), str(output_dir), provider, config)

        translator.run()

        # Output written (provider error = fallback to source, not crash)
        assert (output_dir / "test.srt").exists()
        # Checkpoint cleaned up because final output was successfully written
        assert not (output_dir / "test.partial.srt").exists()

    def test_resume_from_checkpoint(self, tmp_path):
        """Pre-existing checkpoint lets translation skip done chunks."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "test.srt").write_text(SRT_3_BLOCKS, encoding="utf-8")

        # Simulate a checkpoint from a previous interrupted run (1 chunk done)
        checkpoint = output_dir / "test.partial.srt"
        checkpoint.write_text("1\n00:00:01,000 --> 00:00:02,000\nUn\n", encoding="utf-8")

        # Provider only needs to handle chunks 2 and 3
        provider = MockProvider(
            responses=[
                "2\n00:00:03,000 --> 00:00:04,000\nDeux\n",
                "3\n00:00:05,000 --> 00:00:06,000\nTrois\n",
            ]
        )
        config = {"source_lang": "English", "target_lang": "French", "chunk_size": 1, "chunk_delay": 0}
        translator = LLMTranslator(str(input_dir), str(output_dir), provider, config)

        translator.run()

        # Only 2 LLM calls (chunks 2+3), not 3
        assert provider.call_count == 2
        output = (output_dir / "test.srt").read_text(encoding="utf-8")
        assert "Un" in output
        assert "Deux" in output
        assert "Trois" in output
        # Checkpoint cleaned up
        assert not checkpoint.exists()

    def test_corrupt_checkpoint_starts_fresh(self, tmp_path):
        """Unreadable checkpoint is ignored; full translation runs."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "test.srt").write_text(SRT_3_BLOCKS, encoding="utf-8")

        checkpoint = output_dir / "test.partial.srt"
        checkpoint.write_text("NOT VALID SRT {{{{", encoding="utf-8")

        provider = MockProvider(
            responses=[
                "1\n00:00:01,000 --> 00:00:02,000\nUn\n",
                "2\n00:00:03,000 --> 00:00:04,000\nDeux\n",
                "3\n00:00:05,000 --> 00:00:06,000\nTrois\n",
            ]
        )
        config = {"source_lang": "English", "target_lang": "French", "chunk_size": 1, "chunk_delay": 0}
        translator = LLMTranslator(str(input_dir), str(output_dir), provider, config)

        translator.run()

        # All 3 chunks translated from scratch
        assert provider.call_count == 3

    def test_file_level_skip_still_works(self, tmp_path):
        """Fully translated file is skipped (no LLM calls)."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        # Same timestamps in source and output = already translated
        (input_dir / "test.srt").write_text(SRT_3_BLOCKS, encoding="utf-8")
        (output_dir / "test.srt").write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nDéjà traduit\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nDéjà traduit\n\n"
            "3\n00:00:05,000 --> 00:00:06,000\nDéjà traduit\n",
            encoding="utf-8",
        )

        provider = MockProvider(responses=[])
        config = {"source_lang": "English", "target_lang": "French", "chunk_size": 1, "chunk_delay": 0}
        translator = LLMTranslator(str(input_dir), str(output_dir), provider, config)

        translator.run()

        # No LLM calls — file was fully skipped
        assert provider.call_count == 0


# ── Chunk retry on untranslated output ───────────────────────────────


class TestLLMChunkRetry:
    """LLMTranslator retries chunks that appear untranslated."""

    def _make_translator(self, tmp_path, provider, max_retries=2):
        config = {
            "source_lang": "English",
            "target_lang": "French",
            "chunk_size": 50,
            "chunk_delay": 0,
            "max_chunk_retries": max_retries,
        }
        return LLMTranslator(
            input_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            provider=provider,
            config=config,
        )

    def test_retry_on_identical_output(self, tmp_path):
        """When LLM returns source text, chunk is retried."""
        source_srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        translated_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        provider = MockProvider(responses=[source_srt, translated_srt])
        translator = self._make_translator(tmp_path, provider)

        result = translator.translate_logic(source_srt)
        assert "Bonjour" in result
        assert provider.call_count == 2  # 1 initial + 1 retry

    def test_gives_up_after_max_retries(self, tmp_path):
        """After max_chunk_retries, falls back to source text."""
        source_srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        provider = MockProvider(responses=[source_srt, source_srt, source_srt])
        translator = self._make_translator(tmp_path, provider, max_retries=2)

        result = translator.translate_logic(source_srt)
        assert "Hello" in result  # Falls back to source
        assert provider.call_count == 3  # 1 initial + 2 retries

    def test_no_retry_when_translated(self, tmp_path):
        """No retry when translation differs from source."""
        source_srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        translated_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        provider = MockProvider(responses=[translated_srt])
        translator = self._make_translator(tmp_path, provider)

        result = translator.translate_logic(source_srt)
        assert "Bonjour" in result
        assert provider.call_count == 1  # No retry needed

    def test_max_retries_configurable(self, tmp_path):
        """max_chunk_retries config controls retry limit."""
        source_srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        provider = MockProvider(responses=[source_srt, source_srt])
        translator = self._make_translator(tmp_path, provider, max_retries=1)

        translator.translate_logic(source_srt)
        assert provider.call_count == 2  # 1 initial + 1 retry (max_retries=1)
