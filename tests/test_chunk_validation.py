"""Tests for chunk-level translation validation and retry across all engines.

Each translation strategy must verify that translated output actually
differs from the source.  If the output is identical to the source
(i.e. not translated), the chunk is retried up to a configurable limit.
"""

from unittest.mock import MagicMock, patch

from helpers import MockProvider
from modules.legacy_translator import LegacyTranslator
from modules.llm_translator import LLMTranslator
from modules.strategies.hybrid_refiner import HybridRefiner
from modules.translator import BaseTranslator

# ── BaseTranslator._is_chunk_untranslated ────────────────────────────


class TestIsChunkUntranslated:
    """Unit tests for the static validation helper on BaseTranslator."""

    def test_empty_translated_blocks(self):
        """Empty translation list → untranslated."""
        src = [{"text": ["Hello"]}]
        assert BaseTranslator._is_chunk_untranslated(src, []) is True

    def test_identical_text_is_untranslated(self):
        """Same text in source and translation → untranslated."""
        src = [{"text": ["Hello"]}, {"text": ["World"]}]
        tgt = [{"text": ["Hello"]}, {"text": ["World"]}]
        assert BaseTranslator._is_chunk_untranslated(src, tgt) is True

    def test_different_text_is_translated(self):
        """Different text → properly translated."""
        src = [{"text": ["Hello"]}, {"text": ["World"]}]
        tgt = [{"text": ["Bonjour"]}, {"text": ["Monde"]}]
        assert BaseTranslator._is_chunk_untranslated(src, tgt) is False

    def test_case_insensitive_comparison(self):
        """Identity check is case-insensitive."""
        src = [{"text": ["HELLO"]}]
        tgt = [{"text": ["hello"]}]
        assert BaseTranslator._is_chunk_untranslated(src, tgt) is True

    def test_majority_rule_under_threshold(self):
        """If less than half identical → considered translated."""
        src = [{"text": ["A"]}, {"text": ["B"]}, {"text": ["C"]}]
        tgt = [{"text": ["A"]}, {"text": ["Y"]}, {"text": ["Z"]}]
        # 1/3 identical = 33% < 50% → translated
        assert BaseTranslator._is_chunk_untranslated(src, tgt) is False

    def test_majority_rule_over_threshold(self):
        """If more than half identical → considered untranslated."""
        src = [{"text": ["A"]}, {"text": ["B"]}, {"text": ["C"]}]
        tgt = [{"text": ["A"]}, {"text": ["B"]}, {"text": ["Z"]}]
        # 2/3 identical = 67% > 50% → untranslated
        assert BaseTranslator._is_chunk_untranslated(src, tgt) is True

    def test_text_as_string(self):
        """Handles text stored as str (after merge) instead of list."""
        src = [{"text": "Hello"}]
        tgt = [{"text": "Bonjour"}]
        assert BaseTranslator._is_chunk_untranslated(src, tgt) is False

    def test_mixed_text_types(self):
        """Handles mix of list and str text fields."""
        src = [{"text": ["Hello"]}, {"text": "World"}]
        tgt = [{"text": "Bonjour"}, {"text": ["Monde"]}]
        assert BaseTranslator._is_chunk_untranslated(src, tgt) is False

    def test_both_empty_blocks(self):
        """Two empty block lists → untranslated."""
        assert BaseTranslator._is_chunk_untranslated([], []) is True


# ── LLMTranslator chunk retry ───────────────────────────────────────


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
        # First call returns source (untranslated), second returns translated
        provider = MockProvider(responses=[source_srt, translated_srt])
        translator = self._make_translator(tmp_path, provider)

        result = translator.translate_logic(source_srt)
        assert "Bonjour" in result
        assert provider.call_count == 2  # 1 initial + 1 retry

    def test_gives_up_after_max_retries(self, tmp_path):
        """After max_chunk_retries, falls back to source text."""
        source_srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        # All responses return source text (never translates)
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
        # 1 initial + 1 retry = 2 total, then give up
        provider = MockProvider(responses=[source_srt, source_srt])
        translator = self._make_translator(tmp_path, provider, max_retries=1)

        translator.translate_logic(source_srt)
        assert provider.call_count == 2  # 1 initial + 1 retry (max_retries=1)


# ── LegacyTranslator batch retry ────────────────────────────────────


class TestLegacyBatchRetry:
    """LegacyTranslator retries lines that appear untranslated."""

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_retry_untranslated_lines(self, mock_gt_class, _mock_sleep, tmp_path):
        """Lines identical to source after batch are retried."""
        # Setup translator mock
        mock_instance = MagicMock()
        mock_gt_class.return_value = mock_instance

        # First batch returns source text ("Hello"); retry returns translation
        mock_instance.translate.side_effect = [
            "Hello",  # First batch: returns source (untranslated)
            "Bonjour",  # Retry: returns translation
        ]

        config = {
            "translation": {
                "source_lang": "en",
                "target_lang": "fr",
                "cache_file": str(tmp_path / "cache.json"),
                "max_chars_batch": 5000,
            }
        }
        translator = LegacyTranslator(
            input_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            config=config,
        )

        source = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        result = translator.translate_logic(source)
        assert "Bonjour" in result
        assert mock_instance.translate.call_count == 2

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_no_retry_when_translated(self, mock_gt_class, _mock_sleep, tmp_path):
        """No retry when all lines differ from source."""
        mock_instance = MagicMock()
        mock_gt_class.return_value = mock_instance
        mock_instance.translate.return_value = "Bonjour"

        config = {
            "translation": {
                "source_lang": "en",
                "target_lang": "fr",
                "cache_file": str(tmp_path / "cache.json"),
                "max_chars_batch": 5000,
            }
        }
        translator = LegacyTranslator(
            input_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            config=config,
        )

        source = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        result = translator.translate_logic(source)
        assert "Bonjour" in result
        assert mock_instance.translate.call_count == 1  # No retry

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_retry_accepts_second_attempt(self, mock_gt_class, _mock_sleep, tmp_path):
        """If retry still returns source, result is kept (no infinite loop)."""
        mock_instance = MagicMock()
        mock_gt_class.return_value = mock_instance

        # Both calls return source text — retry gives up gracefully
        mock_instance.translate.side_effect = [
            "Hello",  # First batch: source text
            "Hello",  # Retry: still source text
        ]

        config = {
            "translation": {
                "source_lang": "en",
                "target_lang": "fr",
                "cache_file": str(tmp_path / "cache.json"),
                "max_chars_batch": 5000,
            }
        }
        translator = LegacyTranslator(
            input_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            config=config,
        )

        source = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        result = translator.translate_logic(source)
        # Falls back to the untranslated result (not "...")
        assert "Hello" in result
        assert mock_instance.translate.call_count == 2


# ── HybridRefiner window retry ──────────────────────────────────────


class TestHybridWindowRetry:
    """HybridRefiner._refine_window retries when output matches source."""

    def _make_refiner(self, tmp_path, provider):
        config = {"chunk_size": 10, "chunk_delay": 0}
        return HybridRefiner(
            source_dirs={"s1": str(tmp_path / "s1"), "l1": str(tmp_path / "l1"), "mt": str(tmp_path / "mt")},
            output_dir=str(tmp_path / "out"),
            provider=provider,
            config=config,
        )

    def test_retry_on_identical_window(self, tmp_path):
        """When LLM returns S1 text, window is retried once."""
        source_srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        translated_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        # First attempt returns source, second returns translation
        provider = MockProvider(responses=[source_srt, translated_srt])
        refiner = self._make_refiner(tmp_path, provider)
        refiner._active_protocol = "system"

        s1_win = [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello"]}]
        result = refiner._refine_window(s1_win, [0], {"l1": {}, "mt": {}})

        assert result is not None
        assert result[0]["text"] == ["Bonjour"]
        assert provider.call_count == 2

    def test_gives_up_after_two_attempts(self, tmp_path):
        """After 2 failed attempts, returns S1 source blocks."""
        source_srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        # Both attempts return source text
        provider = MockProvider(responses=[source_srt, source_srt])
        refiner = self._make_refiner(tmp_path, provider)
        refiner._active_protocol = "system"

        s1_win = [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello"]}]
        result = refiner._refine_window(s1_win, [0], {"l1": {}, "mt": {}})

        # Falls back to S1 source blocks after 2 failed attempts
        assert result[0]["text"] == ["Hello"]
        assert provider.call_count == 2

    def test_no_retry_when_translated(self, tmp_path):
        """No retry when refinement differs from source."""
        translated_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        provider = MockProvider(responses=[translated_srt])
        refiner = self._make_refiner(tmp_path, provider)
        refiner._active_protocol = "system"

        s1_win = [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello"]}]
        result = refiner._refine_window(s1_win, [0], {"l1": {}, "mt": {}})

        assert result is not None
        assert result[0]["text"] == ["Bonjour"]
        assert provider.call_count == 1

    def test_enforces_s1_timestamps(self, tmp_path):
        """Refined blocks always get S1 timestamps."""
        # Translated block has different timestamps
        translated_srt = "1\n00:00:10,000 --> 00:00:12,000\nBonjour\n"
        provider = MockProvider(responses=[translated_srt])
        refiner = self._make_refiner(tmp_path, provider)
        refiner._active_protocol = "system"

        s1_win = [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello"]}]
        result = refiner._refine_window(s1_win, [0], {"l1": {}, "mt": {}})

        assert result is not None
        assert result[0]["start"] == "00:00:01,000"
        assert result[0]["end"] == "00:00:03,000"

    def test_empty_response_triggers_retry(self, tmp_path):
        """Empty LLM response triggers a retry."""
        translated_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        # First attempt returns empty, second returns valid translation
        provider = MockProvider(responses=["", translated_srt])
        refiner = self._make_refiner(tmp_path, provider)
        refiner._active_protocol = "system"

        s1_win = [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello"]}]
        result = refiner._refine_window(s1_win, [0], {"l1": {}, "mt": {}})

        assert result is not None
        assert result[0]["text"] == ["Bonjour"]
        assert provider.call_count == 2
