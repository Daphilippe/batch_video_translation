"""Tests for BaseTranslator — file-level orchestration in translator.py.

Covers: wait_for_stability(), process_file() (skip logic, translate, write),
and translate_logic() abstract contract.
"""

import time
from unittest.mock import patch

import pytest

from modules.translator import BaseTranslator


class ConcreteTranslator(BaseTranslator):
    """Minimal concrete subclass for testing BaseTranslator logic."""

    def __init__(self, input_dir, output_dir, response="translated"):
        super().__init__(input_dir, output_dir)
        self.response = response
        self.translate_called = False
        self.name = "TestTranslator"

    def translate_logic(self, text):
        self.translate_called = True
        return self.response


# ── wait_for_stability ───────────────────────────────────────────────


class TestWaitForStability:
    """File stability check for slow I/O or antivirus locks."""

    def test_stable_file_returns_true(self, tmp_path):
        """File with stable size returns True quickly."""
        translator = ConcreteTranslator(str(tmp_path / "in"), str(tmp_path / "out"))
        f = tmp_path / "stable.txt"
        f.write_text("content", encoding="utf-8")

        assert translator.wait_for_stability(f, timeout=2) is True

    @patch("modules.translator.time.sleep")
    def test_missing_file_returns_false(self, mock_sleep, tmp_path):
        """Non-existent file returns False after timeout."""
        translator = ConcreteTranslator(str(tmp_path / "in"), str(tmp_path / "out"))

        # Make time.time() advance rapidly to simulate timeout
        call_count = 0
        original_time = time.time

        def fake_time():
            nonlocal call_count
            call_count += 1
            return original_time() + call_count * 5  # Jump 5s per call

        with patch("modules.translator.time.time", side_effect=fake_time):
            result = translator.wait_for_stability(tmp_path / "ghost.txt", timeout=2)

        assert result is False

    def test_empty_file_waits(self, tmp_path):
        """File with size 0 keeps waiting until timeout."""
        translator = ConcreteTranslator(str(tmp_path / "in"), str(tmp_path / "out"))
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        with patch("modules.translator.time.sleep"):
            call_count = 0
            original_time = time.time

            def fake_time():
                nonlocal call_count
                call_count += 1
                return original_time() + call_count * 5

            with patch("modules.translator.time.time", side_effect=fake_time):
                result = translator.wait_for_stability(f, timeout=2)

        assert result is False


# ── process_file ─────────────────────────────────────────────────────


class TestBaseTranslatorProcessFile:
    """File-level orchestration: skip, translate, standardize, write."""

    @patch("modules.translator.time.sleep")
    def test_translates_new_file(self, mock_sleep, tmp_path):
        """New file (no existing output) is translated and written."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        source = input_dir / "test.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        response_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        translator = ConcreteTranslator(str(input_dir), str(output_dir), response=response_srt)
        translator.process_file(source)

        output = output_dir / "test.srt"
        assert output.exists()
        assert "Bonjour" in output.read_text(encoding="utf-8")
        assert translator.translate_called

    @patch("modules.translator.time.sleep")
    def test_skips_already_translated(self, mock_sleep, tmp_path):
        """Existing output with matching timestamps is skipped."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        source = input_dir / "test.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        # Pre-existing output with same timestamp structure
        output_file = output_dir / "test.srt"
        output_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n",
            encoding="utf-8",
        )

        translator = ConcreteTranslator(str(input_dir), str(output_dir))
        translator.process_file(source)

        assert not translator.translate_called  # Skipped

    @patch("modules.translator.time.sleep")
    def test_retranslates_on_structure_mismatch(self, mock_sleep, tmp_path):
        """Existing output with different timestamps is re-translated."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        source = input_dir / "test.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nWorld\n",
            encoding="utf-8",
        )

        # Pre-existing output with only 1 block (mismatch)
        output_file = output_dir / "test.srt"
        output_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n",
            encoding="utf-8",
        )

        response_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n\n2\n00:00:04,000 --> 00:00:06,000\nMonde\n"
        translator = ConcreteTranslator(str(input_dir), str(output_dir), response=response_srt)
        translator.process_file(source)

        assert translator.translate_called
        assert "Monde" in output_file.read_text(encoding="utf-8")

    @patch("modules.translator.time.sleep")
    def test_output_is_standardized(self, mock_sleep, tmp_path):
        """Translated output goes through SRTHandler.standardize."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        source = input_dir / "test.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        # Response with bold markers — standardize should strip them
        response_srt = "1\n00:00:01,000 --> 00:00:03,000\n**Bonjour**\n"
        translator = ConcreteTranslator(str(input_dir), str(output_dir), response=response_srt)
        translator.process_file(source)

        content = (output_dir / "test.srt").read_text(encoding="utf-8")
        assert "**" not in content
        assert "Bonjour" in content


# ── translate_logic abstract ─────────────────────────────────────────


class TestTranslateLogicAbstract:
    """BaseTranslator.translate_logic raises NotImplementedError."""

    def test_raises_not_implemented(self, tmp_path):
        """Direct call to BaseTranslator.translate_logic raises."""
        translator = BaseTranslator(str(tmp_path / "in"), str(tmp_path / "out"))
        with pytest.raises(NotImplementedError):
            translator.translate_logic("1\n00:00:01,000 --> 00:00:03,000\nHello\n")

# ── _is_chunk_untranslated ───────────────────────────────────────────


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
