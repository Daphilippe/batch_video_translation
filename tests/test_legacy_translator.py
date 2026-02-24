import sys
from unittest.mock import MagicMock, patch

# deep_translator is not installed in the dev/CI environment.
# Provide a stub module so that the import chain succeeds.
if "deep_translator" not in sys.modules:
    _dt_stub = MagicMock()
    sys.modules["deep_translator"] = _dt_stub

from modules.legacy_translator import LegacyTranslator
from utils.srt_handler import SRTHandler


def _make_config(tmp_path):
    """Build a minimal LegacyTranslator config dict."""
    return {
        "translation": {
            "source_lang": "en",
            "target_lang": "fr",
            "cache_file": str(tmp_path / "cache.json"),
            "max_chars_batch": 2000,
            "retry_delay": 0,
            "max_retries": 1,
        },
        "technical_dictionary": {
            "neural network": "réseau neuronal",
        },
    }


SIMPLE_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
Goodbye
"""


# ── Dictionary pre-processing ────────────────────────────────────────


class TestApplyDictionary:
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_replaces_terms_case_insensitive(self, mock_gt, tmp_path):
        config = _make_config(tmp_path)
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)

        result = translator._apply_dictionary("This Neural Network is great")
        assert "(réseau neuronal)" in result

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_no_match_returns_lowered_text(self, mock_gt, tmp_path):
        config = _make_config(tmp_path)
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)

        result = translator._apply_dictionary("Hello world")
        assert result == "hello world"


# ── Cache loading ────────────────────────────────────────────────────


class TestCacheManagement:
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_empty_cache_when_no_file(self, mock_gt, tmp_path):
        config = _make_config(tmp_path)
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)
        assert translator.cache == {}

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_loads_existing_cache(self, mock_gt, tmp_path):
        config = _make_config(tmp_path)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text('{"abc123": "Bonjour"}', encoding="utf-8")

        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)
        assert translator.cache == {"abc123": "Bonjour"}

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_corrupt_cache_returns_empty(self, mock_gt, tmp_path):
        config = _make_config(tmp_path)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("NOT VALID JSON {{{", encoding="utf-8")

        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)
        assert translator.cache == {}

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_save_cache_writes_json(self, mock_gt, tmp_path):
        config = _make_config(tmp_path)
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)
        translator.cache = {"key1": "val1"}
        translator.save_cache()

        import json
        with open(tmp_path / "cache.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"key1": "val1"}


# ── Batch translation ────────────────────────────────────────────────


class TestSafeTranslateBatch:
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_successful_batch(self, mock_gt, tmp_path):
        config = _make_config(tmp_path)
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)
        translator.translator = MagicMock()
        translator.translator.translate.return_value = "Bonjour ||| Au revoir"

        result = translator._safe_translate_batch(["Hello", "Goodbye"])
        assert result == ["Bonjour", "Au revoir"]

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_empty_response_returns_empty(self, mock_gt, tmp_path):
        config = _make_config(tmp_path)
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)
        translator.translator = MagicMock()
        translator.translator.translate.return_value = ""

        result = translator._safe_translate_batch(["Hello"])
        assert result == []

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_non_429_error_returns_empty(self, mock_gt, tmp_path):
        config = _make_config(tmp_path)
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)
        translator.translator = MagicMock()
        translator.translator.translate.side_effect = RuntimeError("Network error")

        result = translator._safe_translate_batch(["Hello"])
        assert result == []

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_429_retries_then_fails(self, mock_gt, tmp_path):
        config = _make_config(tmp_path)
        config["translation"]["max_retries"] = 1
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)
        translator.translator = MagicMock()
        translator.translator.translate.side_effect = RuntimeError("HTTP 429 Too Many Requests")

        result = translator._safe_translate_batch(["Hello"])
        assert result == []
        # Should have retried once then given up
        assert translator.translator.translate.call_count == 2


# ── translate_logic ──────────────────────────────────────────────────


class TestTranslateLogic:
    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_translates_text_lines_only(self, mock_gt, mock_sleep, tmp_path):
        """Structural lines (indices, timestamps, blanks) should pass through."""
        config = _make_config(tmp_path)
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)
        translator.translator = MagicMock()
        translator.translator.translate.return_value = "bonjour monde ||| au revoir"

        result = translator.translate_logic(SIMPLE_SRT)
        lines = result.splitlines()

        # Structural lines preserved
        assert lines[0] == "1"
        assert lines[1] == "00:00:01,000 --> 00:00:03,000"
        assert lines[3] == ""
        assert lines[4] == "2"

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_cached_lines_reused(self, mock_gt, mock_sleep, tmp_path):
        """Pre-cached lines should not be sent to Google Translate."""
        config = _make_config(tmp_path)
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)

        # Pre-populate cache with hash of "Hello world"
        h = SRTHandler.get_hash("Hello world")
        translator.cache[h] = "Bonjour le monde"

        translator.translator = MagicMock()
        translator.translator.translate.return_value = "au revoir"

        result = translator.translate_logic(SIMPLE_SRT)

        assert "Bonjour le monde" in result
        # Only "Goodbye" should have been sent to Google
        translator.translator.translate.assert_called_once()

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_failed_batch_produces_ellipsis(self, mock_gt, mock_sleep, tmp_path):
        """When batch translation fails, affected lines become '...'."""
        config = _make_config(tmp_path)
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), config)
        translator.translator = MagicMock()
        translator.translator.translate.return_value = None

        result = translator.translate_logic(SIMPLE_SRT)

        assert "..." in result
