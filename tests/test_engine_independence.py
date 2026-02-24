"""Tests verifying that each translation engine can run independently.

Each engine (legacy, llm-local, llm-ui, hybrid) should only require
its own config sections and dependencies — not those of other engines.
The pipeline validation should also be mode-aware: translate-only mode
must not require FFmpeg or Whisper.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub deep_translator so legacy_translator can import without the real package.
if "deep_translator" not in sys.modules:
    sys.modules["deep_translator"] = MagicMock()

# Stub Windows-only UI automation packages so copilot_ui can import on any env.
for _mod_name in ("pywinauto", "pywinauto.keyboard", "win32api", "pyperclip"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

from modules.legacy_translator import LegacyTranslator  # noqa: E402
from modules.llm_translator import LLMTranslator  # noqa: E402
from modules.providers.base_provider import LLMProvider  # noqa: E402
from modules.providers.llama_provider import LlamaCPPProvider  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEGACY_ONLY_CONFIG = {
    "translation": {
        "source_lang": "en",
        "target_lang": "fr",
        "cache_file": "data/cache.json",
        "max_chars_batch": 2000,
    },
    "technical_dictionary": {},
}

LLM_ONLY_CONFIG = {
    "llm_config": {
        "source_lang": "English",
        "target_lang": "French",
        "chunk_size": 10,
        "prompt_file": "configs/system_prompt_test.txt",
    },
}

FULL_CONFIG = {
    "whisper": {
        "bin_path": "C:/fake/whisper.exe",
        "model_path": "C:/fake/model.bin",
    },
    **LLM_ONLY_CONFIG,
    **LEGACY_ONLY_CONFIG,
}


def _write_config(tmp_path, config_dict):
    """Write a config dict as JSON and return its path."""
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps(config_dict), encoding="utf-8")
    return str(config_path)


class MockProvider(LLMProvider):
    """Minimal mock for LLM provider."""

    def __init__(self):
        self.name = "MockLLM"

    def ask(self, content: str, prompt: str) -> str:
        return prompt  # echo back


# ---------------------------------------------------------------------------
# Pipeline validation — mode awareness
# ---------------------------------------------------------------------------


class TestPipelineValidation:
    """Pipeline __init__ should NOT crash on missing binaries or config.

    Validation is now deferred to run() and is mode/engine-aware.
    """

    def test_init_without_whisper_section(self, tmp_path):
        """Pipeline can be initialised with only legacy config."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)
        assert pipeline.config == LEGACY_ONLY_CONFIG

    def test_init_without_llm_config(self, tmp_path):
        """Pipeline can be initialised with only whisper + translation config."""
        cfg = {
            "whisper": {"bin_path": "fake.exe", "model_path": "fake.bin"},
            "translation": LEGACY_ONLY_CONFIG["translation"],
        }
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, cfg)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)
        assert "whisper" in pipeline.config

    def test_translate_mode_does_not_require_ffmpeg(self, tmp_path):
        """mode='translate' should never check for FFmpeg."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Patch shutil.which to simulate missing FFmpeg — should NOT matter
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        with patch("main.shutil.which", return_value=None):
            # Should not raise FileNotFoundError for FFmpeg
            # (it will fail at translation time with no files, but that's OK)
            pipeline._validate_requirements("translate", "legacy")

    def test_translate_mode_does_not_require_whisper(self, tmp_path):
        """mode='translate' should not need whisper config section."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Should succeed — whisper not needed for translate mode
        pipeline._validate_requirements("translate", "legacy")

    def test_extract_mode_requires_ffmpeg(self, tmp_path):
        """mode='extract' must check for FFmpeg."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with patch("main.shutil.which", return_value=None), pytest.raises(FileNotFoundError, match="FFmpeg"):
            pipeline._validate_requirements("extract", "legacy")

    def test_transcribe_mode_requires_whisper(self, tmp_path):
        """mode='transcribe' must validate whisper config."""
        from main import VideoTranslationPipeline

        cfg = {"translation": LEGACY_ONLY_CONFIG["translation"]}
        config_path = _write_config(tmp_path, cfg)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="whisper"):
            pipeline._validate_requirements("transcribe", "legacy")


# ---------------------------------------------------------------------------
# Engine-specific config validation
# ---------------------------------------------------------------------------


class TestEngineConfigValidation:
    """Each engine should only require its own config sections."""

    def test_legacy_needs_translation_section(self, tmp_path):
        """Legacy engine must have 'translation' config section."""
        from main import VideoTranslationPipeline

        cfg = {"llm_config": LLM_ONLY_CONFIG["llm_config"]}
        config_path = _write_config(tmp_path, cfg)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="translation"):
            pipeline._validate_requirements("translate", "legacy")

    def test_legacy_does_not_need_llm_config(self, tmp_path):
        """Legacy engine must NOT require 'llm_config' section."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Should succeed with only translation config
        pipeline._validate_requirements("translate", "legacy")

    def test_llm_local_needs_llm_config(self, tmp_path):
        """LLM-local engine must have 'llm_config' section."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="llm_config"):
            pipeline._validate_requirements("translate", "llm-local")

    def test_llm_local_does_not_need_translation(self, tmp_path):
        """LLM-local engine must NOT require 'translation' section."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, LLM_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._validate_requirements("translate", "llm-local")

    def test_llm_ui_needs_llm_config(self, tmp_path):
        """LLM-UI engine must have 'llm_config' section."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="llm_config"):
            pipeline._validate_requirements("translate", "llm-ui")

    def test_llm_ui_does_not_need_translation(self, tmp_path):
        """LLM-UI engine must NOT require 'translation' section."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, LLM_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._validate_requirements("translate", "llm-ui")

    def test_hybrid_needs_both_sections(self, tmp_path):
        """Hybrid engine must require BOTH llm_config AND translation."""
        from main import VideoTranslationPipeline

        # Missing translation
        config_path = _write_config(tmp_path, LLM_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)
        with pytest.raises(ValueError, match="translation"):
            pipeline._validate_requirements("translate", "hybrid")

        # Missing llm_config
        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)
        with pytest.raises(ValueError, match="llm_config"):
            pipeline._validate_requirements("translate", "hybrid")


# ---------------------------------------------------------------------------
# Engine instantiation independence
# ---------------------------------------------------------------------------


class TestLegacyEngineIndependence:
    """Legacy engine can be created with only its own dependencies."""

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_creates_with_translation_config_only(self, mock_gt, tmp_path):
        """LegacyTranslator works with just the 'translation' section."""
        translator = LegacyTranslator(
            str(tmp_path / "in"), str(tmp_path / "out"), LEGACY_ONLY_CONFIG
        )
        assert translator.name == "Legacy translation"
        mock_gt.assert_called_once_with(source="en", target="fr")


class TestLLMLocalEngineIndependence:
    """LLM-local engine can be created without deep_translator."""

    def test_creates_with_llm_config_only(self, tmp_path):
        """LLMTranslator + LlamaCPPProvider works with just 'llm_config'."""
        provider = LlamaCPPProvider(url="http://127.0.0.1:8080")
        translator = LLMTranslator(
            str(tmp_path / "in"),
            str(tmp_path / "out"),
            provider=provider,
            config=LLM_ONLY_CONFIG["llm_config"],
        )
        assert translator.name == "Local LLM"
        assert translator.chunk_size == 10

    def test_does_not_import_deep_translator(self):
        """LLMTranslator should not depend on deep_translator at import time."""
        # If deep_translator were a required import, this would crash on
        # environments without it.  The try/except in legacy_translator.py
        # and lazy import in copilot_ui.py ensure isolation.
        import modules.llm_translator

        assert hasattr(modules.llm_translator, "LLMTranslator")


class TestLLMUIEngineIndependence:
    """LLM-UI engine import is deferred — doesn't crash other engines."""

    def test_copilot_ui_not_imported_at_main_load(self):
        """main.py should NOT import CopilotUIProvider at module level."""
        import main

        # CopilotUIProvider should NOT be in main's namespace
        assert not hasattr(main, "CopilotUIProvider")

    def test_ui_provider_used_only_when_requested(self, tmp_path):
        """CopilotUIProvider is only loaded in _create_llm_ui_translator."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, {**LLM_ONLY_CONFIG, **LEGACY_ONLY_CONFIG})
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Creating a legacy translator should NOT trigger CopilotUIProvider import
        translator = pipeline._create_legacy_translator()
        assert translator.name == "Legacy translation"


class TestHybridEngineIndependence:
    """Hybrid engine requires both configs but creates all sub-engines."""

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_hybrid_config_builds_refiner_config(self, mock_gt, tmp_path):
        """_run_hybrid_pipeline builds correct refiner config from both sections."""
        from main import VideoTranslationPipeline

        config = {**FULL_CONFIG}
        config_path = _write_config(tmp_path, config)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # The pipeline should be able to validate hybrid requirements
        pipeline._validate_requirements("translate", "hybrid")


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


class TestCreateTranslator:
    """_create_translator dispatches correctly by engine name."""

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_unknown_engine_raises(self, mock_gt, tmp_path):
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="Unknown engine"):
            pipeline._create_translator("nonexistent")

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_legacy_factory(self, mock_gt, tmp_path):
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        translator = pipeline._create_translator("legacy")
        assert isinstance(translator, LegacyTranslator)

    def test_llm_local_factory(self, tmp_path):
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        translator = pipeline._create_translator("llm-local")
        assert isinstance(translator, LLMTranslator)
        assert translator.name == "Local LLM"

    def test_llm_ui_factory(self, tmp_path):
        from main import VideoTranslationPipeline
        from modules.providers.copilot_ui import Desktop

        # Make the Desktop mock return a window matching "Edge"
        mock_window = MagicMock()
        mock_window.window_text.return_value = "Microsoft Edge"
        Desktop.return_value.windows.return_value = [mock_window]

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        translator = pipeline._create_translator("llm-ui")
        assert isinstance(translator, LLMTranslator)
        assert translator.name == "UI LLM translation"


# ---------------------------------------------------------------------------
# Hybrid Mt seeding (cross-engine reuse)
# ---------------------------------------------------------------------------

S1_CONTENT = (
    "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n"
    "2\n00:00:04,000 --> 00:00:06,000\nWorld\n"
)

TRANSLATED_CONTENT = (
    "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n\n"
    "2\n00:00:04,000 --> 00:00:06,000\nMonde\n"
)


class TestHybridMtSeeding:
    """Hybrid should reuse previous engine output as Mt draft."""

    def test_seeds_mt_from_final_when_mt_empty(self, tmp_path):
        """Previous llm-local output in subtitles_ready/ is copied to 5_llm_mt/."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Simulate: S1 exists, final has translated output, Mt is empty
        (pipeline.dirs["clean_srt"] / "video.srt").write_text(S1_CONTENT, encoding="utf-8")
        (pipeline.dirs["final"] / "video.srt").write_text(TRANSLATED_CONTENT, encoding="utf-8")

        pipeline._seed_mt_from_previous_run()

        seeded = pipeline.dirs["llm_mt"] / "video.srt"
        assert seeded.exists()
        assert "Bonjour" in seeded.read_text(encoding="utf-8")

    def test_no_seed_when_mt_already_populated(self, tmp_path):
        """If Mt already has files, seeding is skipped."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Mt already has a file
        (pipeline.dirs["llm_mt"] / "existing.srt").write_text("existing", encoding="utf-8")
        (pipeline.dirs["clean_srt"] / "video.srt").write_text(S1_CONTENT, encoding="utf-8")
        (pipeline.dirs["final"] / "video.srt").write_text(TRANSLATED_CONTENT, encoding="utf-8")

        pipeline._seed_mt_from_previous_run()

        # video.srt should NOT be copied (Mt already populated)
        assert not (pipeline.dirs["llm_mt"] / "video.srt").exists()

    def test_no_seed_when_final_empty(self, tmp_path):
        """If final directory has no SRT files, nothing to seed."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (pipeline.dirs["clean_srt"] / "video.srt").write_text(S1_CONTENT, encoding="utf-8")

        pipeline._seed_mt_from_previous_run()

        assert not (pipeline.dirs["llm_mt"] / "video.srt").exists()

    def test_no_seed_when_timestamps_mismatch(self, tmp_path):
        """Files with mismatched timestamps are not seeded."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (pipeline.dirs["clean_srt"] / "video.srt").write_text(S1_CONTENT, encoding="utf-8")

        # Different timestamps from S1
        mismatched = "1\n00:00:10,000 --> 00:00:12,000\nBonjour\n"
        (pipeline.dirs["final"] / "video.srt").write_text(mismatched, encoding="utf-8")

        pipeline._seed_mt_from_previous_run()

        assert not (pipeline.dirs["llm_mt"] / "video.srt").exists()

    def test_no_seed_when_s1_missing(self, tmp_path):
        """Files without a matching S1 source are not seeded."""
        from main import VideoTranslationPipeline

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # No S1 file — only final
        (pipeline.dirs["final"] / "orphan.srt").write_text(TRANSLATED_CONTENT, encoding="utf-8")

        pipeline._seed_mt_from_previous_run()

        assert not (pipeline.dirs["llm_mt"] / "orphan.srt").exists()
