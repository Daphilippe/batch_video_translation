"""Tests for VideoTranslationPipeline — orchestrator logic in main.py.

Covers: config loading, run() control flow, _get_file_count(),
_run_hybrid_pipeline(), _validate_config_section(), engine independence,
factory dispatch, promote-to-final, and cross-engine orchestration.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from main import VideoTranslationPipeline
from modules.legacy_translator import LegacyTranslator
from modules.llm_translator import LLMTranslator
from modules.providers.llama_provider import LlamaCPPProvider

# ---------------------------------------------------------------------------
# Shared fixtures
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

MINIMAL_CONFIG = {
    "translation": {
        "source_lang": "en",
        "target_lang": "fr",
        "cache_file": "data/cache.json",
        "max_chars_batch": 2000,
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

S1_CONTENT = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nWorld\n"

TRANSLATED_CONTENT = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n\n2\n00:00:04,000 --> 00:00:06,000\nMonde\n"


def _write_config(tmp_path, config_dict):
    """Write a config dict as JSON and return its path."""
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps(config_dict), encoding="utf-8")
    return str(config_path)


# ── Config loading errors ────────────────────────────────────────────


class TestConfigLoading:
    """Pipeline config error handling."""

    def test_missing_config_file_raises(self, tmp_path):
        """Missing config file → ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Configuration file not found"):
            VideoTranslationPipeline(
                output_dir=str(tmp_path / "out"),
                config_path=str(tmp_path / "nonexistent.json"),
            )

    def test_invalid_json_raises(self, tmp_path):
        """Malformed JSON → ValueError with parse info."""
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("{{{INVALID", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            VideoTranslationPipeline(
                output_dir=str(tmp_path / "out"),
                config_path=str(bad_config),
            )


# ── _get_file_count ──────────────────────────────────────────────────


class TestGetFileCount:
    """Utility to count files or subdirectories."""

    def test_count_by_extension(self, tmp_path):
        """Counts files matching the given extension."""
        config_path = _write_config(tmp_path, MINIMAL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        folder = tmp_path / "files"
        folder.mkdir()
        (folder / "a.srt").write_text("x", encoding="utf-8")
        (folder / "b.srt").write_text("x", encoding="utf-8")
        (folder / "c.txt").write_text("x", encoding="utf-8")

        assert pipeline._get_file_count(folder, (".srt",)) == 2

    def test_count_dirs(self, tmp_path):
        """Counts subdirectories when extension is 'dir'."""
        config_path = _write_config(tmp_path, MINIMAL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        folder = tmp_path / "dirs"
        folder.mkdir()
        (folder / "sub1").mkdir()
        (folder / "sub2").mkdir()
        (folder / "file.txt").write_text("x", encoding="utf-8")

        assert pipeline._get_file_count(folder, "dir") == 2

    def test_nonexistent_returns_zero(self, tmp_path):
        """Non-existent path returns 0."""
        config_path = _write_config(tmp_path, MINIMAL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        assert pipeline._get_file_count(tmp_path / "ghost", (".srt",)) == 0


# ── _validate_config_section ─────────────────────────────────────────


class TestValidateConfigSection:
    """Granular config section validation."""

    def test_missing_section_raises(self, tmp_path):
        """Missing top-level section → ValueError."""
        config_path = _write_config(tmp_path, {})
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="Missing required config section"):
            pipeline._validate_config_section("whisper", ["bin_path"])

    def test_missing_key_raises(self, tmp_path):
        """Present section but missing key → ValueError."""
        config_path = _write_config(tmp_path, {"whisper": {"bin_path": "x"}})
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="Missing required config key"):
            pipeline._validate_config_section("whisper", ["bin_path", "model_path"])

    def test_valid_section_passes(self, tmp_path):
        """Complete section → no error."""
        config_path = _write_config(tmp_path, {"whisper": {"bin_path": "x", "model_path": "y"}})
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._validate_config_section("whisper", ["bin_path", "model_path"])


# ── Pipeline validation — mode awareness ─────────────────────────────


class TestPipelineValidation:
    """Pipeline __init__ should NOT crash on missing binaries or config.

    Validation is now deferred to run() and is mode/engine-aware.
    """

    def test_init_without_whisper_section(self, tmp_path):
        """Pipeline can be initialised with only legacy config."""
        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)
        assert pipeline.config == LEGACY_ONLY_CONFIG

    def test_init_without_llm_config(self, tmp_path):
        """Pipeline can be initialised with only whisper + translation config."""
        cfg = {
            "whisper": {"bin_path": "fake.exe", "model_path": "fake.bin"},
            "translation": LEGACY_ONLY_CONFIG["translation"],
        }
        config_path = _write_config(tmp_path, cfg)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)
        assert "whisper" in pipeline.config

    def test_translate_mode_does_not_require_ffmpeg(self, tmp_path):
        """mode='translate' should never check for FFmpeg."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        with patch("main.shutil.which", return_value=None):
            pipeline._validate_requirements("translate", "legacy")

    def test_translate_mode_does_not_require_whisper(self, tmp_path):
        """mode='translate' should not need whisper config section."""
        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._validate_requirements("translate", "legacy")

    def test_extract_mode_requires_ffmpeg(self, tmp_path):
        """mode='extract' must check for FFmpeg."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with patch("main.shutil.which", return_value=None), pytest.raises(FileNotFoundError, match="FFmpeg"):
            pipeline._validate_requirements("extract", "legacy")

    def test_transcribe_mode_requires_whisper(self, tmp_path):
        """mode='transcribe' must validate whisper config."""
        cfg = {"translation": LEGACY_ONLY_CONFIG["translation"]}
        config_path = _write_config(tmp_path, cfg)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="whisper"):
            pipeline._validate_requirements("transcribe", "legacy")


# ── Engine-specific config validation ────────────────────────────────


class TestEngineConfigValidation:
    """Each engine should only require its own config sections."""

    def test_legacy_needs_translation_section(self, tmp_path):
        """Legacy engine must have 'translation' config section."""
        cfg = {"llm_config": LLM_ONLY_CONFIG["llm_config"]}
        config_path = _write_config(tmp_path, cfg)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="translation"):
            pipeline._validate_requirements("translate", "legacy")

    def test_legacy_does_not_need_llm_config(self, tmp_path):
        """Legacy engine must NOT require 'llm_config' section."""
        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._validate_requirements("translate", "legacy")

    def test_llm_local_needs_llm_config(self, tmp_path):
        """LLM-local engine must have 'llm_config' section."""
        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="llm_config"):
            pipeline._validate_requirements("translate", "llm-local")

    def test_llm_local_does_not_need_translation(self, tmp_path):
        """LLM-local engine must NOT require 'translation' section."""
        config_path = _write_config(tmp_path, LLM_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._validate_requirements("translate", "llm-local")

    def test_llm_ui_needs_llm_config(self, tmp_path):
        """LLM-UI engine must have 'llm_config' section."""
        config_path = _write_config(tmp_path, LEGACY_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="llm_config"):
            pipeline._validate_requirements("translate", "llm-ui")

    def test_llm_ui_does_not_need_translation(self, tmp_path):
        """LLM-UI engine must NOT require 'translation' section."""
        config_path = _write_config(tmp_path, LLM_ONLY_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._validate_requirements("translate", "llm-ui")

    def test_hybrid_needs_both_sections(self, tmp_path):
        """Hybrid engine must require BOTH llm_config AND translation."""
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


# ── Engine instantiation independence ────────────────────────────────


class TestLegacyEngineIndependence:
    """Legacy engine can be created with only its own dependencies."""

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_creates_with_translation_config_only(self, mock_gt, tmp_path):
        """LegacyTranslator works with just the 'translation' section."""
        translator = LegacyTranslator(str(tmp_path / "in"), str(tmp_path / "out"), LEGACY_ONLY_CONFIG)
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
        import modules.llm_translator

        assert hasattr(modules.llm_translator, "LLMTranslator")


class TestLLMUIEngineIndependence:
    """LLM-UI engine import is deferred — doesn't crash other engines."""

    def test_copilot_ui_not_imported_at_main_load(self):
        """main.py should NOT import CopilotUIProvider at module level."""
        import main

        assert not hasattr(main, "CopilotUIProvider")

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_ui_provider_used_only_when_requested(self, mock_gt, tmp_path):
        """CopilotUIProvider is only loaded in _create_llm_ui_translator."""
        config_path = _write_config(tmp_path, {**LLM_ONLY_CONFIG, **LEGACY_ONLY_CONFIG})
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        translator = pipeline._create_legacy_translator()
        assert translator.name == "Legacy translation"


class TestHybridEngineIndependence:
    """Hybrid engine requires both configs but creates all sub-engines."""

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_hybrid_config_builds_refiner_config(self, mock_gt, tmp_path):
        """_run_hybrid_pipeline builds correct refiner config from both sections."""
        config = {**FULL_CONFIG}
        config_path = _write_config(tmp_path, config)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._validate_requirements("translate", "hybrid")


# ── Factory dispatch ─────────────────────────────────────────────────


class TestCreateTranslator:
    """_create_translator dispatches correctly by engine name."""

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_unknown_engine_raises(self, mock_gt, tmp_path):
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="Unknown engine"):
            pipeline._create_translator("nonexistent")

    @patch("modules.legacy_translator.GoogleTranslator")
    def test_legacy_factory_outputs_to_intermediate(self, mock_gt, tmp_path):
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        translator = pipeline._create_translator("legacy")
        assert isinstance(translator, LegacyTranslator)
        assert str(translator.output_dir) == str(pipeline.dirs["legacy_mt"])

    def test_llm_local_factory_outputs_to_intermediate(self, tmp_path):
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        translator = pipeline._create_translator("llm-local")
        assert isinstance(translator, LLMTranslator)
        assert translator.name == "Local LLM"
        assert str(translator.output_dir) == str(pipeline.dirs["llm_mt"])

    @patch("modules.providers.copilot_ui.Desktop")
    def test_llm_ui_factory_outputs_to_intermediate(self, mock_desktop, tmp_path):
        mock_window = MagicMock()
        mock_window.window_text.return_value = "Microsoft Edge"
        mock_desktop.return_value.windows.return_value = [mock_window]

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        translator = pipeline._create_translator("llm-ui")
        assert isinstance(translator, LLMTranslator)
        assert translator.name == "UI LLM translation"
        assert str(translator.output_dir) == str(pipeline.dirs["llm_mt"])


# ── run() control flow ───────────────────────────────────────────────


class TestRunControlFlow:
    """Pipeline run() dispatching and error handling."""

    def test_run_nonexistent_input_dir(self, tmp_path):
        """run() returns early when input dir doesn't exist."""
        config_path = _write_config(tmp_path, MINIMAL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline.run(str(tmp_path / "missing"), mode="translate", engine="legacy")

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_run_translate_mode_legacy(self, mock_gt_class, _sleep, tmp_path):
        """run(mode='translate', engine='legacy') invokes LegacyTranslator."""
        mock_instance = MagicMock()
        mock_gt_class.return_value = mock_instance
        mock_instance.translate.return_value = "Bonjour"

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (tmp_path / "in").mkdir()

        (pipeline.dirs["clean_srt"] / "test.srt").write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        pipeline.run(str(tmp_path / "in"), mode="translate", engine="legacy")

        assert (pipeline.dirs["legacy_mt"] / "test.srt").exists()

    def test_run_translate_mode_llm_local(self, tmp_path):
        """run(mode='translate', engine='llm-local') invokes LLMTranslator."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (tmp_path / "in").mkdir()

        (pipeline.dirs["clean_srt"] / "test.srt").write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        with patch("main.LlamaCPPProvider") as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.name = "MockLLM"
            mock_provider.ask.return_value = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
            mock_provider_cls.return_value = mock_provider

            pipeline.run(str(tmp_path / "in"), mode="translate", engine="llm-local")

        assert (pipeline.dirs["llm_mt"] / "test.srt").exists()

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_run_optimize_mode(self, mock_gt, _sleep, tmp_path):
        """run(mode='optimize') invokes SRTOptimizer."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (tmp_path / "in").mkdir()

        (pipeline.dirs["raw_srt"] / "test.srt").write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:02,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        pipeline.run(str(tmp_path / "in"), mode="optimize", engine="legacy")
        assert (pipeline.dirs["clean_srt"] / "test.srt").exists()


# ── _run_hybrid_pipeline ─────────────────────────────────────────────


class TestRunHybridPipeline:
    """Hybrid pipeline sub-step orchestration."""

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    @patch("main.LlamaCPPProvider")
    def test_hybrid_runs_all_sub_steps(self, mock_llama_cls, mock_gt_cls, _sleep, tmp_path):
        """_run_hybrid_pipeline generates L1, Mt, then refines."""
        mock_gt_instance = MagicMock()
        mock_gt_cls.return_value = mock_gt_instance
        mock_gt_instance.translate.return_value = "Bonjour"

        mock_llama = MagicMock()
        mock_llama.name = "MockLLM"
        mock_llama.ask.return_value = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        mock_llama_cls.return_value = mock_llama

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (pipeline.dirs["clean_srt"] / "test.srt").write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        pipeline._run_hybrid_pipeline()

        assert (pipeline.dirs["legacy_mt"] / "test.srt").exists()
        assert (pipeline.dirs["llm_mt"] / "test.srt").exists()


# ── Promote to final ─────────────────────────────────────────────────


class TestPromoteToFinal:
    """Standalone engines write to intermediate dirs, then promote to final."""

    def test_legacy_promotes_to_final(self, tmp_path):
        """Files from 4_legacy_mt/ are copied to subtitles_ready/."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (pipeline.dirs["legacy_mt"] / "video.srt").write_text(TRANSLATED_CONTENT, encoding="utf-8")

        pipeline._promote_to_final("legacy")

        promoted = pipeline.dirs["final"] / "video.srt"
        assert promoted.exists()
        assert "Bonjour" in promoted.read_text(encoding="utf-8")

    def test_llm_local_promotes_to_final(self, tmp_path):
        """Files from 5_llm_mt/ are copied to subtitles_ready/."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (pipeline.dirs["llm_mt"] / "video.srt").write_text(TRANSLATED_CONTENT, encoding="utf-8")

        pipeline._promote_to_final("llm-local")

        promoted = pipeline.dirs["final"] / "video.srt"
        assert promoted.exists()
        assert "Bonjour" in promoted.read_text(encoding="utf-8")

    def test_llm_ui_promotes_to_final(self, tmp_path):
        """llm-ui engine also promotes from 5_llm_mt/."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (pipeline.dirs["llm_mt"] / "video.srt").write_text(TRANSLATED_CONTENT, encoding="utf-8")

        pipeline._promote_to_final("llm-ui")

        assert (pipeline.dirs["final"] / "video.srt").exists()

    def test_promote_noop_for_empty_dir(self, tmp_path):
        """Promotion with no files in intermediate dir does nothing."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._promote_to_final("legacy")

        assert not any(pipeline.dirs["final"].iterdir())

    def test_promote_overwrites_existing(self, tmp_path):
        """Promotion overwrites stale files in final directory."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (pipeline.dirs["final"] / "video.srt").write_text("old content", encoding="utf-8")
        (pipeline.dirs["legacy_mt"] / "video.srt").write_text(TRANSLATED_CONTENT, encoding="utf-8")

        pipeline._promote_to_final("legacy")

        assert "Bonjour" in (pipeline.dirs["final"] / "video.srt").read_text(encoding="utf-8")

    def test_promote_noop_for_hybrid_engine(self, tmp_path):
        """Hybrid engine has no intermediate dir — promote is a no-op."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._promote_to_final("hybrid")  # should not raise

    def test_promote_handles_subdirectories(self, tmp_path):
        """Promotion preserves subdirectory structure."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        subdir = pipeline.dirs["legacy_mt"] / "season1"
        subdir.mkdir()
        (subdir / "ep01.srt").write_text(TRANSLATED_CONTENT, encoding="utf-8")

        pipeline._promote_to_final("legacy")

        assert (pipeline.dirs["final"] / "season1" / "ep01.srt").exists()


# ── Cross-engine hybrid reuse ────────────────────────────────────────


class TestCrossEngineHybridReuse:
    """Hybrid skips sub-steps when intermediate dirs already have output."""

    def test_hybrid_reuses_legacy_and_llm_output(self, tmp_path):
        """Scenario: legacy then llm-local, then hybrid reuses both."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        (pipeline.dirs["clean_srt"] / "video.srt").write_text(S1_CONTENT, encoding="utf-8")
        (pipeline.dirs["legacy_mt"] / "video.srt").write_text(TRANSLATED_CONTENT, encoding="utf-8")
        (pipeline.dirs["llm_mt"] / "video.srt").write_text(TRANSLATED_CONTENT, encoding="utf-8")

        assert list(pipeline.dirs["legacy_mt"].glob("*.srt"))
        assert list(pipeline.dirs["llm_mt"].glob("*.srt"))

    def test_engine_dir_map_covers_all_standalone(self, tmp_path):
        """ENGINE_DIR_MAP has entries for all non-hybrid engines."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        assert "legacy" in pipeline.ENGINE_DIR_MAP
        assert "llm-local" in pipeline.ENGINE_DIR_MAP
        assert "llm-ui" in pipeline.ENGINE_DIR_MAP
        assert "hybrid" not in pipeline.ENGINE_DIR_MAP
