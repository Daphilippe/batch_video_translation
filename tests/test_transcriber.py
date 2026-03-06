"""Tests for WhisperTranscriber — Whisper.cpp segment transcription."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from modules.transcriber import WhisperTranscriber

# --- Fixtures ---


def _make_transcriber(tmp_path, lang="auto", segment_time=600):
    """Build a WhisperTranscriber with fake binary paths."""
    whisper_bin = tmp_path / "whisper.exe"
    model_path = tmp_path / "model.bin"
    whisper_bin.write_text("fake", encoding="utf-8")
    model_path.write_text("fake", encoding="utf-8")

    return WhisperTranscriber(
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        config={
            "bin_path": str(whisper_bin),
            "model_path": str(model_path),
            "lang": lang,
            "segment_time": segment_time,
        },
    )


# --- Init ---


class TestWhisperTranscriberInit:
    """Initialization and binary validation."""

    def test_init_with_valid_binaries(self, tmp_path):
        """Initializes successfully with existing binary and model."""
        transcriber = _make_transcriber(tmp_path)
        assert transcriber.lang == "auto"
        assert transcriber.segment_time == 600

    def test_missing_whisper_binary_raises(self, tmp_path):
        """Raises FileNotFoundError when whisper binary doesn't exist."""
        model = tmp_path / "model.bin"
        model.write_text("fake", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="Whisper binary"):
            WhisperTranscriber(
                str(tmp_path / "in"),
                str(tmp_path / "out"),
                {"bin_path": str(tmp_path / "missing.exe"), "model_path": str(model)},
            )

    def test_missing_model_raises(self, tmp_path):
        """Raises FileNotFoundError when model file doesn't exist."""
        whisper_bin = tmp_path / "whisper.exe"
        whisper_bin.write_text("fake", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="Whisper model"):
            WhisperTranscriber(
                str(tmp_path / "in"),
                str(tmp_path / "out"),
                {"bin_path": str(whisper_bin), "model_path": str(tmp_path / "missing.bin")},
            )

    def test_custom_lang_and_segment(self, tmp_path):
        """Custom lang and segment_time are stored."""
        transcriber = _make_transcriber(tmp_path, lang="fr", segment_time=300)
        assert transcriber.lang == "fr"
        assert transcriber.segment_time == 300


# --- _get_short_path ---


class TestGetShortPath:
    """Windows short path resolution."""

    def test_non_windows_returns_str(self, tmp_path):
        """On non-Windows, returns str(path) unchanged."""
        transcriber = _make_transcriber(tmp_path)
        path = tmp_path / "file.wav"
        with patch.object(os, "name", "posix"):
            result = transcriber._get_short_path(path)
        assert result == str(path)

    @patch("os.name", "nt")
    def test_windows_calls_kernel32(self, tmp_path):
        """On Windows, calls GetShortPathNameW."""
        transcriber = _make_transcriber(tmp_path)
        path = tmp_path / "file.wav"
        # Just verify the method doesn't crash — actual short path
        # depends on Windows API availability
        result = transcriber._get_short_path(path)
        assert isinstance(result, str)


# --- run() ---


class TestWhisperTranscriberRun:
    """Folder-based run() iteration."""

    def test_run_nonexistent_input(self, tmp_path):
        """run() silently returns when input directory doesn't exist."""
        transcriber = _make_transcriber(tmp_path)
        transcriber.run()  # Should not raise

    @patch("modules.transcriber.subprocess.run")
    def test_run_processes_video_folders(self, mock_subproc, tmp_path):
        """run() iterates over per-video segment folders."""
        transcriber = _make_transcriber(tmp_path)
        input_dir = Path(transcriber.input_dir)
        output_dir = Path(transcriber.output_dir)
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        # Create a video folder with one segment
        video_dir = input_dir / "video1"
        video_dir.mkdir()
        (video_dir / "part000.wav").write_text("audio", encoding="utf-8")

        # Mock subprocess: whisper outputs an SRT file
        def fake_run(cmd, **kwargs):
            # Whisper creates a .srt next to the .wav
            srt_path = video_dir / "part000.srt"
            srt_path.write_text(
                "1\n00:00:01,000 --> 00:00:03,000\nHello\n",
                encoding="utf-8",
            )

        mock_subproc.side_effect = fake_run
        transcriber.run()

        # Final merged SRT should exist
        assert (output_dir / "video1.srt").exists()

    @patch("modules.transcriber.subprocess.run")
    def test_skip_already_transcribed(self, mock_subproc, tmp_path):
        """run() skips videos whose final SRT already exists."""
        transcriber = _make_transcriber(tmp_path)
        input_dir = Path(transcriber.input_dir)
        output_dir = Path(transcriber.output_dir)
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        video_dir = input_dir / "video1"
        video_dir.mkdir()
        (video_dir / "part000.wav").write_text("audio", encoding="utf-8")

        # Pre-create final SRT
        (output_dir / "video1.srt").write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nDéjà fait\n",
            encoding="utf-8",
        )

        transcriber.run()
        mock_subproc.assert_not_called()


# --- process_file ---


class TestWhisperTranscriberProcessFile:
    """Segment-level transcription with caching."""

    @patch("modules.transcriber.subprocess.run")
    def test_uses_cached_segments(self, mock_subproc, tmp_path):
        """Cached realigned SRT segments are reused without re-transcribing."""
        transcriber = _make_transcriber(tmp_path)
        input_dir = Path(transcriber.input_dir)
        output_dir = Path(transcriber.output_dir)
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        video_dir = input_dir / "video1"
        video_dir.mkdir()
        (video_dir / "part000.wav").write_text("audio", encoding="utf-8")

        # Pre-create cache
        cache_dir = video_dir / "srt_cache"
        cache_dir.mkdir()
        (cache_dir / "part000_realigned.srt").write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nCached\n",
            encoding="utf-8",
        )

        transcriber.process_file(video_dir)

        # No subprocess call — used cache
        mock_subproc.assert_not_called()
        assert (output_dir / "video1.srt").exists()
        assert "Cached" in (output_dir / "video1.srt").read_text(encoding="utf-8")

    @patch("modules.transcriber.subprocess.run")
    def test_multi_segment_offset(self, mock_subproc, tmp_path):
        """Multiple segments get time offsets (segment_time * index)."""
        transcriber = _make_transcriber(tmp_path, segment_time=600)
        input_dir = Path(transcriber.input_dir)
        output_dir = Path(transcriber.output_dir)
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        video_dir = input_dir / "video1"
        video_dir.mkdir()
        (video_dir / "part000.wav").write_text("s0", encoding="utf-8")
        (video_dir / "part001.wav").write_text("s1", encoding="utf-8")

        call_count = 0

        def fake_run(cmd, **kwargs):
            nonlocal call_count
            # Determine which segment from the command
            seg_name = f"part{call_count:03d}"
            srt_path = video_dir / f"{seg_name}.srt"
            srt_path.write_text(
                "1\n00:00:01,000 --> 00:00:03,000\nSegment\n",
                encoding="utf-8",
            )
            call_count += 1

        mock_subproc.side_effect = fake_run
        transcriber.process_file(video_dir)

        final = (output_dir / "video1.srt").read_text(encoding="utf-8")
        # Segment 1 should have 600s offset → 00:10:01,000
        assert "00:10:01,000" in final
        assert mock_subproc.call_count == 2

    @patch("modules.transcriber.subprocess.run")
    def test_subprocess_error_does_not_crash(self, mock_subproc, tmp_path):
        """Subprocess error on a segment is logged, other segments continue."""
        import subprocess

        transcriber = _make_transcriber(tmp_path)
        input_dir = Path(transcriber.input_dir)
        output_dir = Path(transcriber.output_dir)
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        video_dir = input_dir / "video1"
        video_dir.mkdir()
        (video_dir / "part000.wav").write_text("audio", encoding="utf-8")

        mock_subproc.side_effect = subprocess.CalledProcessError(1, "whisper", stderr="decode error")

        # Should not raise
        transcriber.process_file(video_dir)
