"""Tests for AudioExtractor — FFmpeg-based audio segmentation."""

from unittest.mock import patch

import pytest

from modules.extractor import AudioExtractor


class TestAudioExtractorInit:
    """Initialization and dependency validation."""

    @patch("modules.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_init_with_ffmpeg_available(self, mock_which, tmp_path):
        """AudioExtractor initializes when FFmpeg is in PATH."""
        extractor = AudioExtractor(str(tmp_path / "in"), str(tmp_path / "out"))
        assert extractor.segment_time == 600
        assert extractor.sample_rate == "16000"

    @patch("modules.extractor.shutil.which", return_value=None)
    def test_init_without_ffmpeg_raises(self, mock_which, tmp_path):
        """AudioExtractor raises FileNotFoundError when FFmpeg is missing."""
        with pytest.raises(FileNotFoundError, match="FFmpeg"):
            AudioExtractor(str(tmp_path / "in"), str(tmp_path / "out"))

    @patch("modules.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_custom_segment_time(self, mock_which, tmp_path):
        """Custom segment_time is stored."""
        extractor = AudioExtractor(str(tmp_path / "in"), str(tmp_path / "out"), segment_time=300)
        assert extractor.segment_time == 300

    @patch("modules.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_custom_extensions(self, mock_which, tmp_path):
        """Custom video extensions are stored."""
        extractor = AudioExtractor(
            str(tmp_path / "in"),
            str(tmp_path / "out"),
            extensions=(".mp4", ".mkv", ".avi", ".mov"),
        )
        assert ".avi" in extractor.extensions


class TestAudioExtractorProcessFile:
    """File-level processing with subprocess mocking."""

    @patch("modules.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("modules.extractor.subprocess.run")
    def test_process_creates_segment_dir(self, mock_run, mock_which, tmp_path):
        """process_file creates the per-video output directory."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()
        video = input_dir / "video.mp4"
        video.write_text("fake", encoding="utf-8")

        extractor = AudioExtractor(str(input_dir), str(output_dir))
        extractor.process_file(video)

        # Should have called subprocess.run with ffmpeg command
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"

    @patch("modules.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("modules.extractor.subprocess.run")
    def test_skip_existing_segments(self, mock_run, mock_which, tmp_path):
        """process_file skips video when part000.wav already exists."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()
        video = input_dir / "video.mp4"
        video.write_text("fake", encoding="utf-8")

        # Pre-create segment output
        seg_dir = output_dir / "video"
        seg_dir.mkdir()
        (seg_dir / "part000.wav").write_text("segment", encoding="utf-8")

        extractor = AudioExtractor(str(input_dir), str(output_dir))
        extractor.process_file(video)

        # subprocess should NOT be called — segments exist
        mock_run.assert_not_called()

    @patch("modules.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("modules.extractor.subprocess.run")
    def test_subprocess_error_logged(self, mock_run, mock_which, tmp_path):
        """process_file handles subprocess errors gracefully."""
        import subprocess

        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()
        video = input_dir / "video.mp4"
        video.write_text("fake", encoding="utf-8")

        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"encode error")

        extractor = AudioExtractor(str(input_dir), str(output_dir))
        # Should not raise — error is logged
        extractor.process_file(video)


class TestAudioExtractorRun:
    """Full run() integration with the directory mirror."""

    @patch("modules.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("modules.extractor.subprocess.run")
    def test_run_processes_matching_files(self, mock_run, mock_which, tmp_path):
        """run() finds .mp4/.mkv files and calls process_file."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "video1.mp4").write_text("v1", encoding="utf-8")
        (input_dir / "video2.mkv").write_text("v2", encoding="utf-8")
        (input_dir / "notes.txt").write_text("skip me", encoding="utf-8")

        extractor = AudioExtractor(str(input_dir), str(output_dir))
        extractor.run()

        # Should have called subprocess twice (one per video)
        assert mock_run.call_count == 2

    @patch("modules.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_run_nonexistent_dir(self, mock_which, tmp_path):
        """run() does not crash when input dir doesn't exist."""
        extractor = AudioExtractor(str(tmp_path / "missing"), str(tmp_path / "out"))
        extractor.run()  # Should not raise
