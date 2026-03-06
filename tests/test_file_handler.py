from pathlib import Path

import pytest

from utils.file_handler import DirectoryMirrorTask


class TestDirectoryMirrorTask:
    def test_get_output_path(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        task = DirectoryMirrorTask(str(input_dir), str(output_dir), (".txt",))
        input_file = input_dir / "subdir" / "file.txt"
        result = task.get_output_path(input_file, ".srt")

        assert result == output_dir / "subdir" / "file.srt"

    def test_run_processes_matching_files(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create test files
        (input_dir / "file.srt").write_text("content", encoding="utf-8")
        (input_dir / "file.txt").write_text("other", encoding="utf-8")

        processed = []

        class TestTask(DirectoryMirrorTask):
            def process_file(self, input_file: Path) -> None:
                processed.append(input_file.name)

        task = TestTask(str(input_dir), str(output_dir), (".srt",))
        task.run()

        assert processed == ["file.srt"]

    def test_run_nonexistent_input_dir(self, tmp_path):
        """Should log error and return without crashing."""
        task = DirectoryMirrorTask(str(tmp_path / "nonexistent"), str(tmp_path / "output"), (".srt",))
        task.run()  # Should not raise

    def test_process_file_raises_not_implemented(self, tmp_path):
        task = DirectoryMirrorTask(str(tmp_path), str(tmp_path), (".txt",))
        with pytest.raises(NotImplementedError):
            task.process_file(tmp_path / "file.txt")
