from modules.srt_optimizer import SRTOptimizer

SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:02,000 --> 00:00:03,000
Hello

3
00:00:04,000 --> 00:00:06,000
World
"""


class TestSRTOptimizerProcessFile:
    def test_standardizes_and_writes(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "test.srt").write_text(SAMPLE_SRT, encoding="utf-8")

        optimizer = SRTOptimizer(str(input_dir), str(output_dir))
        optimizer.run()

        output_file = output_dir / "test.srt"
        assert output_file.exists()

        from utils.srt_handler import SRTHandler

        blocks = SRTHandler.parse_to_blocks(output_file.read_text(encoding="utf-8"))
        # Two "Hello" blocks should be merged → 2 total
        assert len(blocks) == 2
        assert blocks[0]["text"] == ["Hello"]
        assert blocks[0]["end"] == "00:00:03,000"

    def test_skip_when_output_matches(self, tmp_path):
        """Should not rewrite when output hash matches."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "test.srt").write_text(SAMPLE_SRT, encoding="utf-8")

        optimizer = SRTOptimizer(str(input_dir), str(output_dir))
        optimizer.run()

        output_file = output_dir / "test.srt"
        first_mtime = output_file.stat().st_mtime

        # Run again — output should be skipped (not rewritten)
        import time
        time.sleep(0.05)
        optimizer.run()

        assert output_file.stat().st_mtime == first_mtime

    def test_rewrites_when_content_differs(self, tmp_path):
        """Should rewrite when source content changed."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "test.srt").write_text(SAMPLE_SRT, encoding="utf-8")

        optimizer = SRTOptimizer(str(input_dir), str(output_dir))
        optimizer.run()

        output_file = output_dir / "test.srt"

        # Tamper with the output → it should be rewritten on next run
        output_file.write_text("tampered content\n", encoding="utf-8")

        optimizer.run()

        from utils.srt_handler import SRTHandler

        blocks = SRTHandler.parse_to_blocks(output_file.read_text(encoding="utf-8"))
        assert len(blocks) == 2  # Properly re-standardized
