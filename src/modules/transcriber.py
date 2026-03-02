import ctypes
import logging
import os
import subprocess
from pathlib import Path

from utils.file_handler import DirectoryMirrorTask
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class WhisperTranscriber(DirectoryMirrorTask):
    """Whisper.cpp-based audio-to-SRT transcriber with segment caching.

    Processes folders of WAV audio chunks (produced by
    ``AudioExtractor``), transcribes each segment via the
    Whisper.cpp binary, applies a time offset to realign
    timestamps, and merges all segments into a single SRT file.

    Parameters
    ----------
    input_dir : str
        Root directory containing per-video segment folders.
    output_dir : str
        Directory where merged ``.srt`` files are written.
    whisper_bin : str
        Path to the Whisper.cpp executable.
    model_path : str
        Path to the Whisper GGML model file.
    lang : str, optional
        Language code for Whisper (default ``"auto"``).
    segment_time : int, optional
        Duration of each audio segment in seconds (default 600).
        Must match the value used in ``AudioExtractor``.

    Raises
    ------
    FileNotFoundError
        If *whisper_bin* or *model_path* does not exist.
    """
    def __init__(self, input_dir: str, output_dir: str, whisper_bin: str, model_path: str, lang: str = "auto", segment_time: int = 600):
        super().__init__(input_dir, output_dir, extensions=("",))
        self.whisper_bin = Path(whisper_bin)
        self.model_path = Path(model_path)
        self.lang = lang
        self.segment_time = segment_time

        # Validate whisper binary existence
        if not self.whisper_bin.exists():
            raise FileNotFoundError(f"Whisper binary not found at: {self.whisper_bin}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Whisper model not found at: {self.model_path}")

    def _get_short_path(self, path: Path) -> str:
        """
        Obtain the Windows 8.3 short path for a given path.

        Required because Whisper.cpp may not handle paths with
        non-ASCII characters.  Returns the path unchanged on
        non-Windows systems or if the API call fails.

        Parameters
        ----------
        path : Path
            File-system path to shorten.

        Returns
        -------
        str
            Short (8.3) path on Windows, or the original string.
        """
        if os.name != 'nt':
            return str(path)
        buf = ctypes.create_unicode_buffer(260)
        res = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 260)
        return buf.value if res != 0 else str(path)

    def run(self):
        """
        Iterate over per-video segment folders and transcribe each.

        Overrides the base ``run`` because the input structure is
        folder-based (one folder per video) rather than flat files.
        """
        input_path = Path(self.input_dir)
        if not input_path.exists():
            return
        video_folders = [d for d in input_path.iterdir() if d.is_dir()]
        for folder in video_folders:
            self.process_file(folder)

    def process_file(self, input_file: Path):  # pylint: disable=arguments-renamed
        """
        Transcribe all audio segments in a video folder.

        For each ``part*.wav`` segment, runs Whisper.cpp, shifts
        timestamps by the segment offset, caches the realigned SRT,
        and finally merges all segments into a single output file.

        Parameters
        ----------
        input_file : Path
            Path to the per-video segment folder (not a single file).
        """
        final_srt_path = Path(self.output_dir) / f"{input_file.name}.srt"
        # Temporary folder for realigned segments
        cache_dir = input_file / "srt_cache"
        cache_dir.mkdir(exist_ok=True)

        if final_srt_path.exists():
            logger.info(f"[SKIP] Video already fully transcribed: {final_srt_path.name}")
            return

        segments = sorted(list(input_file.glob("part*.wav")))
        all_blocks = []

        logger.info(f"[START] Processing: {input_file.name} ({len(segments)} segments)")

        for i, segment in enumerate(segments):
            # Path for the realigned intermediate SRT
            realigned_cache_srt = cache_dir / f"{segment.stem}_realigned.srt"

            # --- CHECK IF SEGMENT IS ALREADY IN CACHE ---
            if realigned_cache_srt.exists():
                logger.info(f"  -> Using cached segment {i}: {segment.name}")
                with open(realigned_cache_srt, encoding="utf-8") as f:
                    all_blocks.extend(SRTHandler.parse_to_blocks(f.read()))
                continue

            # --- TRANSCRIBE IF NOT IN CACHE ---
            temp_output_prefix = segment.with_suffix("")
            whisper_srt_output = segment.with_suffix(".srt") # whisper adds .srt

            cmd = [
                str(self.whisper_bin), "-m", str(self.model_path),
                "-f", self._get_short_path(segment),
                "-osrt", "-of", self._get_short_path(temp_output_prefix),
                "-l", self.lang, "-p", "4", "-sow", "1"
            ]

            try:
                logger.info(f"  -> Transcribing segment {i}: {segment.name}")
                subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')

                if whisper_srt_output.exists():
                    with open(whisper_srt_output, encoding="utf-8") as f:
                        content = f.read()

                    # Shift and Realign
                    offset_seconds = i * self.segment_time
                    blocks = SRTHandler.parse_to_blocks(content)
                    shifted_blocks = SRTHandler.apply_offset_to_blocks(blocks, offset_seconds)

                    # SAVE TO CACHE
                    with open(realigned_cache_srt, "w", encoding="utf-8") as f:
                        f.write(SRTHandler.render_blocks(shifted_blocks))

                    all_blocks.extend(shifted_blocks)

                    # Delete the raw Whisper SRT (only keep the realigned cache)
                    whisper_srt_output.unlink()

            except subprocess.CalledProcessError as e:
                logger.error(f"[ERROR] Failed segment {segment.name}: {e.stderr}")

        # Final Render
        if all_blocks:
            final_content = SRTHandler.render_blocks(all_blocks)
            with open(final_srt_path, "w", encoding="utf-8") as f:
                f.write(final_content)
            logger.info(f"[SUCCESS] Merged SRT saved to: {final_srt_path.name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", default="auto")
    parser.add_argument("--time", type=int, default=600)
    args = parser.parse_args()

    transcriber = WhisperTranscriber(
        args.input, args.output, args.bin, args.model,
        lang=args.lang, segment_time=args.time
    )
    transcriber.run()
