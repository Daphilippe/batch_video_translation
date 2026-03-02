import logging
import shutil
import subprocess
from pathlib import Path

from utils.file_handler import DirectoryMirrorTask

logger = logging.getLogger(__name__)

class AudioExtractor(DirectoryMirrorTask):
    """Extract and segment audio tracks from video files using FFmpeg.

    For each video, creates a subfolder of WAV chunks
    (``part000.wav``, ``part001.wav``, …) ready for Whisper
    transcription.  Existing segments are skipped automatically.
    """

    def __init__(self, input_dir: str, output_dir: str, extensions: tuple = (".mp4", ".mkv"), segment_time: int = 600):
        """
        Initialize the extractor with segmentation support.

        Parameters
        ----------
        input_dir : str
            Directory containing source video files.
        output_dir : str
            Directory where audio segment folders are created.
        extensions : tuple of str, optional
            Video extensions to process
            (default ``(".mp4", ".mkv")``).
        segment_time : int, optional
            Duration of each audio chunk in seconds (default 600).

        Raises
        ------
        FileNotFoundError
            If FFmpeg is not found in ``PATH``.
        """
        super().__init__(input_dir, output_dir, extensions)
        self.sample_rate = "16000"
        self.segment_time = segment_time

        # Validate FFmpeg availability
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError("FFmpeg not found in PATH. Install it and ensure it's accessible.")

    def process_file(self, input_file: Path):
        """
        Segment audio from a single video file into WAV chunks.

        Creates a dedicated subfolder mirroring the video name,
        then invokes FFmpeg's ``segment`` muxer to produce
        16 kHz mono WAV files of ``self.segment_time`` seconds each.

        Parameters
        ----------
        input_file : Path
            Path to the source video file.
        """
        # Create a specific directory for this video's segments
        # to avoid mixing chunks from different source files.
        video_output_dir = self.get_output_path(input_file, "").with_suffix("")
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # Output pattern: part000.wav, part001.wav, etc.
        output_pattern = video_output_dir / "part%03d.wav"

        # Check if the first segment already exists to prevent redundant processing.
        if (video_output_dir / "part000.wav").exists():
            logger.info(f"Segments already exist for: {input_file.name}")
            return

        # FFmpeg command using the 'segment' muxer
        cmd = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-f", "segment",                 # Enable segmenting mode
            "-segment_time", str(self.segment_time), # Set chunk duration
            "-ar", self.sample_rate,         # Required sample rate for Whisper
            "-ac", "1",                      # Force Mono audio
            "-c:a", "pcm_s16le",             # Uncompressed 16-bit WAV format
            str(output_pattern)
        ]

        try:
            # We use capture_output to keep the logs clean unless an error occurs
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Successfully segmented: {input_file.name} into {self.segment_time}s chunks")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown error"
            logger.error(f"Failed to segment {input_file.name}: {error_msg}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and segment audio from video files.")
    parser.add_argument("--input", required=True, help="Path to raw videos")
    parser.add_argument("--output", required=True, help="Path for audio segments")
    parser.add_argument("--time", type=int, default=600, help="Segment duration in seconds (default: 600)")
    args = parser.parse_args()

    extractor = AudioExtractor(args.input, args.output, segment_time=args.time)
    extractor.run()
