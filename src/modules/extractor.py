import subprocess
import logging
from pathlib import Path
from utils.file_handler import DirectoryMirrorTask

logger = logging.getLogger(__name__)

class AudioExtractor(DirectoryMirrorTask):
    def __init__(self, input_dir, output_dir, extensions=(".mp4", ".mkv")):
        super().__init__(input_dir, output_dir, extensions)
        self.sample_rate = "16000"

    def process_file(self, input_file: Path):
        output_file = self.get_output_path(input_file, ".wav")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists():
            return

        cmd = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-ar", self.sample_rate, "-ac", "1", "-c:a", "pcm_s16le",
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Converted: {input_file.name}")
        except subprocess.CalledProcessError:
            logger.error(f"Failed: {input_file.name}")

if __name__ == "__main__":
    # Test manuel rapide
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    extractor = AudioExtractor(args.input, args.output)
    extractor.run()