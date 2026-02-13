import subprocess
import ctypes
import os
import logging
from pathlib import Path
from utils.file_handler import DirectoryMirrorTask
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class WhisperTranscriber(DirectoryMirrorTask):
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
        if os.name != 'nt': return str(path)
        buf = ctypes.create_unicode_buffer(260)
        res = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 260)
        return buf.value if res != 0 else str(path)

    def run(self):
        input_path = Path(self.input_dir)
        if not input_path.exists(): return
        video_folders = [d for d in input_path.iterdir() if d.is_dir()]
        for folder in video_folders:
            self.process_file(folder)

    def process_file(self, input_folder: Path):
        final_srt_path = Path(self.output_dir) / f"{input_folder.name}.srt"
        # Temporary folder for realigned segments
        cache_dir = input_folder / "srt_cache"
        cache_dir.mkdir(exist_ok=True)

        if final_srt_path.exists():
            logger.info(f"[SKIP] Video already fully transcribed: {final_srt_path.name}")
            return

        segments = sorted(list(input_folder.glob("part*.wav")))
        all_blocks = []

        logger.info(f"[START] Processing: {input_folder.name} ({len(segments)} segments)")

        for i, segment in enumerate(segments):
            # Path for the realigned intermediate SRT
            realigned_cache_srt = cache_dir / f"{segment.stem}_realigned.srt"
            
            # --- CHECK IF SEGMENT IS ALREADY IN CACHE ---
            if realigned_cache_srt.exists():
                logger.info(f"  -> Using cached segment {i}: {segment.name}")
                with open(realigned_cache_srt, "r", encoding="utf-8") as f:
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
                    with open(whisper_srt_output, "r", encoding="utf-8") as f:
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