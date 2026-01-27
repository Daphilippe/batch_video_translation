import subprocess
import ctypes
import os
import logging
from pathlib import Path
from utils.file_handler import DirectoryMirrorTask

logger = logging.getLogger(__name__)

class WhisperTranscriber(DirectoryMirrorTask):
    def __init__(self, input_dir, output_dir, whisper_bin, model_path, lang="auto", extensions=(".wav",)):
        # On initialise la classe mère avec les dossiers et extensions (.wav uniquement)
        super().__init__(input_dir, output_dir, extensions)
        self.whisper_bin = Path(whisper_bin)
        self.model_path = Path(model_path)
        self.lang = lang

    def _get_short_path(self, path: Path) -> str:
        """Windows-specific: handling long or special character paths for CLI."""
        if os.name != 'nt':
            return str(path)
        buf = ctypes.create_unicode_buffer(260)
        res = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 260)
        return buf.value if res != 0 else str(path)

    def process_file(self, input_file: Path):
        """Logic for transcribing a single file using whisper.cpp."""
        # On définit le préfixe de sortie (sans extension car whisper ajoute .srt)
        output_file_full = self.get_output_path(input_file, ".srt")
        output_file_full.parent.mkdir(parents=True, exist_ok=True)

        if output_file_full.exists():
            logger.info(f"[SKIP] Already transcribed: {output_file_full.name}")
            return

        # Conversion en chemins courts pour la compatibilité CLI Windows
        short_input = self._get_short_path(input_file)
        short_output_prefix = self._get_short_path(output_file_full.with_suffix(""))

        cmd = [
            str(self.whisper_bin),
            "-m", str(self.model_path),
            "-f", short_input,
            "-osrt",
            "-of", short_output_prefix,
            "-l", self.lang,
            "-p", "4",  # threads
            "-sow", "1" # split on word
        ]

        try:
            logger.info(f"[TRANSCRIBING] {input_file.name}")
            subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                encoding='utf-8',
                errors='replace'
                )
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Failed to transcribe {input_file.name}: {e.stderr}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone Whisper Transcriber")
    parser.add_argument("--input", required=True, help="WAV files directory")
    parser.add_argument("--output", required=True, help="SRT output directory")
    parser.add_argument("--bin", required=True, help="Path to whisper.cpp executable")
    parser.add_argument("--model", required=True, help="Path to .bin model")
    parser.add_argument("--lang", default="auto", help="Language code (default: auto)")

    args = parser.parse_args()

    # Utilisation de la classe
    transcriber = WhisperTranscriber(
        input_dir=args.input, 
        output_dir=args.output, 
        whisper_bin=args.bin, 
        model_path=args.model,
        lang=args.lang
    )
    transcriber.run()