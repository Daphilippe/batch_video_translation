import argparse
import json
import logging
import sys
import os
from pathlib import Path

# Module Imports
from modules.extractor import AudioExtractor
from modules.transcriber import WhisperTranscriber
from modules.srt_optimizer import SRTOptimizer
from modules.llm_translator import LLMTranslator
from modules.legacy_translator import LegacyTranslator
from modules.providers.copilot_ui import CopilotUIProvider

# Advanced Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VideoPipeline")

class VideoTranslationPipeline:
    def __init__(self, output_dir, config_path="configs/settings.json"):
        logger.info(f"--- Initializing Pipeline ---")
        logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.final_output = Path(output_dir)
        self.work_dir = self.final_output / "internals"

        self.dirs = {
            "audio": self.work_dir / "1_audio",
            "raw_srt": self.work_dir / "2_raw_srt",
            "clean_srt": self.work_dir / "3_clean_srt",
            "final": self.final_output / "subtitles_ready"
        }
        
        # Create directories immediately
        for name, path in self.dirs.items():
            if not path.exists():
                logger.info(f"Creating directory: {path}")
                path.mkdir(parents=True, exist_ok=True)

    def _get_file_count(self, path, extension):
        """Helper to count files for progress tracking."""
        if not path.exists(): return 0
        return len([f for f in os.listdir(path) if f.lower().endswith(extension)])

    def run(self, input_dir, mode="full", engine="llm"):
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_path}")
            return

        logger.info(f"Starting pipeline in '{mode}' mode using '{engine}' engine.")
        logger.info(f"Source folder: {input_path.absolute()}")
        logger.info(f"Output folder: {self.final_output.absolute()}")

        # --- STEP 1: AUDIO EXTRACTION ---
        if mode in ["full", "extract"]:
            video_count = self._get_file_count(input_path, (".mp4", ".mkv", ".avi", ".mov"))
            logger.info(f"Step 1/4: Audio Extraction | Found {video_count} videos.")
            
            extractor = AudioExtractor(
                input_dir=str(input_path),
                output_dir=str(self.dirs["audio"])
            )
            extractor.run()
            logger.info("Step 1/4: Completed successfully.")

        # --- STEP 2: TRANSCRIPTION ---
        if mode in ["full", "transcribe"]:
            audio_count = self._get_file_count(self.dirs["audio"], ".wav")
            logger.info(f"Step 2/4: Transcription (Whisper) | Processing {audio_count} audio files.")
            
            transcriber = WhisperTranscriber(
                input_dir=str(self.dirs["audio"]),
                output_dir=str(self.dirs["raw_srt"]),
                whisper_bin=self.config["whisper"]["bin_path"],
                model_path=self.config["whisper"]["model_path"],
                lang=self.config["whisper"].get("lang", "auto")
            )
            transcriber.run()
            logger.info("Step 2/4: Completed successfully.")

        # --- STEP 3: OPTIMIZATION ---
        if mode in ["full", "optimize"]:
            raw_srt_count = self._get_file_count(self.dirs["raw_srt"], ".srt")
            logger.info(f"Step 3/4: SRT Optimization | Merging duplicates in {raw_srt_count} files.")
            
            optimizer = SRTOptimizer(
                input_dir=str(self.dirs["raw_srt"]),
                output_dir=str(self.dirs["clean_srt"])
            )
            optimizer.run()
            logger.info("Step 3/4: Completed successfully.")

        # --- STEP 4: TRANSLATION ---
        if mode in ["full", "translate"]:
            clean_srt_count = self._get_file_count(self.dirs["clean_srt"], ".srt")
            logger.info(f"Step 4/4: Translation | Engine: {engine} | Source: {clean_srt_count} files.")
            
            if engine == "llm":
                logger.info("Initializing UI Automation Provider (Target: Microsoft Edge)...")
                provider = CopilotUIProvider(window_title="Edge")
                translator = LLMTranslator(
                    input_dir=str(self.dirs["clean_srt"]),
                    output_dir=str(self.dirs["final"]),
                    provider=provider,
                    config=self.config["llm_config"]
                )
            else:
                logger.info("Initializing Legacy Translator (Deep Translator)...")
                translator = LegacyTranslator(
                    input_dir=str(self.dirs["clean_srt"]),
                    output_dir=str(self.dirs["final"]),
                    config_path="configs/settings.json"
                )
            translator.run()
            logger.info(f"Step 4/4: Completed. Final files located in: {self.dirs['final']}")

        logger.info("================================================")
        logger.info("✨ ALL PIPELINE TASKS FINISHED SUCCESSFULLY! ✨")
        logger.info("================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage Video Translation Pipeline")
    parser.add_argument("--input", required=True, help="Path to source video folder")
    parser.add_argument("--output", required=True, help="Path to result folder")
    parser.add_argument("--mode", default="full", choices=["full", "extract", "transcribe", "optimize", "translate"])
    parser.add_argument("--engine", default="llm", choices=["llm", "legacy"])
    
    args = parser.parse_args()
    pipeline = VideoTranslationPipeline(output_dir=args.output)
    try:
        pipeline.run(input_dir=args.input, mode=args.mode, engine=args.engine)
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"A critical error occurred: {str(e)}", exc_info=True)