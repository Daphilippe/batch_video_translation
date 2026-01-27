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
from modules.providers.llama_provider import LlamaCPPProvider

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

        # Define directory structure
        self.dirs = {
            "audio": self.work_dir / "1_audio",      # Now contains FOLDERS of segments
            "raw_srt": self.work_dir / "2_raw_srt",
            "clean_srt": self.work_dir / "3_clean_srt",
            "final": self.final_output / "subtitles_ready"
        }
        
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)

    def _get_file_count(self, path, extension):
        """Helper to count files or directories for progress tracking."""
        if not path.exists(): return 0
        if extension == "dir":
            return len([d for d in path.iterdir() if d.is_dir()])
        return len([f for f in path.iterdir() if f.suffix.lower() in extension])

    def run(self, input_dir, mode="full", engine="llm"):
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_path}")
            return

        # Get segment_time from config or default to 10 mins (600s)
        seg_time = self.config.get("whisper", {}).get("segment_time", 600)

        logger.info(f"Starting pipeline in '{mode}' mode using '{engine}' engine.")
        logger.info(f"Segmentation Interval: {seg_time}s")

        # --- STEP 1: AUDIO EXTRACTION (SEGMENTED) ---
        if mode in ["full", "extract"]:
            video_extensions = (".mp4", ".mkv", ".avi", ".mov")
            video_count = self._get_file_count(input_path, video_extensions)
            logger.info(f"Step 1/4: Audio Extraction | Found {video_count} videos.")
            
            extractor = AudioExtractor(
                input_dir=str(input_path),
                output_dir=str(self.dirs["audio"]),
                segment_time=seg_time
            )
            extractor.run()
            logger.info("Step 1/4: Completed successfully.")

        # --- STEP 2: TRANSCRIPTION (MERGED SEGMENTS) ---
        if mode in ["full", "transcribe"]:
            # We count folders in 1_audio (one folder per video)
            folder_count = self._get_file_count(self.dirs["audio"], "dir")
            logger.info(f"Step 2/4: Transcription | Processing {folder_count} video folders.")
            
            transcriber = WhisperTranscriber(
                input_dir=str(self.dirs["audio"]),
                output_dir=str(self.dirs["raw_srt"]),
                whisper_bin=self.config["whisper"]["bin_path"],
                model_path=self.config["whisper"]["model_path"],
                lang=self.config["whisper"].get("lang", "auto"),
                segment_time=seg_time
            )
            # This will generate: internals/2_raw_srt/VideoName.srt
            transcriber.run()
            logger.info("Step 2/4: Completed. Merged SRTs are in 2_raw_srt.")

        # --- STEP 3: OPTIMIZATION ---
        if mode in ["full", "optimize"]:
            # IMPORTANT: Optimization reads the SINGLE merged files from Step 2
            raw_srt_count = self._get_file_count(self.dirs["raw_srt"], (".srt",))
            logger.info(f"Step 3/4: SRT Optimization | Processing {raw_srt_count} merged files.")
            
            optimizer = SRTOptimizer(
                input_dir=str(self.dirs["raw_srt"]),
                output_dir=str(self.dirs["clean_srt"])
            )
            # This reads 2_raw_srt/VideoName.srt -> writes 3_clean_srt/VideoName.srt
            optimizer.run()

        # --- STEP 4: TRANSLATION ---
        if mode in ["full", "translate"]:
            clean_srt_count = self._get_file_count(self.dirs["clean_srt"], (".srt",))
            logger.info(f"Step 4/4: Translation | Engine: {engine} | Source: {clean_srt_count} files.")
            if (engine == "llm-local"):
                logger.info("Initializing UI Automation Provider...")
                provider = LlamaCPPProvider()
                translator = LLMTranslator(
                    input_dir=str(self.dirs["clean_srt"]),
                    output_dir=str(self.dirs["final"]),
                    provider=provider,
                    config=self.config["llm_config"]
                )
            elif (engine == "llm-ui"):
                logger.info("Initializing UI Automation Provider...")
                provider = CopilotUIProvider(window_title="Edge")
                translator = LLMTranslator(
                    input_dir=str(self.dirs["clean_srt"]),
                    output_dir=str(self.dirs["final"]),
                    provider=provider,
                    config=self.config["llm_config"]
                )
            else:
                logger.info("Initializing Legacy Translator...")
                translator = LegacyTranslator(
                    input_dir=str(self.dirs["clean_srt"]),
                    output_dir=str(self.dirs["final"]),
                    config_path="configs/settings.json"
                )
            translator.run()
            logger.info(f"Step 4/4: Completed. Final files: {self.dirs['final']}")

        logger.info("✨ ALL PIPELINE TASKS FINISHED SUCCESSFULLY! ✨")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage Video Translation Pipeline")
    parser.add_argument("--input", required=True, help="Path to source video folder")
    parser.add_argument("--output", required=True, help="Path to result folder")
    parser.add_argument("--mode", default="full", choices=["full", "extract", "transcribe", "optimize", "translate"])
    parser.add_argument("--engine", default="legacy", choices=["llm-local","llm-ui", "legacy"])
    
    args = parser.parse_args()
    pipeline = VideoTranslationPipeline(output_dir=args.output)
    try:
        pipeline.run(input_dir=args.input, mode=args.mode, engine=args.engine)
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"A critical error occurred: {str(e)}", exc_info=True)