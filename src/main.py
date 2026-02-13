import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

# Module Imports
from modules.extractor import AudioExtractor
from modules.transcriber import WhisperTranscriber
from modules.srt_optimizer import SRTOptimizer
from modules.llm_translator import LLMTranslator
from modules.legacy_translator import LegacyTranslator
from modules.strategies.hybrid_refiner import HybridRefiner
from modules.translator import BaseTranslator
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
    def __init__(self, output_dir: str, config_path: str = "configs/settings.json"):
        logger.info("--- Initializing Pipeline ---")
        logger.info(f"Loading configuration from: {config_path}")
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: '{config_path}'")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file '{config_path}': {e}")
        
        self._validate_config()
        self._validate_binaries()
        self.final_output = Path(output_dir)
        self.work_dir = self.final_output / "internals"

        # Define directory structure - Ajout des dossiers pour L1 et Mt
        self.dirs = {
            "audio": self.work_dir / "1_audio",
            "raw_srt": self.work_dir / "2_raw_srt",
            "clean_srt": self.work_dir / "3_clean_srt",    # S1 (Anchor)
            "legacy_mt": self.work_dir / "4_legacy_mt",   # L1 (Literal)
            "llm_mt": self.work_dir / "5_llm_mt",         # Mt (Draft)
            "final": self.final_output / "subtitles_ready"
        }
        
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)

    def _validate_config(self) -> None:
        """Validates that required configuration sections and keys are present."""
        required_keys = {
            "whisper": ["bin_path", "model_path"],
            "llm_config": ["source_lang", "target_lang"],
        }
        for section, keys in required_keys.items():
            if section not in self.config:
                raise ValueError(f"Missing required config section: '{section}'")
            for key in keys:
                if key not in self.config[section]:
                    raise ValueError(f"Missing required config key: '{section}.{key}'")

    def _validate_binaries(self) -> None:
        """Validates that required external binaries are accessible."""
        # Validate FFmpeg
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError(
                "FFmpeg not found in PATH. Install it and ensure it's accessible."
            )
        
        # Validate Whisper binary
        whisper_bin = Path(self.config["whisper"]["bin_path"])
        if not whisper_bin.exists():
            raise FileNotFoundError(
                f"Whisper binary not found at: {whisper_bin}"
            )
        
        # Validate Whisper model
        model_path = Path(self.config["whisper"]["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"Whisper model file not found at: {model_path}"
            )

    def _get_file_count(self, path, extension):
        """Helper to count files or directories for progress tracking."""
        if not path.exists():
            return 0
        if extension == "dir":
            return len([d for d in path.iterdir() if d.is_dir()])
        return len([f for f in path.iterdir() if f.suffix.lower() in extension])

    def run(self, input_dir, mode="full", engine="llm"):
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_path}")
            return

        seg_time = self.config.get("whisper", {}).get("segment_time", 600)
        logger.info(f"Starting pipeline in '{mode}' mode using '{engine}' engine.")

        # --- STEP 1: AUDIO EXTRACTION ---
        if mode in ["full", "extract"]:
            video_extensions = (".mp4", ".mkv", ".avi", ".mov")
            video_count = self._get_file_count(input_path, video_extensions)
            logger.info(f"Step 1/4: Audio Extraction | Found {video_count} videos.")
            extractor = AudioExtractor(input_dir=str(input_path), output_dir=str(self.dirs["audio"]), segment_time=seg_time)
            extractor.run()

        # --- STEP 2: TRANSCRIPTION ---
        if mode in ["full", "transcribe"]:
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
            transcriber.run()

        # --- STEP 3: OPTIMIZATION (S1) ---
        if mode in ["full", "optimize"]:
            raw_srt_count = self._get_file_count(self.dirs["raw_srt"], (".srt",))
            logger.info(f"Step 3/4: SRT Optimization (Source S1) | Processing {raw_srt_count} files.")
            optimizer = SRTOptimizer(input_dir=str(self.dirs["raw_srt"]), output_dir=str(self.dirs["clean_srt"]))
            optimizer.run()

        # --- STEP 4: TRANSLATION & REFINEMENT ---
        if mode in ["full", "translate"]:
            clean_srt_count = self._get_file_count(self.dirs["clean_srt"], (".srt",))
            logger.info(f"Step 4/4: Translation | Engine: {engine} | Source: {clean_srt_count} files.")

            if engine == "hybrid":
                self._run_hybrid_pipeline()
            else:
                translator = self._create_translator(engine)
                translator.run()

            logger.info(f"Step 4/4: Completed. Final files: {self.dirs['final']}")

        logger.info("✨ ALL PIPELINE TASKS FINISHED SUCCESSFULLY! ✨")

    def _run_hybrid_pipeline(self):
        """Runs the full hybrid protocol: L1 (Legacy) + Mt (LLM Draft) → Refiner."""
        logger.info("Starting Hybrid Protocol (Arbitration S1/L1/Mt)...")

        # A. Generate L1 (Literal translation via Legacy)
        logger.info("Generating L1 (Literal)...")
        legacy = LegacyTranslator(
            input_dir=str(self.dirs["clean_srt"]),
            output_dir=str(self.dirs["legacy_mt"]),
            config=self.config
        )
        legacy.run()

        # B. Generate Mt (LLM Draft)
        logger.info("Generating Mt (LLM Draft)...")
        provider = LlamaCPPProvider()
        llm_config = self.config.get("llm_config", self.config.get("translation", {}))
        llm_draft = LLMTranslator(
            input_dir=str(self.dirs["clean_srt"]),
            output_dir=str(self.dirs["llm_mt"]),
            provider=provider,
            config=llm_config
        )
        llm_draft.run()

        # C. Final Arbitration (Hybrid Refiner)
        logger.info("Performing Final Hybrid Refinement...")
        refiner_config = {
            **self.config.get("translation", {}),
            "chunk_size": llm_config.get("chunk_size",
                          self.config.get("translation", {}).get("chunk_size", 10)),
        }
        refiner = HybridRefiner(
            s1_dir=str(self.dirs["clean_srt"]),
            l1_dir=str(self.dirs["legacy_mt"]),
            mt_dir=str(self.dirs["llm_mt"]),
            output_dir=str(self.dirs["final"]),
            provider=provider,
            config=refiner_config
        )
        refiner.run()

    def _create_translator(self, engine: str) -> BaseTranslator:
        """Factory method for translator engine creation."""
        engine_factories = {
            "llm-local": self._create_llm_local_translator,
            "llm-ui": self._create_llm_ui_translator,
            "legacy": self._create_legacy_translator,
        }
        factory = engine_factories.get(engine)
        if not factory:
            raise ValueError(
                f"Unknown engine: '{engine}'. Available: {list(engine_factories.keys())}"
            )
        return factory()

    def _create_llm_local_translator(self) -> LLMTranslator:
        logger.info("Initializing Local LLM Provider (llama.cpp)...")
        provider = LlamaCPPProvider()
        return LLMTranslator(
            input_dir=str(self.dirs["clean_srt"]),
            output_dir=str(self.dirs["final"]),
            provider=provider,
            config=self.config["llm_config"]
        )

    def _create_llm_ui_translator(self) -> LLMTranslator:
        logger.info("Initializing UI Automation Provider...")
        provider = CopilotUIProvider(window_title="Edge")
        return LLMTranslator(
            input_dir=str(self.dirs["clean_srt"]),
            output_dir=str(self.dirs["final"]),
            provider=provider,
            config=self.config["llm_config"]
        )

    def _create_legacy_translator(self) -> LegacyTranslator:
        logger.info("Initializing Legacy Translator...")
        return LegacyTranslator(
            input_dir=str(self.dirs["clean_srt"]),
            output_dir=str(self.dirs["final"]),
            config=self.config
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage Video Translation Pipeline")
    parser.add_argument("--input", required=True, help="Path to source video folder")
    parser.add_argument("--output", required=True, help="Path to result folder")
    parser.add_argument("--config", default="configs/settings.json", help="Path to configuration file")
    parser.add_argument("--mode", default="full", choices=["full", "extract", "transcribe", "optimize", "translate"])
    parser.add_argument("--engine", default="legacy", choices=["llm-local","llm-ui", "legacy", "hybrid"])
    
    args = parser.parse_args()
    pipeline = VideoTranslationPipeline(output_dir=args.output, config_path=args.config)
    try:
        pipeline.run(input_dir=args.input, mode=args.mode, engine=args.engine)
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"A critical error occurred: {str(e)}", exc_info=True)