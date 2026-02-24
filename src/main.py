import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# --- Pipeline modules ---
from modules.extractor import AudioExtractor
from modules.legacy_translator import LegacyTranslator
from modules.llm_translator import LLMTranslator

# --- LLM providers ---
from modules.providers.llama_provider import LlamaCPPProvider
from modules.srt_optimizer import SRTOptimizer
from modules.strategies.hybrid_refiner import HybridRefiner
from modules.transcriber import WhisperTranscriber
from modules.translator import BaseTranslator
from utils.srt_handler import SRTHandler

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VideoPipeline")

class VideoTranslationPipeline:
    """Main orchestrator for the 4-step video translation pipeline.

    Steps
    -----
    1. **Audio extraction** — FFmpeg segmentation.
    2. **Transcription** — Whisper.cpp (segment-aware).
    3. **SRT optimization** — merge, clean, re-index (S1).
    4. **Translation** — legacy | llm-local | llm-ui | hybrid.

    The hybrid engine performs triple-source arbitration:
    S1 (source SRT) + L1 (Google Translate literal) + Mt (LLM draft)
    → ``HybridRefiner`` with incremental re-run support.

    Parameters
    ----------
    output_dir : str
        Root directory for all pipeline outputs.  An ``internals/``
        subfolder is created for intermediate artefacts.
    config_path : str, optional
        Path to the JSON configuration file
        (default ``"configs/settings.json"``).

    Raises
    ------
    ValueError
        If the configuration file is missing or contains invalid JSON,
        or if required config keys are absent.
    FileNotFoundError
        If FFmpeg, Whisper binary, or Whisper model are not found.
    """

    def __init__(self, output_dir: str, config_path: str = "configs/settings.json"):
        logger.info("--- Initializing Pipeline ---")
        logger.info(f"Loading configuration from: {config_path}")

        try:
            with open(config_path, encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError as exc:
            raise ValueError(f"Configuration file not found: '{config_path}'") from exc
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file '{config_path}': {e}") from e

        self.final_output = Path(output_dir)
        self.work_dir = self.final_output / "internals"

        # Directory structure — numbered for step ordering.
        # S1 = source anchor, L1 = literal legacy MT, Mt = LLM draft.
        self.dirs = {
            "audio":      self.work_dir / "1_audio",
            "raw_srt":    self.work_dir / "2_raw_srt",
            "clean_srt":  self.work_dir / "3_clean_srt",     # S1 (source anchor)
            "legacy_mt":  self.work_dir / "4_legacy_mt",     # L1 (literal reference)
            "llm_mt":     self.work_dir / "5_llm_mt",        # Mt (LLM draft)
            "final":      self.final_output / "subtitles_ready",
        }

        for _name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)

    def _validate_requirements(self, mode: str, engine: str) -> None:
        """
        Validate config sections and external binaries for the selected mode and engine.

        Only checks the requirements actually needed for the given
        pipeline scope (mode) and translation backend (engine),
        allowing each engine to run independently without requiring
        config or binaries for unused components.

        Parameters
        ----------
        mode : str
            Pipeline scope (``"full"``, ``"extract"``, ``"transcribe"``,
            ``"optimize"``, ``"translate"``).
        engine : str
            Translation engine (``"legacy"``, ``"llm-local"``,
            ``"llm-ui"``, ``"hybrid"``).

        Raises
        ------
        ValueError
            If a required config section or key is missing.
        FileNotFoundError
            If required external binaries are not found.
        """
        # Steps 1 (extract) needs FFmpeg
        if mode in ("full", "extract") and not shutil.which("ffmpeg"):
            raise FileNotFoundError(
                "FFmpeg not found in PATH. Install it and ensure it's accessible."
            )

        # Step 2 (transcribe) needs Whisper config + binaries
        if mode in ("full", "transcribe"):
            self._validate_config_section("whisper", ["bin_path", "model_path"])

            whisper_bin = Path(self.config["whisper"]["bin_path"])
            if not whisper_bin.exists():
                raise FileNotFoundError(f"Whisper binary not found at: {whisper_bin}")

            model_path = Path(self.config["whisper"]["model_path"])
            if not model_path.exists():
                raise FileNotFoundError(f"Whisper model file not found at: {model_path}")

        # Step 4 (translate) needs engine-specific config
        if mode in ("full", "translate"):
            if engine in ("llm-local", "llm-ui", "hybrid"):
                self._validate_config_section("llm_config", ["source_lang", "target_lang"])
            if engine in ("legacy", "hybrid"):
                self._validate_config_section("translation", ["source_lang", "target_lang", "cache_file"])

    def _validate_config_section(self, section: str, keys: list[str]) -> None:
        """
        Validate that a configuration section exists with the required keys.

        Parameters
        ----------
        section : str
            Top-level section name in the config dict.
        keys : list of str
            Required keys within the section.

        Raises
        ------
        ValueError
            If the section or any required key is missing.
        """
        if section not in self.config:
            raise ValueError(f"Missing required config section: '{section}'")
        for key in keys:
            if key not in self.config[section]:
                raise ValueError(f"Missing required config key: '{section}.{key}'")

    def _get_file_count(self, path, extension):
        """
        Count files (or subdirectories) in a directory.

        Parameters
        ----------
        path : Path
            Directory to scan.
        extension : str or tuple of str or ``"dir"``
            If ``"dir"``, counts subdirectories.  Otherwise,
            counts files whose suffix matches.

        Returns
        -------
        int
            Number of matching items, or 0 if *path* does not exist.
        """
        if not path.exists():
            return 0
        if extension == "dir":
            return len([d for d in path.iterdir() if d.is_dir()])
        return len([f for f in path.iterdir() if f.suffix.lower() in extension])

    def run(self, input_dir, mode="full", engine="legacy"):
        """
        Run the transcription / translation pipeline.

        Parameters
        ----------
        input_dir : str
            Path to the folder containing source video files.
        mode : str, optional
            Pipeline scope: ``"full"`` (default), ``"extract"``,
            ``"transcribe"``, ``"optimize"``, or ``"translate"``.
        engine : str, optional
            Translation engine to use: ``"legacy"`` (default),
            ``"llm-local"``, ``"llm-ui"``, or ``"hybrid"``.
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_path}")
            return

        self._validate_requirements(mode, engine)

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

    def _seed_mt_from_previous_run(self) -> None:
        """Copy previous engine output to Mt directory for hybrid reuse.

        When a previous non-hybrid engine run (e.g. ``llm-local``)
        produced files in the final output directory, those can serve
        as the Mt (LLM draft) input for the hybrid refiner.  This
        avoids re-translating files that were already processed.

        Only seeds when the Mt directory (``5_llm_mt/``) contains no
        SRT files and the final directory (``subtitles_ready/``) has
        files whose timestamps match their S1 source.
        """
        mt_dir = self.dirs["llm_mt"]
        final_dir = self.dirs["final"]
        s1_dir = self.dirs["clean_srt"]

        if any(mt_dir.rglob("*.srt")):
            return  # Mt already populated — skip seeding

        final_files = list(final_dir.rglob("*.srt"))
        if not final_files:
            return

        seeded = 0
        for final_file in final_files:
            relative = final_file.relative_to(final_dir)
            s1_file = s1_dir / relative
            mt_target = mt_dir / relative

            if not s1_file.exists():
                continue

            s1_ts = SRTHandler.extract_timestamps(s1_file)
            final_ts = SRTHandler.extract_timestamps(final_file)

            if s1_ts == final_ts and len(s1_ts) > 0:
                mt_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(final_file, mt_target)
                seeded += 1

        if seeded:
            logger.info(
                f"Seeded {seeded} Mt file(s) from previous translation output. "
                f"LLM draft step will skip these files."
            )

    def _run_hybrid_pipeline(self):
        """
        Execute the full hybrid protocol.

        Sequentially generates L1 (literal via ``LegacyTranslator``),
        Mt (LLM draft via ``LLMTranslator``), then runs the
        ``HybridRefiner`` for triple-source arbitration.

        If a previous engine run left files in the final output
        directory, they are seeded into the Mt directory so the LLM
        draft step can skip already-translated files.
        """
        logger.info("Starting Hybrid Protocol (Arbitration S1/L1/Mt)...")

        # 0. Seed Mt from previous non-hybrid run (e.g. llm-local)
        self._seed_mt_from_previous_run()

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
        llm_url = self.config.get("llm_config", {}).get("server_url", "http://127.0.0.1:8080")
        provider = LlamaCPPProvider(url=llm_url)
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
        """
        Instantiate the appropriate translator for the given engine.

        Parameters
        ----------
        engine : str
            One of ``"llm-local"``, ``"llm-ui"``, ``"legacy"``.

        Returns
        -------
        BaseTranslator
            Configured translator instance.

        Raises
        ------
        ValueError
            If *engine* is not a recognised engine name.
        """
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
        """
        Create an ``LLMTranslator`` backed by a local llama.cpp server.

        Returns
        -------
        LLMTranslator
            Translator configured with ``LlamaCPPProvider``.
        """
        logger.info("Initializing Local LLM Provider (llama.cpp)...")
        llm_url = self.config.get("llm_config", {}).get("server_url", "http://127.0.0.1:8080")
        provider = LlamaCPPProvider(url=llm_url)
        return LLMTranslator(
            input_dir=str(self.dirs["clean_srt"]),
            output_dir=str(self.dirs["final"]),
            provider=provider,
            config=self.config["llm_config"]
        )

    def _create_llm_ui_translator(self) -> LLMTranslator:
        """
        Create an ``LLMTranslator`` backed by browser UI automation.

        Imports ``CopilotUIProvider`` lazily so that Windows-only
        dependencies (``pywinauto``, ``win32api``) are not required
        unless this engine is actually selected.

        Returns
        -------
        LLMTranslator
            Translator configured with ``CopilotUIProvider``.

        Raises
        ------
        ImportError
            If required UI automation packages are not installed.
        """
        logger.info("Initializing UI Automation Provider...")
        from modules.providers.copilot_ui import CopilotUIProvider  # pylint: disable=import-outside-toplevel

        provider = CopilotUIProvider(window_title="Edge")
        return LLMTranslator(
            input_dir=str(self.dirs["clean_srt"]),
            output_dir=str(self.dirs["final"]),
            provider=provider,
            config=self.config["llm_config"]
        )

    def _create_legacy_translator(self) -> LegacyTranslator:
        """
        Create a ``LegacyTranslator`` using Google Translate.

        Returns
        -------
        LegacyTranslator
            Translator configured from the pipeline settings.
        """
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
        logger.error(f"A critical error occurred: {e!s}", exc_info=True)
