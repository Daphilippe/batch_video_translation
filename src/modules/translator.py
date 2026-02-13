import time
import logging
from pathlib import Path
from typing import Optional
from utils.file_handler import DirectoryMirrorTask
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class BaseTranslator(DirectoryMirrorTask):
    def __init__(self, input_dir: str, output_dir: str, extensions: tuple = (".srt",)):
        super().__init__(input_dir, output_dir, extensions)
        self.name = "Base"

    def wait_for_stability(self, path: Path, timeout: int = 10) -> bool:
        """
        Ensures the file is fully written to disk before proceeding.
        Useful on Windows where antivirus/indexers may hold file locks,
        or when I/O is slow on network-mounted drives.
        """
        start = time.time()
        last_size = -1
        while time.time() - start < timeout:
            if path.exists():
                size = path.stat().st_size
                if size == last_size and size > 0:
                    return True
                last_size = size
            time.sleep(0.1)
        return False

    def process_file(self, input_file: Path) -> None:
        output_file = self.get_output_path(input_file, ".srt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 1. ROBUST SKIP LOGIC
        # We compare timestamps to ensure structural identity
        if output_file.exists():
            source_ts = SRTHandler.extract_timestamps(input_file)
            target_ts = SRTHandler.extract_timestamps(output_file)
            
            if source_ts == target_ts and len(source_ts) > 0:
                logger.info(f"[SKIP] {input_file.name} is already translated and matches source structure.")
                return
            logger.warning(f"[REDO] {input_file.name} structure mismatch or missing. Re-translating...")

        # 2. READING CONTENT
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"[TRANSLATING] {input_file.name} using engine: {self.name}...")
        
        # 3. CORE TRANSLATION (Overridden by LLMTranslator or LegacyTranslator)
        start_time = time.time()
        translated_raw = self.translate_logic(content)

        # 4. POST-PROCESS CONDITIONING
        # We standardize the LLM output (clean ghost boxes, fix indices, strip markdown)
        # using the EXACT same logic applied in Step 3 (Optimizer).
        final_standardized = SRTHandler.standardize(translated_raw)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_standardized)
        
        self.wait_for_stability(output_file)
        
        duration = (time.time() - start_time) / 60
        logger.info(f"[DONE] {output_file.name} in {duration:.2f} minutes.")

    def translate_logic(self, text: str) -> str:
        """To be implemented by specific translation modules."""
        raise NotImplementedError