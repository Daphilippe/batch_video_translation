import time
import logging
from pathlib import Path
from utils.file_handler import DirectoryMirrorTask
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class BaseTranslator(DirectoryMirrorTask):
    def __init__(self, input_dir, output_dir, bot=None, extensions=(".srt",)):
        super().__init__(input_dir, output_dir, extensions)
        self.bot = bot
        self.name = ""

    def wait_for_stability(self, path: Path, timeout=30):
        start = time.time()
        last_size = -1
        while time.time() - start < timeout:
            if path.exists():
                size = path.stat().st_size
                if size == last_size and size > 0: return True
                last_size = size
            time.sleep(0.1)
        return False

    def process_file(self, input_file: Path):
        output_file = self.get_output_path(input_file, ".srt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if translation is actually needed
        if output_file.exists():
            if SRTHandler.extract_timestamps(input_file) == SRTHandler.extract_timestamps(output_file):
                logger.info(f"[SKIP] {input_file.name} is already translated and valid.")
                return

        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"[TRANSLATING] {input_file.name} with {self.name}...")
        translated_text = self.translate_logic(content)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(translated_text)

        self.wait_for_stability(output_file)
        SRTHandler.normalize_file(output_file, output_file)
        logger.info(f"[DONE] {output_file.name}")

    def translate_logic(self, text: str):
        """To be implemented by specific translation modules."""
        raise NotImplementedError

class LLMTranslator(BaseTranslator):
    def translate_logic(self, text: str, chunk_size=40):
        """Implements the large-scale chunking for LLM context windows."""
        lines = text.splitlines()
        blocks, current_block = [], []

        for line in lines:
            if line.strip().isdigit() and current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
            current_block.append(line)
        if current_block: blocks.append("\n".join(current_block).strip())

        translated_chunks = []
        for i in range(0, len(blocks), chunk_size):
            chunk = "\n\n".join(blocks[i : i + chunk_size])
            logger.info(f"  -> Sending chunk {i//chunk_size + 1} to Copilot")
            translated_chunks.append(self.bot.answer(chunk))

        return "\n\n".join(translated_chunks).strip()