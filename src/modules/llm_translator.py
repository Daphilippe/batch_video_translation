import logging
import time
from pathlib import Path

from modules.providers.base_provider import LLMProvider
from modules.translator import BaseTranslator
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class LLMTranslator(BaseTranslator):
    """
    Standalone LLM translator: splits SRT into chunks, sends each to the provider,
    and reassembles the result. Works through BaseTranslator.process_file() which
    handles skip logic, standardization, and wait_for_stability.
    """
    def __init__(self, input_dir: str, output_dir: str, provider: LLMProvider, config: dict):
        super().__init__(input_dir, output_dir, extensions=(".srt",))
        self.provider = provider
        self.config = config
        self.chunk_size = config.get("chunk_size", 30)
        self.chunk_delay = config.get("chunk_delay", 1.0)  # Seconds between LLM calls
        self.name = provider.name if hasattr(provider, 'name') else "LLM"
        self.system_instructions = self._load_custom_prompt(config)

    def _load_custom_prompt(self, config: dict) -> str:
        """Reads the prompt from a text file and formats it with languages."""
        prompt_path = Path(config.get("prompt_file", "configs/system_prompt.txt"))
        if not prompt_path.exists():
            logger.warning(f"Prompt file not found at {prompt_path}. Using fallback.")
            raw_prompt = "Translate from {source_lang} to {target_lang}:"
        else:
            with open(prompt_path, encoding="utf-8") as f:
                raw_prompt = f.read()

        return raw_prompt.format(
            source_lang=config.get("source_lang", "English"),
            target_lang=config.get("target_lang", "French")
        )

    def _split_into_chunks(self, blocks: list[dict]) -> list[list[dict]]:
        """Groups SRT blocks into manageable chunks for the LLM context window."""
        return [blocks[i : i + self.chunk_size] for i in range(0, len(blocks), self.chunk_size)]

    def translate_logic(self, text: str) -> str:
        """Core orchestration: SRT -> Blocks -> Chunks -> LLM -> Merged SRT"""
        all_blocks = SRTHandler.parse_to_blocks(text)
        chunks = self._split_into_chunks(all_blocks)

        translated_full_blocks = []
        total_chunks = len(chunks)

        logger.info(f"Starting translation: {len(all_blocks)} blocks in {total_chunks} chunks.")

        for idx, chunk in enumerate(chunks, 1):
            chunk_text = SRTHandler.render_blocks(chunk)
            prompt = f"CONTENT TO TRANSLATE:\n{chunk_text}"

            logger.info(f"Processing Chunk {idx}/{total_chunks}...")

            try:
                raw_response = self.provider.ask(self.system_instructions, prompt)
            except Exception as e:
                logger.error(f"Chunk {idx}: Provider error: {e}. Keeping source text.")
                translated_full_blocks.extend(chunk)
                continue

            try:
                translated_blocks = SRTHandler.parse_to_blocks(raw_response)

                if not translated_blocks:
                    logger.warning(f"Chunk {idx}: LLM returned empty/unparseable response. Keeping source.")
                    translated_full_blocks.extend(chunk)
                    continue

                if len(translated_blocks) != len(chunk):
                    logger.warning(
                        f"Chunk {idx}: LLM returned {len(translated_blocks)} blocks "
                        f"instead of {len(chunk)}."
                    )

                translated_full_blocks.extend(translated_blocks)
            except Exception as e:
                logger.error(f"Failed to parse LLM response for chunk {idx}: {e}")
                translated_full_blocks.extend(chunk)

            # Rate-limit protection: pause between chunks to avoid overloading
            # the LLM server (local or remote). Configurable via "chunk_delay".
            if idx < total_chunks:
                time.sleep(self.chunk_delay)

        return SRTHandler.render_blocks(translated_full_blocks)
