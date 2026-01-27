import logging
import time
from pathlib import Path
from typing import List, Dict
from modules.translator import BaseTranslator
from utils.srt_handler import SRTHandler
from modules.providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

class LLMTranslator(BaseTranslator):
    def __init__(self, input_dir, output_dir, provider: LLMProvider, config: Dict):
        # BaseTranslator handles file loops and skip logic
        super().__init__(input_dir, output_dir, bot=provider, extensions=(".srt",))
        
        self.provider = provider
        self.config = config
        self.chunk_size = config.get("chunk_size", 30)
        
        # Load Instructions from external .txt file
        self.system_instructions = self._load_custom_prompt(config)

    def _load_custom_prompt(self, config: Dict) -> str:
        """Reads the prompt from a text file and formats it with languages."""
        prompt_path = Path(config.get("prompt_file", "configs/system_prompt.txt"))
        
        if not prompt_path.exists():
            logger.error(f"Prompt file not found at {prompt_path}. Using default fallback.")
            raw_prompt = "Translate from {source_lang} to {target_lang}:"
        else:
            with open(prompt_path, "r", encoding="utf-8") as f:
                raw_prompt = f.read()

        # Inject languages into the template
        return raw_prompt.format(
            source_lang=config.get("source_lang", "English"),
            target_lang=config.get("target_lang", "French")
        )

    def _split_into_chunks(self, blocks: List[Dict]) -> List[List[Dict]]:
        """Groups SRT blocks into manageable chunks for the LLM context window."""
        return [blocks[i : i + self.chunk_size] for i in range(0, len(blocks), self.chunk_size)]

    def translate_logic(self, content: str) -> str:
        """Core orchestration: SRT -> Blocks -> Chunks -> LLM -> Merged SRT"""
        all_blocks = SRTHandler.parse_to_blocks(content)
        chunks = self._split_into_chunks(all_blocks)
        
        translated_full_blocks = []
        total_chunks = len(chunks)

        logger.info(f"Starting translation: {len(all_blocks)} blocks in {total_chunks} chunks.")

        for idx, chunk in enumerate(chunks, 1):
            chunk_text = SRTHandler.render_blocks(chunk)
            
            # Use the loaded and formatted prompt
            prompt = f"CONTENT TO TRANSLATE:\n{chunk_text}"
            
            logger.info(f"Processing Chunk {idx}/{total_chunks}...")
            raw_response = self.provider.ask(self.system_instructions,prompt)
            
            try:
                # Parse response and validate format
                translated_blocks = SRTHandler.parse_to_blocks(raw_response)
                
                if len(translated_blocks) != len(chunk):
                    logger.warning(
                        f"Chunk {idx}: LLM returned {len(translated_blocks)} blocks "
                        f"instead of {len(chunk)}."
                    )
                
                translated_full_blocks.extend(translated_blocks)
            except Exception as e:
                logger.error(f"Failed to parse LLM response for chunk {idx}: {e}")
                translated_full_blocks.extend(chunk)

        return SRTHandler.render_blocks(translated_full_blocks)