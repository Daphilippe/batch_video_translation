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
        super().__init__(input_dir, output_dir,bot=provider, extensions=(".srt",))
        self.provider = provider
        self.config = config
        self.chunk_size = config.get("chunk_size", 30)  # Number of SRT blocks per request
        
        # This prompt ensures the LLM doesn't break your file
        self.system_instructions = (
            "You are a professional translator. Translate the following SRT subtitle blocks "
            f"from {config['source_lang']} to {config['target_lang']}. "
            "CRITICAL: Keep the exact same timestamp and index for every block. "
            "Put it in a Markdown section to make copying easier."
            "Return ONLY the valid SRT content. Do not add conversational text."
        )
        self.name="LLM translation"

    def _split_into_chunks(self, blocks: List[Dict]) -> List[List[Dict]]:
        """Groups SRT blocks into manageable chunks for the LLM context window."""
        return [blocks[i : i + self.chunk_size] for i in range(0, len(blocks), self.chunk_size)]

    def translate_logic(self, content: str) -> str:
        """
        The core orchestration: 
        SRT -> Blocks -> Chunks -> LLM -> Merged SRT
        """
        # 1. Parse raw text into structured objects
        all_blocks = SRTHandler.parse_to_blocks(content)
        chunks = self._split_into_chunks(all_blocks)
        
        translated_full_blocks = []
        total_chunks = len(chunks)

        logger.info(f"Starting translation: {len(all_blocks)} blocks in {total_chunks} chunks.")

        for idx, chunk in enumerate(chunks, 1):
            # 2. Render chunk back to SRT text to send as a prompt
            chunk_text = SRTHandler.render_blocks(chunk)
            
            prompt = f"{self.system_instructions}\n\nCONTENT TO TRANSLATE:\n{chunk_text}"
            
            logger.info(f"Processing Chunk {idx}/{total_chunks}...")
            
            # 3. Call the provider (Copilot UI, OpenAI, etc.)
            # If it's CopilotUI, it will trigger the manual clicks/copy
            raw_response = self.provider.ask(prompt)
            
            # 4. Parse the LLM response back into blocks
            # This is safer than just appending text because it validates the SRT format
            try:
                translated_blocks = SRTHandler.parse_to_blocks(raw_response)
                
                # Check if the LLM missed any blocks
                if len(translated_blocks) != len(chunk):
                    logger.warning(
                        f"Chunk {idx}: LLM returned {len(translated_blocks)} blocks "
                        f"but we sent {len(chunk)}. Attempting to align..."
                    )
                
                translated_full_blocks.extend(translated_blocks)
            except Exception as e:
                logger.error(f"Failed to parse LLM response for chunk {idx}: {e}")
                # Fallback: add untranslated chunk to avoid losing data
                translated_full_blocks.extend(chunk)

        # 5. Final Rendering
        return SRTHandler.render_blocks(translated_full_blocks)

    def process_file(self, input_file: Path):
        """Overrides the parent process to add file-specific logging."""
        output_file = self.get_output_path(input_file, ".srt")
        
        # Check if already done (Timestamp comparison)
        if output_file.exists():
            if SRTHandler.extract_timestamps(input_file) == SRTHandler.extract_timestamps(output_file):
                logger.info(f"Subtitles already translated: {output_file.name}")
                return

        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        start_time = time.time()
        translated_srt = self.translate_logic(content)
        
        # Final formatting cleanup
        optimized_srt = SRTHandler.render_blocks(
            SRTHandler.merge_identical_blocks(
                SRTHandler.parse_to_blocks(translated_srt)
            )
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(optimized_srt)

        duration = (time.time() - start_time) / 60
        logger.info(f"Successfully translated {input_file.name} in {duration:.2f} minutes.")