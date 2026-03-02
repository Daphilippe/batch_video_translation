import logging
import time
from pathlib import Path

from modules.providers.base_provider import LLMProvider, LLMProviderError
from modules.translator import BaseTranslator
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class LLMTranslator(BaseTranslator):
    """
    Standalone LLM translator with chunked processing.

    Splits SRT content into manageable chunks, sends each to an
    ``LLMProvider``, and reassembles the translated result.
    Inherits the file-level orchestration from ``BaseTranslator``
    (skip logic, standardization, disk-write stability).

    Supports **mid-file recovery**: after each successful chunk,
    a ``.partial.srt`` checkpoint file is written.  If the process
    is interrupted, the next run resumes from the last saved chunk
    instead of re-translating everything.

    Parameters
    ----------
    input_dir : str
        Directory containing source ``.srt`` files.
    output_dir : str
        Directory where translated ``.srt`` files are written.
    provider : LLMProvider
        Backend that implements ``ask(content, prompt) -> str``.
    config : dict
        Translation settings (``chunk_size``, ``chunk_delay``,
        ``prompt_file``, ``source_lang``, ``target_lang``, …).
    """
    def __init__(self, input_dir: str, output_dir: str, provider: LLMProvider, config: dict):
        super().__init__(input_dir, output_dir, extensions=(".srt",))
        self.provider = provider
        self.config = config
        self.chunk_size = config.get("chunk_size", 30)
        self.chunk_delay = config.get("chunk_delay", 1.0)  # Seconds between LLM calls
        self.name = provider.name if hasattr(provider, 'name') else "LLM"
        self.system_instructions = self._load_custom_prompt(config)
        self._checkpoint_file: Path | None = None

    def _load_custom_prompt(self, config: dict) -> str:
        """
        Load and format the system prompt template from a text file.

        Reads the prompt file specified in *config* and substitutes
        ``{source_lang}`` / ``{target_lang}`` placeholders.  Falls
        back to a minimal inline template if the file is missing.

        Parameters
        ----------
        config : dict
            Must contain ``prompt_file``, ``source_lang``, and
            ``target_lang`` keys.

        Returns
        -------
        str
            Formatted system instructions for the LLM.
        """
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
        """
        Group SRT blocks into fixed-size chunks.

        Each chunk is a contiguous slice of *blocks* whose length
        does not exceed ``self.chunk_size``.

        Parameters
        ----------
        blocks : list of dict
            Parsed SRT blocks.

        Returns
        -------
        list of list of dict
            Ordered list of block chunks.

        Examples
        --------
        >>> t = LLMTranslator.__new__(LLMTranslator)
        >>> t.chunk_size = 2
        >>> t._split_into_chunks([{'a': 1}, {'a': 2}, {'a': 3}])
        [[{'a': 1}, {'a': 2}], [{'a': 3}]]
        """
        return [blocks[i : i + self.chunk_size] for i in range(0, len(blocks), self.chunk_size)]

    def process_file(self, input_file: Path) -> None:
        """
        Translate a single SRT file with checkpoint support.

        Sets up a ``.partial.srt`` checkpoint path before delegating
        to the parent's orchestration.  On successful completion the
        checkpoint is removed; on interruption it is preserved so
        that the next run can resume from the last saved chunk.

        Parameters
        ----------
        input_file : Path
            Path to the source ``.srt`` file.
        """
        output_file = self.get_output_path(input_file, ".srt")
        self._checkpoint_file = output_file.with_suffix(".partial.srt")
        try:
            super().process_file(input_file)
        finally:
            # Clean up checkpoint only when the final output was written
            if output_file.exists() and self._checkpoint_file and self._checkpoint_file.exists():
                self._checkpoint_file.unlink(missing_ok=True)
            self._checkpoint_file = None

    def _translate_chunk(self, chunk: list[dict], idx: int, total: int, _attempt: int = 0) -> list[dict]:
        """
        Translate a single chunk, falling back to source on error.

        After translation, validates that the output differs from
        the source.  If the chunk appears untranslated, retries up
        to ``max_chunk_retries`` times (config, default 2).

        Parameters
        ----------
        chunk : list of dict
            SRT blocks to translate.
        idx : int
            1-based chunk index (for logging).
        total : int
            Total number of chunks (for logging).
        _attempt : int, optional
            Internal retry counter (default 0).

        Returns
        -------
        list of dict
            Translated blocks, or original *chunk* on failure.
        """
        chunk_text = SRTHandler.render_blocks(chunk)
        prompt = f"CONTENT TO TRANSLATE:\n{chunk_text}"

        logger.info(f"Processing Chunk {idx}/{total}...")

        try:
            raw_response = self.provider.ask(self.system_instructions, prompt)
        except LLMProviderError as e:
            logger.error(f"Chunk {idx}: Provider error: {e}. Keeping source text.")
            return list(chunk)

        try:
            translated_blocks = SRTHandler.parse_to_blocks(raw_response)

            if not translated_blocks:
                logger.warning(f"Chunk {idx}: LLM returned empty/unparseable response. Keeping source.")
                return list(chunk)

            if len(translated_blocks) != len(chunk):
                logger.warning(
                    f"Chunk {idx}: LLM returned {len(translated_blocks)} blocks "
                    f"instead of {len(chunk)}."
                )

            # Validate: translated content must differ from source
            if self._is_chunk_untranslated(chunk, translated_blocks):
                max_retries = self.config.get("max_chunk_retries", 2)
                if _attempt < max_retries:
                    logger.warning(
                        f"Chunk {idx}: translation identical to source. "
                        f"Retrying ({_attempt + 1}/{max_retries})..."
                    )
                    time.sleep(self.chunk_delay)
                    return self._translate_chunk(chunk, idx, total, _attempt + 1)
                logger.warning(
                    f"Chunk {idx}: still identical after {max_retries} retries. Keeping source."
                )
                return list(chunk)

            return translated_blocks
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse LLM response for chunk {idx}: {e}")
            return list(chunk)

    def _save_checkpoint(self, blocks: list[dict]) -> None:
        """
        Persist partial translation progress to a checkpoint file.

        Parameters
        ----------
        blocks : list of dict
            All translated blocks accumulated so far.
        """
        if not self._checkpoint_file:
            return
        try:
            self._checkpoint_file.write_text(
                SRTHandler.render_blocks(blocks), encoding="utf-8"
            )
        except OSError as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, total_source_blocks: int) -> tuple[list[dict], int]:
        """
        Load partial progress from a previous interrupted run.

        Reads the ``.partial.srt`` checkpoint, validates its block
        count against the source, and calculates the chunk index
        to resume from.

        Parameters
        ----------
        total_source_blocks : int
            Total blocks in the source SRT (used for validation).

        Returns
        -------
        recovered_blocks : list of dict
            Blocks recovered from the checkpoint (complete chunks only).
        resume_from : int
            1-based chunk index to resume from (0 = start fresh).
        """
        if not self._checkpoint_file or not self._checkpoint_file.exists():
            return [], 0

        try:
            partial_text = self._checkpoint_file.read_text(encoding="utf-8")
            partial_blocks = SRTHandler.parse_to_blocks(partial_text)

            if not partial_blocks or len(partial_blocks) > total_source_blocks:
                logger.warning("Checkpoint invalid (block count mismatch). Starting fresh.")
                return [], 0

            # Keep only complete chunks to ensure consistency
            resume_from = len(partial_blocks) // self.chunk_size
            keep = resume_from * self.chunk_size
            recovered = partial_blocks[:keep]

            if resume_from > 0:
                logger.info(
                    f"Resuming from chunk {resume_from + 1} "
                    f"({keep} blocks recovered from checkpoint)"
                )

            return recovered, resume_from
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to read checkpoint: {e}. Starting fresh.")
            return [], 0

    def translate_logic(self, text: str) -> str:
        """
        Translate full SRT content via chunked LLM calls.

        Pipeline: parse SRT → check checkpoint → split into chunks
        → resume or start → send each chunk to the provider → save
        checkpoint → concatenate all blocks → render back to SRT.

        On provider or parse failure the source chunk is kept
        unchanged.

        Parameters
        ----------
        text : str
            Raw SRT file content.

        Returns
        -------
        str
            Translated SRT content (un-standardized).
        """
        all_blocks = SRTHandler.parse_to_blocks(text)
        chunks = self._split_into_chunks(all_blocks)
        total_chunks = len(chunks)

        # Attempt to resume from a previous interrupted run
        translated_full_blocks, resume_from = self._load_checkpoint(len(all_blocks))

        if not resume_from:
            logger.info(f"Starting translation: {len(all_blocks)} blocks in {total_chunks} chunks.")

        for idx, chunk in enumerate(chunks, 1):
            if idx <= resume_from:
                continue

            result = self._translate_chunk(chunk, idx, total_chunks)
            translated_full_blocks.extend(result)
            self._save_checkpoint(translated_full_blocks)

            # Rate-limit protection: pause between chunks to avoid overloading
            # the LLM server (local or remote). Configurable via "chunk_delay".
            if idx < total_chunks:
                time.sleep(self.chunk_delay)

        return SRTHandler.render_blocks(translated_full_blocks)
