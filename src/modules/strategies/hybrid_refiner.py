import logging
import time
from pathlib import Path

from modules.translator import BaseTranslator
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)


class HybridRefiner(BaseTranslator):
    """
    Triple-source arbitration translator (S1 + L1 + Mt → Final).

    Aligns three SRT streams — S1 (source anchor / Whisper), L1
    (literal Google Translate), and Mt (LLM draft) — using a
    two-pointer sweep on start-timestamps, then sends windowed
    prompts to the LLM for final refinement.

    Supports **incremental re-runs**: on subsequent executions only
    windows containing problematic blocks (empty, untranslated, or
    missing) are re-processed.

    Parameters
    ----------
    s1_dir : str
        Directory of source SRT files (master timeline).
    l1_dir : str
        Directory of literal translations (Google Translate).
    mt_dir : str
        Directory of LLM draft translations.
    output_dir : str
        Directory for final refined SRT files.
    provider : LLMProvider
        Backend used for refinement prompts.
    config : dict
        Settings including ``chunk_size``, ``chunk_delay``, and
        ``refinement_protocol_file``.
    """

    # Maximum allowed drift (seconds) between an S1 block and its L1/Mt match.
    # Beyond this, the block is considered unmatched.
    ALIGNMENT_TOLERANCE = 5.0

    def __init__(self, s1_dir: str, l1_dir: str, mt_dir: str, output_dir: str, provider, config: dict):
        super().__init__(s1_dir, output_dir)
        self.l1_dir = Path(l1_dir)
        self.mt_dir = Path(mt_dir)
        self.config = config
        self.bot = provider
        self.name = "Hybrid Refiner (Triple-Source Arbitration)"
        self.chunk_delay = config.get("chunk_delay", 1.0)

    # ------------------------------------------------------------------
    # File-level orchestration
    # ------------------------------------------------------------------

    def process_file(self, input_file: Path):  # pylint: disable=arguments-renamed
        """
        Refine a single SRT file via triple-source arbitration.

        Loads S1, L1, and Mt streams, optionally loads existing
        output for incremental refinement, runs ``refine_logic``,
        standardizes and writes the result.

        Parameters
        ----------
        input_file : Path
            Path to the S1 (source) ``.srt`` file.
        """
        output_file = self.get_output_path(input_file, ".srt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        relative_path = input_file.relative_to(self.input_dir)
        l1_file = self.l1_dir / relative_path
        mt_file = self.mt_dir / relative_path

        if not l1_file.exists() or not mt_file.exists():
            logger.error(f"Missing input streams for {input_file.name}. L1: {l1_file.exists()}, Mt: {mt_file.exists()}")
            return

        logger.info(f"Starting triple-source refinement for: {input_file.name}")

        # Load existing output for incremental refinement
        existing_refined_text = None
        if output_file.exists():
            try:
                with open(output_file, encoding="utf-8") as f:
                    existing_refined_text = f.read()
                logger.info(f"Existing output found for {input_file.name}. Attempting incremental refinement.")
            except Exception as e:
                logger.warning(f"Failed to read existing output: {e}. Performing full refinement.")

        try:
            with open(input_file, encoding="utf-8") as f:
                s1_raw = f.read()
            with open(l1_file, encoding="utf-8") as f:
                l1_raw = f.read()
            with open(mt_file, encoding="utf-8") as f:
                mt_raw = f.read()
        except Exception as e:
            logger.error(f"Failed to read input files for refinement: {e}")
            return

        final_srt_content = self.refine_logic(s1_raw, l1_raw, mt_raw, existing_refined_text)

        if final_srt_content is None:
            logger.info(f"No changes needed for {input_file.name}. Skipping write.")
            return

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(SRTHandler.standardize(final_srt_content))
            self.wait_for_stability(output_file)
            logger.info(f"Successfully refined and saved: {output_file.name}")
        except Exception as e:
            logger.error(f"Failed to save refined SRT: {e}")

    # ------------------------------------------------------------------
    # Alignment engine
    # ------------------------------------------------------------------

    @staticmethod
    def _ts(block: dict) -> float:
        """
        Convert a block's start timestamp to seconds.

        Parameters
        ----------
        block : dict
            SRT block with a ``start`` key.

        Returns
        -------
        float
            Start time in fractional seconds.
        """
        return SRTHandler.timestamp_to_seconds(block['start'])

    def _build_alignment_map(self, s1_blocks: list[dict], target_blocks: list[dict]) -> dict[int, dict]:
        """
        Map each S1 block index to the closest target block.

        Uses a two-pointer sweep (both lists are time-ordered)
        for *O(n+m)* complexity.  A target block may be reused
        across multiple S1 indices when it spans a merged region.

        Parameters
        ----------
        s1_blocks : list of dict
            Parsed S1 (source) SRT blocks.
        target_blocks : list of dict
            Parsed L1 or Mt blocks to align against S1.

        Returns
        -------
        dict of int to dict
            ``{s1_index: closest_target_block}`` for all matches
            within ``ALIGNMENT_TOLERANCE`` seconds.
        """
        alignment: dict[int, dict] = {}
        if not target_blocks:
            return alignment

        t_ptr = 0
        for s_idx, s1_blk in enumerate(s1_blocks):
            s_start = self._ts(s1_blk)
            best = None
            best_dist = float('inf')

            # Scan forward from current pointer
            scan = t_ptr
            while scan < len(target_blocks):
                t_start = self._ts(target_blocks[scan])
                dist = abs(s_start - t_start)
                if dist < best_dist:
                    best_dist = dist
                    best = scan
                elif t_start > s_start + self.ALIGNMENT_TOLERANCE:
                    # Past any reasonable match, stop scanning
                    break
                scan += 1

            # Also check one block before t_ptr (edge case when blocks overlap)
            if t_ptr > 0:
                prev_dist = abs(s_start - self._ts(target_blocks[t_ptr - 1]))
                if prev_dist < best_dist:
                    best_dist = prev_dist
                    best = t_ptr - 1

            if best is not None and best_dist <= self.ALIGNMENT_TOLERANCE:
                alignment[s_idx] = target_blocks[best]
                # Advance pointer to avoid rescanning far-behind blocks
                t_ptr = best

        return alignment

    def _collect_window_targets(self, window_indices: list[int], alignment_map: dict[int, dict]) -> list[dict]:
        """
        Collect unique, ordered target blocks for a window of S1 indices.

        Deduplicates by object identity so that a merged target block
        mapped to several S1 blocks appears only once in the prompt.

        Parameters
        ----------
        window_indices : list of int
            S1 block indices belonging to the current window.
        alignment_map : dict of int to dict
            Alignment map produced by ``_build_alignment_map``.

        Returns
        -------
        list of dict
            Unique target blocks in order of first appearance.
        """
        seen_ids = set()
        result = []
        for idx in window_indices:
            blk = alignment_map.get(idx)
            if blk is not None and id(blk) not in seen_ids:
                seen_ids.add(id(blk))
                result.append(blk)
        return result

    def _log_alignment_quality(self, label: str, s1_blocks: list[dict], alignment_map: dict[int, dict]):
        """
        Log alignment match statistics for diagnostics.

        Parameters
        ----------
        label : str
            Human-readable stream name (e.g. ``"L1"`` or ``"Mt"``).
        s1_blocks : list of dict
            Full list of S1 blocks.
        alignment_map : dict of int to dict
            Mapping from S1 index to matched target block.
        """
        total = len(s1_blocks)
        matched = len(alignment_map)
        if matched == total:
            logger.info(f"Alignment [{label}]: {matched}/{total} blocks matched (100%).")
        else:
            pct = (matched / total * 100) if total else 0
            unmatched = [i for i in range(total) if i not in alignment_map]
            logger.warning(
                f"Alignment [{label}]: {matched}/{total} blocks matched ({pct:.0f}%). "
                f"Unmatched S1 indices: {unmatched[:10]}{'...' if len(unmatched) > 10 else ''}"
            )

    # ------------------------------------------------------------------
    # Incremental refinement — detect problematic blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _block_text(block: dict) -> str:
        """
        Extract a block's text content as a single stripped string.

        Parameters
        ----------
        block : dict
            SRT block with a ``text`` key (str or list of str).

        Returns
        -------
        str
            Concatenated, stripped text.

        Examples
        --------
        >>> len(HybridRefiner._block_text({"text": ["Hello", "World"]}))
        11

        >>> HybridRefiner._block_text({"text": "Single line"})
        'Single line'
        """
        text = block.get('text', '')
        if isinstance(text, list):
            return "\n".join(text).strip()
        return str(text).strip()

    def _identify_problematic_indices(
        self, s1_blocks: list[dict], existing_blocks: list[dict]
    ) -> tuple:
        """
        Detect blocks that need re-refinement.

        Compares existing refined output against the S1 reference.
        A block is *problematic* when:

        - No existing block aligns to that S1 index.
        - The aligned block has empty text.
        - The aligned block text is identical to S1 (not translated).

        Parameters
        ----------
        s1_blocks : list of dict
            Parsed S1 (source) blocks.
        existing_blocks : list of dict
            Parsed blocks from the previous refined output.

        Returns
        -------
        problematic : set of int
            S1 indices that require re-processing.
        existing_map : dict of int to dict
            Alignment map of S1 indices to existing blocks.
        """
        existing_map = self._build_alignment_map(s1_blocks, existing_blocks)
        problematic = set()

        for i, s1_blk in enumerate(s1_blocks):
            ex_blk = existing_map.get(i)

            if ex_blk is None:
                problematic.add(i)
                continue

            ex_text = self._block_text(ex_blk)
            s1_text = self._block_text(s1_blk)

            if not ex_text:
                problematic.add(i)
                continue

            if ex_text == s1_text:
                problematic.add(i)
                continue

        return problematic, existing_map

    @staticmethod
    def _expand_problematic_indices(
        problematic: set, total_blocks: int, context_padding: int = 2
    ) -> set:
        """
        Expand problematic indices with surrounding context.

        Adds *context_padding* neighbours on each side so that the
        LLM receives enough local context to produce coherent
        translations around problematic areas.

        Parameters
        ----------
        problematic : set of int
            Indices flagged as needing re-refinement.
        total_blocks : int
            Total number of S1 blocks (used for clamping).
        context_padding : int, optional
            Number of extra blocks on each side (default 2).

        Returns
        -------
        set of int
            Expanded set of indices to include in refinement.

        Examples
        --------
        >>> sorted(HybridRefiner._expand_problematic_indices({5}, 20, 2))
        [3, 4, 5, 6, 7]

        >>> sorted(HybridRefiner._expand_problematic_indices({0}, 10, 2))
        [0, 1, 2]
        """
        expanded = set()
        for idx in problematic:
            lo = max(0, idx - context_padding)
            hi = min(total_blocks, idx + context_padding + 1)
            expanded.update(range(lo, hi))
        return expanded

    # ------------------------------------------------------------------
    # Core arbitration loop
    # ------------------------------------------------------------------

    def refine_logic(self, s1_text: str, l1_text: str, mt_text: str, existing_refined_text: str | None = None) -> str:
        """
        Perform windowed triple-source arbitration via LLM.

        Aligns L1 and Mt blocks to S1, slices the SRT into windows
        of ``chunk_size`` blocks, and sends each window to the LLM
        with an arbitration prompt.  S1 timestamps are always
        enforced on the output.

        When *existing_refined_text* is provided (re-run scenario),
        only windows containing problematic blocks are sent to the
        LLM; already-good blocks are reused from the existing output.

        Parameters
        ----------
        s1_text : str
            Raw SRT content from the source (Whisper) stream.
        l1_text : str
            Raw SRT content from the literal (Google) stream.
        mt_text : str
            Raw SRT content from the LLM draft stream.
        existing_refined_text : str or None, optional
            Previously refined SRT content for incremental mode
            (default ``None`` → full refinement).

        Returns
        -------
        str or None
            Refined SRT content, or ``None`` if incremental mode
            determined that no re-processing is needed.
        """
        s1_blocks = SRTHandler.parse_to_blocks(s1_text)
        l1_blocks = SRTHandler.parse_to_blocks(l1_text)
        mt_blocks = SRTHandler.parse_to_blocks(mt_text)

        logger.info(f"Block counts — S1: {len(s1_blocks)}, L1: {len(l1_blocks)}, Mt: {len(mt_blocks)}")

        # Build alignment maps (once per file, not per window)
        l1_map = self._build_alignment_map(s1_blocks, l1_blocks)
        mt_map = self._build_alignment_map(s1_blocks, mt_blocks)
        self._log_alignment_quality("L1", s1_blocks, l1_map)
        self._log_alignment_quality("Mt", s1_blocks, mt_map)

        # --- Incremental refinement: detect problematic blocks ---
        indices_to_refine = None   # None = process all windows (first run)
        existing_map = {}

        if existing_refined_text:
            existing_blocks = SRTHandler.parse_to_blocks(existing_refined_text)
            problematic, existing_map = self._identify_problematic_indices(s1_blocks, existing_blocks)

            if not problematic:
                logger.info("All blocks are already properly refined. Nothing to re-process.")
                return None  # Signal to process_file: skip write

            indices_to_refine = self._expand_problematic_indices(problematic, len(s1_blocks))
            logger.info(
                f"Incremental refinement: {len(problematic)} problematic blocks "
                f"→ {len(indices_to_refine)} blocks to process (with context padding)."
            )

        # Load the system protocol
        protocol_path = Path(self.config.get("refinement_protocol_file", "configs/refinement_protocol.txt"))
        if not protocol_path.exists():
            logger.warning("Refinement protocol file not found. Using fallback instructions.")
            system_instructions = "You are a professional translator. Refine the translation using the provided sources."
        else:
            system_instructions = protocol_path.read_text(encoding="utf-8")

        final_blocks = []
        step = self.config.get("chunk_size", 10)
        total_blocks = len(s1_blocks)
        total_windows = (total_blocks - 1) // step + 1
        skipped_windows = 0

        logger.info(f"Arbitrating {total_blocks} blocks using windows of {step}.")

        for i in range(0, total_blocks, step):
            window_indices = list(range(i, min(i + step, total_blocks)))
            s1_win = [s1_blocks[j] for j in window_indices]
            win_label = f"{s1_win[0]['start']} → {s1_win[-1]['end']}"

            # --- Incremental: skip windows where all blocks are OK ---
            if indices_to_refine is not None:
                window_needs_work = any(idx in indices_to_refine for idx in window_indices)
                if not window_needs_work:
                    skipped_windows += 1
                    logger.info(
                        f"Window [{i // step + 1}/{total_windows}] "
                        f"{win_label} | SKIP — reusing existing translation"
                    )
                    for j in window_indices:
                        ex_blk = existing_map.get(j)
                        if ex_blk is not None:
                            reused = dict(ex_blk)
                            reused['start'] = s1_blocks[j]['start']
                            reused['end'] = s1_blocks[j]['end']
                            final_blocks.append(reused)
                        else:
                            final_blocks.append(dict(s1_blocks[j]))
                    continue

            l1_win = self._collect_window_targets(window_indices, l1_map)
            mt_win = self._collect_window_targets(window_indices, mt_map)

            if not l1_win:
                logger.warning(f"Window {i}: No L1 blocks aligned. Legacy stream may be incomplete.")
            if not mt_win:
                logger.warning(f"Window {i}: No Mt blocks aligned. LLM draft stream may be incomplete.")

            user_prompt = self._build_arbitration_prompt(s1_win, l1_win, mt_win)
            logger.info(
                f"Window [{i // step + 1}/{total_windows}] "
                f"{win_label} | S1={len(s1_win)} L1={len(l1_win)} Mt={len(mt_win)}"
            )

            try:
                response = self.bot.ask(system_instructions, user_prompt)
                refined_blocks = SRTHandler.parse_to_blocks(response)

                if not refined_blocks:
                    logger.warning(f"Window {i}: LLM returned empty/unparseable response. Keeping S1 source.")
                    final_blocks.extend(s1_win)
                    continue

                if len(refined_blocks) != len(s1_win):
                    logger.warning(
                        f"Window {i}: LLM returned {len(refined_blocks)} blocks, "
                        f"expected {len(s1_win)}. Forcing S1 timestamps onto result."
                    )
                    refined_blocks = self._force_align_to_s1(s1_win, refined_blocks)

                # Always enforce S1 timestamps on the final output
                for s1_blk, ref_blk in zip(s1_win, refined_blocks):
                    ref_blk['start'] = s1_blk['start']
                    ref_blk['end'] = s1_blk['end']

                final_blocks.extend(refined_blocks)
            except Exception as e:
                logger.error(f"Error during arbitration at window {i}: {e}")
                final_blocks.extend(s1_win)

            if i + step < total_blocks:
                time.sleep(self.chunk_delay)

        if indices_to_refine is not None:
            processed = total_windows - skipped_windows
            logger.info(
                f"Incremental refinement complete: {processed}/{total_windows} windows "
                f"sent to LLM, {skipped_windows} reused from existing output."
            )

        return SRTHandler.render_blocks(final_blocks)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_arbitration_prompt(self, s1: list[dict], l1: list[dict], mt: list[dict]) -> str:
        """
        Build the XML-structured arbitration prompt for the LLM.

        Includes the three source streams with explicit block counts
        so the LLM knows exactly how many blocks to output and that
        L1/Mt may differ in count.

        Parameters
        ----------
        s1 : list of dict
            S1 (source) blocks for the current window.
        l1 : list of dict
            L1 (literal) blocks aligned to this window.
        mt : list of dict
            Mt (LLM draft) blocks aligned to this window.

        Returns
        -------
        str
            Formatted prompt ready to be sent to the LLM.
        """
        return f"""<S1_SOURCE_ORIGINAL count="{len(s1)}">
{SRTHandler.render_blocks(s1)}
</S1_SOURCE_ORIGINAL>

<L1_LITERAL_REFERENCE count="{len(l1)}">
{SRTHandler.render_blocks(l1)}
</L1_LITERAL_REFERENCE>

<MT_DRAFT_STYLE count="{len(mt)}">
{SRTHandler.render_blocks(mt)}
</MT_DRAFT_STYLE>

CRITICAL: Output EXACTLY {len(s1)} SRT blocks with the EXACT timestamps from S1_SOURCE.
L1 and MT may have a different block count due to merging — use them as semantic/stylistic reference only.
Do NOT merge, split, reorder, or skip any block. One S1 block = one output block."""

    # ------------------------------------------------------------------
    # Safety: force-align LLM output to S1 block count
    # ------------------------------------------------------------------

    @staticmethod
    def _force_align_to_s1(s1_win: list[dict], refined: list[dict]) -> list[dict]:
        """
        Pad or trim refined blocks to match the S1 block count.

        If the LLM returned too few blocks, S1 source blocks are
        appended as untranslated fallbacks.  If too many, the list
        is trimmed.

        Parameters
        ----------
        s1_win : list of dict
            S1 blocks for the current window (defines target count).
        refined : list of dict
            LLM-produced blocks.

        Returns
        -------
        list of dict
            Block list with exactly ``len(s1_win)`` elements.

        Examples
        --------
        >>> s1 = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        >>> ref = [{"text": "x"}]
        >>> result = HybridRefiner._force_align_to_s1(s1, ref)
        >>> len(result)
        3
        >>> result[0]['text']
        'x'
        >>> result[2]['text']
        'c'
        """
        result = refined[:len(s1_win)]
        while len(result) < len(s1_win):
            # Pad with the corresponding S1 block as fallback
            result.append(dict(s1_win[len(result)]))
        return result
