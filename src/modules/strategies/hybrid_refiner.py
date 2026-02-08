import logging
import time
from pathlib import Path
from typing import List, Dict
from modules.translator import BaseTranslator
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)


class HybridRefiner(BaseTranslator):
    """
    Triple-source arbitration translator (S1 + L1 + Mt → Final).

    Alignment strategy:
      S1 is the master timeline. Each S1 block is matched to its closest
      L1 and Mt block by start-timestamp proximity. This handles the fact
      that L1/Mt may have fewer blocks (due to merge_identical_blocks in
      standardize) or slightly different timestamps.
    """

    # Maximum allowed drift (seconds) between an S1 block and its L1/Mt match.
    # Beyond this, the block is considered unmatched.
    ALIGNMENT_TOLERANCE = 5.0

    def __init__(self, s1_dir: str, l1_dir: str, mt_dir: str, output_dir: str, provider, config: Dict):
        super().__init__(s1_dir, output_dir, bot=provider)
        self.l1_dir = Path(l1_dir)
        self.mt_dir = Path(mt_dir)
        self.config = config
        self.bot = provider
        self.name = "Hybrid Refiner (Triple-Source Arbitration)"
        self.chunk_delay = config.get("chunk_delay", 1.0)

    # ------------------------------------------------------------------
    # File-level orchestration
    # ------------------------------------------------------------------

    def process_file(self, s1_file: Path):
        """Processes a single file by arbitrating between S1, L1, and Mt sources."""
        output_file = self.get_output_path(s1_file, ".srt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        relative_path = s1_file.relative_to(self.input_dir)
        l1_file = self.l1_dir / relative_path
        mt_file = self.mt_dir / relative_path

        if not l1_file.exists() or not mt_file.exists():
            logger.error(f"Missing input streams for {s1_file.name}. L1: {l1_file.exists()}, Mt: {mt_file.exists()}")
            return

        logger.info(f"Starting triple-source refinement for: {s1_file.name}")

        try:
            with open(s1_file, "r", encoding="utf-8") as f:
                s1_raw = f.read()
            with open(l1_file, "r", encoding="utf-8") as f:
                l1_raw = f.read()
            with open(mt_file, "r", encoding="utf-8") as f:
                mt_raw = f.read()
        except Exception as e:
            logger.error(f"Failed to read input files for refinement: {e}")
            return

        final_srt_content = self.refine_logic(s1_raw, l1_raw, mt_raw)

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
    def _ts(block: Dict) -> float:
        """Shortcut to convert a block's start timestamp to seconds."""
        return SRTHandler.timestamp_to_seconds(block['start'])

    def _build_alignment_map(self, s1_blocks: List[Dict], target_blocks: List[Dict]) -> Dict[int, Dict]:
        """
        Maps each S1 block index → closest target block by start-timestamp.

        Uses a two-pointer sweep (both lists are time-ordered) for O(n+m)
        complexity instead of O(n*m). A target block may be reused across
        multiple S1 indices (merged block case).
        """
        alignment: Dict[int, Dict] = {}
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

    def _collect_window_targets(self, window_indices: List[int], alignment_map: Dict[int, Dict]) -> List[Dict]:
        """
        Collects unique, ordered target blocks for a given window of S1 indices.
        Deduplicates by object identity (a merged target block may map to
        several S1 blocks but should appear only once in the prompt).
        """
        seen_ids = set()
        result = []
        for idx in window_indices:
            blk = alignment_map.get(idx)
            if blk is not None and id(blk) not in seen_ids:
                seen_ids.add(id(blk))
                result.append(blk)
        return result

    def _log_alignment_quality(self, label: str, s1_blocks: List[Dict], alignment_map: Dict[int, Dict]):
        """Logs alignment statistics for diagnostics."""
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
    # Core arbitration loop
    # ------------------------------------------------------------------

    def refine_logic(self, s1_text: str, l1_text: str, mt_text: str) -> str:
        """Slices the SRT into windows and performs the arbitration via LLM."""
        s1_blocks = SRTHandler.parse_to_blocks(s1_text)
        l1_blocks = SRTHandler.parse_to_blocks(l1_text)
        mt_blocks = SRTHandler.parse_to_blocks(mt_text)

        logger.info(f"Block counts — S1: {len(s1_blocks)}, L1: {len(l1_blocks)}, Mt: {len(mt_blocks)}")

        # Build alignment maps (once per file, not per window)
        l1_map = self._build_alignment_map(s1_blocks, l1_blocks)
        mt_map = self._build_alignment_map(s1_blocks, mt_blocks)
        self._log_alignment_quality("L1", s1_blocks, l1_map)
        self._log_alignment_quality("Mt", s1_blocks, mt_map)

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

        logger.info(f"Arbitrating {total_blocks} blocks using windows of {step}.")

        for i in range(0, total_blocks, step):
            window_indices = list(range(i, min(i + step, total_blocks)))
            s1_win = [s1_blocks[j] for j in window_indices]
            l1_win = self._collect_window_targets(window_indices, l1_map)
            mt_win = self._collect_window_targets(window_indices, mt_map)

            if not l1_win:
                logger.warning(f"Window {i}: No L1 blocks aligned. Legacy stream may be incomplete.")
            if not mt_win:
                logger.warning(f"Window {i}: No Mt blocks aligned. LLM draft stream may be incomplete.")

            user_prompt = self._build_arbitration_prompt(s1_win, l1_win, mt_win)
            win_label = f"{s1_win[0]['start']} → {s1_win[-1]['end']}"
            logger.info(
                f"Window [{i // step + 1}/{(total_blocks - 1) // step + 1}] "
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
                    # Best-effort: re-stamp what we can, pad/trim to match S1 count
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

        return SRTHandler.render_blocks(final_blocks)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_arbitration_prompt(self, s1: List[Dict], l1: List[Dict], mt: List[Dict]) -> str:
        """
        Formats the data sources with explicit block counts so the LLM
        knows exactly how many blocks to output and that L1/Mt may differ.
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
    def _force_align_to_s1(s1_win: List[Dict], refined: List[Dict]) -> List[Dict]:
        """
        Pads or trims the refined blocks to match S1 block count exactly.
        - If LLM returned too few: pad with S1 source blocks (untranslated).
        - If LLM returned too many: trim to S1 count.
        """
        result = refined[:len(s1_win)]
        while len(result) < len(s1_win):
            # Pad with the corresponding S1 block as fallback
            result.append(dict(s1_win[len(result)]))
        return result