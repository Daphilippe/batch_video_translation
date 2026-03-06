from helpers import MockProvider
from modules.providers.base_provider import LLMProviderError
from modules.strategies.hybrid_refiner import HybridRefiner
from utils.srt_handler import SRTHandler

# --- Sample SRT data ---

S1_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
This is a test

3
00:00:07,000 --> 00:00:09,000
Goodbye
"""

L1_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Bonjour le monde

2
00:00:04,000 --> 00:00:06,000
Ceci est un test

3
00:00:07,000 --> 00:00:09,000
Au revoir
"""

MT_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Salut tout le monde

2
00:00:04,000 --> 00:00:06,000
C'est un test

3
00:00:07,000 --> 00:00:09,000
À bientôt
"""


def _make_refiner(provider, chunk_size=10, chunk_delay=0):
    """Helper to build a HybridRefiner without filesystem deps."""
    config = {
        "chunk_size": chunk_size,
        "chunk_delay": chunk_delay,
        "refinement_protocol_file": "nonexistent.txt",  # triggers fallback
    }
    return HybridRefiner(
        source_dirs={"s1": "/dummy/s1", "l1": "/dummy/l1", "mt": "/dummy/mt"},
        output_dir="/dummy/out",
        provider=provider,
        config=config,
    )


# ── Alignment engine ─────────────────────────────────────────────────


class TestBuildAlignmentMap:
    def test_perfect_alignment(self):
        """When S1 and target share identical timestamps, all blocks are mapped."""
        refiner = _make_refiner(MockProvider())
        s1_blocks = SRTHandler.parse_to_blocks(S1_SRT)
        l1_blocks = SRTHandler.parse_to_blocks(L1_SRT)

        alignment = refiner._build_alignment_map(s1_blocks, l1_blocks)

        assert len(alignment) == 3
        for i in range(3):
            assert i in alignment

    def test_empty_target_returns_empty_map(self):
        refiner = _make_refiner(MockProvider())
        s1_blocks = SRTHandler.parse_to_blocks(S1_SRT)

        alignment = refiner._build_alignment_map(s1_blocks, [])
        assert alignment == {}

    def test_empty_s1_returns_empty_map(self):
        refiner = _make_refiner(MockProvider())
        l1_blocks = SRTHandler.parse_to_blocks(L1_SRT)

        alignment = refiner._build_alignment_map([], l1_blocks)
        assert alignment == {}

    def test_target_with_drift_within_tolerance(self):
        """Target blocks shifted by < ALIGNMENT_TOLERANCE are still matched."""
        refiner = _make_refiner(MockProvider())
        s1_blocks = SRTHandler.parse_to_blocks(S1_SRT)
        # Shift L1 by 2 seconds (within the 5s tolerance)
        l1_blocks = SRTHandler.parse_to_blocks(L1_SRT)
        l1_shifted = SRTHandler.apply_offset_to_blocks(l1_blocks, 2)

        alignment = refiner._build_alignment_map(s1_blocks, l1_shifted)
        assert len(alignment) == 3

    def test_target_with_drift_beyond_tolerance(self):
        """Target blocks shifted beyond tolerance are not matched."""
        refiner = _make_refiner(MockProvider())
        s1_blocks = SRTHandler.parse_to_blocks(S1_SRT)
        l1_blocks = SRTHandler.parse_to_blocks(L1_SRT)
        l1_far = SRTHandler.apply_offset_to_blocks(l1_blocks, 20)

        alignment = refiner._build_alignment_map(s1_blocks, l1_far)
        assert len(alignment) == 0


class TestCollectWindowTargets:
    def test_deduplicates_by_identity(self):
        refiner = _make_refiner(MockProvider())
        s1_blocks = SRTHandler.parse_to_blocks(S1_SRT)
        l1_blocks = SRTHandler.parse_to_blocks(L1_SRT)
        alignment = refiner._build_alignment_map(s1_blocks, l1_blocks)

        result = refiner._collect_window_targets([0, 1, 2], alignment)
        assert len(result) == 3

    def test_missing_indices_skipped(self):
        refiner = _make_refiner(MockProvider())
        # Empty alignment map — nothing aligns
        result = refiner._collect_window_targets([0, 1, 2], {})
        assert result == []


# ── Incremental refinement ───────────────────────────────────────────


class TestIdentifyProblematicIndices:
    def test_all_good_returns_empty(self):
        """Existing output matching L1 → 0 problematic blocks."""
        refiner = _make_refiner(MockProvider())
        s1_blocks = SRTHandler.parse_to_blocks(S1_SRT)
        # Existing blocks are properly translated (different from S1)
        existing_blocks = SRTHandler.parse_to_blocks(L1_SRT)

        problematic, _ = refiner._identify_problematic_indices(s1_blocks, existing_blocks)
        assert len(problematic) == 0

    def test_untranslated_blocks_detected(self):
        """Existing output identical to S1 → flagged as untranslated."""
        refiner = _make_refiner(MockProvider())
        s1_blocks = SRTHandler.parse_to_blocks(S1_SRT)
        # Same text as S1 = not translated
        existing_blocks = SRTHandler.parse_to_blocks(S1_SRT)

        problematic, _ = refiner._identify_problematic_indices(s1_blocks, existing_blocks)
        assert len(problematic) == 3

    def test_empty_text_detected(self):
        """Existing block with empty text → flagged."""
        refiner = _make_refiner(MockProvider())
        s1_blocks = SRTHandler.parse_to_blocks(S1_SRT)
        empty_srt = """\
1
00:00:01,000 --> 00:00:03,000
Bonjour

2
00:00:04,000 --> 00:00:06,000


3
00:00:07,000 --> 00:00:09,000
Au revoir
"""
        existing_blocks = SRTHandler.parse_to_blocks(empty_srt)

        problematic, _ = refiner._identify_problematic_indices(s1_blocks, existing_blocks)
        # Block 1 (index 1) has empty text → problematic
        assert 1 in problematic

    def test_missing_alignment_detected(self):
        """If existing output has fewer blocks, unaligned ones are flagged."""
        refiner = _make_refiner(MockProvider())
        s1_blocks = SRTHandler.parse_to_blocks(S1_SRT)
        # Only 1 block in existing — block at 00:00:01 aligns to S1[0],
        # S1[1] at 00:00:04 is within 5s tolerance so also aligns (same text = untranslated handled elsewhere),
        # but S1[2] at 00:00:07 is too far → missing alignment
        partial_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        existing_blocks = SRTHandler.parse_to_blocks(partial_srt)

        problematic, _ = refiner._identify_problematic_indices(s1_blocks, existing_blocks)
        # At minimum, block 2 (index 2) must be flagged as missing
        assert 2 in problematic


class TestExpandProblematicIndices:
    def test_adds_context_padding(self):
        result = HybridRefiner._expand_problematic_indices({5}, 20, context_padding=2)
        assert result == {3, 4, 5, 6, 7}

    def test_clamps_to_zero(self):
        result = HybridRefiner._expand_problematic_indices({0}, 10, context_padding=2)
        assert result == {0, 1, 2}

    def test_clamps_to_total(self):
        result = HybridRefiner._expand_problematic_indices({9}, 10, context_padding=2)
        assert result == {7, 8, 9}

    def test_empty_input(self):
        result = HybridRefiner._expand_problematic_indices(set(), 10, context_padding=2)
        assert result == set()


# ── Force-align ──────────────────────────────────────────────────────


class TestForceAlignToS1:
    def test_trim_excess_blocks(self):
        s1 = [{"text": "a"}, {"text": "b"}]
        refined = [{"text": "x"}, {"text": "y"}, {"text": "z"}]
        result = HybridRefiner._force_align_to_s1(s1, refined)
        assert len(result) == 2
        assert result[0]["text"] == "x"
        assert result[1]["text"] == "y"

    def test_pad_missing_blocks(self):
        s1 = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        refined = [{"text": "x"}]
        result = HybridRefiner._force_align_to_s1(s1, refined)
        assert len(result) == 3
        assert result[0]["text"] == "x"
        assert result[1]["text"] == "b"  # fallback to S1
        assert result[2]["text"] == "c"

    def test_exact_match_unchanged(self):
        s1 = [{"text": "a"}, {"text": "b"}]
        refined = [{"text": "x"}, {"text": "y"}]
        result = HybridRefiner._force_align_to_s1(s1, refined)
        assert len(result) == 2
        assert result[0]["text"] == "x"
        assert result[1]["text"] == "y"

    def test_empty_refined_uses_all_s1(self):
        s1 = [{"text": "a"}, {"text": "b"}]
        result = HybridRefiner._force_align_to_s1(s1, [])
        assert len(result) == 2
        assert result[0]["text"] == "a"
        assert result[1]["text"] == "b"


# ── Prompt construction ──────────────────────────────────────────────


class TestBuildArbitrationPrompt:
    def test_prompt_contains_all_sources(self):
        refiner = _make_refiner(MockProvider())
        s1 = SRTHandler.parse_to_blocks("1\n00:00:01,000 --> 00:00:03,000\nHello\n")
        l1 = SRTHandler.parse_to_blocks("1\n00:00:01,000 --> 00:00:03,000\nBonjour\n")
        mt = SRTHandler.parse_to_blocks("1\n00:00:01,000 --> 00:00:03,000\nSalut\n")

        prompt = refiner._build_arbitration_prompt(s1, l1, mt)

        assert "<S1_SOURCE_ORIGINAL" in prompt
        assert "<L1_LITERAL_REFERENCE" in prompt
        assert "<MT_DRAFT_STYLE" in prompt
        assert "Hello" in prompt
        assert "Bonjour" in prompt
        assert "Salut" in prompt
        assert 'count="1"' in prompt

    def test_prompt_specifies_exact_block_count(self):
        refiner = _make_refiner(MockProvider())
        s1 = SRTHandler.parse_to_blocks(S1_SRT)
        l1 = SRTHandler.parse_to_blocks(L1_SRT)
        mt = SRTHandler.parse_to_blocks(MT_SRT)

        prompt = refiner._build_arbitration_prompt(s1, l1, mt)
        assert f"EXACTLY {len(s1)} SRT blocks" in prompt


# ── refine_logic (full arbitration loop) ─────────────────────────────


class TestRefineLogic:
    def test_full_refinement_single_window(self):
        """With chunk_size >= block count, one LLM call is made."""
        response_srt = """\
1
00:00:01,000 --> 00:00:03,000
Bonjour le monde

2
00:00:04,000 --> 00:00:06,000
Ceci est un test

3
00:00:07,000 --> 00:00:09,000
Au revoir
"""
        provider = MockProvider(responses=[response_srt])
        refiner = _make_refiner(provider, chunk_size=10, chunk_delay=0)

        result = refiner.refine_logic(S1_SRT, L1_SRT, MT_SRT)

        assert result is not None
        assert provider.call_count == 1
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 3
        assert blocks[0]["text"] == ["Bonjour le monde"]

    def test_s1_timestamps_enforced(self):
        """Even if LLM shifts timestamps, S1 originals are restored."""
        shifted_response = """\
1
00:00:02,000 --> 00:00:04,000
Bonjour

2
00:00:05,000 --> 00:00:07,000
Test

3
00:00:08,000 --> 00:00:10,000
Salut
"""
        provider = MockProvider(responses=[shifted_response])
        refiner = _make_refiner(provider, chunk_size=10, chunk_delay=0)

        result = refiner.refine_logic(S1_SRT, L1_SRT, MT_SRT)
        blocks = SRTHandler.parse_to_blocks(result)

        # Timestamps must match S1, not the LLM output
        assert blocks[0]["start"] == "00:00:01,000"
        assert blocks[1]["start"] == "00:00:04,000"
        assert blocks[2]["start"] == "00:00:07,000"

    def test_llm_error_falls_back_to_s1(self):
        """Provider error → S1 source blocks are kept."""
        provider = MockProvider(responses=[LLMProviderError("LLM down")])
        refiner = _make_refiner(provider, chunk_size=10, chunk_delay=0)

        result = refiner.refine_logic(S1_SRT, L1_SRT, MT_SRT)
        blocks = SRTHandler.parse_to_blocks(result)

        assert len(blocks) == 3
        assert blocks[0]["text"] == ["Hello world"]

    def test_empty_llm_response_falls_back_to_s1(self):
        """Empty/unparseable LLM response → S1 kept."""
        provider = MockProvider(responses=[""])
        refiner = _make_refiner(provider, chunk_size=10, chunk_delay=0)

        result = refiner.refine_logic(S1_SRT, L1_SRT, MT_SRT)
        blocks = SRTHandler.parse_to_blocks(result)

        assert len(blocks) == 3
        assert blocks[0]["text"] == ["Hello world"]

    def test_incremental_no_problems_returns_none(self):
        """When existing output is fully translated, returns None (skip)."""
        provider = MockProvider(responses=[])
        refiner = _make_refiner(provider, chunk_size=10, chunk_delay=0)

        result = refiner.refine_logic(S1_SRT, L1_SRT, MT_SRT, existing_refined_text=L1_SRT)

        assert result is None
        assert provider.call_count == 0

    def test_incremental_reprocesses_problematic_windows(self):
        """Only windows with problematic blocks get sent to LLM."""
        # Existing has block 2 identical to S1 (not translated)
        existing_srt = """\
1
00:00:01,000 --> 00:00:03,000
Bonjour le monde

2
00:00:04,000 --> 00:00:06,000
This is a test

3
00:00:07,000 --> 00:00:09,000
Au revoir
"""
        corrected_response = """\
1
00:00:01,000 --> 00:00:03,000
Bonjour le monde

2
00:00:04,000 --> 00:00:06,000
Ceci est un test

3
00:00:07,000 --> 00:00:09,000
Au revoir
"""
        provider = MockProvider(responses=[corrected_response])
        refiner = _make_refiner(provider, chunk_size=10, chunk_delay=0)

        result = refiner.refine_logic(S1_SRT, L1_SRT, MT_SRT, existing_refined_text=existing_srt)

        assert result is not None
        # The window containing the problematic block was sent
        assert provider.call_count == 1

    def test_multi_window_chunking(self):
        """With chunk_size=1, each block gets its own LLM call."""
        responses = [
            "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n",
            "1\n00:00:04,000 --> 00:00:06,000\nTest\n",
            "1\n00:00:07,000 --> 00:00:09,000\nSalut\n",
        ]
        provider = MockProvider(responses=responses)
        refiner = _make_refiner(provider, chunk_size=1, chunk_delay=0)

        result = refiner.refine_logic(S1_SRT, L1_SRT, MT_SRT)

        assert provider.call_count == 3
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 3

    def test_block_count_mismatch_triggers_force_align(self):
        """LLM returns wrong block count → force-aligned to S1."""
        # Return 2 blocks instead of 3
        response = """\
1
00:00:01,000 --> 00:00:03,000
Bonjour

2
00:00:04,000 --> 00:00:06,000
Test
"""
        provider = MockProvider(responses=[response])
        refiner = _make_refiner(provider, chunk_size=10, chunk_delay=0)

        result = refiner.refine_logic(S1_SRT, L1_SRT, MT_SRT)
        blocks = SRTHandler.parse_to_blocks(result)

        # Should still have 3 blocks (3rd padded from S1)
        assert len(blocks) == 3
        assert blocks[0]["text"] == ["Bonjour"]
        assert blocks[2]["text"] == ["Goodbye"]  # S1 fallback


# ── _block_text helper ───────────────────────────────────────────────


class TestBlockText:
    def test_list_text(self):
        assert HybridRefiner._block_text({"text": ["Hello", "World"]}) == "Hello\nWorld"

    def test_string_text(self):
        assert HybridRefiner._block_text({"text": "Single line"}) == "Single line"

    def test_empty_text(self):
        assert HybridRefiner._block_text({"text": ""}) == ""
        assert HybridRefiner._block_text({"text": []}) == ""

    def test_missing_text_key(self):
        assert HybridRefiner._block_text({}) == ""


# ── process_file (file-level orchestration) ──────────────────────────


class TestHybridRefinerProcessFile:
    """Integration tests for HybridRefiner.process_file."""

    def _setup_dirs(self, tmp_path):
        """Create all four required directories."""
        s1_dir = tmp_path / "s1"
        l1_dir = tmp_path / "l1"
        mt_dir = tmp_path / "mt"
        out_dir = tmp_path / "out"
        for d in (s1_dir, l1_dir, mt_dir, out_dir):
            d.mkdir()
        return s1_dir, l1_dir, mt_dir, out_dir

    def test_process_file_full_refinement(self, tmp_path):
        """process_file reads S1/L1/Mt, calls refine_logic, writes output."""
        s1_dir, l1_dir, mt_dir, out_dir = self._setup_dirs(tmp_path)

        (s1_dir / "video.srt").write_text(S1_SRT, encoding="utf-8")
        (l1_dir / "video.srt").write_text(L1_SRT, encoding="utf-8")
        (mt_dir / "video.srt").write_text(MT_SRT, encoding="utf-8")

        response_srt = (
            "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nTest\n\n"
            "3\n00:00:07,000 --> 00:00:09,000\nSalut\n"
        )
        provider = MockProvider(responses=[response_srt])

        config = {"chunk_size": 10, "chunk_delay": 0, "refinement_protocol_file": "nonexistent.txt"}
        refiner = HybridRefiner(
            source_dirs={"s1": str(s1_dir), "l1": str(l1_dir), "mt": str(mt_dir)},
            output_dir=str(out_dir),
            provider=provider,
            config=config,
        )
        refiner.process_file(s1_dir / "video.srt")

        output = out_dir / "video.srt"
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "Bonjour" in content

    def test_process_file_missing_l1_skips(self, tmp_path):
        """process_file returns early when L1 file is missing."""
        s1_dir, l1_dir, mt_dir, out_dir = self._setup_dirs(tmp_path)

        (s1_dir / "video.srt").write_text(S1_SRT, encoding="utf-8")
        (mt_dir / "video.srt").write_text(MT_SRT, encoding="utf-8")
        # L1 missing intentionally

        provider = MockProvider()
        config = {"chunk_size": 10, "chunk_delay": 0}
        refiner = HybridRefiner(
            source_dirs={"s1": str(s1_dir), "l1": str(l1_dir), "mt": str(mt_dir)},
            output_dir=str(out_dir),
            provider=provider,
            config=config,
        )
        refiner.process_file(s1_dir / "video.srt")

        # No output should be written
        assert not (out_dir / "video.srt").exists()
        assert provider.call_count == 0

    def test_process_file_incremental_skip(self, tmp_path):
        """process_file skips write when existing output is fully translated."""
        s1_dir, l1_dir, mt_dir, out_dir = self._setup_dirs(tmp_path)

        (s1_dir / "video.srt").write_text(S1_SRT, encoding="utf-8")
        (l1_dir / "video.srt").write_text(L1_SRT, encoding="utf-8")
        (mt_dir / "video.srt").write_text(MT_SRT, encoding="utf-8")
        # Pre-existing output that is fully translated
        (out_dir / "video.srt").write_text(L1_SRT, encoding="utf-8")

        provider = MockProvider()
        config = {"chunk_size": 10, "chunk_delay": 0, "refinement_protocol_file": "nonexistent.txt"}
        refiner = HybridRefiner(
            source_dirs={"s1": str(s1_dir), "l1": str(l1_dir), "mt": str(mt_dir)},
            output_dir=str(out_dir),
            provider=provider,
            config=config,
        )
        refiner.process_file(s1_dir / "video.srt")

        # No LLM calls — output already good
        assert provider.call_count == 0


# ── _log_alignment_quality ───────────────────────────────────────────


class TestLogAlignmentQuality:
    """Diagnostics logging for alignment stats."""

    def test_perfect_alignment_logged(self, tmp_path):
        """100% alignment logs info-level message."""
        refiner = _make_refiner(MockProvider())
        s1 = SRTHandler.parse_to_blocks(S1_SRT)
        l1 = SRTHandler.parse_to_blocks(L1_SRT)
        alignment = refiner._build_alignment_map(s1, l1)

        # Should not raise — just verifies the code path runs
        refiner._log_alignment_quality("L1", s1, alignment)

    def test_partial_alignment_logged(self, tmp_path):
        """Partial alignment logs warning with unmatched indices."""
        refiner = _make_refiner(MockProvider())
        s1 = SRTHandler.parse_to_blocks(S1_SRT)
        # Empty map = 0% alignment
        refiner._log_alignment_quality("L1", s1, {})

    def test_empty_s1_no_crash(self, tmp_path):
        """Empty S1 blocks don't crash the logging."""
        refiner = _make_refiner(MockProvider())
        refiner._log_alignment_quality("L1", [], {})


# ── Window retry on untranslated output ──────────────────────────────


class TestHybridWindowRetry:
    """HybridRefiner._refine_window retries when output matches source."""

    def _make_refiner(self, tmp_path, provider):
        config = {"chunk_size": 10, "chunk_delay": 0}
        return HybridRefiner(
            source_dirs={"s1": str(tmp_path / "s1"), "l1": str(tmp_path / "l1"), "mt": str(tmp_path / "mt")},
            output_dir=str(tmp_path / "out"),
            provider=provider,
            config=config,
        )

    def test_retry_on_identical_window(self, tmp_path):
        """When LLM returns S1 text, window is retried once."""
        source_srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        translated_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        provider = MockProvider(responses=[source_srt, translated_srt])
        refiner = self._make_refiner(tmp_path, provider)
        refiner._active_protocol = "system"

        s1_win = [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello"]}]
        result = refiner._refine_window(s1_win, [0], {"l1": {}, "mt": {}})

        assert result is not None
        assert result[0]["text"] == ["Bonjour"]
        assert provider.call_count == 2

    def test_gives_up_after_two_attempts(self, tmp_path):
        """After 2 failed attempts, returns S1 source blocks."""
        source_srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        provider = MockProvider(responses=[source_srt, source_srt])
        refiner = self._make_refiner(tmp_path, provider)
        refiner._active_protocol = "system"

        s1_win = [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello"]}]
        result = refiner._refine_window(s1_win, [0], {"l1": {}, "mt": {}})

        assert result[0]["text"] == ["Hello"]
        assert provider.call_count == 2

    def test_no_retry_when_translated(self, tmp_path):
        """No retry when refinement differs from source."""
        translated_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        provider = MockProvider(responses=[translated_srt])
        refiner = self._make_refiner(tmp_path, provider)
        refiner._active_protocol = "system"

        s1_win = [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello"]}]
        result = refiner._refine_window(s1_win, [0], {"l1": {}, "mt": {}})

        assert result is not None
        assert result[0]["text"] == ["Bonjour"]
        assert provider.call_count == 1

    def test_enforces_s1_timestamps(self, tmp_path):
        """Refined blocks always get S1 timestamps."""
        translated_srt = "1\n00:00:10,000 --> 00:00:12,000\nBonjour\n"
        provider = MockProvider(responses=[translated_srt])
        refiner = self._make_refiner(tmp_path, provider)
        refiner._active_protocol = "system"

        s1_win = [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello"]}]
        result = refiner._refine_window(s1_win, [0], {"l1": {}, "mt": {}})

        assert result is not None
        assert result[0]["start"] == "00:00:01,000"
        assert result[0]["end"] == "00:00:03,000"

    def test_empty_response_triggers_retry(self, tmp_path):
        """Empty LLM response triggers a retry."""
        translated_srt = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        provider = MockProvider(responses=["", translated_srt])
        refiner = self._make_refiner(tmp_path, provider)
        refiner._active_protocol = "system"

        s1_win = [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello"]}]
        result = refiner._refine_window(s1_win, [0], {"l1": {}, "mt": {}})

        assert result is not None
        assert result[0]["text"] == ["Bonjour"]
        assert provider.call_count == 2
