from utils.srt_handler import SRTHandler


# --- Fixtures & Sample Data ---

SAMPLE_SRT = """\
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

SAMPLE_SRT_WITH_DUPLICATES = """\
1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:02,000 --> 00:00:03,000
Hello

3
00:00:04,000 --> 00:00:06,000
World
"""


# --- shift_timestamp ---

class TestShiftTimestamp:
    def test_basic_offset(self):
        result = SRTHandler.shift_timestamp("00:00:01,000", 60)
        assert result == "00:01:01,000"

    def test_zero_offset(self):
        result = SRTHandler.shift_timestamp("00:05:30,500", 0)
        assert result == "00:05:30,500"

    def test_large_offset_crosses_hour(self):
        result = SRTHandler.shift_timestamp("00:00:00,000", 3661)
        assert result == "01:01:01,000"

    def test_millisecond_preservation(self):
        result = SRTHandler.shift_timestamp("00:00:00,750", 1)
        assert result == "00:00:01,750"

    def test_invalid_format_returns_unchanged(self):
        result = SRTHandler.shift_timestamp("invalid", 10)
        assert result == "invalid"

    def test_hour_boundary(self):
        result = SRTHandler.shift_timestamp("00:59:59,999", 1)
        assert result == "01:00:00,999"


# --- apply_offset_to_blocks ---

class TestApplyOffsetToBlocks:
    def test_with_offset(self):
        blocks = [{"start": "00:00:01,000", "end": "00:00:03,000", "text": "Hello"}]
        result = SRTHandler.apply_offset_to_blocks(blocks, 600)
        assert result[0]["start"] == "00:10:01,000"
        assert result[0]["end"] == "00:10:03,000"

    def test_zero_offset_returns_same_reference(self):
        blocks = [{"start": "00:00:01,000", "end": "00:00:03,000", "text": "Hello"}]
        result = SRTHandler.apply_offset_to_blocks(blocks, 0)
        assert result is blocks

    def test_malformed_blocks_skipped(self):
        blocks = [
            {"start": "00:00:01,000", "end": "00:00:03,000", "text": "Valid"},
            {"start": None, "end": "00:00:05,000", "text": "Invalid"},
        ]
        result = SRTHandler.apply_offset_to_blocks(blocks, 10)
        assert len(result) == 1
        assert result[0]["text"] == "Valid"

    def test_empty_list(self):
        result = SRTHandler.apply_offset_to_blocks([], 100)
        assert result == []


# --- clean_text ---

class TestCleanText:
    def test_removes_bold_markers(self):
        assert SRTHandler.clean_text("**Hello**") == "Hello"

    def test_replaces_box_characters(self):
        assert SRTHandler.clean_text("□ item") == "- item"
        assert SRTHandler.clean_text("▪ item") == "- item"

    def test_replaces_ellipsis(self):
        assert SRTHandler.clean_text("wait…") == "wait..."

    def test_collapses_whitespace(self):
        assert SRTHandler.clean_text("  too   many  spaces  ") == "too many spaces"

    def test_empty_string(self):
        assert SRTHandler.clean_text("") == ""


# --- canonicalize ---

class TestCanonicalize:
    def test_removes_bom(self):
        result = SRTHandler.canonicalize("\ufeffHello")
        assert result == "Hello\n"

    def test_removes_zero_width_chars(self):
        result = SRTHandler.canonicalize("He\u200bllo")
        assert result == "Hello\n"

    def test_normalizes_line_endings(self):
        result = SRTHandler.canonicalize("line1\r\nline2")
        assert result == "line1\nline2\n"

    def test_strips_trailing_whitespace_per_line(self):
        result = SRTHandler.canonicalize("hello   \nworld  ")
        assert result == "hello\nworld\n"

    def test_removes_nbsp(self):
        result = SRTHandler.canonicalize("hello\xa0world")
        assert result == "helloworld\n"


# --- get_hash ---

class TestGetHash:
    def test_deterministic(self):
        h1 = SRTHandler.get_hash("Hello")
        h2 = SRTHandler.get_hash("Hello")
        assert h1 == h2

    def test_different_text_different_hash(self):
        h1 = SRTHandler.get_hash("Hello")
        h2 = SRTHandler.get_hash("World")
        assert h1 != h2

    def test_returns_hex_string(self):
        h = SRTHandler.get_hash("test")
        assert all(c in "0123456789abcdef" for c in h)
        assert len(h) == 64  # SHA-256 hex digest


# --- parse_to_blocks ---

class TestParseToBlocks:
    def test_standard_srt(self):
        blocks = SRTHandler.parse_to_blocks(SAMPLE_SRT)
        assert len(blocks) == 3
        assert blocks[0]["index"] == 1
        assert blocks[0]["start"] == "00:00:01,000"
        assert blocks[0]["end"] == "00:00:03,000"
        assert blocks[0]["text"] == ["Hello world"]

    def test_strips_markdown_fences(self):
        content = "```srt\n1\n00:00:01,000 --> 00:00:02,000\nHello\n```"
        blocks = SRTHandler.parse_to_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["text"] == ["Hello"]

    def test_empty_content(self):
        blocks = SRTHandler.parse_to_blocks("")
        assert blocks == []

    def test_multiline_text_block(self):
        content = "1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two\n"
        blocks = SRTHandler.parse_to_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["text"] == ["Line one", "Line two"]

    def test_preserves_brackets(self):
        """Brackets like [Music] should be preserved as legitimate SRT annotations."""
        content = "1\n00:00:01,000 --> 00:00:02,000\n[Music]\n"
        blocks = SRTHandler.parse_to_blocks(content)
        assert blocks[0]["text"] == ["[Music]"]

    def test_strips_wrapping_quotes(self):
        """Full-line wrapping quotes (LLM artifact) should be stripped."""
        content = '1\n00:00:01,000 --> 00:00:02,000\n"Hello world"\n'
        blocks = SRTHandler.parse_to_blocks(content)
        assert blocks[0]["text"] == ["Hello world"]

    def test_preserves_internal_quotes(self):
        """Quotes that are part of dialogue should be preserved."""
        content = '1\n00:00:01,000 --> 00:00:02,000\nHe said "hello" to me\n'
        blocks = SRTHandler.parse_to_blocks(content)
        assert blocks[0]["text"] == ['He said "hello" to me']

    def test_skips_lines_before_timestamp(self):
        content = "Random header text\n1\n00:00:01,000 --> 00:00:02,000\nHello\n"
        blocks = SRTHandler.parse_to_blocks(content)
        assert len(blocks) == 1


# --- merge_identical_blocks ---

class TestMergeIdenticalBlocks:
    def test_merges_consecutive_identical(self):
        blocks = [
            {"start": "00:00:01,000", "end": "00:00:02,000", "text": ["Hello"]},
            {"start": "00:00:02,000", "end": "00:00:03,000", "text": ["Hello"]},
        ]
        merged = SRTHandler.merge_identical_blocks(blocks)
        assert len(merged) == 1
        assert merged[0]["start"] == "00:00:01,000"
        assert merged[0]["end"] == "00:00:03,000"

    def test_keeps_different_blocks_separate(self):
        blocks = [
            {"start": "00:00:01,000", "end": "00:00:02,000", "text": ["Hello"]},
            {"start": "00:00:02,000", "end": "00:00:03,000", "text": ["World"]},
        ]
        merged = SRTHandler.merge_identical_blocks(blocks)
        assert len(merged) == 2

    def test_non_consecutive_identical_not_merged(self):
        blocks = [
            {"start": "00:00:01,000", "end": "00:00:02,000", "text": ["Hello"]},
            {"start": "00:00:02,000", "end": "00:00:03,000", "text": ["World"]},
            {"start": "00:00:03,000", "end": "00:00:04,000", "text": ["Hello"]},
        ]
        merged = SRTHandler.merge_identical_blocks(blocks)
        assert len(merged) == 3

    def test_empty_input(self):
        merged = SRTHandler.merge_identical_blocks([])
        assert merged == []

    def test_three_consecutive_identical(self):
        blocks = [
            {"start": "00:00:01,000", "end": "00:00:02,000", "text": ["Same"]},
            {"start": "00:00:02,000", "end": "00:00:03,000", "text": ["Same"]},
            {"start": "00:00:03,000", "end": "00:00:04,000", "text": ["Same"]},
        ]
        merged = SRTHandler.merge_identical_blocks(blocks)
        assert len(merged) == 1
        assert merged[0]["end"] == "00:00:04,000"


# --- render_blocks ---

class TestRenderBlocks:
    def test_renders_valid_srt_format(self):
        blocks = [
            {"start": "00:00:01,000", "end": "00:00:03,000", "text": ["Hello world"]},
            {"start": "00:00:04,000", "end": "00:00:06,000", "text": ["Goodbye"]},
        ]
        result = SRTHandler.render_blocks(blocks)
        lines = result.strip().split("\n")
        assert lines[0] == "1"
        assert lines[1] == "00:00:01,000 --> 00:00:03,000"
        assert lines[2] == "Hello world"
        assert lines[4] == "2"

    def test_renders_string_text(self):
        blocks = [{"start": "00:00:01,000", "end": "00:00:02,000", "text": "Single line"}]
        result = SRTHandler.render_blocks(blocks)
        assert "Single line" in result

    def test_reindexes_blocks_from_one(self):
        blocks = [
            {"start": "00:00:01,000", "end": "00:00:02,000", "text": "A"},
            {"start": "00:00:03,000", "end": "00:00:04,000", "text": "B"},
            {"start": "00:00:05,000", "end": "00:00:06,000", "text": "C"},
        ]
        result = SRTHandler.render_blocks(blocks)
        lines = result.strip().split("\n")
        indices = [line for line in lines if line.strip().isdigit()]
        assert indices == ["1", "2", "3"]

    def test_empty_blocks(self):
        result = SRTHandler.render_blocks([])
        assert result == ""


# --- standardize (full pipeline) ---

class TestStandardize:
    def test_merges_duplicate_blocks(self):
        result = SRTHandler.standardize(SAMPLE_SRT_WITH_DUPLICATES)
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 2  # Two "Hello" blocks merged into one
        assert blocks[0]["text"] == ["Hello"]
        assert blocks[0]["end"] == "00:00:03,000"
        assert blocks[1]["text"] == ["World"]

    def test_removes_empty_text_blocks(self):
        content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,000\n\n\n3\n00:00:05,000 --> 00:00:06,000\nWorld\n"
        result = SRTHandler.standardize(content)
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 2

    def test_reindexes_after_merge(self):
        content = "5\n00:00:01,000 --> 00:00:02,000\nTest\n\n10\n00:00:03,000 --> 00:00:04,000\nOther\n"
        result = SRTHandler.standardize(content)
        lines = result.strip().split("\n")
        assert lines[0] == "1"
        assert "2" in lines

    def test_roundtrip_preserves_content(self):
        """Standardize on already-clean content should not lose data."""
        result = SRTHandler.standardize(SAMPLE_SRT)
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 3
        assert blocks[0]["text"] == ["Hello world"]
        assert blocks[1]["text"] == ["This is a test"]
        assert blocks[2]["text"] == ["Goodbye"]

    def test_handles_llm_markdown_artifacts(self):
        content = "```srt\n1\n00:00:01,000 --> 00:00:02,000\nBonjour\n```"
        result = SRTHandler.standardize(content)
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 1
        assert blocks[0]["text"] == ["Bonjour"]

    def test_cleans_bold_markers_via_clean_text(self):
        """standardize() should now apply clean_text, removing **bold** markers."""
        content = "1\n00:00:01,000 --> 00:00:02,000\n**Hello world**\n"
        result = SRTHandler.standardize(content)
        blocks = SRTHandler.parse_to_blocks(result)
        assert blocks[0]["text"] == ["Hello world"]

    def test_cleans_box_characters_via_clean_text(self):
        """standardize() should now apply clean_text, replacing □ and ▪."""
        content = "1\n00:00:01,000 --> 00:00:02,000\n□ item one\n"
        result = SRTHandler.standardize(content)
        blocks = SRTHandler.parse_to_blocks(result)
        assert blocks[0]["text"] == ["- item one"]

    def test_collapses_whitespace_via_clean_text(self):
        """standardize() should collapse excessive whitespace."""
        content = "1\n00:00:01,000 --> 00:00:02,000\ntoo   many  spaces\n"
        result = SRTHandler.standardize(content)
        blocks = SRTHandler.parse_to_blocks(result)
        assert blocks[0]["text"] == ["too many spaces"]


# --- apply_offset_to_blocks (immutability) ---

class TestApplyOffsetImmutability:
    def test_does_not_mutate_original_blocks(self):
        """apply_offset_to_blocks should return new block dicts, not mutate originals."""
        blocks = [{"start": "00:00:01,000", "end": "00:00:03,000", "text": "Hello"}]
        original_start = blocks[0]["start"]
        original_end = blocks[0]["end"]
        result = SRTHandler.apply_offset_to_blocks(blocks, 600)
        # Original should be unchanged
        assert blocks[0]["start"] == original_start
        assert blocks[0]["end"] == original_end
        # Result should be shifted
        assert result[0]["start"] == "00:10:01,000"
        assert result[0]["end"] == "00:10:03,000"

    def test_returned_blocks_are_separate_dicts(self):
        """Returned blocks should not be the same object references."""
        blocks = [{"start": "00:00:01,000", "end": "00:00:03,000", "text": "Hello"}]
        result = SRTHandler.apply_offset_to_blocks(blocks, 10)
        assert result[0] is not blocks[0]


# --- Non-western character support ---

CYRILLIC_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Привет мир

2
00:00:04,000 --> 00:00:06,000
Это тест

3
00:00:07,000 --> 00:00:09,000
До свидания
"""

CHINESE_SRT = """\
1
00:00:01,000 --> 00:00:03,000
你好世界

2
00:00:04,000 --> 00:00:06,000
这是一个测试

3
00:00:07,000 --> 00:00:09,000
再见
"""

MIXED_SCRIPT_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Hello 你好 Привет

2
00:00:04,000 --> 00:00:06,000
Test with café, naïve, and 日本語
"""


class TestCyrillicSupport:
    def test_parse_cyrillic_srt(self):
        blocks = SRTHandler.parse_to_blocks(CYRILLIC_SRT)
        assert len(blocks) == 3
        assert blocks[0]["text"] == ["Привет мир"]
        assert blocks[1]["text"] == ["Это тест"]
        assert blocks[2]["text"] == ["До свидания"]

    def test_standardize_cyrillic(self):
        result = SRTHandler.standardize(CYRILLIC_SRT)
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 3
        assert blocks[0]["text"] == ["Привет мир"]

    def test_merge_identical_cyrillic(self):
        content = (
            "1\n00:00:01,000 --> 00:00:02,000\nПривет\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nПривет\n\n"
            "3\n00:00:04,000 --> 00:00:05,000\nМир\n"
        )
        result = SRTHandler.standardize(content)
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 2
        assert blocks[0]["end"] == "00:00:03,000"

    def test_hash_cyrillic_deterministic(self):
        h1 = SRTHandler.get_hash("Привет мир")
        h2 = SRTHandler.get_hash("Привет мир")
        assert h1 == h2

    def test_hash_cyrillic_differs_from_latin(self):
        h1 = SRTHandler.get_hash("Привет")
        h2 = SRTHandler.get_hash("Hello")
        assert h1 != h2

    def test_render_roundtrip_cyrillic(self):
        blocks = SRTHandler.parse_to_blocks(CYRILLIC_SRT)
        rendered = SRTHandler.render_blocks(blocks)
        re_parsed = SRTHandler.parse_to_blocks(rendered)
        assert len(re_parsed) == 3
        assert re_parsed[0]["text"] == ["Привет мир"]


class TestChineseSupport:
    def test_parse_chinese_srt(self):
        blocks = SRTHandler.parse_to_blocks(CHINESE_SRT)
        assert len(blocks) == 3
        assert blocks[0]["text"] == ["你好世界"]
        assert blocks[1]["text"] == ["这是一个测试"]
        assert blocks[2]["text"] == ["再见"]

    def test_standardize_chinese(self):
        result = SRTHandler.standardize(CHINESE_SRT)
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 3
        assert blocks[2]["text"] == ["再见"]

    def test_chinese_ellipsis_replacement(self):
        """Chinese often uses … (U+2026). clean_text replaces it with '...'."""
        result = SRTHandler.clean_text("等一下…")
        assert result == "等一下..."

    def test_fullwidth_digits_not_treated_as_index(self):
        """Full-width digits (１２３) must NOT be parsed as SRT block indices."""
        content = "1\n00:00:01,000 --> 00:00:02,000\n１２３\n"
        blocks = SRTHandler.parse_to_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["text"] == ["１２３"]

    def test_merge_identical_chinese(self):
        content = (
            "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\n你好\n"
        )
        result = SRTHandler.standardize(content)
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 1
        assert blocks[0]["end"] == "00:00:03,000"

    def test_render_roundtrip_chinese(self):
        blocks = SRTHandler.parse_to_blocks(CHINESE_SRT)
        rendered = SRTHandler.render_blocks(blocks)
        re_parsed = SRTHandler.parse_to_blocks(rendered)
        assert len(re_parsed) == 3
        assert re_parsed[1]["text"] == ["这是一个测试"]


class TestMixedScriptSupport:
    def test_parse_mixed_scripts(self):
        blocks = SRTHandler.parse_to_blocks(MIXED_SCRIPT_SRT)
        assert len(blocks) == 2
        assert blocks[0]["text"] == ["Hello 你好 Привет"]
        assert blocks[1]["text"] == ["Test with café, naïve, and 日本語"]

    def test_standardize_mixed(self):
        result = SRTHandler.standardize(MIXED_SCRIPT_SRT)
        blocks = SRTHandler.parse_to_blocks(result)
        assert len(blocks) == 2

    def test_arabic_digits_not_treated_as_index(self):
        """Arabic-Indic digits (٣٢١) must NOT be parsed as SRT block indices."""
        content = "1\n00:00:01,000 --> 00:00:02,000\n٣٢١\n"
        blocks = SRTHandler.parse_to_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["text"] == ["٣٢١"]

    def test_devanagari_digits_not_treated_as_index(self):
        """Devanagari digits (१२३) must NOT be parsed as SRT block indices."""
        content = "1\n00:00:01,000 --> 00:00:02,000\n१२३\n"
        blocks = SRTHandler.parse_to_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["text"] == ["१२३"]

    def test_korean_text(self):
        content = "1\n00:00:01,000 --> 00:00:02,000\n안녕하세요\n"
        blocks = SRTHandler.parse_to_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["text"] == ["안녕하세요"]

    def test_japanese_mixed_text(self):
        """Japanese uses Kanji + Hiragana + Katakana."""
        content = "1\n00:00:01,000 --> 00:00:02,000\n東京タワーへようこそ\n"
        blocks = SRTHandler.parse_to_blocks(content)
        assert len(blocks) == 1
        assert blocks[0]["text"] == ["東京タワーへようこそ"]

    def test_chinese_quotes_preserved(self):
        """Chinese-style quotes (「」) should be preserved, not stripped."""
        content = '1\n00:00:01,000 --> 00:00:02,000\n他说「你好」\n'
        blocks = SRTHandler.parse_to_blocks(content)
        assert blocks[0]["text"] == ['他说「你好」']

    def test_russian_quotes_preserved(self):
        """Russian-style quotes (« ») should be preserved, not stripped."""
        content = '1\n00:00:01,000 --> 00:00:02,000\nОн сказал «привет»\n'
        blocks = SRTHandler.parse_to_blocks(content)
        assert blocks[0]["text"] == ['Он сказал «привет»']

    def test_clean_text_preserves_cjk(self):
        """clean_text should not corrupt CJK characters."""
        assert SRTHandler.clean_text("你好世界") == "你好世界"
        assert SRTHandler.clean_text("Привет мир") == "Привет мир"
        assert SRTHandler.clean_text("**太棒了**") == "太棒了"

    def test_canonicalize_nfc_normalization(self):
        """NFC normalization should handle composed vs decomposed forms."""
        # é can be U+00E9 (precomposed) or U+0065 + U+0301 (decomposed)
        decomposed = "caf\u0065\u0301"  # e + combining acute
        composed = "café"              # precomposed é
        h1 = SRTHandler.get_hash(decomposed)
        h2 = SRTHandler.get_hash(composed)
        assert h1 == h2  # NFC normalization should make these identical
