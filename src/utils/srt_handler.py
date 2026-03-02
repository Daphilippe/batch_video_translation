import hashlib
import logging
import re
import unicodedata
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class SRTHandler:
    """Stateless SRT subtitle file manipulation toolkit.

    Provides static and class methods for parsing, rendering,
    cleaning, hashing, and aligning SRT subtitle blocks.
    All methods are side-effect-free (no file I/O except
    ``extract_timestamps``).
    """

    TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}")

    @staticmethod
    def shift_timestamp(ts_str: str, offset_seconds: int) -> str:
        """
        Add a time offset to an SRT timestamp string.

        Used to realign segmented audio transcripts after
        Whisper processes each audio chunk independently.

        Parameters
        ----------
        ts_str : str
            Timestamp in SRT format ``HH:MM:SS,mmm``.
        offset_seconds : int
            Number of seconds to add (can be negative).

        Returns
        -------
        str
            Shifted timestamp in the same ``HH:MM:SS,mmm`` format.
            Returns the original string unchanged if it cannot be parsed.

        Examples
        --------
        >>> SRTHandler.shift_timestamp("00:00:05,000", 10)
        '00:00:15,000'

        >>> SRTHandler.shift_timestamp("00:09:50,500", 600)
        '00:19:50,500'
        """
        # Parse the HH:MM:SS,mmm format
        match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts_str.strip())
        if not match:
            return ts_str

        h, m, s, ms = map(int, match.groups())

        # Use timedelta for robust time math
        td = timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)
        new_td = td + timedelta(seconds=offset_seconds)

        # Extract new components
        total_seconds = int(new_td.total_seconds())
        new_h = total_seconds // 3600
        new_m = (total_seconds % 3600) // 60
        new_s = total_seconds % 60
        new_ms = int(new_td.microseconds / 1000)

        return f"{new_h:02d}:{new_m:02d}:{new_s:02d},{new_ms:03d}"

    @classmethod
    def apply_offset_to_blocks(cls, blocks: list[dict], offset_seconds: int) -> list[dict]:
        """
        Apply a time offset to a list of parsed SRT blocks.

        Returns new block copies with shifted timestamps.
        Blocks missing ``start`` or ``end`` keys are silently skipped
        with a warning.

        Parameters
        ----------
        blocks : list of dict
            Parsed SRT blocks (each with ``start``, ``end``, ``text``).
        offset_seconds : int
            Seconds to add to every timestamp.

        Returns
        -------
        list of dict
            New block dicts with adjusted timestamps.

        Examples
        --------
        >>> blocks = [{"index": 1, "start": "00:00:01,000",
        ...            "end": "00:00:03,000", "text": ["Hi"]}]
        >>> SRTHandler.apply_offset_to_blocks(blocks, 60)
        [{'index': 1, 'start': '00:01:01,000', 'end': '00:01:03,000', 'text': ['Hi']}]

        >>> SRTHandler.apply_offset_to_blocks(blocks, 0)
        [{'index': 1, 'start': '00:00:01,000', 'end': '00:00:03,000', 'text': ['Hi']}]
        """
        if offset_seconds == 0:
            return blocks

        valid_blocks = []
        for b in blocks:
            # SAFETY CHECK: Skip blocks that are missing timestamps
            if b.get('start') is None or b.get('end') is None:
                logger.warning(f"Skipping malformed block: {b}")
                continue

            shifted = {
                **b,
                'start': cls.shift_timestamp(b['start'], offset_seconds),
                'end': cls.shift_timestamp(b['end'], offset_seconds),
            }
            valid_blocks.append(shifted)
        return valid_blocks

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Remove common LLM artifacts from subtitle text.

        Strips bold markers (``**``), replaces box-drawing characters
        with dashes, normalizes ellipsis, and collapses whitespace.

        Parameters
        ----------
        text : str
            Raw text potentially containing LLM formatting artifacts.

        Returns
        -------
        str
            Cleaned text with artifacts removed.

        Examples
        --------
        >>> SRTHandler.clean_text("**Hello** world")
        'Hello world'

        >>> SRTHandler.clean_text("Multiple   spaces   here")
        'Multiple spaces here'

        >>> SRTHandler.clean_text("Item □ one ▪ two")
        'Item - one - two'
        """
        replacements = {"**": "", "□": "-", "▪": "-", "…": "..."}
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return " ".join(text.split()).strip()

    @staticmethod
    def canonicalize(text: str) -> str:
        """
        Normalize text for deterministic hashing.

        Applies NFC Unicode normalization, removes BOM and
        zero-width characters, normalizes line endings, strips
        trailing whitespace per line, and ensures a single
        trailing newline.

        Parameters
        ----------
        text : str
            Raw text to canonicalize.

        Returns
        -------
        str
            Canonical form suitable for reproducible hashing.

        Examples
        --------
        >>> SRTHandler.canonicalize("Hello ") == SRTHandler.canonicalize("Hello")
        True

        >>> SRTHandler.canonicalize("Hello").endswith("\\n")
        True
        """
        text = unicodedata.normalize("NFC", text)
        text = text.lstrip("\ufeff") # Remove BOM
        for ch in ["\u200b", "\u200c", "\u200d", "\xa0"]:
            text = text.replace(ch, "")
        lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
        return "\n".join(lines).strip() + "\n"

    @classmethod
    def get_hash(cls, text: str) -> str:
        """
        Compute a SHA-256 hash of the canonicalized text.

        Uses ``canonicalize`` first so that cosmetically different
        inputs (trailing spaces, BOM, etc.) produce the same hash.

        Parameters
        ----------
        text : str
            Text to hash.

        Returns
        -------
        str
            64-character lowercase hexadecimal SHA-256 digest.

        Examples
        --------
        >>> len(SRTHandler.get_hash("Hello"))
        64

        >>> SRTHandler.get_hash("Hello") == SRTHandler.get_hash("Hello ")
        True
        """
        return hashlib.sha256(cls.canonicalize(text).encode("utf-8")).hexdigest()

    @classmethod
    def extract_timestamps(cls, path: Path) -> list[str]:
        """
        Extract all valid timestamp lines from an SRT file.

        Reads the file at *path* and returns every line matching the
        SRT timestamp pattern ``HH:MM:SS,mmm --> HH:MM:SS,mmm``.
        Used for structural comparison between source and translated
        files (skip-logic in ``BaseTranslator``).

        Parameters
        ----------
        path : Path
            Path to the ``.srt`` file.

        Returns
        -------
        list of str
            Stripped timestamp lines, in file order.
            Empty list if the file does not exist.
        """
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if "-->" in line and cls.TIMESTAMP_RE.fullmatch(line.strip())]

    @staticmethod
    def parse_to_blocks(content: str) -> list:
        """
        Parse raw SRT content into a list of block dicts.

        Applies aggressive cleaning of LLM artifacts (fenced code
        blocks, wrapping quotes) during parsing.  Each block dict
        has keys ``index`` (int), ``start`` (str), ``end`` (str),
        and ``text`` (list of str).

        Parameters
        ----------
        content : str
            Raw SRT content (may contain LLM markdown artifacts).

        Returns
        -------
        list of dict
            Parsed blocks in file order.

        Examples
        --------
        >>> blocks = SRTHandler.parse_to_blocks(
        ...     "1\\n00:00:01,000 --> 00:00:03,000\\nHello world\\n")
        >>> blocks[0]['index']
        1
        >>> blocks[0]['text']
        ['Hello world']
        """
        content = re.sub(r"```[a-z]*", "", content)
        content = content.replace("```", "")

        blocks = []
        current = {"index": None, "start": None, "end": None, "text": []}

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue

            if re.match(r"^[0-9]+$", line):
                if current["index"] is not None and current["start"] is not None:
                    blocks.append(current)
                current = {"index": int(line), "start": None, "end": None, "text": []}

            elif "-->" in line:
                times = [x.strip() for x in line.split("-->")]
                if len(times) == 2:
                    current["start"], current["end"] = times[0], times[1]

            else:
                if current["start"] is not None:
                    # Only strip full-line wrapping quotes (LLM artifact),
                    # preserve legitimate brackets like [Music] and internal quotes
                    clean_line = line.strip()
                    if len(clean_line) >= 2 and (
                        (clean_line[0] == '"' and clean_line[-1] == '"')
                        or (clean_line[0] == "'" and clean_line[-1] == "'")
                    ):
                        clean_line = clean_line[1:-1].strip()
                    if clean_line:
                        current["text"].append(clean_line)
                else:
                    pass

        # Final check
        if current["index"] is not None and current["start"] is not None:
            blocks.append(current)

        return blocks

    @staticmethod
    def merge_identical_blocks(blocks: list) -> list:
        """
        Merge consecutive blocks that share identical text.

        When two adjacent blocks have the same text content, they are
        collapsed into a single block spanning from the first block's
        ``start`` to the last block's ``end``.

        Parameters
        ----------
        blocks : list of dict
            Parsed SRT blocks (each with ``start``, ``end``, ``text``).

        Returns
        -------
        list of dict
            Merged blocks with ``text`` stored as a plain string.

        Examples
        --------
        >>> blocks = [
        ...     {"start": "00:00:01,000", "end": "00:00:02,000", "text": ["Hi"]},
        ...     {"start": "00:00:02,000", "end": "00:00:03,000", "text": ["Hi"]},
        ...     {"start": "00:00:03,000", "end": "00:00:04,000", "text": ["Bye"]},
        ... ]
        >>> merged = SRTHandler.merge_identical_blocks(blocks)
        >>> len(merged)
        2
        >>> merged[0]['end']
        '00:00:03,000'
        """
        merged = []
        prev = None

        for b in blocks:
            text = "\n".join(b["text"]).strip() if isinstance(b["text"], list) else b["text"]
            if prev and text == prev["text"]:
                prev["end"] = b["end"]
            else:
                prev = {"start": b["start"], "end": b["end"], "text": text}
                merged.append(prev)
        return merged

    @staticmethod
    def render_blocks(blocks: list) -> str:
        """
        Convert a list of block dicts into a valid SRT string.

        Blocks are re-indexed sequentially starting from 1,
        regardless of their original ``index`` values.

        Parameters
        ----------
        blocks : list of dict
            Blocks with ``start``, ``end``, and ``text`` keys.
            ``text`` may be a list of str or a plain str.

        Returns
        -------
        str
            Formatted SRT content ready to be written to a file.

        Examples
        --------
        >>> blocks = [{"start": "00:00:01,000", "end": "00:00:03,000",
        ...            "text": ["Hello world"]}]
        >>> print(SRTHandler.render_blocks(blocks))
        1
        00:00:01,000 --> 00:00:03,000
        Hello world
        <BLANKLINE>
        """
        out = []
        for i, b in enumerate(blocks, start=1):
            if isinstance(b['text'], list):
                text_content = "\n".join(b['text']).strip()
            else:
                text_content = str(b['text']).strip()

            out.append(f"{i}")
            out.append(f"{b['start']} --> {b['end']}")
            out.append(text_content)
            out.append("")

        return "\n".join(out)

    @classmethod
    def standardize(cls, content: str) -> str:
        """
        Apply the full SRT post-processing pipeline.

        Sequentially: parse → merge identical consecutive blocks →
        remove empty / malformed blocks → clean LLM artifacts →
        re-index → render.

        Parameters
        ----------
        content : str
            Raw SRT content (possibly produced by an LLM).

        Returns
        -------
        str
            Clean, well-formed SRT string.

        Examples
        --------
        >>> raw = "1\\n00:00:01,000 --> 00:00:02,000\\n**Hello**\\n"
        >>> "**" in SRTHandler.standardize(raw)
        False
        """
        # 1. Parse raw string to blocks
        blocks = cls.parse_to_blocks(content)

        # 2. Merge consecutive blocks with same text
        merged = cls.merge_identical_blocks(blocks)

        # 3. Remove blocks with no text or invalid data, apply clean_text
        cleaned = []
        for b in merged:
            text_content = "".join(b['text']).strip() if isinstance(b['text'], list) else str(b['text']).strip()
            if b.get('start') and b.get('end') and text_content:
                # Clean LLM artifacts (bold markers, special chars, extra whitespace)
                b['text'] = cls.clean_text(text_content)
                cleaned.append(b)

        # 4. Render back to string with fresh indexing (1, 2, 3...)
        return cls.render_blocks(cleaned)

    @staticmethod
    def timestamp_to_seconds(ts: str) -> float:
        """
        Convert an SRT timestamp to total seconds.

        Parameters
        ----------
        ts : str
            Timestamp in ``HH:MM:SS,mmm`` format.

        Returns
        -------
        float
            Equivalent time in fractional seconds.

        Examples
        --------
        >>> SRTHandler.timestamp_to_seconds("00:01:30,500")
        90.5

        >>> SRTHandler.timestamp_to_seconds("01:00:00,000")
        3600.0
        """
        h, m, s_ms = ts.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    @classmethod
    def get_blocks_in_range(cls, blocks: list, start_sec: float, end_sec: float, margin: float = 0.05) -> list:
        """
        Filter blocks that overlap with a specific time range.

        Uses overlap logic (not strict containment) so that blocks
        whose timestamps were shifted by standardize/merge across
        different translation streams (S1, L1, Mt) are still captured.

        Parameters
        ----------
        blocks : list of dict
            Parsed SRT blocks with ``start`` and ``end`` timestamps.
        start_sec : float
            Start of the target range in seconds.
        end_sec : float
            End of the target range in seconds.
        margin : float, optional
            Tolerance in seconds added to both boundaries
            (default 0.05, i.e. 50 ms).

        Returns
        -------
        list of dict
            Blocks whose time span overlaps ``[start_sec, end_sec]``
            (with *margin* tolerance).

        Examples
        --------
        >>> blocks = [
        ...     {"start": "00:00:01,000", "end": "00:00:03,000", "text": "A"},
        ...     {"start": "00:00:05,000", "end": "00:00:07,000", "text": "B"},
        ... ]
        >>> result = SRTHandler.get_blocks_in_range(blocks, 0.0, 4.0)
        >>> len(result)
        1
        >>> result[0]['text']
        'A'
        """
        return [
            b for b in blocks
            if cls.timestamp_to_seconds(b['start']) <= end_sec + margin
            and cls.timestamp_to_seconds(b['end']) >= start_sec - margin
        ]
