import hashlib
import logging
import re
import unicodedata
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class SRTHandler:
    TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}")

    @staticmethod
    def shift_timestamp(ts_str: str, offset_seconds: int) -> str:
        """
        Adds offset_seconds to an SRT timestamp string (00:00:00,000).
        Used to realign segmented audio transcripts.
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
        """Applies a time offset to a list of parsed SRT blocks (returns new copies)."""
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
        """Removes LLM artifacts (bold markers, special chars, extra whitespace)."""
        replacements = {"**": "", "□": "-", "▪": "-", "…": "..."}
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return " ".join(text.split()).strip()

    @staticmethod
    def canonicalize(text: str) -> str:
        """Normalizes text for deterministic hashing (NFC, strip BOM/zero-width)."""
        text = unicodedata.normalize("NFC", text)
        text = text.lstrip("\ufeff") # Remove BOM
        for ch in ["\u200b", "\u200c", "\u200d", "\xa0"]:
            text = text.replace(ch, "")
        lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
        return "\n".join(lines).strip() + "\n"

    @classmethod
    def get_hash(cls, text: str) -> str:
        """Returns a SHA-256 hash of the canonicalized text."""
        return hashlib.sha256(cls.canonicalize(text).encode("utf-8")).hexdigest()

    @classmethod
    def extract_timestamps(cls, path: Path) -> list[str]:
        """Extracts all valid timestamp lines from an SRT file."""
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if "-->" in line and cls.TIMESTAMP_RE.fullmatch(line.strip())]

    @staticmethod
    def parse_to_blocks(content: str) -> list:
        """Parses SRT content with aggressive cleaning of LLM artifacts."""
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
        """Merges consecutive blocks if the text is identical."""
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
        """Converts blocks back into a valid SRT string."""
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
        Full post-processing pipeline:
        Parse -> Merge Identical -> Clean Empty -> Re-index -> Render
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
        """Convert SRT timestamp to total seconds."""
        h, m, s_ms = ts.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    @classmethod
    def get_blocks_in_range(cls, blocks: list, start_sec: float, end_sec: float, margin: float = 0.05) -> list:
        """Filters blocks that overlap with a specific time range.

        Uses overlap logic (not strict containment) so that blocks whose
        timestamps were shifted by standardize/merge across different
        translation streams (S1, L1, Mt) are still captured.
        A small margin (default 50ms) provides additional tolerance.
        """
        return [
            b for b in blocks
            if cls.timestamp_to_seconds(b['start']) <= end_sec + margin
            and cls.timestamp_to_seconds(b['end']) >= start_sec - margin
        ]
