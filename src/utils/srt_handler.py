import re
import hashlib
import unicodedata
import logging
from pathlib import Path
from datetime import timedelta

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
    def apply_offset_to_blocks(cls, blocks: list, offset_seconds: int) -> list:
        """Applies a time offset to a list of parsed SRT blocks."""
        if offset_seconds == 0:
            return blocks
            
        valid_blocks = []
        for b in blocks:
            # SAFETY CHECK: Skip blocks that are missing timestamps
            if b.get('start') is None or b.get('end') is None:
                logger.warning(f"Skipping malformed block: {b}")
                continue
                
            b['start'] = cls.shift_timestamp(b['start'], offset_seconds)
            b['end'] = cls.shift_timestamp(b['end'], offset_seconds)
            valid_blocks.append(b)
        return valid_blocks

    @staticmethod
    def clean_text(text: str) -> str:
        replacements = {"**": "", "□": "-", "▪": "-", "…": "..."}
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return " ".join(text.split()).strip()

    @staticmethod
    def canonicalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.lstrip("\ufeff") # Remove BOM
        for ch in ["\u200b", "\u200c", "\u200d", "\xa0"]:
            text = text.replace(ch, "")
        lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
        return "\n".join(lines).strip() + "\n"

    @classmethod
    def get_hash(cls, text: str) -> str:
        return hashlib.sha256(cls.canonicalize(text).encode("utf-8")).hexdigest()

    @classmethod
    def extract_timestamps(cls, path: Path) -> list:
        if not path.exists(): return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if "-->" in line and cls.TIMESTAMP_RE.fullmatch(line.strip())]

    @classmethod
    def normalize_file(cls, path_in: Path, path_out: Path):
        """Rebuilds the SRT to ensure perfect 1, 2, 3 numbering and clean spacing."""
        with open(path_in, "r", encoding="utf-8") as f:
            content = f.read()
        
        blocks = []
        current_ts = None
        current_text = []
        
        for line in content.splitlines():
            line = line.strip()
            if not line: continue
            if "-->" in line and cls.TIMESTAMP_RE.fullmatch(line):
                if current_ts and current_text:
                    blocks.append((current_ts, " ".join(current_text)))
                current_ts, current_text = line, []
            elif not line.isdigit():
                current_text.append(line)
        
        if current_ts and current_text:
            blocks.append((current_ts, " ".join(current_text)))

        output = [f"{i}\n{ts}\n{txt}\n" for i, (ts, txt) in enumerate(blocks, 1)]
        final_text = "\n".join(output)

        with open(path_out, "w", encoding="utf-8") as f:
            f.write(final_text)
            
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

            if re.match(r"^\d+$", line):
                if current["index"] is not None and current["start"] is not None:
                    blocks.append(current)
                current = {"index": int(line), "start": None, "end": None, "text": []}

            elif "-->" in line:
                times = [x.strip() for x in line.split("-->")]
                if len(times) == 2:
                    current["start"], current["end"] = times[0], times[1]

            else:
                if current["start"] is not None:
                    clean_line = re.sub(r"^['\"\[\s]+|['\"\]\s]+$", "", line)
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
        
        # 3. Remove blocks with no text or invalid data
        cleaned = []
        for b in merged:
            text_content = "".join(b['text']).strip() if isinstance(b['text'], list) else str(b['text']).strip()
            if b.get('start') and b.get('end') and text_content:
                # Ensure text is stored as a clean string for final rendering
                b['text'] = text_content
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